#!/usr/bin/env python3
import os
import time
import random
import requests
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from socks import ProxyConnectionError
from requests.exceptions import ConnectionError as ReqConnError, HTTPError
import gc
import sys
import traceback
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
import threading

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
DATABASE_URL         = os.getenv("DATABASE_URL")
CREATORS            = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS              = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ     = 100
env_batch           = int(os.getenv("BATCH_SIZE","50"))
BATCH_SIZE          = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY    = float(os.getenv("RATE_LIMIT_DELAY","0.5"))  # Reduced for multi-threading
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

# Multi-threading configuration
MAX_WORKERS         = int(os.getenv("MAX_WORKERS", "8"))  # Number of concurrent threads
IMAGE_WORKERS       = int(os.getenv("IMAGE_WORKERS", "4"))  # Separate pool for image downloads
API_WORKERS         = int(os.getenv("API_WORKERS", "6"))    # Separate pool for API calls

# Existing configuration
CHECKPOINT_FREQUENCY = int(os.getenv("CHECKPOINT_FREQUENCY", "100"))
MAX_IMAGE_SIZE_MB   = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT", "false").lower() == "true"
SKIP_IMAGES         = os.getenv("SKIP_IMAGES", "false").lower() == "true"

# Thread-safe counters and locks
progress_lock = Lock()
processed_count = 0
total_count = 0

print(f"[startup] batch size={BATCH_SIZE}, rate_delay={RATE_LIMIT_DELAY}")
print(f"[startup] workers: api={API_WORKERS}, images={IMAGE_WORKERS}, max={MAX_WORKERS}")
print(f"[startup] checkpoint_freq={CHECKPOINT_FREQUENCY}, max_image_mb={MAX_IMAGE_SIZE_MB}")

# ─── Thread-safe Database Helpers ─────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def ensure_tables():
    print("[db] Ensuring tables exist…")
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Create games table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS games (
                  id   BIGINT PRIMARY KEY,
                  name TEXT NOT NULL
                );
            """)
            
            # Create processing_checkpoint table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processing_checkpoint (
                  id            SERIAL PRIMARY KEY,
                  run_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                  processed_games INTEGER NOT NULL,
                  total_games   INTEGER NOT NULL,
                  status        TEXT NOT NULL,
                  last_game_id  BIGINT
                );
            """)
            
            # Check if snapshots table exists and what columns it has
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'snapshots'
            """)
            existing_columns = {row[0] for row in cur.fetchall()}
            
            if not existing_columns:
                # Fresh table - create with BYTEA columns
                print("[db] Creating new snapshots table with BYTEA columns...")
                cur.execute("""
                    CREATE TABLE snapshots (
                      game_id       BIGINT      NOT NULL,
                      snapshot_time TIMESTAMP   NOT NULL DEFAULT NOW(),
                      playing       INTEGER     NOT NULL,
                      visits        BIGINT      NOT NULL,
                      favorites     INTEGER     NOT NULL,
                      likes         INTEGER     NOT NULL,
                      dislikes      INTEGER     NOT NULL,
                      icon_data     BYTEA,
                      thumbnail_data BYTEA,
                      PRIMARY KEY (game_id, snapshot_time)
                    );
                """)
            else:
                # Table exists - check if it needs migration
                has_old_columns = 'icon_url' in existing_columns or 'thumbnail_url' in existing_columns
                has_new_columns = 'icon_data' in existing_columns and 'thumbnail_data' in existing_columns
                
                if has_old_columns and not has_new_columns:
                    print("[db] Migrating snapshots table to BYTEA columns...")
                    
                    # Add new BYTEA columns
                    if 'icon_data' not in existing_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN icon_data BYTEA;")
                    if 'thumbnail_data' not in existing_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN thumbnail_data BYTEA;")
                    
                    # Remove old URL columns
                    if 'icon_url' in existing_columns:
                        cur.execute("ALTER TABLE snapshots DROP COLUMN icon_url;")
                    if 'thumbnail_url' in existing_columns:
                        cur.execute("ALTER TABLE snapshots DROP COLUMN thumbnail_url;")
                    
                    print("[db] Migration completed automatically")
                
                elif not has_new_columns:
                    # Missing BYTEA columns - add them
                    print("[db] Adding missing BYTEA columns...")
                    if 'icon_data' not in existing_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN icon_data BYTEA;")
                    if 'thumbnail_data' not in existing_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN thumbnail_data BYTEA;")
        
        conn.commit()
    print("[db] Tables ready.")

def save_checkpoint(processed: int, total: int, status: str, last_game_id: Optional[int] = None):
    """Thread-safe checkpoint saving"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processing_checkpoint (processed_games, total_games, status, last_game_id)
                    VALUES (%s, %s, %s, %s)
                """, (processed, total, status, last_game_id))
            conn.commit()
        print(f"[checkpoint] Saved: {processed}/{total} games, status={status}")
    except Exception as e:
        print(f"[checkpoint] Failed to save: {e!r}")

def save_single_snapshot(snap: Tuple):
    """Save a single snapshot - thread safe"""
    sql = """
    INSERT INTO snapshots
      (game_id, playing, visits, favorites, likes, dislikes, icon_data, thumbnail_data)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (game_id, snapshot_time) DO NOTHING
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, snap)
            conn.commit()
        return True
    except Exception as e:
        print(f"[db] Error saving snapshot for game {snap[0]}: {e!r}")
        return False

def update_progress():
    """Thread-safe progress tracking"""
    global processed_count
    with progress_lock:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"[progress] {processed_count}/{total_count} games completed ({processed_count/total_count*100:.1f}%)")
        
        if processed_count % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(processed_count, total_count, "processing", None)

# ─── Thread-safe HTTP Helpers ─────────────────────────────────────────────────
def fetch_proxies_from_api():
    if not PROXY_API_KEY:
        return []
    print("[proxy] Fetching proxy list from API…")
    try:
        resp = requests.get(f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[proxy] API error: {e!r}")
        return []
    proxies = []
    for e in data.get("ipv4", []):
        ip, auth = e["ip"], e["authInfo"]
        for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
            if port := e.get(port_key):
                proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    print(f"[proxy] Retrieved {len(proxies)} proxies from API.")
    return proxies

PROXIES = fetch_proxies_from_api() or PROXY_URLS_FALLBACK
proxy_lock = Lock()
print(f"[proxy] Using {len(PROXIES)} proxies total.")

# Thread-local storage for cookies
thread_local = threading.local()

def get_cookie():
    """Thread-safe cookie rotation"""
    if not hasattr(thread_local, 'cookie_idx'):
        thread_local.cookie_idx = random.randint(0, len(TOKENS)-1) if TOKENS else 0
    
    if not TOKENS:
        return {}
    
    tok = TOKENS[thread_local.cookie_idx % len(TOKENS)]
    thread_local.cookie_idx += 1
    return {".ROBLOSECURITY": tok}

def get_user_agent():
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36/Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36/Chrome/126.0.0.0 Safari/537.36",
    ])

def get_session():
    """Thread-safe session with proxy rotation"""
    s = requests.Session()
    if PROXIES:
        with proxy_lock:
            p = random.choice(PROXIES)
        s.proxies.update({"http":p, "https":p})
    return s

def safe_get(url, retries=3):
    """Thread-safe HTTP GET"""
    last_err = None
    for attempt in range(1, retries+1):
        # Randomized delay to spread out requests
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.5)
        sess = get_session()
        
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except (ProxyConnectionError, ReqConnError) as e:
            proxy = sess.proxies.get("https") or sess.proxies.get("http") or "direct"
            if proxy != "direct" and proxy in PROXIES:
                with proxy_lock:
                    if proxy in PROXIES:
                        PROXIES.remove(proxy)
        except Exception as e:
            last_err = e
        
        if attempt < retries:
            time.sleep((2 ** attempt) + random.random())
    
    raise RuntimeError(f"GET failed after {retries} attempts: {url!r}\nLast error: {last_err!r}")

def resize_image_to_limit(image_data, max_size_bytes):
    """Resize image by reducing resolution until it fits within size limit"""
    try:
        from PIL import Image
        from io import BytesIO
        
        img = Image.open(BytesIO(image_data))
        original_size = img.size
        
        resize_attempts = [
            (1.0, 85), (1.0, 70), (1.0, 50),
            (0.8, 85), (0.8, 70),
            (0.6, 85), (0.6, 70),
            (0.5, 85), (0.5, 70),
            (0.4, 85), (0.3, 85), (0.2, 85),
        ]
        
        for scale, quality in resize_attempts:
            try:
                new_width = int(original_size[0] * scale)
                new_height = int(original_size[1] * scale)
                
                if new_width < 50 or new_height < 50:
                    continue
                
                if scale < 1.0:
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    resized_img = img
                
                output = BytesIO()
                
                if img.format == 'PNG' and quality < 100:
                    if resized_img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", resized_img.size, (255, 255, 255))
                        if resized_img.mode == "P":
                            resized_img = resized_img.convert("RGBA")
                        background.paste(resized_img, mask=resized_img.split()[-1] if resized_img.mode == "RGBA" else None)
                        resized_img = background
                    resized_img.save(output, format='JPEG', quality=quality, optimize=True)
                elif img.format == 'JPEG' or quality < 100:
                    resized_img.save(output, format='JPEG', quality=quality, optimize=True)
                else:
                    resized_img.save(output, format=img.format or 'PNG', optimize=True)
                
                output_data = output.getvalue()
                
                if len(output_data) <= max_size_bytes:
                    return output_data
                
            except Exception:
                continue
        
        return None
        
    except ImportError:
        print("[resize_image] ⚠ PIL not available, cannot resize images")
        return None
    except Exception as e:
        print(f"[resize_image] ⚠ Resize failed: {e!r}")
        return None

def download_image(url, retries=2):
    """Thread-safe image download with auto-resize"""
    if SKIP_IMAGES:
        return None
        
    max_size = MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    for attempt in range(1, retries+1):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent()},
                cookies=get_cookie(),
                timeout=30,
                stream=True
            )
            r.raise_for_status()
            
            image_data = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    image_data += chunk
            
            if len(image_data) <= max_size:
                return image_data
            
            # Try to resize if too large
            resized_data = resize_image_to_limit(image_data, max_size)
            return resized_data
            
        except Exception as e:
            if attempt == retries:
                print(f"[download_image] Failed: {e!r}")
            
        if attempt < retries:
            time.sleep(1)
    
    return None

# ─── Multi-threaded Processing Functions ──────────────────────────────────────
def process_single_game(game_data):
    """Process a single game - this runs in parallel"""
    game_id, game_meta, votes_dict, icons_dict, thumbs_dict = game_data
    
    try:
        uid = str(game_id)
        icon_data = thumb_data = None
        
        # Download images concurrently if available
        if not SKIP_IMAGES:
            icon_futures = []
            thumb_futures = []
            
            with ThreadPoolExecutor(max_workers=2) as img_executor:
                if icon_url := icons_dict.get(uid):
                    icon_futures.append(img_executor.submit(download_image, icon_url))
                
                if thumb_url := thumbs_dict.get(uid):
                    thumb_futures.append(img_executor.submit(download_image, thumb_url))
                
                # Get results
                if icon_futures:
                    icon_data = icon_futures[0].result()
                if thumb_futures:
                    thumb_data = thumb_futures[0].result()
        
        # Create snapshot
        snap = (
            int(uid),
            game_meta.get("playing", 0),
            game_meta.get("visits", 0),
            game_meta.get("favoritedCount", 0),
            votes_dict.get(uid, {}).get("upVotes", 0),
            votes_dict.get(uid, {}).get("downVotes", 0),
            icon_data,
            thumb_data,
        )
        
        # Save to database
        success = save_single_snapshot(snap)
        if success:
            update_progress()
        
        return success
        
    except Exception as e:
        print(f"[process_game] Error processing game {game_id}: {e!r}")
        return False

def fetch_batch_data(universe_ids):
    """Fetch all data for a batch of universe IDs using concurrent requests"""
    print(f"[fetch_batch] Processing {len(universe_ids)} games with {API_WORKERS} workers")
    
    # Prepare API calls
    api_calls = [
        ("details", universe_ids),
        ("votes", universe_ids),
        ("icons", universe_ids),
        ("thumbnails", universe_ids),
    ]
    
    results = {}
    
    def fetch_api_type(call_data):
        api_type, ids = call_data
        try:
            if api_type == "details":
                return api_type, get_game_details_concurrent(ids)
            elif api_type == "votes":
                return api_type, get_game_votes_concurrent(ids)
            elif api_type == "icons":
                return api_type, fetch_icons_concurrent(ids)
            elif api_type == "thumbnails":
                return api_type, fetch_thumbnails_concurrent(ids)
        except Exception as e:
            print(f"[fetch_batch] Error fetching {api_type}: {e!r}")
            return api_type, {} if api_type in ["votes", "icons", "thumbnails"] else []
    
    # Execute API calls concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_type = {executor.submit(fetch_api_type, call): call[0] for call in api_calls}
        
        for future in as_completed(future_to_type):
            api_type, data = future.result()
            results[api_type] = data
    
    return results

def get_game_details_concurrent(universe_ids):
    """Concurrent version of get_game_details"""
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://games.roblox.com/v1/games?universeIds={s}"
        try:
            return safe_get(url).get("data", [])
        except Exception:
            if len(ids) == 1:
                return []
            m = len(ids) // 2
            return fetch_chunk(ids[:m]) + fetch_chunk(ids[m:])
    
    chunks = [universe_ids[i:i+BATCH_SIZE] for i in range(0, len(universe_ids), BATCH_SIZE)]
    
    with ThreadPoolExecutor(max_workers=min(len(chunks), API_WORKERS)) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        results = []
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"[get_game_details_concurrent] Chunk failed: {e!r}")
    
    return results

def get_game_votes_concurrent(universe_ids):
    """Concurrent version of get_game_votes"""
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://games.roblox.com/v1/games/votes?universeIds={s}"
        try:
            data = safe_get(url).get("data", [])
            return {str(v["id"]): {"upVotes": v["upVotes"], "downVotes": v["downVotes"]} for v in data}
        except Exception:
            if len(ids) == 1:
                return {}
            m = len(ids) // 2
            a = fetch_chunk(ids[:m])
            b = fetch_chunk(ids[m:])
            a.update(b)
            return a
    
    chunks = [universe_ids[i:i+BATCH_SIZE] for i in range(0, len(universe_ids), BATCH_SIZE)]
    
    votes = {}
    with ThreadPoolExecutor(max_workers=min(len(chunks), API_WORKERS)) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                votes.update(future.result())
            except Exception as e:
                print(f"[get_game_votes_concurrent] Chunk failed: {e!r}")
    
    return votes

def fetch_icons_concurrent(universe_ids):
    """Concurrent version of fetch_icons"""
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://thumbnails.roblox.com/v1/games/icons?universeIds={s}&size=512x512&format=Png"
        try:
            data = safe_get(url).get("data", [])
            return {str(e["targetId"]): e["imageUrl"] for e in data}
        except Exception:
            return {}
    
    chunks = [universe_ids[i:i+BATCH_SIZE] for i in range(0, len(universe_ids), BATCH_SIZE)]
    
    icons = {}
    with ThreadPoolExecutor(max_workers=min(len(chunks), API_WORKERS)) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                icons.update(future.result())
            except Exception as e:
                print(f"[fetch_icons_concurrent] Chunk failed: {e!r}")
    
    return icons

def fetch_thumbnails_concurrent(universe_ids):
    """Concurrent version of fetch_thumbnails"""
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://thumbnails.roblox.com/v1/games/multiget/thumbnails?universeIds={s}&size=768x432&format=Png"
        try:
            response = safe_get(url)
            data = response.get("data", [])
            thumbs = {}
            
            for game_data in data:
                universe_id = str(game_data.get("universeId", ""))
                thumbnails_list = game_data.get("thumbnails", [])
                
                if thumbnails_list and len(thumbnails_list) > 0:
                    first_thumbnail = thumbnails_list[0]
                    if first_thumbnail.get("state") == "Completed":
                        if image_url := first_thumbnail.get("imageUrl", ""):
                            thumbs[universe_id] = image_url
            
            return thumbs
        except Exception:
            return {}
    
    chunks = [universe_ids[i:i+BATCH_SIZE] for i in range(0, len(universe_ids), BATCH_SIZE)]
    
    thumbs = {}
    with ThreadPoolExecutor(max_workers=min(len(chunks), API_WORKERS)) as executor:
        futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                thumbs.update(future.result())
            except Exception as e:
                print(f"[fetch_thumbnails_concurrent] Chunk failed: {e!r}")
    
    return thumbs

# ─── Legacy Functions (for collecting initial data) ───────────────────────────
def fetch_creator_games(user_id):
    print(f"[fetch_creator_games] user={user_id}")
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/users/{user_id}/games"
    page_count = 0
    
    while True:
        page_count += 1
        if page_count > 100:
            break
            
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[fetch_creator_games] Failed: {e!r}")
            break
            
        chunk = data.get("data", [])
        for it in chunk:
            games.append({"universeId": str(it["id"]), "name": it.get("name","")})
        
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    
    return games

def fetch_user_groups(user_id):
    try:
        data = safe_get(f"https://groups.roblox.com/v2/users/{user_id}/groups/roles")
        return [str(g["group"]["id"]) for g in data.get("data",[]) if "group" in g]
    except Exception as e:
        print(f"[fetch_user_groups] Failed: {e!r}")
        return []

def fetch_group_games(group_id):
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/groups/{group_id}/games"
    page_count = 0
    
    while True:
        page_count += 1
        if page_count > 100:
            break
            
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            break
            
        chunk = data.get("data",[])
        for it in chunk:
            games.append({
                "universeId": str(it.get("id") or it.get("universeId")),
                "name": it.get("name","")
            })
        
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    
    return games

def upsert_games(games: List[Tuple[int, str]]):
    if not games:
        return
    
    chunk_size = 100
    for i in range(0, len(games), chunk_size):
        chunk = games[i:i+chunk_size]
        sql = """
        INSERT INTO games(id,name) VALUES %s
          ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, chunk)
                conn.commit()
        except Exception as e:
            print(f"[db] Error upserting games: {e!r}")

# ─── Main Multi-threaded Processing ───────────────────────────────────────────
def scrape_and_snapshot():
    global total_count, processed_count
    
    print("[main] Starting multi-threaded scrape run")
    
    try:
        # Collect all game IDs (single-threaded for simplicity)
        all_ids = set()
        master_games = []

        for uid in CREATORS:
            print(f"[main] Fetching games for creator ID: {uid}")
            try:
                own = fetch_creator_games(uid)
                groups = fetch_user_groups(uid)
                grp = []
                for g in groups:
                    grp.extend(fetch_group_games(g))
                
                for g in own + grp:
                    if g["universeId"] and g["universeId"] != "0":
                        all_ids.add(g["universeId"])
                        master_games.append((int(g["universeId"]), g["name"][:255]))
            except Exception as e:
                print(f"[main] Error fetching games for creator {uid}: {e!r}")

        print(f"[main] Found {len(master_games)} entries ({len(all_ids)} unique)")
        all_list = list(all_ids)
        
        if not all_list:
            print("[main] No games found; exiting.")
            return

        total_count = len(all_list)
        processed_count = 0
        save_checkpoint(0, total_count, "starting", None)

        # Upsert games
        unique_map = {gid: name for gid, name in master_games}
        deduped = list(unique_map.items())
        upsert_games(deduped)

        # Process in concurrent chunks
        chunk_size = 500  # Larger chunks for multi-threading
        
        for chunk_start in range(0, len(all_list), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_list))
            chunk_ids = all_list[chunk_start:chunk_end]
            
            print(f"[main] Processing chunk {chunk_start//chunk_size+1}: games {chunk_start+1}-{chunk_end}")
            
            try:
                # Fetch all data for this chunk concurrently
                batch_data = fetch_batch_data(chunk_ids)
                
                meta = batch_data.get("details", [])
                votes = batch_data.get("votes", {})
                icons = batch_data.get("icons", {})
                thumbs = batch_data.get("thumbnails", {})
                
                # Create lookup for meta data
                meta_dict = {str(g.get("universeId") or g.get("id")): g for g in meta}
                
                # Prepare game data for parallel processing
                game_tasks = []
                for game_id in chunk_ids:
                    if game_id in meta_dict:
                        game_tasks.append((
                            game_id,
                            meta_dict[game_id],
                            votes,
                            icons,
                            thumbs
                        ))
                
                # Process games in parallel
                print(f"[main] Processing {len(game_tasks)} games with {MAX_WORKERS} workers")
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(process_single_game, task) for task in game_tasks]
                    
                    # Wait for all to complete
                    completed = 0
                    for future in as_completed(futures):
                        try:
                            success = future.result()
                            completed += 1
                            if completed % 50 == 0:
                                print(f"[main] Chunk progress: {completed}/{len(game_tasks)} games")
                        except Exception as e:
                            print(f"[main] Game processing error: {e!r}")
                
                print(f"[main] Completed chunk: {processed_count}/{total_count} total games processed")
                gc.collect()
                
            except Exception as e:
                print(f"[main] Error processing chunk: {e!r}")
                traceback.print_exc()

        save_checkpoint(processed_count, total_count, "completed", None)
        print(f"[main] 🕒 Completed {total_count} games at {datetime.utcnow()}")
        print(f"[main] Successfully processed {processed_count} games with multi-threading")

    except Exception as e:
        print(f"[main] Fatal error: {e!r}")
        traceback.print_exc()
        save_checkpoint(processed_count, total_count, f"fatal_error: {str(e)[:200]}", None)
        raise

def main():
    try:
        ensure_tables()
        scrape_and_snapshot()
        
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user")
        save_checkpoint(processed_count, total_count, "interrupted", None)
    except Exception as e:
        print(f"[main] Unhandled error: {e!r}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
