#!/usr/bin/env python3
"""
Updated scraper with game descriptions support
This includes both the migration logic and the enhanced scraper
"""

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

# ─── Configuration (same as before) ────────────────────────────────────────────
DATABASE_URL         = os.getenv("DATABASE_URL")
CREATORS            = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS              = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ     = 100
env_batch           = int(os.getenv("BATCH_SIZE","50"))
BATCH_SIZE          = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY    = float(os.getenv("RATE_LIMIT_DELAY","0.5"))
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

# Multi-threading configuration
MAX_WORKERS         = int(os.getenv("MAX_WORKERS", "8"))
IMAGE_WORKERS       = int(os.getenv("IMAGE_WORKERS", "4"))
API_WORKERS         = int(os.getenv("API_WORKERS", "6"))

# Other configuration
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

# ─── Enhanced Database Helpers with Description Support ───────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def ensure_tables():
    print("[db] Ensuring tables exist with description support…")
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Check existing schema
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'games'
            """)
            games_columns = {row[0]: row[1] for row in cur.fetchall()}
            
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'snapshots'
            """)
            snapshots_columns = {row[0]: row[1] for row in cur.fetchall()}
            
            # Create games table with description
            if not games_columns:
                print("[db] Creating new games table with description...")
                cur.execute("""
                    CREATE TABLE games (
                      id          BIGINT PRIMARY KEY,
                      name        TEXT NOT NULL,
                      description TEXT,
                      created_at  TIMESTAMP DEFAULT NOW(),
                      updated_at  TIMESTAMP DEFAULT NOW()
                    );
                """)
            else:
                # Add description column if missing
                if 'description' not in games_columns:
                    print("[db] Adding description column to games table...")
                    cur.execute("ALTER TABLE games ADD COLUMN description TEXT;")
                
                if 'created_at' not in games_columns:
                    cur.execute("ALTER TABLE games ADD COLUMN created_at TIMESTAMP DEFAULT NOW();")
                
                if 'updated_at' not in games_columns:
                    cur.execute("ALTER TABLE games ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();")
            
            # Handle snapshots table (same as before)
            if not snapshots_columns:
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
                # Migration logic for snapshots (URL -> BYTEA)
                has_old_columns = 'icon_url' in snapshots_columns or 'thumbnail_url' in snapshots_columns
                has_new_columns = 'icon_data' in snapshots_columns and 'thumbnail_data' in snapshots_columns
                
                if has_old_columns and not has_new_columns:
                    print("[db] Migrating snapshots table to BYTEA columns...")
                    if 'icon_data' not in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN icon_data BYTEA;")
                    if 'thumbnail_data' not in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN thumbnail_data BYTEA;")
                    if 'icon_url' in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots DROP COLUMN icon_url;")
                    if 'thumbnail_url' in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots DROP COLUMN thumbnail_url;")
                    print("[db] Snapshots migration completed")
                
                elif not has_new_columns:
                    if 'icon_data' not in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN icon_data BYTEA;")
                    if 'thumbnail_data' not in snapshots_columns:
                        cur.execute("ALTER TABLE snapshots ADD COLUMN thumbnail_data BYTEA;")
            
            # Create processing checkpoint table
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
            
            # Create indexes for better performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_name ON games(name);
                CREATE INDEX IF NOT EXISTS idx_games_updated_at ON games(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_snapshots_game_time ON snapshots(game_id, snapshot_time DESC);
                CREATE INDEX IF NOT EXISTS idx_snapshots_recent ON snapshots(snapshot_time DESC);
            """)
        
        conn.commit()
    print("[db] Tables ready with description support.")

def upsert_games_with_description(games: List[Tuple[int, str, str]]):
    """Upsert games with name and description"""
    print(f"[db] Upserting {len(games)} games with descriptions…")
    if not games:
        return
    
    chunk_size = 100
    for i in range(0, len(games), chunk_size):
        chunk = games[i:i+chunk_size]
        sql = """
        INSERT INTO games(id, name, description, updated_at) VALUES %s
          ON CONFLICT(id) DO UPDATE SET 
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            updated_at = EXCLUDED.updated_at
        """
        try:
            # Prepare data with current timestamp
            chunk_with_timestamp = [(game_id, name, desc, datetime.utcnow()) for game_id, name, desc in chunk]
            
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, chunk_with_timestamp)
                conn.commit()
        except Exception as e:
            print(f"[db] Error upserting games chunk {i//chunk_size+1}: {e!r}")
            # Try individual inserts
            for game_id, name, desc in chunk:
                try:
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO games(id, name, description, updated_at) VALUES (%s, %s, %s, %s)
                                ON CONFLICT(id) DO UPDATE SET 
                                  name = EXCLUDED.name,
                                  description = EXCLUDED.description,
                                  updated_at = EXCLUDED.updated_at
                            """, (game_id, name, desc, datetime.utcnow()))
                        conn.commit()
                except Exception as e2:
                    print(f"[db] Failed to insert individual game {game_id}: {e2!r}")
    print("[db] Games upsert with descriptions complete.")

# ─── HTTP Helpers (same as before) ─────────────────────────────────────────────
def fetch_proxies_from_api():
    if not PROXY_API_KEY:
        return []
    try:
        resp = requests.get(f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        proxies = []
        for e in data.get("ipv4", []):
            ip, auth = e["ip"], e["authInfo"]
            for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
                if port := e.get(port_key):
                    proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
        return proxies
    except Exception as e:
        print(f"[proxy] API error: {e!r}")
        return []

PROXIES = fetch_proxies_from_api() or PROXY_URLS_FALLBACK
proxy_lock = Lock()
thread_local = threading.local()

def get_cookie():
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
    s = requests.Session()
    if PROXIES:
        with proxy_lock:
            p = random.choice(PROXIES)
        s.proxies.update({"http":p, "https":p})
    return s

def safe_get(url, retries=3):
    last_err = None
    for attempt in range(1, retries+1):
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

# ─── Enhanced Game Data Fetching with Descriptions ────────────────────────────
def get_game_details_with_descriptions(universe_ids):
    """Enhanced version that fetches both game details AND descriptions"""
    print(f"[get_game_details] Fetching details + descriptions for {len(universe_ids)} games")
    
    def fetch_basic_details_chunk(ids):
        """Fetch basic game details (playing, visits, etc.)"""
        s = ",".join(ids)
        url = f"https://games.roblox.com/v1/games?universeIds={s}"
        try:
            data = safe_get(url).get("data", [])
            print(f"[get_game_details] Basic details chunk: {len(ids)} → {len(data)} records")
            return data
        except Exception as e:
            print(f"[get_game_details] Basic details chunk failed: {e!r}")
            if len(ids) == 1:
                return []
            m = len(ids) // 2
            return fetch_basic_details_chunk(ids[:m]) + fetch_basic_details_chunk(ids[m:])
    
    def fetch_descriptions_chunk(ids):
        """Fetch game descriptions from the games endpoint"""
        descriptions = {}
        
        # Convert universe IDs to place IDs first (needed for descriptions)
        place_ids = []
        universe_to_place = {}
        
        # Get place IDs from universe IDs
        try:
            s = ",".join(ids)
            url = f"https://games.roblox.com/v1/games?universeIds={s}"
            data = safe_get(url).get("data", [])
            
            for game in data:
                universe_id = str(game.get("universeId") or game.get("id", ""))
                root_place_id = game.get("rootPlaceId")
                if root_place_id:
                    place_ids.append(str(root_place_id))
                    universe_to_place[universe_id] = str(root_place_id)
        except Exception as e:
            print(f"[get_descriptions] Failed to get place IDs: {e!r}")
            return {}
        
        if not place_ids:
            return {}
        
        # Fetch descriptions using place IDs
        try:
            # Split into smaller batches for descriptions
            desc_batch_size = 50
            for i in range(0, len(place_ids), desc_batch_size):
                batch_place_ids = place_ids[i:i+desc_batch_size]
                s = ",".join(batch_place_ids)
                url = f"https://games.roblox.com/v1/games/multiget-place-details?placeIds={s}"
                
                try:
                    data = safe_get(url)
                    for place_data in data:
                        place_id = str(place_data.get("placeId", ""))
                        description = place_data.get("description", "")
                        
                        # Find corresponding universe ID
                        for universe_id, mapped_place_id in universe_to_place.items():
                            if mapped_place_id == place_id:
                                descriptions[universe_id] = description
                                break
                except Exception as e:
                    print(f"[get_descriptions] Batch failed: {e!r}")
                    continue
                    
        except Exception as e:
            print(f"[get_descriptions] Failed to fetch descriptions: {e!r}")
        
        print(f"[get_descriptions] Got {len(descriptions)} descriptions for {len(ids)} games")
        return descriptions
    
    # Fetch both basic details and descriptions concurrently
    chunks = [universe_ids[i:i+BATCH_SIZE] for i in range(0, len(universe_ids), BATCH_SIZE)]
    
    all_details = []
    all_descriptions = {}
    
    with ThreadPoolExecutor(max_workers=min(len(chunks), API_WORKERS)) as executor:
        # Submit both types of requests
        detail_futures = [executor.submit(fetch_basic_details_chunk, chunk) for chunk in chunks]
        desc_futures = [executor.submit(fetch_descriptions_chunk, chunk) for chunk in chunks]
        
        # Collect basic details
        for future in as_completed(detail_futures):
            try:
                all_details.extend(future.result())
            except Exception as e:
                print(f"[get_game_details] Detail future failed: {e!r}")
        
        # Collect descriptions
        for future in as_completed(desc_futures):
            try:
                all_descriptions.update(future.result())
            except Exception as e:
                print(f"[get_game_details] Description future failed: {e!r}")
    
    # Combine details with descriptions
    for game in all_details:
        universe_id = str(game.get("universeId") or game.get("id", ""))
        game["description"] = all_descriptions.get(universe_id, "")
    
    print(f"[get_game_details] Combined {len(all_details)} games with descriptions")
    return all_details

# ─── Enhanced Game Collection Functions ────────────────────────────────────────
def fetch_creator_games(user_id):
    """Fetch creator games with descriptions"""
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
            games.append({
                "universeId": str(it["id"]), 
                "name": it.get("name", ""),
                "description": it.get("description", "")  # Get description from API if available
            })
        
        cursor = data.get("nextPageCursor", "")
        if not cursor:
            break
    
    print(f"[fetch_creator_games] Got {len(games)} games for user {user_id}")
    return games

def fetch_group_games(group_id):
    """Fetch group games with descriptions"""
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
            
        chunk = data.get("data", [])
        for it in chunk:
            games.append({
                "universeId": str(it.get("id") or it.get("universeId")),
                "name": it.get("name", ""),
                "description": it.get("description", "")
            })
        
        cursor = data.get("nextPageCursor", "")
        if not cursor:
            break
    
    return games

def fetch_user_groups(user_id):
    try:
        data = safe_get(f"https://groups.roblox.com/v2/users/{user_id}/groups/roles")
        return [str(g["group"]["id"]) for g in data.get("data", []) if "group" in g]
    except Exception as e:
        print(f"[fetch_user_groups] Failed: {e!r}")
        return []

# ─── Keep all other functions the same (image processing, votes, icons, etc.) ──
def resize_image_to_limit(image_data, max_size_bytes):
    try:
        from PIL import Image
        from io import BytesIO
        
        img = Image.open(BytesIO(image_data))
        original_size = img.size
        
        resize_attempts = [
            (1.0, 85), (1.0, 70), (1.0, 50),
            (0.8, 85), (0.8, 70), (0.6, 85), (0.6, 70),
            (0.5, 85), (0.5, 70), (0.4, 85), (0.3, 85), (0.2, 85),
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
        print("[resize_image] ⚠ PIL not available")
        return None
    except Exception as e:
        print(f"[resize_image] ⚠ Resize failed: {e!r}")
        return None

def download_image(url, retries=2):
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
            
            resized_data = resize_image_to_limit(image_data, max_size)
            return resized_data
            
        except Exception as e:
            if attempt == retries:
                print(f"[download_image] Failed: {e!r}")
            
        if attempt < retries:
            time.sleep(1)
    
    return None

# ─── Keep other functions the same (votes, icons, thumbnails, etc.) ────────────
def get_game_votes_concurrent(universe_ids):
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

# ─── Keep checkpoint and progress functions the same ───────────────────────────
def save_checkpoint(processed: int, total: int, status: str, last_game_id: Optional[int] = None):
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

def update_progress():
    global processed_count
    with progress_lock:
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"[progress] {processed_count}/{total_count} games completed ({processed_count/total_count*100:.1f}%)")
        if processed_count % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(processed_count, total_count, "processing", None)

def save_single_snapshot(snap: Tuple):
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

def process_single_game(game_data):
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
    
    # Prepare API calls - now using enhanced description fetching
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
                return api_type, get_game_details_with_descriptions(ids)  # Enhanced version
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

# ─── Enhanced Main Processing Function ─────────────────────────────────────────
def scrape_and_snapshot():
    global total_count, processed_count
    
    print("[main] Starting multi-threaded scrape run with description support")
    
    try:
        # Collect all game IDs with descriptions
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
                        # Now includes description
                        master_games.append((
                            int(g["universeId"]), 
                            g["name"][:255],  # Truncate long names
                            g.get("description", "")[:2000]  # Truncate long descriptions
                        ))
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

        # Upsert games with descriptions
        unique_map = {}
        for gid, name, desc in master_games:
            if gid not in unique_map:
                unique_map[gid] = (name, desc)
            else:
                # Keep the longer description if we have duplicates
                existing_desc = unique_map[gid][1]
                if len(desc) > len(existing_desc):
                    unique_map[gid] = (name, desc)
        
        deduped = [(gid, name, desc) for gid, (name, desc) in unique_map.items()]
        print(f"[main] Deduped to {len(deduped)} unique games with descriptions")
        upsert_games_with_description(deduped)

        # Process in concurrent chunks
        chunk_size = 500
        
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
                
                # Also update game descriptions from the detailed metadata
                games_to_update = []
                for game in meta:
                    universe_id = int(game.get("universeId") or game.get("id", 0))
                    if universe_id > 0:
                        name = game.get("name", "")[:255]
                        description = game.get("description", "")[:2000]
                        games_to_update.append((universe_id, name, description))
                
                if games_to_update:
                    print(f"[main] Updating {len(games_to_update)} games with fresh descriptions")
                    upsert_games_with_description(games_to_update)
                
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
        print(f"[main] Successfully processed {processed_count} games with descriptions!")

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
