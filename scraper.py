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

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
DATABASE_URL         = os.getenv("DATABASE_URL")
CREATORS            = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS              = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ     = 100
env_batch           = int(os.getenv("BATCH_SIZE","50"))  # Reduced default batch size
BATCH_SIZE          = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY    = float(os.getenv("RATE_LIMIT_DELAY","1.0"))  # Increased delay
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

# New configuration for robustness
CHECKPOINT_FREQUENCY = int(os.getenv("CHECKPOINT_FREQUENCY", "50"))  # Save progress every N games
MAX_IMAGE_SIZE_MB   = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))     # Skip images larger than this
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT", "false").lower() == "true"
SKIP_IMAGES         = os.getenv("SKIP_IMAGES", "false").lower() == "true"  # Option to skip image downloads

print(f"[startup] batch size={BATCH_SIZE}, rate_delay={RATE_LIMIT_DELAY}")
print(f"[startup] checkpoint_freq={CHECKPOINT_FREQUENCY}, max_image_mb={MAX_IMAGE_SIZE_MB}")
print(f"[startup] resume_from_checkpoint={RESUME_FROM_CHECKPOINT}, skip_images={SKIP_IMAGES}")

# ─── Database Helpers ──────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def ensure_tables():
    print("[db] Ensuring tables exist…")
    ddl = """
    CREATE TABLE IF NOT EXISTS games (
      id   BIGINT PRIMARY KEY,
      name TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS snapshots (
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
    CREATE TABLE IF NOT EXISTS processing_checkpoint (
      id            SERIAL PRIMARY KEY,
      run_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
      processed_games INTEGER NOT NULL,
      total_games   INTEGER NOT NULL,
      status        TEXT NOT NULL,
      last_game_id  BIGINT
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("[db] Tables ready.")

def save_checkpoint(processed: int, total: int, status: str, last_game_id: Optional[int] = None):
    """Save processing checkpoint for resume capability"""
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

def get_last_checkpoint():
    """Get the last checkpoint for resume capability"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT processed_games, total_games, last_game_id 
                    FROM processing_checkpoint 
                    ORDER BY run_timestamp DESC 
                    LIMIT 1
                """)
                row = cur.fetchone()
                return row if row else (0, 0, None)
    except Exception as e:
        print(f"[checkpoint] Failed to get: {e!r}")
        return (0, 0, None)

def upsert_games(games: List[Tuple[int, str]]):
    print(f"[db] Upserting {len(games)} games…")
    if not games:
        return
    
    # Process in smaller chunks to avoid memory issues
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
            print(f"[db] Error upserting games chunk {i//chunk_size+1}: {e!r}")
            # Try individual inserts for this chunk
            for game_id, name in chunk:
                try:
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO games(id,name) VALUES (%s,%s)
                                ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name
                            """, (game_id, name))
                        conn.commit()
                except Exception as e2:
                    print(f"[db] Failed to insert individual game {game_id}: {e2!r}")
    print("[db] Upsert complete.")

def save_snapshots_batch(snaps: List[Tuple]):
    """Save snapshots in smaller batches with error handling"""
    if not snaps:
        return
    
    print(f"[db] Saving {len(snaps)} snapshots in batches…")
    batch_size = 25  # Smaller batches for BYTEA data
    
    for i in range(0, len(snaps), batch_size):
        batch = snaps[i:i+batch_size]
        sql = """
        INSERT INTO snapshots
          (game_id, playing, visits, favorites, likes, dislikes, icon_data, thumbnail_data)
        VALUES %s
        """
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, sql, batch)
                conn.commit()
            print(f"[db] Saved batch {i//batch_size+1}/{(len(snaps)-1)//batch_size+1}")
        except Exception as e:
            print(f"[db] Error saving batch {i//batch_size+1}: {e!r}")
            # Try individual inserts for this batch
            for snap in batch:
                try:
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(sql.replace('%s', '(%s,%s,%s,%s,%s,%s,%s,%s)'), snap)
                        conn.commit()
                except Exception as e2:
                    print(f"[db] Failed to insert individual snapshot for game {snap[0]}: {e2!r}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    print("[db] All snapshots saved.")

def prune_stale(current_ids: List[int]):
    print(f"[db] Pruning stale IDs not in current set of {len(current_ids)} games…")
    if not current_ids:
        return
    
    # Process pruning in chunks to avoid SQL parameter limits
    chunk_size = 1000
    all_current = set(current_ids)
    
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Get all existing game IDs
                cur.execute("SELECT DISTINCT game_id FROM snapshots")
                existing_ids = {row[0] for row in cur.fetchall()}
                
                # Find IDs to delete
                to_delete = existing_ids - all_current
                print(f"[db] Found {len(to_delete)} stale games to prune")
                
                if to_delete:
                    for i in range(0, len(to_delete), chunk_size):
                        chunk = list(to_delete)[i:i+chunk_size]
                        ids = tuple(chunk)
                        cur.execute("DELETE FROM snapshots WHERE game_id = ANY(%s)", (list(ids),))
                        cur.execute("DELETE FROM games     WHERE id       = ANY(%s)", (list(ids),))
                        print(f"[db] Pruned chunk {i//chunk_size+1}")
            conn.commit()
    except Exception as e:
        print(f"[db] Error during pruning: {e!r}")
    
    print("[db] Prune complete.")

# ─── Proxy Handling ────────────────────────────────────────────────────────────
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
print(f"[proxy] Using {len(PROXIES)} proxies total.")

# ─── HTTP Helpers ─────────────────────────────────────────────────────────────
_cookie_idx = 0
def get_cookie():
    global _cookie_idx
    if not TOKENS:
        return {}
    tok = TOKENS[_cookie_idx % len(TOKENS)]
    _cookie_idx += 1
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
        p = random.choice(PROXIES)
        s.proxies.update({"http":p, "https":p})
    return s

def safe_get(url, retries=3):
    last_err = None
    for attempt in range(1, retries+1):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess  = get_session()
        proxy = sess.proxies.get("https") or sess.proxies.get("http") or "direct"
        
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                timeout=45  # Increased timeout
            )
            r.raise_for_status()
            return r.json()
        except (ProxyConnectionError, ReqConnError) as e:
            print(f"[safe_get] proxy error (attempt {attempt}): {e!r}")
            if proxy != "direct" and proxy in PROXIES:
                PROXIES.remove(proxy)
            if not PROXIES:
                sess.proxies.clear()
        except Exception as e:
            print(f"[safe_get] error (attempt {attempt}): {e!r}")
            last_err = e
        
        # Exponential backoff
        if attempt < retries:
            time.sleep(2 ** attempt)
    
    raise RuntimeError(f"GET failed after {retries} attempts: {url!r}\nLast error: {last_err!r}")

def download_image(url, retries=3):
    """Download image and return binary data with size limits and auto-resize"""
    if SKIP_IMAGES:
        return None
        
    last_err = None
    max_size = MAX_IMAGE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    for attempt in range(1, retries+1):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent()},
                cookies=get_cookie(),
                timeout=45,
                stream=True
            )
            r.raise_for_status()
            
            # Download the full image first
            image_data = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    image_data += chunk
            
            print(f"[download_image] Downloaded {len(image_data)} bytes")
            
            # If image is within size limit, return as-is
            if len(image_data) <= max_size:
                print(f"[download_image] ✓ Image within size limit")
                return image_data
            
            # Image is too large, try to resize it
            print(f"[download_image] Image too large ({len(image_data)} bytes), attempting resize...")
            resized_data = resize_image_to_limit(image_data, max_size)
            
            if resized_data:
                print(f"[download_image] ✓ Resized to {len(resized_data)} bytes")
                return resized_data
            else:
                print(f"[download_image] ⚠ Could not resize image to acceptable size")
                return None
            
        except Exception as e:
            print(f"[download_image] Error (attempt {attempt}): {e!r}")
            last_err = e
            
        if attempt < retries:
            time.sleep(2 ** attempt)
    
    print(f"[download_image] Failed after {retries} attempts: {last_err!r}")
    return None

def resize_image_to_limit(image_data, max_size_bytes):
    """Resize image by reducing resolution until it fits within size limit"""
    try:
        from PIL import Image
        from io import BytesIO
        
        # Load image from binary data
        img = Image.open(BytesIO(image_data))
        original_size = img.size
        print(f"[resize_image] Original size: {original_size[0]}x{original_size[1]}")
        
        # Try different quality and resolution combinations
        resize_attempts = [
            # (scale_factor, quality)
            (1.0, 85),    # Reduce quality first
            (1.0, 70),    # Lower quality
            (1.0, 50),    # Even lower quality
            (0.8, 85),    # 80% size, good quality
            (0.8, 70),    # 80% size, medium quality
            (0.6, 85),    # 60% size, good quality
            (0.6, 70),    # 60% size, medium quality
            (0.5, 85),    # 50% size, good quality
            (0.5, 70),    # 50% size, medium quality
            (0.4, 85),    # 40% size, good quality
            (0.3, 85),    # 30% size, good quality
            (0.2, 85),    # 20% size, good quality
        ]
        
        for scale, quality in resize_attempts:
            try:
                # Calculate new dimensions
                new_width = int(original_size[0] * scale)
                new_height = int(original_size[1] * scale)
                
                # Skip if image would be too small
                if new_width < 50 or new_height < 50:
                    continue
                
                # Resize image
                if scale < 1.0:
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    resized_img = img
                
                # Convert to bytes with specified quality
                output = BytesIO()
                
                # Handle different image formats
                if img.format == 'PNG' and quality < 100:
                    # Convert PNG to JPEG for better compression when reducing quality
                    if resized_img.mode in ("RGBA", "LA", "P"):
                        # Create white background for transparent images
                        background = Image.new("RGB", resized_img.size, (255, 255, 255))
                        if resized_img.mode == "P":
                            resized_img = resized_img.convert("RGBA")
                        background.paste(resized_img, mask=resized_img.split()[-1] if resized_img.mode == "RGBA" else None)
                        resized_img = background
                    resized_img.save(output, format='JPEG', quality=quality, optimize=True)
                elif img.format == 'JPEG' or quality < 100:
                    resized_img.save(output, format='JPEG', quality=quality, optimize=True)
                else:
                    # Keep original format for high quality
                    resized_img.save(output, format=img.format or 'PNG', optimize=True)
                
                output_data = output.getvalue()
                output_size = len(output_data)
                
                print(f"[resize_image] Attempt {scale:.1f}x scale, {quality}% quality: {new_width}x{new_height} = {output_size} bytes")
                
                # Check if this version fits
                if output_size <= max_size_bytes:
                    print(f"[resize_image] ✓ Success: {original_size[0]}x{original_size[1]} → {new_width}x{new_height} ({len(image_data)} → {output_size} bytes)")
                    return output_data
                
            except Exception as resize_error:
                print(f"[resize_image] Error with scale {scale}, quality {quality}: {resize_error!r}")
                continue
        
        print(f"[resize_image] ⚠ Could not resize image below {max_size_bytes} bytes")
        return None
        
    except ImportError:
        print("[resize_image] ⚠ PIL not available, cannot resize images")
        return None
    except Exception as e:
        print(f"[resize_image] ⚠ Resize failed: {e!r}")
        return None

# ─── Roblox endpoints (unchanged but with better error handling) ───────────────
def fetch_creator_games(user_id):
    print(f"[fetch_creator_games] user={user_id}")
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/users/{user_id}/games"
    page_count = 0
    
    while True:
        page_count += 1
        if page_count > 100:  # Safety limit
            print(f"[fetch_creator_games] Hit page limit for user {user_id}")
            break
            
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[fetch_creator_games] Failed for user {user_id}: {e!r}")
            break
            
        chunk = data.get("data", [])
        print(f"[fetch_creator_games] page {page_count}: got {len(chunk)} games")
        
        for it in chunk:
            games.append({"universeId": str(it["id"]), "name": it.get("name","")})
        
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    
    return games

def fetch_user_groups(user_id):
    print(f"[fetch_user_groups] user={user_id}")
    try:
        data = safe_get(f"https://groups.roblox.com/v2/users/{user_id}/groups/roles")
        groups = [str(g["group"]["id"]) for g in data.get("data",[]) if "group" in g]
        print(f"[fetch_user_groups] found {len(groups)} groups")
        return groups
    except Exception as e:
        print(f"[fetch_user_groups] Failed for user {user_id}: {e!r}")
        return []

def fetch_group_games(group_id):
    print(f"[fetch_group_games] group={group_id}")
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/groups/{group_id}/games"
    page_count = 0
    
    while True:
        page_count += 1
        if page_count > 100:  # Safety limit
            print(f"[fetch_group_games] Hit page limit for group {group_id}")
            break
            
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[fetch_group_games] Failed for group {group_id}: {e!r}")
            break
            
        chunk = data.get("data",[])
        print(f"[fetch_group_games] page {page_count}: got {len(chunk)} games")
        
        for it in chunk:
            games.append({
                "universeId": str(it.get("id") or it.get("universeId")),
                "name": it.get("name","")
            })
        
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    
    return games

def get_game_details(universe_ids):
    print(f"[get_game_details] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://games.roblox.com/v1/games?universeIds={s}"
        try:
            data = safe_get(url).get("data",[])
            print(f"[get_game_details] chunk size={len(ids)} → got {len(data)} records")
            return data
        except Exception as e:
            print(f"[get_game_details] Chunk failed: {e!r}")
            if len(ids) == 1:
                print(f"[get_game_details] Skipping single {ids[0]}")
                return []
            # Split and retry
            m = len(ids) // 2
            return fetch_chunk(ids[:m]) + fetch_chunk(ids[m:])
    
    out = []
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        print(f"[get_game_details] fetching batch {i//BATCH_SIZE+1}/{(len(universe_ids)-1)//BATCH_SIZE+1}")
        try:
            out.extend(fetch_chunk(batch))
        except Exception as e:
            print(f"[get_game_details] Batch {i//BATCH_SIZE+1} completely failed: {e!r}")
        
        # Memory cleanup
        if i % (BATCH_SIZE * 10) == 0:
            gc.collect()
    
    return out

def get_game_votes(universe_ids):
    print(f"[get_game_votes] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    
    def fetch_chunk(ids):
        s = ",".join(ids)
        url = f"https://games.roblox.com/v1/games/votes?universeIds={s}"
        try:
            data = safe_get(url).get("data",[])
            print(f"[get_game_votes] chunk size={len(ids)} → got {len(data)} votes")
            return {str(v["id"]):{"upVotes":v["upVotes"],"downVotes":v["downVotes"]} for v in data}
        except Exception as e:
            print(f"[get_game_votes] Chunk failed: {e!r}")
            if len(ids) == 1:
                return {}
            m = len(ids) // 2
            a = fetch_chunk(ids[:m])
            b = fetch_chunk(ids[m:])
            a.update(b)
            return a
    
    votes = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        print(f"[get_game_votes] fetching batch {i//BATCH_SIZE+1}")
        try:
            votes.update(fetch_chunk(batch))
        except Exception as e:
            print(f"[get_game_votes] Batch {i//BATCH_SIZE+1} completely failed: {e!r}")
    
    return votes

def fetch_icons(universe_ids):
    print(f"[fetch_icons] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    icons = {}
    
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        s = ",".join(batch)
        url = f"https://thumbnails.roblox.com/v1/games/icons?universeIds={s}&size=512x512&format=Png"
        print(f"[fetch_icons] batch {i//BATCH_SIZE+1}")
        
        try:
            for e in safe_get(url).get("data",[]):
                icons[str(e["targetId"])] = e["imageUrl"]
            print(f"[fetch_icons] got {len(batch)} URLs")
        except Exception as e:
            print(f"[fetch_icons] Batch failed: {e!r}")
    
    return icons

def fetch_thumbnails(universe_ids):
    print(f"[fetch_thumbnails] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    thumbs = {}

    def fetch_batch(batch):
        s = ",".join(batch)
        url = f"https://thumbnails.roblox.com/v1/games/multiget/thumbnails?universeIds={s}&size=768x432&format=Png"
        
        try:
            response = safe_get(url)
            data = response.get("data", [])
            print(f"[fetch_thumbnails] got {len(data)} thumbnail responses")
            
            for game_data in data:
                universe_id = str(game_data.get("universeId", ""))
                thumbnails_list = game_data.get("thumbnails", [])
                
                if thumbnails_list and len(thumbnails_list) > 0:
                    first_thumbnail = thumbnails_list[0]
                    if first_thumbnail.get("state") == "Completed":
                        image_url = first_thumbnail.get("imageUrl", "")
                        if image_url:
                            thumbs[universe_id] = image_url
                            
        except Exception as e:
            print(f"[fetch_thumbnails] batch error: {e!r}")
            if len(batch) > 1:
                mid = len(batch) // 2
                fetch_batch(batch[:mid])
                fetch_batch(batch[mid:])

    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        print(f"[fetch_thumbnails] processing batch {i//BATCH_SIZE+1}")
        fetch_batch(batch)

    return thumbs

# ─── Main processing with checkpoints ──────────────────────────────────────────
def scrape_and_snapshot():
    print("[main] Starting scrape run")
    
    try:
        # Get all game IDs first
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
                        master_games.append((int(g["universeId"]), g["name"][:255]))  # Truncate long names
            except Exception as e:
                print(f"[main] Error fetching games for creator {uid}: {e!r}")
                traceback.print_exc()

        print(f"[main] Found {len(master_games)} entries ({len(all_ids)} unique)")
        all_list = list(all_ids)
        
        if not all_list:
            print("[main] No games found; exiting.")
            return

        save_checkpoint(0, len(all_list), "starting", None)

        # Dedupe and upsert games
        unique_map = {gid: name for gid, name in master_games}
        deduped = list(unique_map.items())
        print(f"[main] Deduped to {len(deduped)} unique games")
        upsert_games(deduped)

        # Process in chunks to manage memory
        chunk_size = 200  # Process 200 games at a time
        total_processed = 0
        
        for chunk_start in range(0, len(all_list), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_list))
            chunk_ids = all_list[chunk_start:chunk_end]
            
            print(f"[main] Processing chunk {chunk_start//chunk_size+1}: games {chunk_start+1}-{chunk_end}")
            
            try:
                # Fetch metadata for this chunk
                meta = get_game_details(chunk_ids)
                votes = get_game_votes(chunk_ids)
                icons = fetch_icons(chunk_ids) if not SKIP_IMAGES else {}
                thumbs = fetch_thumbnails(chunk_ids) if not SKIP_IMAGES else {}

                # Process snapshots for this chunk
                snaps = []
                for g in meta:
                    uid = str(g.get("universeId") or g.get("id"))
                    icon_data = thumb_data = None

                    # Download images if not skipping
                    if not SKIP_IMAGES:
                        if icon_url := icons.get(uid):
                            try:
                                icon_data = download_image(icon_url)
                            except Exception as e:
                                print(f"[download] Icon failed for {uid}: {e!r}")

                        if thumb_url := thumbs.get(uid):
                            try:
                                thumb_data = download_image(thumb_url)
                            except Exception as e:
                                print(f"[download] Thumbnail failed for {uid}: {e!r}")

                    snap = (
                        int(uid),
                        g.get("playing", 0),
                        g.get("visits", 0),
                        g.get("favoritedCount", 0),
                        votes.get(uid, {}).get("upVotes", 0),
                        votes.get(uid, {}).get("downVotes", 0),
                        icon_data,
                        thumb_data,
                    )
                    snaps.append(snap)

                # Save this chunk
                save_snapshots_batch(snaps)
                total_processed += len(snaps)
                
                # Save checkpoint
                if total_processed % CHECKPOINT_FREQUENCY == 0:
                    save_checkpoint(total_processed, len(all_list), "processing", int(uid) if snaps else None)
                
                print(f"[main] Completed chunk: {total_processed}/{len(all_list)} games processed")
                
                # Cleanup
                del meta, votes, icons, thumbs, snaps
                gc.collect()
                
            except Exception as e:
                print(f"[main] Error processing chunk {chunk_start//chunk_size+1}: {e!r}")
                traceback.print_exc()
                save_checkpoint(total_processed, len(all_list), f"error_chunk_{chunk_start//chunk_size+1}", None)

        # Final operations
        save_checkpoint(total_processed, len(all_list), "pruning", None)
        prune_stale([int(x) for x in all_list])
        save_checkpoint(total_processed, len(all_list), "completed", None)
        
        print(f"[main] 🕒 Completed {len(all_list)} games at {datetime.utcnow()}")

    except Exception as e:
        print(f"[main] Fatal error: {e!r}")
        traceback.print_exc()
        save_checkpoint(0, 0, f"fatal_error: {str(e)[:200]}", None)
        raise

def main():
    try:
        ensure_tables()
        
        # Check for resume
        if RESUME_FROM_CHECKPOINT:
            processed, total, last_game = get_last_checkpoint()
            print(f"[main] Resume capability: last checkpoint had {processed}/{total} games")
        
        scrape_and_snapshot()
        
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user")
        save_checkpoint(0, 0, "interrupted", None)
    except Exception as e:
        print(f"[main] Unhandled error: {e!r}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
