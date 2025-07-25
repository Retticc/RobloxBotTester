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

# Load environment variables from .env file
load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
CREATORS = [u.strip() for u in os.getenv("TARGET_CREATORS", "").split(",") if u.strip()]
TOKENS = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS", "").split(",") if t.strip()]
MAX_IDS_PER_REQ = 100
BATCH_SIZE = max(1, min(int(os.getenv("BATCH_SIZE", "100")), MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
PROXY_API_BASE = os.getenv("PROXY_API_BASE", "https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

# ─── Database Helpers ──────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def ensure_tables():
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
      icon_url      TEXT,
      thumbnail_url TEXT,
      PRIMARY KEY (game_id, snapshot_time)
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

def upsert_games(games):
    sql = """
    INSERT INTO games(id, name) VALUES %s
    ON CONFLICT(id) DO UPDATE SET name = EXCLUDED.name
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, games)
        conn.commit()

def save_snapshots(snaps):
    sql = """
    INSERT INTO snapshots
      (game_id, playing, visits, favorites, likes, dislikes, icon_url, thumbnail_url)
    VALUES %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, snaps)
        conn.commit()

def prune_stale(current_ids):
    ids = tuple(current_ids) or (0,)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM snapshots WHERE game_id NOT IN %s", (ids,))
            cur.execute("DELETE FROM games WHERE id NOT IN %s", (ids,))
        conn.commit()

# ─── HTTP and Proxy Logic omitted here for brevity ─────────────────────────────
# Keep your existing fetch_proxies_from_api, get_session, safe_get, etc.

# ─── Main scrape + snapshot + prune ────────────────────────────────────────────
def scrape_and_snapshot():
    print("[main] Starting scrape run")
    all_ids = set()
    master_games = []

    for uid in CREATORS:
        print(f"[main] Processing creator: {uid}")
        
        # Actually call the API functions instead of empty lists
        own = fetch_creator_games(uid)  # This was []
        groups = fetch_user_groups(uid)  # This was []
        
        grp = []
        for group_id in groups:
            group_games = fetch_group_games(group_id)  # This was missing
            grp.extend(group_games)
        
        print(f"[main] Creator {uid}: {len(own)} own games, {len(grp)} group games")
        
        # Process all games (own + group games)
        for g in own + grp:
            universe_id = g["universeId"]
            all_ids.add(universe_id)
            master_games.append((int(universe_id), g["name"]))

    print(f"[main] Found {len(master_games)} entries ({len(all_ids)} unique)")
    all_list = list(all_ids)
    if not all_list:
        print("[main] No games found; exiting.")
        return

    # Actually call the API functions instead of empty placeholders
    print("[main] Fetching game details...")
    meta = get_game_details(all_list)  # This was []
    
    print("[main] Fetching game votes...")
    votes = get_game_votes(all_list)   # This was {}
    
    print("[main] Fetching game icons...")
    icons = fetch_icons(all_list)      # This was {}
    
    print("[main] Fetching game thumbnails...")
    thumbs = fetch_thumbnails(all_list)  # This was {}

    # Dedupe before upsert
    unique_map = {gid: name for gid, name in master_games}
    deduped = list(unique_map.items())
    print(f"[main] Deduped to {len(deduped)} unique games")
    upsert_games(deduped)

    # Create snapshots
    snaps = []
    os.makedirs("thumbnails", exist_ok=True)
    os.makedirs("icons", exist_ok=True)
    
    for g in meta:
        uid = str(g.get("universeId") or g.get("id"))
        icon_url = icons.get(uid)
        thumb_url = thumbs.get(uid)
        icon_path = thumb_path = None

        # Download icon if available
        if icon_url:
            try:
                icon_path = f"icons/{uid}.png"
                with requests.get(icon_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(icon_path, "wb") as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                print(f"[download] ✓ saved icon for {uid}")
            except Exception as e:
                print(f"[download] ⚠ failed to download icon for {uid}: {e!r}")
                icon_path = None

        # Download thumbnail if available
        if thumb_url and thumb_url.strip():
            try:
                thumb_path = f"thumbnails/{uid}.png"
                print(f"[download] downloading thumbnail for {uid}")
                with requests.get(thumb_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(thumb_path, "wb") as f:
                        for chunk in r.iter_content(1024):
                            if chunk:
                                f.write(chunk)
                print(f"[download] ✓ saved thumbnail for {uid}")
            except Exception as e:
                print(f"[download] ⚠ failed to download thumbnail for {uid}: {e!r}")
                thumb_path = None

        # Create snapshot record
        snaps.append((
            int(uid),
            g.get("playing", 0),
            g.get("visits", 0),
            g.get("favoritedCount", 0),
            votes.get(uid, {}).get("upVotes", 0),
            votes.get(uid, {}).get("downVotes", 0),
            icon_path,
            thumb_path,
        ))

    save_snapshots(snaps)
    prune_stale([int(x) for x in all_list])
    print(f"[main] 🕒 Completed {len(all_list)} games at {datetime.utcnow()}")
    
def main():
    ensure_tables()
    scrape_and_snapshot()

if __name__ == "__main__":
    main()
