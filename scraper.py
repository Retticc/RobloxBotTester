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

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
print(f"[debug] Using database: {DATABASE_URL}")
CREATORS = [u.strip() for u in os.getenv("TARGET_CREATORS", "").split(",") if u.strip()]
TOKENS = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS", "").split(",") if t.strip()]
MAX_IDS_PER_REQ = 100
env_batch = int(os.getenv("BATCH_SIZE", "100"))
BATCH_SIZE = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
PROXY_API_BASE = os.getenv("PROXY_API_BASE", "https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

print(f"[startup] batch size={BATCH_SIZE}, rate_delay={RATE_LIMIT_DELAY}")

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
      icon_url      TEXT,
      thumbnail_url TEXT,
      PRIMARY KEY (game_id, snapshot_time)
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("[db] Tables ready.")

def upsert_games(games):
    print(f"[db] Upserting {len(games)} games…")
    sql = """
    INSERT INTO games(id,name) VALUES %s
      ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, games)
            cur.execute("SELECT COUNT(*) FROM games")
            print(f"[db] Total games in table now: {cur.fetchone()[0]}")
        conn.commit()
    print("[db] Upsert complete.")

def save_snapshots(snaps):
    print(f"[db] Saving {len(snaps)} snapshots…")
    if not snaps:
        print("[db] ⚠ No snapshots to save.")
        return
    sql = """
    INSERT INTO snapshots
      (game_id, playing, visits, favorites, likes, dislikes, icon_url, thumbnail_url)
    VALUES %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, snaps)
            cur.execute("SELECT COUNT(*) FROM snapshots")
            print(f"[db] Total snapshots in table now: {cur.fetchone()[0]}")
        conn.commit()
    print("[db] Snapshots saved.")

def prune_stale(current_ids):
    print(f"[db] Pruning stale IDs not in current set of {len(current_ids)} games…")
    ids = tuple(current_ids) or (0,)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM snapshots WHERE game_id NOT IN %s", (ids,))
            cur.execute("DELETE FROM games WHERE id NOT IN %s", (ids,))
        conn.commit()
    print("[db] Prune complete.")

# Leave the rest of your proxy and HTTP logic untouched

# ─── Main scrape + snapshot + prune ────────────────────────────────────────────
def scrape_and_snapshot():
    print("[main] Starting scrape run")
    all_ids = set()
    master_games = []

    for uid in CREATORS:
        own = fetch_creator_games(uid)
        groups = fetch_user_groups(uid)
        grp = []
        for g in groups:
            grp.extend(fetch_group_games(g))
        for g in own + grp:
            all_ids.add(g["universeId"])
            master_games.append((int(g["universeId"]), g["name"]))

    print(f"[main] Found {len(master_games)} entries ({len(all_ids)} unique)")
    all_list = list(all_ids)
    if not all_list:
        print("[main] No games found; exiting.")
        return

    meta = get_game_details(all_list)
    votes = get_game_votes(all_list)
    icons = fetch_icons(all_list)
    thumbs = fetch_thumbnails(all_list)

    unique_map = {gid: name for gid, name in master_games}
    deduped = list(unique_map.items())
    print(f"[main] Deduped to {len(deduped)} unique games")
    upsert_games(deduped)

    snaps = []
    for g in meta:
        uid = str(g.get("universeId") or g.get("id"))
        snaps.append((
            int(uid),
            g.get("playing", 0),
            g.get("visits", 0),
            g.get("favoritedCount", 0),
            votes.get(uid, {}).get("upVotes", 0),
            votes.get(uid, {}).get("downVotes", 0),
            icons.get(uid),
            thumbs.get(uid)
        ))

    print(f"[main] Preparing to insert {len(snaps)} snapshots…")
    save_snapshots(snaps)
    prune_stale([int(x) for x in all_list])
    print(f"[main] 🕒 Completed {len(all_list)} games at {datetime.utcnow()}")

def main():
    ensure_tables()
    scrape_and_snapshot()

if __name__ == "__main__":
    main()
