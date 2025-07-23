#!/usr/bin/env python3
import os, time, random, requests, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
DATABASE_URL      = os.getenv("DATABASE_URL")
CREATORS         = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS           = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ  = 100
env_batch        = int(os.getenv("BATCH_SIZE","100"))
BATCH_SIZE       = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY","0.7"))
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

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
    """games = list of (id,name)"""
    sql = "INSERT INTO games(id,name) VALUES %s ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name"
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, games)
            conn.commit()

def save_snapshots(snaps):
    """
    snaps = list of tuples:
      (game_id, playing, visits, favorites, likes, dislikes, icon_url, thumbnail_url)
    """
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
    """Remove any games & snapshots not in current_ids."""
    ids = tuple(current_ids) or (0,)  # avoid empty tuple
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM snapshots WHERE game_id NOT IN %s", (ids,))
            cur.execute("DELETE FROM games     WHERE id       NOT IN %s", (ids,))
        conn.commit()

# ─── Proxy Handling ────────────────────────────────────────────────────────────
def fetch_proxies_from_api():
    if not PROXY_API_KEY:
        return []
    try:
        resp = requests.get(f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ProxyAPI] error: {e}")
        return []
    proxies = []
    for e in data.get("ipv4", []):
        ip, auth = e["ip"], e["authInfo"]
        for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
            if port := e.get(port_key):
                proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    for order in data.get("ipv6", []):
        for ipinfo in order.get("ips", []):
            proto, ipport, auth = ipinfo["protocol"].lower(), ipinfo["ip"], ipinfo["authInfo"]
            proxies.append(f"{proto}://{auth['login']}:{auth['password']}@{ipport}")
    for key in ("isp","mobile"):
        for e in data.get(key, []):
            ip, auth = e["ip"], e["authInfo"]
            for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
                if port := e.get(port_key):
                    proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    return proxies

PROXIES = fetch_proxies_from_api() or PROXY_URLS_FALLBACK
print(f"→ Using {len(PROXIES)} proxies")

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
        "Mozilla/5.0 (Windows NT 10; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        "Mozilla/5.0 (X11; Linux x86_64)...",
    ])

def get_session():
    s = requests.Session()
    if PROXIES:
        p = random.choice(PROXIES)
        s.proxies.update({"http":p,"https":p})
    return s

def safe_get(url, retries=3):
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.get(url,
                         headers={"User-Agent":get_user_agent(),"Accept":"application/json"},
                         cookies=get_cookie(), timeout=30)
            if r.ok:
                return r.json()
            r.raise_for_status()
        except Exception as e:
            if i==retries-1:
                raise
            time.sleep(2**i)
    raise RuntimeError(f"GET failed: {url}")

def safe_post(url, json=None, retries=3):
    headers = {"User-Agent":get_user_agent(),
               "Accept":"application/json",
               "Content-Type":"application/json"}
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.post(url, headers=headers, cookies=get_cookie(), json=json, timeout=30)
            if r.ok:
                return r.json()
            r.raise_for_status()
        except Exception:
            if i==retries-1:
                raise
            time.sleep(2**i)
    raise RuntimeError(f"POST failed: {url}")

# ─── Roblox endpoints ─────────────────────────────────────────────────────────
def fetch_creator_games(user_id):
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/users/{user_id}/games"
    while True:
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        data = safe_get(f"{base}?{qs}")
        for it in data.get("data",[]):
            games.append({"universeId":str(it["id"]), "name":it.get("name","")})
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

def fetch_user_groups(user_id):
    data = safe_get(f"https://groups.roblox.com/v2/users/{user_id}/groups/roles")
    return [str(g["group"]["id"]) for g in data.get("data",[]) if "group" in g]

def fetch_group_games(group_id):
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/groups/{group_id}/games"
    while True:
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        data = safe_get(f"{base}?{qs}")
        for it in data.get("data",[]):
            games.append({"universeId":str(it["id"]), "name":it.get("name","")})
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

def get_game_details(universe_ids):
    """POST to /v1/games with JSON body {'universeIds': [...]}."""
    details = []

    def fetch_chunk(ids):
        resp = safe_post(
            "https://games.roblox.com/v1/games",
            json={"universeIds": ids}
        )
        return resp.get("data", [])

    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        try:
            details.extend(fetch_chunk(chunk))
        except Exception as e:
            print(f"[Meta] chunk {i//BATCH_SIZE+1} failed: {e}")
    return details

def get_game_votes(universe_ids):
    """POST to /v1/games/votes with JSON body {'universeIds': [...]}."""
    votes = {}

    def fetch_chunk(ids):
        resp = safe_post(
            "https://games.roblox.com/v1/games/votes",
            json={"universeIds": ids}
        )
        return resp.get("data", [])

    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        try:
            for v in fetch_chunk(chunk):
                uid = str(v["id"])
                votes[uid] = {
                    "upVotes":   v.get("upVotes", 0),
                    "downVotes": v.get("downVotes", 0),
                }
        except Exception as e:
            print(f"[Votes] chunk {i//BATCH_SIZE+1} failed: {e}")
    return votes

def fetch_icons(universe_ids):
    ids_str = ",".join(universe_ids)
    url = f"https://thumbnails.roblox.com/v1/games/icons?universeIds={ids_str}&size=512x512&format=Png"
    return {str(i["targetId"]):i["imageUrl"] for i in safe_get(url).get("data",[])}

def fetch_thumbnails(universe_ids):
    ids_str = ",".join(universe_ids)
    url = f"https://thumbnails.roblox.com/v1/games/thumbnails?universeIds={ids_str}&size=768x432&format=Png"
    return {str(i["targetId"]):i["imageUrl"] for i in safe_get(url).get("data",[])}

# ─── Core scrape + snapshot + prune ────────────────────────────────────────────
def scrape_and_snapshot():
    all_ids = set()
    master_games = []

    for uid in CREATORS:
        own   = fetch_creator_games(uid)
        groups= fetch_user_groups(uid)
        grp   = []
        for g in groups:
            grp.extend(fetch_group_games(g))
        for g in own+grp:
            all_ids.add(g["universeId"])
            master_games.append((int(g["universeId"]), g["name"]))

    if not all_ids:
        print("No games found; exiting this run.")
        return

    all_list = list(all_ids)
    meta     = get_game_details(all_list)
    votes    = get_game_votes(all_list)
    icons    = fetch_icons(all_list)
    thumbs   = fetch_thumbnails(all_list)

    # upsert master games
    upsert_games(master_games)

    # build snapshots
    snaps = []
    for g in meta:
        uid = str(g.get("universeId") or g.get("id"))
        snaps.append((
            int(uid),
            g.get("playing",0),
            g.get("visits",0),
            g.get("favoritedCount",0),
            votes.get(uid,{}).get("upVotes",0),
            votes.get(uid,{}).get("downVotes",0),
            icons.get(uid),
            thumbs.get(uid),
        ))
    save_snapshots(snaps)

    # prune any that disappeared
    prune_stale([int(x) for x in all_list])

    print(f"🕒 {len(all_list)} games snapped at {datetime.utcnow()}")

def main():
    scrape_and_snapshot()

if __name__=="__main__":
    main()

