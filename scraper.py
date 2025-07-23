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

from socks import ProxyConnectionError
from requests.exceptions import ConnectionError as ReqConnError

def safe_get(url, retries=3):
    last_err = None

    for attempt in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random() * 0.3)
        sess = get_session()
        proxy = sess.proxies.get("https") or sess.proxies.get("http")

        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            if r.ok:
                return r.json()
            r.raise_for_status()

        except (ProxyConnectionError, ReqConnError) as e:
            last_err = e
            # if it was a proxy we just tried, drop it
            if proxy in PROXIES:
                print(f"[proxy] removing dead proxy {proxy}")
                PROXIES.remove(proxy)
            # if we still have proxies, retry immediately
            if PROXIES:
                continue
            # otherwise fall through to retry without proxy
            print("[proxy] no proxies left, trying direct connection")
            sess.proxies.clear()
            continue

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    # if we get here, everything failed
    raise RuntimeError(f"GET failed after {retries} attempts: {url}\nLast error: {last_err}")


from socks import ProxyConnectionError
from requests.exceptions import ConnectionError as ReqConnError

def safe_post(url, json=None, retries=3):
    headers = {
        "User‑Agent": get_user_agent(),
        "Accept":      "application/json",
        "Content-Type":"application/json"
    }
    last_err = None

    for attempt in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random() * 0.3)
        sess = get_session()
        proxy = sess.proxies.get("https") or sess.proxies.get("http")

        try:
            r = sess.post(url, headers=headers, cookies=get_cookie(), json=json, timeout=30)
            if r.ok:
                return r.json()
            r.raise_for_status()

        except (ProxyConnectionError, ReqConnError) as e:
            last_err = e
            if proxy in PROXIES:
                print(f"[proxy] removing dead proxy {proxy}")
                PROXIES.remove(proxy)
            if PROXIES:
                continue
            print("[proxy] no proxies left, trying direct")
            sess.proxies.clear()
            continue

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError(f"POST failed after {retries} attempts: {url}\nLast error: {last_err}")

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

# ─── Roblox Metadata & Votes (GET + recursive chunking) ────────────────────────
def get_game_details(universe_ids):
    """Fetch metadata via GET /v1/games?universeIds=… in batches and
       fall back by splitting a batch in half on failure."""
    def fetch_chunk(ids):
        ids_str = ",".join(ids)
        url     = f"https://games.roblox.com/v1/games?universeIds={ids_str}"
        try:
            return safe_get(url).get("data", [])
        except Exception:
            if len(ids) == 1:
                return []  # give up on this single universe
            mid = len(ids) // 2
            return fetch_chunk(ids[:mid]) + fetch_chunk(ids[mid:])

    details = []
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i : i + BATCH_SIZE]
        details.extend(fetch_chunk(batch))
    return details

def get_game_votes(universe_ids):
    """Fetch votes via GET /v1/games/votes?universeIds=… in batches."""
    def fetch_chunk(ids):
        ids_str = ",".join(ids)
        url     = f"https://games.roblox.com/v1/games/votes?universeIds={ids_str}"
        try:
            data = safe_get(url).get("data", [])
            return { str(v["id"]): {"upVotes": v["upVotes"], "downVotes": v["downVotes"]} 
                     for v in data }
        except Exception:
            if len(ids) == 1:
                return {}
            mid = len(ids) // 2
            a = fetch_chunk(ids[:mid])
            b = fetch_chunk(ids[mid:])
            a.update(b)
            return a

    votes = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i : i + BATCH_SIZE]
        votes.update(fetch_chunk(batch))
    return votes

# ─── Chunked Thumbnails & Icons ───────────────────────────────────────────────

from requests.exceptions import HTTPError

def fetch_icons(universe_ids):
    icons = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i : i + BATCH_SIZE]
        qs    = ",".join(batch)
        url   = (
            "https://thumbnails.roblox.com/v1/games/icons"
            f"?universeIds={qs}&size=512x512&format=Png"
        )
        try:
            data = safe_get(url).get("data", [])
            for entry in data:
                icons[str(entry["targetId"])] = entry["imageUrl"]
        except HTTPError as e:
            print(f"[Icons] batch {i//BATCH_SIZE+1} failed: {e}")
            if len(batch) > 1:
                # split and retry smaller chunks
                mid = len(batch)//2
                icons.update(fetch_icons(batch[:mid]))
                icons.update(fetch_icons(batch[mid:]))
    return icons

def fetch_thumbnails(universe_ids):
    thumbs = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i : i + BATCH_SIZE]
        qs    = ",".join(batch)
        url   = (
            "https://thumbnails.roblox.com/v1/games/thumbnails"
            f"?universeIds={qs}&size=768x432&format=Png"
        )
        try:
            data = safe_get(url).get("data", [])
            for entry in data:
                thumbs[str(entry["targetId"])] = entry["imageUrl"]
        except HTTPError as e:
            print(f"[Thumbnails] batch {i//BATCH_SIZE+1} failed: {e}")
            if len(batch) > 1:
                mid = len(batch)//2
                thumbs.update(fetch_thumbnails(batch[:mid]))
                thumbs.update(fetch_thumbnails(batch[mid:]))
    return thumbs


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

