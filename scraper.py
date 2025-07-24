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
DATABASE_URL         = os.getenv("DATABASE_URL")
CREATORS            = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS              = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ     = 100
env_batch           = int(os.getenv("BATCH_SIZE","100"))
BATCH_SIZE          = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY    = float(os.getenv("RATE_LIMIT_DELAY","0.7"))
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

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
        conn.commit()
    print("[db] Upsert complete.")

def save_snapshots(snaps):
    print(f"[db] Saving {len(snaps)} snapshots…")
    sql = """
    INSERT INTO snapshots
      (game_id, playing, visits, favorites, likes, dislikes, icon_url, thumbnail_url)
    VALUES %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, snaps)
        conn.commit()
    print("[db] Snapshots saved.")

def prune_stale(current_ids):
    print(f"[db] Pruning stale IDs not in current set of {len(current_ids)} games…")
    ids = tuple(current_ids) or (0,)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM snapshots WHERE game_id NOT IN %s", (ids,))
            cur.execute("DELETE FROM games     WHERE id       NOT IN %s", (ids,))
        conn.commit()
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
    # Pure ASCII User‑Agents only
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36/"
        "Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36/"
        "Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36/"
        "Chrome/126.0.0.0 Safari/537.36",
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
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.2)
        sess  = get_session()
        proxy = sess.proxies.get("https") or sess.proxies.get("http") or "direct"
        print(f"[safe_get] try #{attempt} → GET {url} via {proxy}")
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            r.raise_for_status()
            print("[safe_get]  ✓ success")
            return r.json()
        except (ProxyConnectionError, ReqConnError) as e:
            print(f"[safe_get]  proxy error: {e!r}")
            if proxy != "direct" and proxy in PROXIES:
                PROXIES.remove(proxy)
            if PROXIES:
                continue
            sess.proxies.clear()
        except Exception as e:
            print(f"[safe_get]  error: {e!r}")
            last_err = e
    raise RuntimeError(f"GET failed after {retries} attempts: {url!r}\nLast error: {last_err!r}")

def safe_post(url, json=None, retries=3):
    headers = {
        "User-Agent":   get_user_agent(),
        "Accept":       "application/json",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, retries+1):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.2)
        sess  = get_session()
        proxy = sess.proxies.get("https") or sess.proxies.get("http") or "direct"
        print(f"[safe_post] try #{attempt} → POST {url} via {proxy}")
        try:
            r = sess.post(url, headers=headers, cookies=get_cookie(), json=json, timeout=30)
            r.raise_for_status()
            print("[safe_post]  ✓ success")
            return r.json()
        except (ProxyConnectionError, ReqConnError) as e:
            print(f"[safe_post]  proxy error: {e!r}")
            if proxy != "direct" and proxy in PROXIES:
                PROXIES.remove(proxy)
            if PROXIES:
                continue
            sess.proxies.clear()
        except Exception as e:
            print(f"[safe_post]  error: {e!r}")
            last_err = e
    raise RuntimeError(f"POST failed after {retries} attempts: {url!r}\nLast error: {last_err!r}")

# ─── Roblox endpoints ─────────────────────────────────────────────────────────
def fetch_creator_games(user_id):
    print(f"[fetch_creator_games] user={user_id}")
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/users/{user_id}/games"
    while True:
        qs   = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        data = safe_get(f"{base}?{qs}")
        chunk = data.get("data", [])
        print(f"[fetch_creator_games]   got {len(chunk)} games")
        for it in chunk:
            games.append({"universeId": str(it["id"]), "name": it.get("name","")})
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

def fetch_user_groups(user_id):
    print(f"[fetch_user_groups] user={user_id}")
    data = safe_get(f"https://groups.roblox.com/v2/users/{user_id}/groups/roles")
    groups = [str(g["group"]["id"]) for g in data.get("data",[]) if "group" in g]
    print(f"[fetch_user_groups]   found {len(groups)} groups")
    return groups

def fetch_group_games(group_id):
    print(f"[fetch_group_games] group={group_id}")
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/groups/{group_id}/games"
    while True:
        qs = f"accessFilter=Public&sortOrder=Asc&limit=50" + (f"&cursor={cursor}" if cursor else "")
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[fetch_group_games]  FAILED: {e!r}")
            break
        chunk = data.get("data",[])
        print(f"[fetch_group_games]   got {len(chunk)} games")
        for it in chunk:
            games.append({
                "universeId": str(it.get("id") or it.get("universeId")),
                "name":       it.get("name","")
            })
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

# ─── Metadata & Votes ──────────────────────────────────────────────────────────
def get_game_details(universe_ids):
    print(f"[get_game_details] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    def fetch_chunk(ids):
        s   = ",".join(ids)
        url = f"https://games.roblox.com/v1/games?universeIds={s}"
        try:
            data = safe_get(url).get("data",[])
            print(f"[get_game_details]   chunk size={len(ids)} → got {len(data)} records")
            return data
        except Exception:
            if len(ids)==1:
                print(f"[get_game_details]    skipping single {ids[0]}")
                return []
            m = len(ids)//2
            return fetch_chunk(ids[:m]) + fetch_chunk(ids[m:])
    out = []
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        print(f"[get_game_details] fetching batch {i//BATCH_SIZE+1}")
        out.extend(fetch_chunk(batch))
    return out

def get_game_votes(universe_ids):
    print(f"[get_game_votes] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    def fetch_chunk(ids):
        s   = ",".join(ids)
        url = f"https://games.roblox.com/v1/games/votes?universeIds={s}"
        try:
            data = safe_get(url).get("data",[])
            print(f"[get_game_votes]   chunk size={len(ids)} → got {len(data)} votes")
            return { str(v["id"]):{"upVotes":v["upVotes"],"downVotes":v["downVotes"]} for v in data }
        except Exception:
            if len(ids)==1:
                return {}
            m = len(ids)//2
            a = fetch_chunk(ids[:m]); b = fetch_chunk(ids[m:])
            a.update(b); return a
    votes = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        print(f"[get_game_votes] fetching batch {i//BATCH_SIZE+1}")
        votes.update(fetch_chunk(batch))
    return votes

# ─── Thumbnails & Icons ────────────────────────────────────────────────────────
def fetch_icons(universe_ids):
    print(f"[fetch_icons] total IDs={len(universe_ids)}, batches of {BATCH_SIZE}")
    icons = {}
    os.makedirs("icons", exist_ok=True)
    for i in range(0, len(universe_ids), BATCH_SIZE):
        batch = universe_ids[i:i+BATCH_SIZE]
        s     = ",".join(batch)
        url   = f"https://thumbnails.roblox.com/v1/games/icons?universeIds={s}&size=512x512&format=Png"
        print(f"[fetch_icons] batch {i//BATCH_SIZE+1}")
        try:
            for e in safe_get(url).get("data",[]):
                icons[str(e["targetId"])] = e["imageUrl"]
            print(f"[fetch_icons]   got {len(batch)} URLs")
        except HTTPError as e:
            print(f"[fetch_icons]   HTTPError, skipping batch: {e!r}")
    return icons

def fetch_thumbnails(universe_ids):
    """
    For each universeId, build the direct image URL:
    https://www.roblox.com/asset-thumbnail/image?assetId={id}&width=768&height=432&format=png
    """
    print(f"[fetch_thumbnails] total IDs={len(universe_ids)}")
    thumbs = {}
    os.makedirs("thumbnails", exist_ok=True)

    for uid in universe_ids:
        url = (
            "https://www.roblox.com/asset-thumbnail/image"
            f"?assetId={uid}&width=768&height=432&format=png"
        )
        thumbs[str(uid)] = url
        print(f"[fetch_thumbnails]  → {uid}: {url}")

    return thumbs



# ─── Main scrape + snapshot + prune ────────────────────────────────────────────
def scrape_and_snapshot():
    print("[main] Starting scrape run")
    all_ids      = set()
    master_games = []

    for uid in CREATORS:
        own    = fetch_creator_games(uid)
        groups = fetch_user_groups(uid)
        grp    = []
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

    meta   = get_game_details(all_list)
    votes  = get_game_votes(all_list)
    icons  = fetch_icons(all_list)
    thumbs = fetch_thumbnails(all_list)

    # Dedupe before upsert
    unique_map = { gid:name for gid,name in master_games }
    deduped    = list(unique_map.items())
    print(f"[main] Deduped to {len(deduped)} unique games")
    upsert_games(deduped)

    # Download & snapshot
    snaps = []
    os.makedirs("thumbnails", exist_ok=True)
    for g in meta:
        uid       = str(g.get("universeId") or g.get("id"))
        icon_url  = icons.get(uid)
        thumb_url = thumbs.get(uid)
        icon_path = thumb_path = None

        if icon_url:
            try:
                icon_path = f"icons/{uid}.png"
                with requests.get(icon_url, stream=True) as r:
                    r.raise_for_status()
                    with open(icon_path, "wb") as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
            except HTTPError:
                icon_path = None

        if thumb_url:
            try:
                thumb_path = f"thumbnails/{uid}.png"
                with requests.get(thumb_url, stream=True) as r:
                    r.raise_for_status()
                    with open(thumb_path, "wb") as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
            except HTTPError as e:
                print(f"[download] no thumbnail for {uid}, skipping ({e!r})")
                thumb_path = None

        snaps.append((
            int(uid),
            g.get("playing",0),
            g.get("visits",0),
            g.get("favoritedCount",0),
            votes.get(uid,{}).get("upVotes",0),
            votes.get(uid,{}).get("downVotes",0),
            icon_path,
            thumb_path,
        ))

    save_snapshots(snaps)
    prune_stale([int(x) for x in all_list])
    print(f"[main] 🕒 Completed {len(all_list)} games at {datetime.utcnow()}")

def main():
    ensure_tables()
    scrape_and_snapshot()

if __name__=="__main__":
    main()
