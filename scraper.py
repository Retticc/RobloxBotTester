import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
CREATORS         = [u.strip() for u in os.getenv("TARGET_CREATORS","").split(",") if u.strip()]
TOKENS           = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS","").split(",") if t.strip()]
MAX_IDS_PER_REQ  = 100
env_batch        = int(os.getenv("BATCH_SIZE","100"))
BATCH_SIZE       = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY","0.7"))

PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

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
    for entry in data.get("ipv4", []):
        ip, auth = entry["ip"], entry["authInfo"]
        for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
            if port := entry.get(port_key):
                proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    for order in data.get("ipv6", []):
        for ipinfo in order.get("ips", []):
            proto, ipport, auth = ipinfo["protocol"].lower(), ipinfo["ip"], ipinfo["authInfo"]
            proxies.append(f"{proto}://{auth['login']}:{auth['password']}@{ipport}")
    for key in ("isp","mobile"):
        for entry in data.get(key, []):
            ip, auth = entry["ip"], entry["authInfo"]
            for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
                if port := entry.get(port_key):
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
        "Mozilla/5.0 (Windows NT 10; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
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
            r = sess.get(
                url,
                headers={"User-Agent":get_user_agent(),"Accept":"application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            if r.ok:
                return r.json()
            print(f"[GET] {r.status_code} @ {url}")
            r.raise_for_status()
        except Exception as e:
            print(f"[GET] attempt {i+1} error: {e}")
            if i < retries-1:
                time.sleep(2**i)
    raise RuntimeError(f"GET failed: {url}")

def safe_post(url, json=None, retries=3):
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "application/json",
        "Content-Type": "application/json"       # ← add this
    }
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.post(url, headers=headers, cookies=get_cookie(), json=json, timeout=30)
            if r.ok:
                return r.json()
            print(f"[POST] HTTP {r.status_code} @ {url}")
            r.raise_for_status()
        except Exception as e:
            print(f"[POST] attempt {i+1} error: {e}")
            if i < retries - 1:
                time.sleep(2**i)
    raise RuntimeError(f"POST failed after {retries} attempts: {url}")


# ─── Roblox: a user’s own games ────────────────────────────────────────────────
def fetch_creator_games(user_id):
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/users/{user_id}/games"
    while True:
        params = {"accessFilter":"Public","sortOrder":"Asc","limit":50}
        if cursor:
            params["cursor"] = cursor
        qs = "&".join(f"{k}={v}" for k,v in params.items())
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[UserGames] {e}")
            return games
        for item in data.get("data",[]):
            games.append({"universeId":str(item["id"]), "name":item.get("name","")})
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

# ─── Roblox: groups a user belongs to ─────────────────────────────────────────
def fetch_user_groups(user_id):
    url = f"https://groups.roblox.com/v2/users/{user_id}/groups/roles"
    try:
        data = safe_get(url)
        return [str(g["group"]["id"]) for g in data.get("data",[]) if "group" in g]
    except Exception as e:
        print(f"[UserGroups] {e}")
        return []

# ─── Roblox: games owned by a group ───────────────────────────────────────────
def fetch_group_games(group_id):
    games, cursor = [], ""
    base = f"https://games.roblox.com/v2/groups/{group_id}/games"
    while True:
        params = {"accessFilter":"Public","sortOrder":"Asc","limit":50}
        if cursor:
            params["cursor"] = cursor
        qs = "&".join(f"{k}={v}" for k,v in params.items())
        try:
            data = safe_get(f"{base}?{qs}")
        except Exception as e:
            print(f"[GroupGames] {e}")
            return games
        for item in data.get("data",[]):
            games.append({"universeId":str(item["id"]), "name":item.get("name","")})
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

# ─── Roblox: metadata & votes ──────────────────────────────────────────────────

def get_game_details(universe_ids):
    def fetch_chunk(ids):
        ids_str = ",".join(ids)
        url = f"https://games.roblox.com/v1/games?universeIds={ids_str}"
        try:
            return safe_get(url).get("data", [])
        except RuntimeError as e:
            if len(ids) == 1:
                # skip this single universeId if it truly fails
                return []
            # split in half and retry
            mid = len(ids) // 2
            return fetch_chunk(ids[:mid]) + fetch_chunk(ids[mid:])

    details = []
    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        details.extend(fetch_chunk(chunk))
    return details



def get_game_votes(universe_ids):
    def fetch_chunk(ids):
        ids_str = ",".join(ids)
        url = f"https://games.roblox.com/v1/games/votes?universeIds={ids_str}"
        try:
            data = safe_get(url).get("data", [])
            return { str(v["id"]): {"upVotes":v["upVotes"],"downVotes":v["downVotes"]} for v in data }
        except RuntimeError:
            if len(ids) == 1:
                return {}
            mid = len(ids) // 2
            a = fetch_chunk(ids[:mid])
            b = fetch_chunk(ids[mid:])
            a.update(b)
            return a

    votes = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        votes.update(fetch_chunk(chunk))
    return votes



# ─── Main Workflow ─────────────────────────────────────────────────────────────
def main():
    print("Starting data collection...\n")
    all_ids = set()

    for uid in CREATORS:
        print(f"> Creator {uid}")
        own = fetch_creator_games(uid)
        print(f"  → {len(own)} personal games")

        groups = fetch_user_groups(uid)
        print(f"  → {len(groups)} groups")

        grp_games = []
        for gid in groups:
            gms = fetch_group_games(gid)
            print(f"    • Group {gid}: {len(gms)} games")
            grp_games.extend(gms)

        for g in own + grp_games:
            all_ids.add(g["universeId"])
        print(f"  → Total unique games: {len(all_ids)}")

    all_list = list(all_ids)
    if not all_list:
        print("No games found; exiting.")
        return

    print("\n> Fetching metadata…")
    meta  = get_game_details(all_list)

    print("> Fetching votes…")
    votes = get_game_votes(all_list)

    records = []
    for g in meta:
        uid = str(g.get("universeId") or g.get("id"))
        records.append({
            "universeId": uid,
            "name":       g.get("name",""),
            "playing":    g.get("playing",0),
            "visits":     g.get("visits",0),
            "favorites":  g.get("favoritedCount",0),
            "likeCount":  votes.get(uid,{}).get("upVotes",0),
            "dislikeCount":votes.get(uid,{}).get("downVotes",0),
            "genre":       g.get("genre",""),
            "price":       g.get("price",0),
            "creatorType": g.get("creator",{}).get("type",""),
            "creator":     g.get("creator",{}).get("name","")
        })

    df = pd.DataFrame(records)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Wrote {len(df)} records to test_data.csv")
    print("\nPreview:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
