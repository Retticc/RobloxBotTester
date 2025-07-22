import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

# Roblox creators to test (comma‑separated, whitespace stripped)
CREATORS = [uid.strip() for uid in os.getenv("TARGET_CREATORS", "").split(",") if uid.strip()]

# Auth tokens, batch size, rate‑limit delay
TOKENS           = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS", "").split(",") if t.strip()]
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "100"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))

# Proxy API / static list
PROXY_API_KEY = os.getenv("API_KEY")
PROXY_URLS    = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

API_BASE_URL = "https://api.ipv4proxy.com/client-api/v1"

# Fields to capture from /v1/games
FIELDS = [
    "universeId", "name", "playing", "visits",
    "favorites", "likeCount", "dislikeCount",
    "genre", "price", "creatorType"
]

# ─── Proxy Handling ────────────────────────────────────────────────────────────

def fetch_ipv6_proxies():
    url = f"{API_BASE_URL}/{PROXY_API_KEY}/get/proxies"
    params = {"proxyType": "ipv6"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    out = []
    for order in data.get("proxies", []):
        for ip_info in order.get("ips", []):
            proto = ip_info.get("protocol", "HTTP").lower()
            hostport = ip_info["ip"]
            auth = ip_info.get("authInfo", {})
            user = auth.get("login")
            pw   = auth.get("password")
            if user and pw:
                out.append(f"{proto}://{user}:{pw}@{hostport}")
            else:
                out.append(f"{proto}://{hostport}")
    return out

# build PROXIES list
if PROXY_API_KEY:
    try:
        PROXIES = fetch_ipv6_proxies()
        print(f"Fetched {len(PROXIES)} proxies via API")
    except Exception as e:
        print(f"Proxy API error: {e}")
        PROXIES = PROXY_URLS
else:
    PROXIES = PROXY_URLS

# ─── Request Utilities ─────────────────────────────────────────────────────────

_cookie_idx = 0
def get_cookie():
    global _cookie_idx
    token = TOKENS[_cookie_idx % len(TOKENS)]
    _cookie_idx += 1
    return {".ROBLOSECURITY": token}

def get_user_agent():
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        "Mozilla/5.0 (X11; Linux x86_64)..."
    ])

def get_proxy_session():
    sess = requests.Session()
    if PROXIES:
        proxy = random.choice(PROXIES)
        sess.proxies.update({"http": proxy, "https": proxy})
    return sess

def safe_get(url):
    headers = {"User-Agent": get_user_agent(), "Accept": "application/json"}
    cookies = get_cookie()
    time.sleep(RATE_LIMIT_DELAY)
    sess = get_proxy_session()
    resp = sess.get(url, headers=headers, cookies=cookies, timeout=20)
    if not resp.ok:
        print(f"  ! HTTP {resp.status_code} for URL: {url}")
        resp.raise_for_status()
    return resp.json()

# ─── Roblox Data Fetching ─────────────────────────────────────────────────────

def fetch_user_games(uid):
    games, cursor = [], ""
    while True:
        url = (
            f"https://games.roblox.com/v1/users/{uid}/games"
            f"?accessFilter=All&limit=50&sortOrder=Asc&cursor={cursor}"
        )
        data = safe_get(url)
        for g in data.get("data", []):
            games.append({"id": str(g["id"]), "name": g["name"]})
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
    return games

def fetch_group_games(group_id):
    games, cursor = [], ""
    while True:
        url = (
            f"https://games.roblox.com/v1/groups/{group_id}/games"
            f"?accessFilter=Public&limit=50&sortOrder=Asc&cursor={cursor}"
        )
        data = safe_get(url)
        for g in data.get("data", []):
            games.append({"id": str(g["id"]), "name": g["name"]})
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
    return games

def fetch_game_metadata(ids):
    chunk = ",".join(ids)
    url = f"https://games.roblox.com/v1/games?universeIds={chunk}"
    data = safe_get(url)
    return data.get("data", [])

# ─── Main Orchestration ───────────────────────────────────────────────────────

def main():
    # 1) Gather game IDs from users
    all_ids = set()
    for uid in CREATORS:
        print(f"> Fetching games for user {uid}")
        try:
            for g in fetch_user_games(uid):
                all_ids.add(g["id"])
            # Optionally fetch group games too:
            # for gid in fetch_user_groups(uid): ...
        except Exception as e:
            print(f"  ! Error for {uid}: {e}")
        print(f"  → {len(all_ids)} IDs so far")

    all_ids = list(all_ids)
    print(f"\nTotal unique games: {len(all_ids)}\n")

    # 2) Metadata batches
    rows = []
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]
        print(f"> Meta batch {i//BATCH_SIZE + 1} ({len(batch)} IDs)")
        try:
            for g in fetch_game_metadata(batch):
                rows.append({f: g.get(f) for f in FIELDS})
        except Exception as e:
            print(f"  ! Batch error: {e}")

    # 3) Save CSV
    df = pd.DataFrame(rows)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Saved {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
