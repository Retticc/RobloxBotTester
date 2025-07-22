import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

# Roblox creators to test (comma‑separated)
CREATORS = os.getenv("TARGET_CREATORS", "").split(",")

# Five .ROBLOSECURITY tokens (comma‑separated)
TOKENS = os.getenv("ROBLOSECURITY_TOKENS", "").split(",")

# How many universeIds per /v1/games batch call
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Delay between each HTTP request (seconds)
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))

# Proxy setup: either a static list or fetched via your provider’s API
PROXY_API_KEY = os.getenv("PROXY_API_KEY")      # your VbHjBpTosZvZ key
PROXY_URLS    = os.getenv("PROXY_URLS", "").split(",")  # optional fallback

API_BASE_URL = "https://api.ipv4proxy.com/client-api/v1"

# Fields to capture from the Roblox /v1/games endpoint
FIELDS = [
    "universeId", "name", "playing", "visits",
    "favorites", "likeCount", "dislikeCount",
    "genre", "price", "creatorType"
]

# ─── Proxy Handling ────────────────────────────────────────────────────────────

def fetch_ipv6_proxies():
    """Fetch active IPv6 proxies from provider API."""
    url = f"{API_BASE_URL}/{PROXY_API_KEY}/get/proxies"
    params = {"proxyType": "ipv6"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    proxies = []
    for order in data.get("proxies", []):
        for ip_info in order.get("ips", []):
            proto = ip_info.get("protocol", "HTTP").lower()
            hostport = ip_info["ip"]
            auth = ip_info.get("authInfo", {})
            user = auth.get("login")
            pw   = auth.get("password")
            if user and pw:
                proxies.append(f"{proto}://{user}:{pw}@{hostport}")
            else:
                # no auth info, just host:port
                proxies.append(f"{proto}://{hostport}")
    return proxies

# Build the PROXIES list
if PROXY_API_KEY:
    try:
        PROXIES = fetch_ipv6_proxies()
        print(f"Fetched {len(PROXIES)} proxies via API")
    except Exception as e:
        print(f"Error fetching proxies via API: {e}")
        PROXIES = [p for p in PROXY_URLS if p]
else:
    PROXIES = [p for p in PROXY_URLS if p]

# ─── Request Utilities ─────────────────────────────────────────────────────────

_cookie_idx = 0
def get_cookie():
    """Round‑robin selection of .ROBLOSECURITY tokens."""
    global _cookie_idx
    token = TOKENS[_cookie_idx % len(TOKENS)]
    _cookie_idx += 1
    return {".ROBLOSECURITY": token}

def get_user_agent():
    """Random User‑Agent from a small fallback list."""
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        "Mozilla/5.0 (X11; Linux x86_64)..."
    ])

def get_proxy_session():
    """Return a requests.Session routed through a random proxy (if any)."""
    sess = requests.Session()
    if PROXIES:
        proxy = random.choice(PROXIES)
        sess.proxies.update({
            "http":  proxy,
            "https": proxy
        })
    return sess

def safe_get(url):
    """Perform a GET with UA rotation, proxy rotation, and rate‑limit delay."""
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "application/json"
    }
    cookies = get_cookie()
    time.sleep(RATE_LIMIT_DELAY)
    sess = get_proxy_session()
    resp = sess.get(url, headers=headers, cookies=cookies, timeout=20)
    resp.raise_for_status()
    return resp.json()

# ─── Roblox Data Fetching ─────────────────────────────────────────────────────

def fetch_user_games(uid):
    """Fetch all games owned by a user via /v2/users/{uid}/games."""
    games, cursor = [], ""
    while True:
        url = (
            f"https://games.roblox.com/v2/users/{uid}/games"
            f"?accessFilter=All&limit=50&sortOrder=Asc&cursor={cursor}"
        )
        data = safe_get(url)
        for g in data.get("data", []):
            games.append({"id": str(g["id"]), "name": g["name"]})
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
    return games

def fetch_game_metadata(ids):
    """Batch fetch game metadata for a list of universeIds."""
    chunk = ",".join(ids)
    url = f"https://games.roblox.com/v1/games?universeIds={chunk}"
    data = safe_get(url)
    return data.get("data", [])

# ─── Main Orchestration ───────────────────────────────────────────────────────

def main():
    # 1) Gather all game IDs from the target creators
    all_ids = set()
    for uid in CREATORS:
        print(f"> Fetching games for user {uid}")
        try:
            for g in fetch_user_games(uid):
                all_ids.add(g["id"])
        except Exception as e:
            print(f"  ! Error fetching games for {uid}: {e}")
        print(f"  → {len(all_ids)} total IDs so far")

    all_ids = list(all_ids)
    print(f"\nTotal unique games to fetch metadata for: {len(all_ids)}\n")

    # 2) Batch metadata fetch
    rows = []
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]
        print(f"> Fetching metadata batch {i//BATCH_SIZE + 1} ({len(batch)} IDs)")
        try:
            meta = fetch_game_metadata(batch)
            for g in meta:
                rows.append({f: g.get(f) for f in FIELDS})
        except Exception as e:
            print(f"  ! Error fetching metadata for batch {i//BATCH_SIZE + 1}: {e}")

    # 3) Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Saved {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
