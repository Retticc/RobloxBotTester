import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Config from env
CREATORS = os.getenv("TARGET_CREATORS", "").split(",")
TOKENS   = os.getenv("ROBLOSECURITY_TOKENS", "").split(",")
BATCH    = int(os.getenv("BATCH_SIZE", "100"))
DELAY    = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))
PROXIES  = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

# Data fields to capture
FIELDS = [
    "universeId", "name", "playing", "visits",
    "favorites", "likeCount", "dislikeCount",
    "genre", "price", "creatorType"
]

# Round‑robin index for cookies
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
        "Mozilla/5.0 (X11; Linux x86_64)…"
    ])

def get_proxy_session():
    """Return a requests.Session() that routes via a random proxy (if configured)."""
    sess = requests.Session()
    if PROXIES:
        proxy = random.choice(PROXIES)
        sess.proxies.update({
            "http":  proxy,
            "https": proxy
        })
    return sess

def safe_get(url):
    """Fetch URL with random UA, proxy rotation, and rate‑limit delay."""
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "application/json"
    }
    cookies = get_cookie()
    time.sleep(DELAY)
    sess = get_proxy_session()
    resp = sess.get(url, headers=headers, cookies=cookies, timeout=20)
    resp.raise_for_status()
    return resp.json()

def fetch_user_games(uid):
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
    chunk = ",".join(ids)
    url   = f"https://games.roblox.com/v1/games?universeIds={chunk}"
    data  = safe_get(url)
    return data.get("data", [])

def main():
    # 1) Gather all game IDs from target creators
    all_ids = set()
    for uid in CREATORS:
        print(f"> Fetching games for user {uid}")
        for g in fetch_user_games(uid):
            all_ids.add(g["id"])
        print(f"  → {len(all_ids)} total IDs so far")

    # 2) Batch metadata fetch
    rows = []
    all_ids = list(all_ids)
    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i : i + BATCH]
        print(f"> Fetching metadata batch {i//BATCH + 1} ({len(batch)} IDs)")
        meta = fetch_game_metadata(batch)
        for g in meta:
            row = {f: g.get(f) for f in FIELDS}
            rows.append(row)

    # 3) Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv("test_data.csv", index=False)
    print(f"✅ Saved {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
