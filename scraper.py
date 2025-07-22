import os, time, random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Config from env
CREATORS = os.getenv("TARGET_CREATORS", "").split(",")
TOKENS   = os.getenv("ROBLOSECURITY_TOKENS", "").split(",")
BATCH    = int(os.getenv("BATCH_SIZE", "100"))
DELAY    = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))

# Fields we want from the /v1/games API
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

def get_headers():
    ua = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."
    ])
    return {
        "User-Agent": ua,
        "Accept": "application/json"
    }

def fetch_user_games(uid):
    games, cursor = [], ""
    while True:
        url = (
            f"https://games.roblox.com/v2/users/{uid}/games"
            f"?accessFilter=All&limit=50&sortOrder=Asc&cursor={cursor}"
        )
        res = requests.get(url, headers=get_headers(), cookies=get_cookie())
        data = res.json()
        for g in data.get("data", []):
            games.append({"id": str(g["id"]), "name": g["name"]})
        cursor = data.get("nextPageCursor") or ""
        if not cursor:
            break
        time.sleep(DELAY)
    return games

def fetch_game_metadata(ids):
    chunk = ",".join(ids)
    url = f"https://games.roblox.com/v1/games?universeIds={chunk}"
    res = requests.get(url, headers=get_headers(), cookies=get_cookie())
    return res.json().get("data", [])

def main():
    all_ids = set()
    for uid in CREATORS:
        for g in fetch_user_games(uid):
            all_ids.add(g["id"])
        print(f"> Collected {len(all_ids)} total game IDs so far")
    all_ids = list(all_ids)

    rows = []
    for i in range(0, len(all_ids), BATCH):
        batch = all_ids[i : i + BATCH]
        meta = fetch_game_metadata(batch)
        for g in meta:
            row = {f: g.get(f, None) for f in FIELDS}
            rows.append(row)
        print(f"> Fetched metadata for batch {i//BATCH+1}")
        time.sleep(DELAY)

    df = pd.DataFrame(rows)
    df.to_csv("test_data.csv", index=False)
    print(f"✅ Saved {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
