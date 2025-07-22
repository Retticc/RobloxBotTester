import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
CREATORS         = [u.strip() for u in os.getenv("TARGET_CREATORS", "").split(",") if u.strip()]
TOKENS           = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS", "").split(",") if t.strip()]
# Roblox's API only accepts up to 100 IDs per request
MAX_IDS_PER_REQ  = 100
env_batch        = int(os.getenv("BATCH_SIZE", "100"))
BATCH_SIZE       = max(1, min(env_batch, MAX_IDS_PER_REQ))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))

# Proxy service settings
PROXY_API_KEY       = os.getenv("PROXY_API_KEY", "")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE", "https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

# Fields to pull into CSV
FIELDS = [
    "universeId", "name", "playing", "visits",
    "favorites", "likeCount", "dislikeCount",
    "genre", "price", "creatorType", "creator"
]

# ─── Proxy Handling ────────────────────────────────────────────────────────────
def fetch_proxies_from_api():
    """Fetch proxies from your provider's API."""
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
    # IPv4
    for e in data.get("ipv4", []):
        ip   = e["ip"]
        auth = e["authInfo"]
        for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
            if port := e.get(port_key):
                proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    # IPv6
    for order in data.get("ipv6", []):
        for ipinfo in order.get("ips", []):
            proto  = ipinfo["protocol"].lower()
            ipport = ipinfo["ip"]
            auth   = ipinfo["authInfo"]
            proxies.append(f"{proto}://{auth['login']}:{auth['password']}@{ipport}")
    # ISP & Mobile
    for key in ("isp","mobile"):
        for e in data.get(key, []):
            ip   = e["ip"]
            auth = e["authInfo"]
            for port_key, scheme in (("httpsPort","http"),("socks5Port","socks5")):
                if port := e.get(port_key):
                    proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    return proxies

_proxies = fetch_proxies_from_api()
PROXIES = _proxies if _proxies else PROXY_URLS_FALLBACK
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
        "Mozilla/5.0 (Windows NT 10; Win64; x64)…",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)…",
        "Mozilla/5.0 (X11; Linux x86_64)…"
    ])

def get_session():
    sess = requests.Session()
    if PROXIES:
        p = random.choice(PROXIES)
        sess.proxies.update({"http": p, "https": p})
    return sess

def safe_get(url, retries=3):
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            if r.ok:
                return r.json()
            print(f"[GET] HTTP {r.status_code} @ {url}")
            r.raise_for_status()
        except Exception as e:
            print(f"[GET] attempt {i+1} error: {e}")
            if i < retries - 1:
                time.sleep(2**i)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}")

def safe_post(url, json=None, retries=3):
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.post(
                url,
                headers={"User-Agent": get_user_agent(), "Accept": "application/json"},
                cookies=get_cookie(),
                json=json,
                timeout=30
            )
            if r.ok:
                return r.json()
            print(f"[POST] HTTP {r.status_code} @ {url} payload={json}")
            r.raise_for_status()
        except Exception as e:
            print(f"[POST] attempt {i+1} error: {e}")
            if i < retries - 1:
                time.sleep(2**i)
    raise RuntimeError(f"POST failed after {retries} attempts: {url}")

# ─── Roblox API – Game Listing ────────────────────────────────────────────────
def fetch_creator_games(user_id):
    """
    POST to /v1/universes/list to list all universes (games) by a creator.
    """
    games, cursor = [], ""
    url = "https://games.roblox.com/v1/universes/list"
    while True:
        payload = {
            "model": {
                "creatorType": "User",
                "creatorId":   int(user_id),
                "limit":       50,
                "cursor":      cursor
            }
        }
        try:
            data = safe_post(url, json=payload)
        except Exception as e:
            print(f"[UniversesList] failed for {user_id}: {e}")
            return games

        for uni in data.get("universes", []):
            games.append({
                "universeId": str(uni.get("id") or uni.get("universeId")),
                "name":       uni.get("name", "")
            })
        cursor = data.get("nextPageCursor", "")
        if not cursor:
            break

    return games

# ─── Roblox API – Metadata & Votes ────────────────────────────────────────────
def get_game_details(universe_ids):
    """Fetch metadata for a list of universe IDs."""
    details = []
    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        ids = ",".join(str(uid) for uid in chunk)
        url = f"https://games.roblox.com/v1/games?universeIds={ids}"
        try:
            data = safe_get(url)
            details.extend(data.get("data", []))
        except Exception as e:
            print(f"[Meta] chunk {i//BATCH_SIZE+1} failed: {e}")
    return details

def get_game_votes(universe_ids):
    """Fetch vote counts for a list of universe IDs."""
    votes = {}
    for i in range(0, len(universe_ids), BATCH_SIZE):
        chunk = universe_ids[i : i + BATCH_SIZE]
        ids = ",".join(str(uid) for uid in chunk)
        url = f"https://games.roblox.com/v1/games/votes?universeIds={ids}"
        try:
            data = safe_get(url)
            for v in data.get("data", []):
                uid = str(v.get("id", ""))
                votes[uid] = {
                    "upVotes": v.get("upVotes", 0),
                    "downVotes": v.get("downVotes", 0),
                }
        except Exception as e:
            print(f"[Votes] chunk {i//BATCH_SIZE+1} failed: {e}")
    return votes

# ─── Main Workflow ─────────────────────────────────────────────────────────────
def main():
    print("Starting Roblox data collection...\n")
    all_ids = set()

    for uid in CREATORS:
        print(f"> Creator {uid}")
        games = fetch_creator_games(uid)
        print(f"  → Found {len(games)}")
        for g in games:
            all_ids.add(g["universeId"])
        print(f"  → Total unique so far: {len(all_ids)}")

    all_ids = list(all_ids)
    if not all_ids:
        print("No games found; exiting.")
        return

    print("\n> Fetching metadata…")
    meta = get_game_details(all_ids)

    print("> Fetching votes…")
    votes = get_game_votes(all_ids)

    records = []
    for g in meta:
        uid = str(g.get("universeId") or g.get("id"))
        records.append({
            "universeId":  uid,
            "name":        g.get("name",""),
            "playing":     g.get("playing",0),
            "visits":      g.get("visits",0),
            "favorites":   g.get("favoritedCount",0),
            "likeCount":   votes.get(uid,{}).get("upVotes",0),
            "dislikeCount":votes.get(uid,{}).get("downVotes",0),
            "genre":       g.get("genre",""),
            "price":       g.get("price",0),
            "creatorType": g.get("creator",{}).get("type",""),
            "creator":     g.get("creator",{}).get("name","")
        })

    df = pd.DataFrame(records)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Wrote {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
