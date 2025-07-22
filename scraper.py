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
BATCH_SIZE       = int(os.getenv("BATCH_SIZE","100"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY","0.7"))
PROXY_API_KEY    = os.getenv("PROXY_API_KEY","")
# Correct base per your provider docs:
PROXY_API_BASE   = os.getenv("PROXY_API_BASE","https://proxy‑ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [
    p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()
]

FIELDS = [
    "universeId","name","playing","visits",
    "favorites","likeCount","dislikeCount",
    "genre","price","creatorType"
]

# ─── Proxy Handling ────────────────────────────────────────────────────────────

def fetch_proxies_from_api():
    """Call GET /client-api/v1/<key>/get/proxies to list all proxy types."""
    if not PROXY_API_KEY:
        return []
    try:
        url = f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Proxy API error, using fallback list: {e}")
        return []
    data = resp.json()
    proxies = []
    # IPv4
    for entry in data.get("ipv4", []):
        ip = entry["ip"]
        login = entry["authInfo"]["login"]
        pw    = entry["authInfo"]["password"]
        if port := entry.get("httpsPort"):
            proxies.append(f"http://{login}:{pw}@{ip}:{port}")
        if port := entry.get("socks5Port"):
            proxies.append(f"socks5://{login}:{pw}@{ip}:{port}")
    # IPv6
    for order in data.get("ipv6", []):
        for ip_info in order.get("ips", []):
            proto = ip_info["protocol"].lower()
            hostport = ip_info["ip"]
            auth = ip_info["authInfo"]
            proxies.append(f"{proto}://{auth['login']}:{auth['password']}@{hostport}")
    # ISP
    for entry in data.get("isp", []):
        ip = entry["ip"]
        login,pw = entry["authInfo"]["login"], entry["authInfo"]["password"]
        if port := entry.get("httpsPort"):
            proxies.append(f"http://{login}:{pw}@{ip}:{port}")
        if port := entry.get("socks5Port"):
            proxies.append(f"socks5://{login}:{pw}@{ip}:{port}")
    # Mobile
    for entry in data.get("mobile", []):
        ip = entry["ip"]
        login,pw = entry["authInfo"]["login"], entry["authInfo"]["password"]
        if port := entry.get("httpsPort"):
            proxies.append(f"http://{login}:{pw}@{ip}:{port}")
        if port := entry.get("socks5Port"):
            proxies.append(f"socks5://{login}:{pw}@{ip}:{port}")
    return proxies

# build PROXIES
_prox = fetch_proxies_from_api()
PROXIES = _prox if _prox else PROXY_URLS_FALLBACK
print(f"→ Using {len(PROXIES)} proxies")

# ─── HTTP Utilities ────────────────────────────────────────────────────────────

_cookie_idx = 0
def get_cookie():
    global _cookie_idx
    tok = TOKENS[_cookie_idx % len(TOKENS)]
    _cookie_idx += 1
    return {".ROBLOSECURITY": tok}

def get_user_agent():
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        "Mozilla/5.0 (X11; Linux x86_64)..."
    ])

def get_session():
    sess = requests.Session()
    if PROXIES:
        p = random.choice(PROXIES)
        sess.proxies.update({"http":p,"https":p})
    return sess

def safe_get(url):
    headers = {"User-Agent":get_user_agent(),"Accept":"application/json"}
    cookies = get_cookie()
    time.sleep(RATE_LIMIT_DELAY)
    sess = get_session()
    try:
        r = sess.get(url, headers=headers, cookies=cookies, timeout=20)
        if not r.ok:
            print(f"  ! HTTP {r.status_code} @ {url}")
            r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ! Request error: {e}")
        raise

# ─── Roblox Endpoints ─────────────────────────────────────────────────────────

def fetch_creator_games(uid):
    """Use v1/games?creatorIds=... without empty cursor param."""
    games, cursor = [], None
    while True:
        url = (f"https://games.roblox.com/v1/games"
               f"?creatorIds={uid}&limit=50&sortOrder=Asc")
        if cursor:
            url += f"&cursor={cursor}"
        data = safe_get(url)
        for g in data.get("data",[]):
            games.append({
                "id": str(g.get("universeId") or g.get("id")),
                "name": g.get("name")
            })
        cursor = data.get("nextPageCursor")
        if not cursor:
            break
    return games

def fetch_game_meta(ids):
    chunk = ",".join(ids)
    url = f"https://games.roblox.com/v1/games?universeIds={chunk}"
    data = safe_get(url)
    return data.get("data",[])

# ─── Main Workflow ─────────────────────────────────────────────────────────────

def main():
    all_ids = set()
    for uid in CREATORS:
        print(f"> Creator {uid}")
        try:
            for g in fetch_creator_games(uid):
                all_ids.add(g["id"])
        except:
            print(f"  ! Skip {uid}")
        print(f"  → {len(all_ids)} IDs so far")

    all_ids = list(all_ids)
    print(f"\nTotal games: {len(all_ids)}\n")

    rows = []
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i:i+BATCH_SIZE]
        print(f"> Batch {i//BATCH_SIZE+1} ({len(batch)})")
        try:
            for g in fetch_game_meta(batch):
                rows.append({f:g.get(f) for f in FIELDS})
        except:
            print(f"  ! Batch {i//BATCH_SIZE+1} error")

    df = pd.DataFrame(rows)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Wrote {len(df)} records to test_data.csv")

if __name__=="__main__":
    main()
