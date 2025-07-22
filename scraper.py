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

# Proxy service settings
PROXY_API_KEY       = os.getenv("PROXY_API_KEY","")
PROXY_API_BASE      = os.getenv("PROXY_API_BASE","https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS","").split(",") if p.strip()]

# Fields to pull into CSV
FIELDS = [
    "universeId","name","playing","visits",
    "favorites","likeCount","dislikeCount",
    "genre","price","creatorType","creator"
]

# ─── Proxy Handling ────────────────────────────────────────────────────────────
def fetch_proxies_from_api():
    if not PROXY_API_KEY:
        return []
    try:
        resp = requests.get(
            f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ProxyAPI] error: {e}")
        return []
    proxies = []
    # IPv4
    for e in data.get("ipv4", []):
        ip = e["ip"]
        auth = e["authInfo"]
        for port_key, scheme in (("httpsPort","http"), ("socks5Port","socks5")):
            if port := e.get(port_key):
                proxies.append(f"{scheme}://{auth['login']}:{auth['password']}@{ip}:{port}")
    # IPv6
    for order in data.get("ipv6", []):
        for ipinfo in order.get("ips", []):
            proto = ipinfo["protocol"].lower()
            ipport = ipinfo["ip"]
            auth  = ipinfo["authInfo"]
            proxies.append(f"{proto}://{auth['login']}:{auth['password']}@{ipport}")
    # ISP & Mobile
    for key in ("isp","mobile"):
        for e in data.get(key, []):
            ip   = e["ip"]
            auth = e["authInfo"]
            for port_key, scheme in (("httpsPort","http"), ("socks5Port","socks5")):
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
        sess.proxies.update({"http":p,"https":p})
    return sess

def safe_get(url, retries=3):
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.get(
                url,
                headers={"User-Agent": get_user_agent(), "Accept":"application/json"},
                cookies=get_cookie(),
                timeout=30
            )
            if r.ok:
                return r.json()
            print(f"[GET] HTTP {r.status_code} @ {url}")
            r.raise_for_status()
        except Exception as e:
            print(f"[GET] attempt {i+1} error: {e}")
            time.sleep(2**i)
    raise RuntimeError(f"GET failed after {retries} retries: {url}")

def safe_post(url, json=None, retries=3):
    for i in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.random()*0.3)
        sess = get_session()
        try:
            r = sess.post(
                url,
                headers={"User-Agent": get_user_agent(), "Accept":"application/json"},
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
            time.sleep(2**i)
    raise RuntimeError(f"POST failed after {retries} retries: {url}")

# ─── Roblox Game Listing ──────────────────────────────────────────────────────
def fetch_creator_games(user_id):
    """POST to /v1/games/list to get all games by a creator."""
    games, cursor = [], ""
    url = "https://games.roblox.com/v1/games/list"
    while True:
        payload = {
            "model": {
                "creatorId":   int(user_id),
                "creatorType": "User",
                "limit":       50,
                "cursor":      cursor
            }
        }
        try:
            data = safe_post(url, json=payload)
        except Exception as e:
            print(f"[GamesList] failed for {user_id}: {e}")
            return games
        for g in data.get("games", []):
            games.append({
                "universeId": str(g.get("universeId") or g.get("id")),
                "name":       g.get("name","")
            })
        cursor = data.get("nextPageCursor","")
        if not cursor:
            break
    return games

# ─── Roblox Metadata & Votes ─────────────────────────────────────────────────
def fetch_game_meta(ids):
    """POST to /v1/games with JSON body {'universeIds': [...] }."""
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i : i+100]
        try:
            data = safe_post(
                "https://games.roblox.com/v1/games",
                json={"universeIds": chunk}
            )
            out.extend(data.get("data", []))
        except Exception:
            print(f"[Meta] chunk {i//100+1} failed")
    return out

def fetch_game_votes(ids):
    """POST to /v1/games/votes with JSON body {'universeIds': [...] }."""
    votes = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i+100]
        try:
            data = safe_post(
                "https://games.roblox.com/v1/games/votes",
                json={"universeIds": chunk}
            )
            for v in data.get("data", []):
                uid = str(v.get("id",""))
                votes[uid] = {
                    "upVotes":   v.get("upVotes", 0),
                    "downVotes": v.get("downVotes", 0)
                }
        except Exception:
            print(f"[Votes] chunk {i//100+1} failed")
    return votes

# ─── Main Workflow ─────────────────────────────────────────────────────────────
def main():
    print("Starting collection…\n")
    all_ids = set()

    # 1) List games per creator
    for uid in CREATORS:
        print(f"> Creator {uid}")
        gms = fetch_creator_games(uid)
        print(f"  → Found {len(gms)} games")
        for g in gms:
            all_ids.add(g["universeId"])
        print(f"  → Total unique so far: {len(all_ids)}")

    all_ids = list(all_ids)
    if not all_ids:
        print("No games found; exiting.")
        return

    # 2) Fetch metadata
    print("\n> Fetching metadata…")
    games_meta = fetch_game_meta(all_ids)

    # 3) Fetch votes
    print("> Fetching votes…")
    votes = fetch_game_votes(all_ids)

    # 4) Merge & save
    records = []
    for gm in games_meta:
        uid = str(gm.get("universeId") or gm.get("id"))
        rec = {
            "universeId":  uid,
            "name":        gm.get("name",""),
            "playing":     gm.get("playing",0),
            "visits":      gm.get("visits",0),
            "favorites":   gm.get("favoritedCount",0),
            "likeCount":   votes.get(uid,{}).get("upVotes",0),
            "dislikeCount":votes.get(uid,{}).get("downVotes",0),
            "genre":       gm.get("genre",""),
            "price":       gm.get("price",0),
            "creatorType": gm.get("creator",{}).get("type",""),
            "creator":     gm.get("creator",{}).get("name","")
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv("test_data.csv", index=False)
    print(f"\n✅ Wrote {len(df)} records to test_data.csv")

if __name__ == "__main__":
    main()
