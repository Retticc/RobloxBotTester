import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

CREATORS = [u.strip() for u in os.getenv("TARGET_CREATORS", "").split(",") if u.strip()]
TOKENS = [t.strip() for t in os.getenv("ROBLOSECURITY_TOKENS", "").split(",") if t.strip()]
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))

# Proxy settings
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
PROXY_API_BASE = os.getenv("PROXY_API_BASE", "https://proxy-ipv4.com/client-api/v1")
PROXY_URLS_FALLBACK = [p.strip() for p in os.getenv("PROXY_URLS", "").split(",") if p.strip()]

# Fields to extract from game metadata
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
        url = f"{PROXY_API_BASE}/{PROXY_API_KEY}/get/proxies"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        proxies = []
        
        # Parse IPv4 proxies
        for entry in data.get("ipv4", []):
            ip = entry["ip"]
            auth = entry["authInfo"]
            login, password = auth["login"], auth["password"]
            
            if https_port := entry.get("httpsPort"):
                proxies.append(f"http://{login}:{password}@{ip}:{https_port}")
            if socks5_port := entry.get("socks5Port"):
                proxies.append(f"socks5://{login}:{password}@{ip}:{socks5_port}")
        
        # Parse IPv6 proxies
        for order in data.get("ipv6", []):
            for ip_info in order.get("ips", []):
                ip = ip_info["ip"]
                protocol = ip_info["protocol"].lower()
                auth = ip_info["authInfo"]
                login, password = auth["login"], auth["password"]
                proxies.append(f"{protocol}://{login}:{password}@{ip}")
        
        # Parse ISP proxies
        for entry in data.get("isp", []):
            ip = entry["ip"]
            auth = entry["authInfo"]
            login, password = auth["login"], auth["password"]
            
            if https_port := entry.get("httpsPort"):
                proxies.append(f"http://{login}:{password}@{ip}:{https_port}")
            if socks5_port := entry.get("socks5Port"):
                proxies.append(f"socks5://{login}:{password}@{ip}:{socks5_port}")
        
        # Parse Mobile proxies
        for entry in data.get("mobile", []):
            ip = entry["ip"]
            auth = entry["authInfo"]
            login, password = auth["login"], auth["password"]
            
            if https_port := entry.get("httpsPort"):
                proxies.append(f"http://{login}:{password}@{ip}:{https_port}")
            if socks5_port := entry.get("socks5Port"):
                proxies.append(f"socks5://{login}:{password}@{ip}:{socks5_port}")
        
        return proxies
        
    except Exception as e:
        print(f"Proxy API error: {e}")
        return []

# Initialize proxies
_fetched_proxies = fetch_proxies_from_api()
PROXIES = _fetched_proxies if _fetched_proxies else PROXY_URLS_FALLBACK
print(f"→ Using {len(PROXIES)} proxies")

# ─── HTTP Utilities ────────────────────────────────────────────────────────────

_cookie_idx = 0
def get_cookie():
    global _cookie_idx
    if not TOKENS:
        return {}
    token = TOKENS[_cookie_idx % len(TOKENS)]
    _cookie_idx += 1
    return {".ROBLOSECURITY": token}

def get_user_agent():
    return random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ])

def get_session():
    sess = requests.Session()
    if PROXIES:
        proxy = random.choice(PROXIES)
        sess.proxies.update({"http": proxy, "https": proxy})
    return sess

def safe_get(url, retries=3):
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    }
    cookies = get_cookie()
    
    for attempt in range(retries):
        time.sleep(RATE_LIMIT_DELAY + random.uniform(0, 0.3))  # Add jitter
        sess = get_session()
        
        try:
            response = sess.get(url, headers=headers, cookies=cookies, timeout=30)
            if response.ok:
                return response.json()
            else:
                print(f"  ! HTTP {response.status_code} for {url}")
                if attempt == retries - 1:
                    response.raise_for_status()
        except Exception as e:
            print(f"  ! Attempt {attempt + 1} failed for {url}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# ─── Roblox API Functions ──────────────────────────────────────────────────────

def get_user_created_games(user_id):
    """Get games created by a specific user using the catalog API."""
    games = []
    cursor = ""
    
    while True:
        # Use the catalog search API to find games by creator
        url = f"https://catalog.roblox.com/v1/search/items"
        params = {
            "category": "Experiences",
            "creatorName": "",  # We'll get creator info separately
            "limit": 30,
            "sortType": "Relevance"
        }
        
        if cursor:
            params["cursor"] = cursor
            
        try:
            # First, let's try a different approach - get user info and then games
            user_url = f"https://users.roblox.com/v1/users/{user_id}"
            user_data = safe_get(user_url)
            username = user_data.get("name", "")
            
            if not username:
                print(f"  ! Could not get username for user {user_id}")
                break
                
            # Now search for games by this creator using the games API
            games_url = f"https://games.roblox.com/v2/users/{user_id}/games"
            games_params = {
                "accessFilter": "2",  # Public games only
                "limit": 50,
                "sortOrder": "Asc"
            }
            
            if cursor:
                games_params["cursor"] = cursor
                
            data = safe_get(games_url)
            
            for game in data.get("data", []):
                universe_id = game.get("id")
                if universe_id:
                    games.append({
                        "universeId": str(universe_id),
                        "name": game.get("name", ""),
                        "rootPlaceId": game.get("rootPlaceId", "")
                    })
            
            cursor = data.get("nextPageCursor", "")
            if not cursor:
                break
                
        except Exception as e:
            print(f"  ! Error fetching games for user {user_id}: {e}")
            # Try alternative method - use the experiences endpoint
            try:
                alt_url = f"https://games.roblox.com/v1/games/list"
                alt_params = {
                    "model.creatorId": user_id,
                    "model.creatorType": "User",
                    "model.sortOrder": "Asc",
                    "model.limit": 50
                }
                
                if cursor:
                    alt_params["model.cursor"] = cursor
                    
                alt_data = safe_get(alt_url)
                
                for game in alt_data.get("games", []):
                    universe_id = game.get("universeId")
                    if universe_id:
                        games.append({
                            "universeId": str(universe_id),
                            "name": game.get("name", ""),
                            "rootPlaceId": game.get("rootPlaceId", "")
                        })
                
                cursor = alt_data.get("nextPageCursor", "")
                if not cursor:
                    break
                    
            except Exception as alt_e:
                print(f"  ! Alternative method also failed: {alt_e}")
                break
    
    return games

# ─── Roblox Metadata & Votes (fixed) ──────────────────────────────────────────

def get_game_details(universe_ids):
    """POST to /v1/games with JSON body {'universeIds': [...]}."""
    details = []
    # chunk into 100s to avoid huge payloads
    for i in range(0, len(universe_ids), 100):
        chunk = universe_ids[i : i + 100]
        try:
            data = safe_post(
                "https://games.roblox.com/v1/games",
                json={"universeIds": chunk}
            )
            details.extend(data.get("data", []))
        except Exception as e:
            print(f"  ! Metadata chunk {i//100+1} failed: {e}")
    return details

def get_game_votes(universe_ids):
    """POST to /v1/games/votes with JSON body {'universeIds': [...]}."""
    votes = {}
    for i in range(0, len(universe_ids), 100):
        chunk = universe_ids[i : i + 100]
        try:
            data = safe_post(
                "https://games.roblox.com/v1/games/votes",
                json={"universeIds": chunk}
            )
            for v in data.get("data", []):
                uid = str(v.get("id", ""))
                votes[uid] = {
                    "upVotes":   v.get("upVotes", 0),
                    "downVotes": v.get("downVotes", 0)
                }
        except Exception as e:
            print(f"  ! Votes chunk {i//100+1} failed: {e}")
    return votes


# ─── Main Workflow ─────────────────────────────────────────────────────────────

def main():
    print("Starting Roblox game data collection...\n")
    
    all_universe_ids = set()
    
    # Collect games from each creator
    for creator_id in CREATORS:
        print(f"> Fetching games for creator {creator_id}")
        
        try:
            games = get_user_created_games(creator_id)
            for game in games:
                if game["universeId"]:
                    all_universe_ids.add(game["universeId"])
            
            print(f"  → Found {len(games)} games for creator {creator_id}")
            
        except Exception as e:
            print(f"  ! Failed to fetch games for creator {creator_id}: {e}")
        
        print(f"  → Total unique games so far: {len(all_universe_ids)}")
    
    all_universe_ids = list(all_universe_ids)
    print(f"\nTotal unique games to process: {len(all_universe_ids)}")
    
    if not all_universe_ids:
        print("No games found. Exiting.")
        return
    
    # Get detailed game information
    print("\nFetching detailed game information...")
    games_data = get_game_details(all_universe_ids)
    
    # Get vote information
    print("Fetching vote information...")
    votes_data = get_game_votes(all_universe_ids)
    
    # Process and combine data
    processed_games = []
    for game in games_data:
        universe_id = str(game.get("universeId", ""))
        
        # Get vote data
        vote_info = votes_data.get(universe_id, {})
        
        processed_game = {
            "universeId": universe_id,
            "name": game.get("name", ""),
            "playing": game.get("playing", 0),
            "visits": game.get("visits", 0),
            "favorites": game.get("favoritedCount", 0),
            "likeCount": vote_info.get("upVotes", 0),
            "dislikeCount": vote_info.get("downVotes", 0),
            "genre": game.get("genre", ""),
            "price": game.get("price", 0),
            "creatorType": game.get("creator", {}).get("type", ""),
            "creator": game.get("creator", {}).get("name", "")
        }
        
        processed_games.append(processed_game)
    
    # Save to CSV
    if processed_games:
        df = pd.DataFrame(processed_games)
        df.to_csv("test_data.csv", index=False)
        print(f"\n✅ Successfully saved {len(df)} games to test_data.csv")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total games: {len(df)}")
        print(f"- Total visits: {df['visits'].sum():,}")
        print(f"- Average visits per game: {df['visits'].mean():.0f}")
        print(f"- Games currently being played: {df[df['playing'] > 0].shape[0]}")
    else:
        print("No game data was collected.")

if __name__ == "__main__":
    main()
