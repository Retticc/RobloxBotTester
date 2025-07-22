def get_game_details(universe_ids):
    """Get detailed information for games by universe IDs."""
    if not universe_ids:
        return []
        
    chunk_size = 100
    all_games = []
    
    for i in range(0, len(universe_ids), chunk_size):
        chunk = universe_ids[i:i + chunk_size]
        universe_ids_str = ",".join(chunk)
        url = f"https://games.roblox.com/v1/games?universeIds={universe_ids_str}"
        
        try:
            data = safe_get(url)
            if data is not None:
                games = data.get("data", [])
                all_games.extend(games)
        except Exception as e:
            print(f"  ! Error fetching game details for chunk: {e}")
            continue
    return all_games

# --- REMOVE this block ---
# def get_game_votes(universe_ids):
#     ...
# --- END REMOVE ---

# ... (rest of your code) ...

def main():
    print("Starting Roblox game data collection...\n")
    all_universe_ids = set()
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

    # Fetch detailed game info (includes votes)
    print("\nFetching detailed game information...")
    games_data = get_game_details(all_universe_ids)

    # Remove get_game_votes, just use votes from games_data
    processed_games = []
    for game in games_data:
        universe_id = str(game.get("universeId", ""))
        processed_game = {
            "universeId": universe_id,
            "name": game.get("name", ""),
            "playing": game.get("playing", 0),
            "visits": game.get("visits", 0),
            "favorites": game.get("favoritedCount", 0),
            "likeCount": game.get("upVotes", game.get("likeCount", 0)),
            "dislikeCount": game.get("downVotes", game.get("dislikeCount", 0)),
            "genre": game.get("genre", ""),
            "price": game.get("price", 0),
            "creatorType": game.get("creator", {}).get("type", ""),
            "creator": game.get("creator", {}).get("name", "")
        }
        processed_games.append(processed_game)
    if processed_games:
        df = pd.DataFrame(processed_games)
        df.to_csv("test_data.csv", index=False)
        print(f"\n✅ Successfully saved {len(df)} games to test_data.csv")
        print(f"\nSummary:")
        print(f"- Total games: {len(df)}")
        print(f"- Total visits: {df['visits'].sum():,}")
        print(f"- Average visits per game: {df['visits'].mean():.0f}")
        print(f"- Games currently being played: {df[df['playing'] > 0].shape[0]}")
    else:
        print("No game data was collected.")
