def create_style_summary_image(self, style_games, style_name, style_data, image_type):
        """Create a summary image showing examples of a style"""
        try:
            # Create a summary image with examples and stats
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Top Performing {image_type.title()} Style: {style_name.replace('_', ' ').title()}", fontsize=16)
            
            # Plot example images
            for i, (_, game) in enumerate(style_games.head(3).iterrows()):
                if i < 3:
                    ax = axes[0, i]
                    # Here you would display the actual image if stored
                    ax.text(0.5, 0.5, f"Game: {game['game_name'][:20]}...\nPlayers: {game['avg_playing']:.0f}\nScore: {game['performance_score']:.1f}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"Example {i+1}")
                    ax.axis('off')
            
            # Plot statistics
            stats_ax = axes[1, :]
            stats_text = f"""
Style Characteristics:
• Performance Score: {style_data['performance_score_mean']:.2f}
• Average Players: {style_data['avg_playing_mean']:.0f}
• Like Ratio: {style_data['like_ratio_mean']:.1%}
• Brightness: {style_data['brightness_mean']:.0f}/255
• Saturation: {style_data['saturation_mean']:.0f}%
• Has Faces: {style_data['has_faces']*100:.0f}% of games
• Games Analyzed: {style_data['performance_score_count']}
            """
            
            for ax in stats_ax:
                ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12, va='center')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/reports/{image_type}_{style_name}_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[examples] Error creating summary for {style_name}: {e}")
    
    def save_trending_assets_to_database(self):
        """Save currently trending visual assets and keywords to database tables"""
        print("[database] Saving trending assets and keywords to database...")
        
        with self.get_conn() as conn:
            # Create trending assets tables
            with conn.cursor() as cur:
                # Table for trending visual assets
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trending_visual_assets (
                        id SERIAL PRIMARY KEY,
                        game_id BIGINT NOT NULL,
                        game_name TEXT NOT NULL,
                        asset_type TEXT NOT NULL, -- 'icon' or 'thumbnail'
                        visual_style TEXT NOT NULL,
                        performance_score FLOAT NOT NULL,
                        avg_players FLOAT NOT NULL,
                        like_ratio FLOAT NOT NULL,
                        brightness FLOAT,
                        saturation FLOAT,
                        has_faces BOOLEAN,
                        is_minimalist BOOLEAN,
                        trend_rank INTEGER,
                        analysis_date TIMESTAMP DEFAULT NOW(),
                        image_hash TEXT,
                        UNIQUE(game_id, asset_type, analysis_date::date)
                    );
                """)
                
                # Table for trending keywords with detailed metrics
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trending_keywords_archive (
                        id SERIAL PRIMARY KEY,
                        keyword TEXT NOT NULL,
                        popularity_rank INTEGER NOT NULL,
                        success_correlation FLOAT NOT NULL,
                        total_games INTEGER NOT NULL,
                        successful_games INTEGER NOT NULL,
                        avg_players_with_keyword FLOAT NOT NULL,
                        avg_players_without_keyword FLOAT NOT NULL,
                        performance_lift FLOAT NOT NULL, -- how much better games with this keyword perform
                        trend_direction TEXT, -- 'rising', 'stable', 'declining'
                        analysis_date TIMESTAMP DEFAULT NOW(),
                        category TEXT, -- 'genre', 'theme', 'style', etc.
                        UNIQUE(keyword, analysis_date::date)
                    );
                """)
                
                # Table for top performing individual assets with metadata
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS top_performing_assets (
                        id SERIAL PRIMARY KEY,
                        game_id BIGINT NOT NULL,
                        game_name TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        performance_rank INTEGER NOT NULL,
                        performance_score FLOAT NOT NULL,
                        visual_style TEXT,
                        dominant_colors JSONB, -- Store RGB values of dominant colors
                        visual_features JSONB, -- All extracted visual features
                        success_metrics JSONB, -- Player count, growth, engagement
                        image_data BYTEA, -- Store the actual image
                        image_hash TEXT UNIQUE,
                        date_added TIMESTAMP DEFAULT NOW(),
                        last_performance_update TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trending_assets_style_score 
                    ON trending_visual_assets(visual_style, performance_score DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_trending_keywords_rank 
                    ON trending_keywords_archive(popularity_rank, analysis_date DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_top_assets_rank 
                    ON top_performing_assets(asset_type, performance_rank);
                """)
            
            conn.commit()
        
        # Now populate the tables with current trending data
        self.populate_trending_tables()
    
    def populate_trending_tables(self):
        """Populate trending tables with current analysis"""
        print("[populate] Populating trending tables with current data...")
        
        # Get current high-performing games
        with self.get_conn() as conn:
            # Get top performing games from last 30 days
            query = """
            SELECT DISTINCT ON (g.id)
                g.id, g.name, g.description,
                s.icon_data, s.thumbnail_data,
                AVG(s.playing) OVER (PARTITION BY g.id) as avg_playing,
                AVG(s.visits) OVER (PARTITION BY g.id) as avg_visits,
                AVG(s.likes::float / NULLIF(s.likes + s.dislikes, 0)) OVER (PARTITION BY g.id) as like_ratio,
                COUNT(s.game_id) OVER (PARTITION BY g.id) as snapshot_count
            FROM games g
            JOIN snapshots s ON g.id = s.game_id
            WHERE s.snapshot_time > NOW() - INTERVAL '30 days'
            AND (s.icon_data IS NOT NULL OR s.thumbnail_data IS NOT NULL)
            ORDER BY g.id, s.snapshot_time DESC
            """
            
            games_df = pd.read_sql(query, conn)
        
        # Filter for games with sufficient data
        games_df = games_df[games_df['snapshot_count'] >= 5]
        
        print(f"[populate] Processing {len(games_df)} games for trending analysis...")
        
        # Extract features and populate tables
        self.populate_visual_assets_table(games_df)
        self.populate_keywords_table(games_df)
        self.populate_top_assets_table(games_df)
    
    def populate_visual_assets_table(self, games_df):
        """Populate trending visual assets table"""
        visual_assets = []
        
        for idx, row in games_df.iterrows():
            if idx % 25 == 0:
                print(f"[visual_assets] Processing game {idx+1}/{len(games_df)}")
            
            performance_score = self.calculate_performance_score(row)
            
            # Process icon
            if row['icon_data']:
                icon_features = self.extract_enhanced_visual_features(row['icon_data'])
                image_hash = self.calculate_image_hash(row['icon_data'])
                
                visual_assets.append({
                    'game_id': row['id'],
                    'game_name': row['name'],
                    'asset_type': 'icon',
                    'visual_style': icon_features['visual_style'],
                    'performance_score': performance_score,
                    'avg_players': row['avg_playing'],
                    'like_ratio': row['like_ratio'] or 0.5,
                    'brightness': icon_features['brightness_mean'],
                    'saturation': icon_features['saturation_mean'],
                    'has_faces': icon_features['has_faces'],
                    'is_minimalist': icon_features['is_minimalist'],
                    'image_hash': image_hash
                })
            
            # Process thumbnail
            if row['thumbnail_data']:
                thumb_features = self.extract_enhanced_visual_features(row['thumbnail_data'])
                image_hash = self.calculate_image_hash(row['thumbnail_data'])
                
                visual_assets.append({
                    'game_id': row['id'],
                    'game_name': row['name'],
                    'asset_type': 'thumbnail',
                    'visual_style': thumb_features['visual_style'],
                    'performance_score': performance_score,
                    'avg_players': row['avg_playing'],
                    'like_ratio': row['like_ratio'] or 0.5,
                    'brightness': thumb_features['brightness_mean'],
                    'saturation': thumb_features['saturation_mean'],
                    'has_faces': thumb_features['has_faces'],
                    'is_minimalist': thumb_features['is_minimalist'],
                    'image_hash': image_hash
                })
        
        # Sort by performance and assign ranks
        visual_assets.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Separate icons and thumbnails for ranking
        icons = [asset for asset in visual_assets if asset['asset_type'] == 'icon']
        thumbnails = [asset for asset in visual_assets if asset['asset_type'] == 'thumbnail']
        
        # Assign ranks
        for i, asset in enumerate(icons):
            asset['trend_rank'] = i + 1
        for i, asset in enumerate(thumbnails):
            asset['trend_rank'] = i + 1
        
        # Insert into database
        self.insert_trending_assets(visual_assets)
    
    def populate_keywords_table(self, games_df):
        """Analyze and populate trending keywords table"""
        print("[keywords] Analyzing keyword performance...")
        
        # Extract keywords from successful vs unsuccessful games
        success_threshold = games_df['avg_playing'].quantile(0.6)  # Top 40%
        successful_games = games_df[games_df['avg_playing'] >= success_threshold]
        all_games = games_df
        
        # Extract all text content
        all_text = []
        successful_text = []
        
        for _, game in all_games.iterrows():
            text = f"{game['name']} {game['description'] or ''}".lower()
            all_text.append((game['id'], text, game['avg_playing']))
        
        for _, game in successful_games.iterrows():
            text = f"{game['name']} {game['description'] or ''}".lower()
            successful_text.append((game['id'], text, game['avg_playing']))
        
        # Extract keywords using multiple methods
        keyword_analysis = self.analyze_keyword_performance(all_text, successful_text, success_threshold)
        
        # Insert into database
        self.insert_trending_keywords(keyword_analysis)
    
    def analyze_keyword_performance(self, all_text, successful_text, success_threshold):
        """Perform comprehensive keyword performance analysis"""
        from collections import defaultdict
        import re
        
        # Extract keywords using various patterns
        gaming_patterns = [
            r'\b(simulator)\b', r'\b(tycoon)\b', r'\b(obby)\b', r'\b(roleplay)\b', r'\b(rp)\b',
            r'\b(adventure)\b', r'\b(survival)\b', r'\b(racing)\b', r'\b(fighting)\b', r'\b(horror)\b',
            r'\b(anime)\b', r'\b(adopt)\b', r'\b(pet)\b', r'\b(story)\b', r'\b(world)\b',
            r'\b(life)\b', r'\b(battle)\b', r'\b(war)\b', r'\b(school)\b', r'\b(hospital)\b',
            r'\b(restaurant)\b', r'\b(hotel)\b', r'\b(city)\b', r'\b(island)\b', r'\b(prison)\b',
            r'\b(zombie)\b', r'\b(magic)\b', r'\b(fantasy)\b', r'\b(space)\b', r'\b(ninja)\b',
            r'\b(pirate)\b', r'\b(kingdom)\b', r'\b(empire)\b', r'\b(factory)\b', r'\b(mining)\b',
            r'\b(cooking)\b', r'\b(fashion)\b', r'\b(dance)\b', r'\b(music)\b', r'\b(art)\b',
            r'\b(building)\b', r'\b(crafting)\b', r'\b(exploration)\b', r'\b(puzzle)\b'
        ]
        
        keyword_stats = defaultdict(lambda: {
            'total_games': 0,
            'successful_games': 0,
            'total_players': 0,
            'successful_players': 0,
            'games_with_keyword': [],
            'games_without_keyword': []
        })
        
        # Analyze all games
        for game_id, text, avg_players in all_text:
            found_keywords = set()
            
            # Extract gaming keywords
            for pattern in gaming_patterns:
                matches = re.findall(pattern, text)
                found_keywords.update(matches)
            
            # Also extract frequent meaningful words
            words = re.findall(r'\b[a-z]{4,12}\b', text)
            word_counts = Counter(words)
            
            # Add high-frequency meaningful words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'your', 'with', 'have', 'this', 'that', 'from', 'they', 'been', 'will', 'more', 'like', 'time', 'very', 'when', 'much', 'game', 'play', 'free'}
            
            for word, count in word_counts.items():
                if word not in stop_words and count >= 2:  # Appears multiple times
                    found_keywords.add(word)
            
            # Update keyword statistics
            is_successful = avg_players >= success_threshold
            
            for keyword in found_keywords:
                keyword_stats[keyword]['total_games'] += 1
                keyword_stats[keyword]['total_players'] += avg_players
                keyword_stats[keyword]['games_with_keyword'].append(avg_players)
                
                if is_successful:
                    keyword_stats[keyword]['successful_games'] += 1
                    keyword_stats[keyword]['successful_players'] += avg_players
            
            # Track games without each keyword (for comparison)
            all_keywords = set(keyword_stats.keys())
            for keyword in all_keywords:
                if keyword not in found_keywords:
                    keyword_stats[keyword]['games_without_keyword'].append(avg_players)
        
        # Calculate performance metrics
        keyword_analysis = []
        
        for keyword, stats in keyword_stats.items():
            if stats['total_games'] < 3:  # Need minimum games for reliable analysis
                continue
            
            # Calculate averages
            avg_players_with = np.mean(stats['games_with_keyword']) if stats['games_with_keyword'] else 0
            avg_players_without = np.mean(stats['games_without_keyword']) if stats['games_without_keyword'] else avg_players_with
            
            # Performance lift calculation
            performance_lift = (avg_players_with / avg_players_without - 1) if avg_players_without > 0 else 0
            
            # Success correlation
            success_rate = stats['successful_games'] / stats['total_games'] if stats['total_games'] > 0 else 0
            
            # Categorize keywords
            category = self.categorize_keyword(keyword)
            
            # Determine trend direction (simplified - could be enhanced with historical data)
            trend_direction = 'stable'  # Default, could analyze historical trends
            
            keyword_analysis.append({
                'keyword': keyword,
                'success_correlation': success_rate,
                'total_games': stats['total_games'],
                'successful_games': stats['successful_games'],
                'avg_players_with_keyword': avg_players_with,
                'avg_players_without_keyword': avg_players_without,
                'performance_lift': performance_lift,
                'trend_direction': trend_direction,
                'category': category
            })
        
        # Sort by performance lift and assign ranks
        keyword_analysis.sort(key=lambda x: x['performance_lift'], reverse=True)
        
        for i, keyword_data in enumerate(keyword_analysis):
            keyword_data['popularity_rank'] = i + 1
        
        return keyword_analysis[:100]  # Top 100 keywords
    
    def categorize_keyword(self, keyword):
        """Categorize keywords into different types"""
        genre_keywords = {'simulator', 'tycoon', 'obby', 'roleplay', 'rp', 'adventure', 'survival', 'racing', 'fighting', 'horror', 'puzzle'}
        theme_keywords = {'anime', 'zombie', 'magic', 'fantasy', 'space', 'ninja', 'pirate', 'medieval', 'futuristic'}
        setting_keywords = {'school', 'hospital', 'restaurant', 'hotel', 'city', 'island', 'prison', 'kingdom', 'empire', 'world'}
        activity_keywords = {'adopt', 'pet', 'cooking', 'fashion', 'dance', 'music', 'art', 'building', 'crafting', 'exploration', 'mining'}
        
        if keyword in genre_keywords:
            return 'genre'
        elif keyword in theme_keywords:
            return 'theme'
        elif keyword in setting_keywords:
            return 'setting'
        elif keyword in activity_keywords:
            return 'activity'
        else:
            return 'general'
    
    def populate_top_assets_table(self, games_df):
        """Populate top performing individual assets table"""
        print("[top_assets] Identifying top performing individual assets...")
        
        top_assets = []
        
        # Get top 50 performers by each asset type
        sorted_games = games_df.nlargest(100, 'avg_playing')
        
        for idx, row in sorted_games.iterrows():
            performance_score = self.calculate_performance_score(row)
            
            # Process icon
            if row['icon_data']:
                icon_features = self.extract_enhanced_visual_features(row['icon_data'])
                image_hash = self.calculate_image_hash(row['icon_data'])
                
                # Create dominant colors array
                dominant_colors = []
                for i in range(5):
                    dominant_colors.append({
                        'r': icon_features.get(f'dominant_color_{i}_r', 128),
                        'g': icon_features.get(f'dominant_color_{i}_g', 128),
                        'b': icon_features.get(f'dominant_color_{i}_b', 128)
                    })
                
                success_metrics = {
                    'avg_playing': row['avg_playing'],
                    'avg_visits': row['avg_visits'],
                    'like_ratio': row['like_ratio'] or 0.5,
                    'performance_score': performance_score
                }
                
                top_assets.append({
                    'game_id': row['id'],
                    'game_name': row['name'],
                    'asset_type': 'icon',
                    'performance_score': performance_score,
                    'visual_style': icon_features['visual_style'],
                    'dominant_colors': json.dumps(dominant_colors),
                    'visual_features': json.dumps({k: v for k, v in icon_features.items() if isinstance(v, (int, float, bool))}),
                    'success_metrics': json.dumps(success_metrics),
                    'image_data': row['icon_data'],
                    'image_hash': image_hash
                })
            
            # Process thumbnail
            if row['thumbnail_data']:
                thumb_features = self.extract_enhanced_visual_features(row['thumbnail_data'])
                image_hash = self.calculate_image_hash(row['thumbnail_data'])
                
                dominant_colors = []
                for i in range(5):
                    dominant_colors.append({
                        'r': thumb_features.get(f'dominant_color_{i}_r', 128),
                        'g': thumb_features.get(f'dominant_color_{i}_g', 128),
                        'b': thumb_features.get(f'dominant_color_{i}_b', 128)
                    })
                
                success_metrics = {
                    'avg_playing': row['avg_playing'],
                    'avg_visits': row['avg_visits'],
                    'like_ratio': row['like_ratio'] or 0.5,
                    'performance_score': performance_score
                }
                
                top_assets.append({
                    'game_id': row['id'],
                    'game_name': row['name'],
                    'asset_type': 'thumbnail',
                    'performance_score': performance_score,
                    'visual_style': thumb_features['visual_style'],
                    'dominant_colors': json.dumps(dominant_colors),
                    'visual_features': json.dumps({k: v for k, v in thumb_features.items() if isinstance(v, (int, float, bool))}),
                    'success_metrics': json.dumps(success_metrics),
                    'image_data': row['thumbnail_data'],
                    'image_hash': image_hash
                })
        
        # Sort and rank
        icons = [asset for asset in top_assets if asset['asset_type'] == 'icon']
        thumbnails = [asset for asset in top_assets if asset['asset_type'] == 'thumbnail']
        
        icons.sort(key=lambda x: x['performance_score'], reverse=True)
        thumbnails.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Assign ranks and limit to top 50 each
        for i, asset in enumerate(icons[:50]):
            asset['performance_rank'] = i + 1
        for i, asset in enumerate(thumbnails[:50]):
            asset['performance_rank'] = i + 1
        
        # Insert into database
        self.insert_top_assets(icons[:50] + thumbnails[:50])
    
    def calculate_image_hash(self, image_data):
        """Calculate hash for image identification"""
        if not image_data:
            return None
        import hashlib
        return hashlib.md5(image_data).hexdigest()
    
    def insert_trending_assets(self, visual_assets):
        """Insert trending visual assets into database"""
        if not visual_assets:
            return
        
        print(f"[insert] Inserting {len(visual_assets)} trending visual assets...")
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                # Clear today's data first
                cur.execute("DELETE FROM trending_visual_assets WHERE analysis_date::date = CURRENT_DATE")
                
                for asset in visual_assets:
                    try:
                        cur.execute("""
                            INSERT INTO trending_visual_assets 
                            (game_id, game_name, asset_type, visual_style, performance_score, 
                             avg_players, like_ratio, brightness, saturation, has_faces, 
                             is_minimalist, trend_rank, image_hash)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (game_id, asset_type, analysis_date::date) DO UPDATE SET
                                performance_score = EXCLUDED.performance_score,
                                trend_rank = EXCLUDED.trend_rank
                        """, (
                            asset['game_id'], asset['game_name'], asset['asset_type'],
                            asset['visual_style'], asset['performance_score'], asset['avg_players'],
                            asset['like_ratio'], asset['brightness'], asset['saturation'],
                            asset['has_faces'], asset['is_minimalist'], asset['trend_rank'],
                            asset['image_hash']
                        ))
                    except Exception as e:
                        print(f"[insert] Error inserting asset {asset['game_id']}: {e}")
                
                conn.commit()
        
        print(f"[insert] Successfully inserted trending visual assets")
    
    def insert_trending_keywords(self, keyword_analysis):
        """Insert trending keywords into database"""
        if not keyword_analysis:
            return
        
        print(f"[insert] Inserting {len(keyword_analysis)} trending keywords...")
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                # Clear today's data first
                cur.execute("DELETE FROM trending_keywords_archive WHERE analysis_date::date = CURRENT_DATE")
                
                for keyword_data in keyword_analysis:
                    try:
                        cur.execute("""
                            INSERT INTO trending_keywords_archive 
                            (keyword, popularity_rank, success_correlation, total_games, 
                             successful_games, avg_players_with_keyword, avg_players_without_keyword,
                             performance_lift, trend_direction, category)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            keyword_data['keyword'], keyword_data['popularity_rank'],
                            keyword_data['success_correlation'], keyword_data['total_games'],
                            keyword_data['successful_games'], keyword_data['avg_players_with_keyword'],
                            keyword_data['avg_players_without_keyword'], keyword_data['performance_lift'],
                            keyword_data['trend_direction'], keyword_data['category']
                        ))
                    except Exception as e:
                        print(f"[insert] Error inserting keyword {keyword_data['keyword']}: {e}")
                
                conn.commit()
        
        print(f"[insert] Successfully inserted trending keywords")
    
    def insert_top_assets(self, top_assets):
        """Insert top performing assets into database"""
        if not top_assets:
            return
        
        print(f"[insert] Inserting {len(top_assets)} top performing assets...")
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                for asset in top_assets:
                    try:
                        cur.execute("""
                            INSERT INTO top_performing_assets 
                            (game_id, game_name, asset_type, performance_rank, performance_score,
                             visual_style, dominant_colors, visual_features, success_metrics,
                             image_data, image_hash)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (image_hash) DO UPDATE SET
                                performance_score = EXCLUDED.performance_score,
                                performance_rank = EXCLUDED.performance_rank,
                                last_performance_update = NOW()
                        """, (
                            asset['game_id'], asset['game_name'], asset['asset_type'],
                            asset['performance_rank'], asset['performance_score'], asset['visual_style'],
                            asset['dominant_colors'], asset['visual_features'], asset['success_metrics'],
                            asset['image_data'], asset['image_hash']
                        ))
                    except Exception as e:
                        print(f"[insert] Error inserting top asset {asset['game_id']}: {e}")
                
                conn.commit()
        
        print(f"[insert] Successfully inserted top performing assets")
    
    def save_trend_analysis(self, trend_report, icon_trends, thumbnail_trends):
        """Save comprehensive trend analysis to files and database"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        with open(f"{self.output_dir}/reports/trend_analysis_{timestamp}.json", 'w') as f:
            json.dump(trend_report, f, indent=2)
        
        # Save detailed CSV reports
        if icon_trends:
            icon_df = pd.DataFrame.from_dict(icon_trends, orient='index')
            icon_df.to_csv(f"{self.output_dir}/reports/icon_trends_{timestamp}.csv")
        
        if thumbnail_trends:
            thumb_df = pd.DataFrame.from_dict(thumbnail_trends, orient='index')
            thumb_df.to_csv(f"{self.output_dir}/reports/thumbnail_trends_{timestamp}.csv")
        
        print(f"[save] Trend analysis saved to {self.output_dir}/reports/")
    
    def generate_trending_report(self):
        """Generate a comprehensive trending report with database insights"""
        print("\n📊 GENERATING COMPREHENSIVE TRENDING REPORT")
        print("=" * 60)
        
        # Run visual trend analysis
        trend_report = self.analyze_trending_visuals(days_back=30)
        
        # Save trending data to database
        self.save_trending_assets_to_database()
        
        # Generate summary report
        self.print_trending_summary()
        
        return trend_report
    
    def print_trending_summary(self):
        """Print a comprehensive summary of current trends"""
        with self.get_conn() as conn:
            print("\n🎨 TOP VISUAL STYLES TRENDING NOW:")
            print("-" * 40)
            
            # Top visual styles
            visual_query = """
            SELECT visual_style, asset_type, COUNT(*) as games, 
                   AVG(performance_score) as avg_score, AVG(avg_players) as avg_players
            FROM trending_visual_assets 
            WHERE analysis_date::date = CURRENT_DATE
            GROUP BY visual_style, asset_type
            ORDER BY avg_score DESC
            LIMIT 10
            """
            
            visual_trends = pd.read_sql(visual_query, conn)
            
            for _, row in visual_trends.iterrows():
                print(f"  🎯 {row['visual_style'].replace('_', ' ').title()} ({row['asset_type']})")
                print(f"     • {row['games']} games | Score: {row['avg_score']:.2f} | Avg Players: {row['avg_players']:.0f}")
            
            print("\n🔥 TOP TRENDING KEYWORDS:")
            print("-" * 40)
            
            # Top keywords
            keyword_query = """
            SELECT keyword, category, performance_lift, total_games, 
                   avg_players_with_keyword, success_correlation
            FROM trending_keywords_archive 
            WHERE analysis_date::date = CURRENT_DATE
            ORDER BY performance_lift DESC
            LIMIT 15
            """
            
            keyword_trends = pd.read_sql(keyword_query, conn)
            
            for _, row in keyword_trends.iterrows():
                lift_pct = row['performance_lift'] * 100
                print(f"  📈 '{row['keyword']}' ({row['category']})")
                print(f"     • +{lift_pct:.1f}% performance lift | {row['total_games']} games | {row['avg_players_with_keyword']:.0f} avg players")
            
            print("\n🏆 TOP PERFORMING INDIVIDUAL ASSETS:")
            print("-" * 40)
            
            # Top individual assets
            assets_query = """
            SELECT game_name, asset_type, visual_style, performance_score, 
                   (success_metrics->>'avg_playing')::float as avg_playing
            FROM top_performing_assets 
            WHERE date_added::date = CURRENT_DATE
            ORDER BY performance_rank
            LIMIT 10
            """
            
            top_assets = pd.read_sql(assets_query, conn)
            
            for _, row in top_assets.iterrows():
                print(f"  🥇 {row['game_name'][:30]} ({row['asset_type']})")
                print(f"     • Style: {row['visual_style']} | Score: {row['performance_score']:.2f} | Players: {row['avg_playing']:.0f}")

def get_trending_recommendations():
    """Get actionable recommendations from trending data"""
    analyzer = VisualTrendAnalyzer()
    
    with analyzer.get_conn() as conn:
        print("\n💡 ACTIONABLE RECOMMENDATIONS FOR SUCCESS:")
        print("=" * 50)
        
        # Visual recommendations
        visual_rec_query = """
        SELECT visual_style, asset_type, AVG(performance_score) as avg_score,
               AVG(brightness) as avg_brightness, AVG(saturation) as avg_saturation,
               AVG(CASE WHEN has_faces THEN 1 ELSE 0 END) * 100 as face_percentage
        FROM trending_visual_assets 
        WHERE analysis_date::date = CURRENT_DATE AND trend_rank <= 10
        GROUP BY visual_style, asset_type
        ORDER BY avg_score DESC
        LIMIT 5
        """
        
        visual_recs = pd.read_sql(visual_rec_query, conn)
        
        print("\n🎨 Visual Style Recommendations:")
        for _, rec in visual_recs.iterrows():
            print(f"\n  ✅ Use '{rec['visual_style'].replace('_', ' ').title()}' style for {rec['asset_type']}s")
            print(f"     • Target brightness: {rec['avg_brightness']:.0f}/255")
            print(f"     • Target saturation: {rec['avg_saturation']:.0f}%")
            print(f"     • Include characters/faces: {'Yes' if rec['face_percentage'] > 50 else 'No'} ({rec['face_percentage']:.0f}% of top games)")
        
        # Keyword recommendations
        keyword_rec_query = """
        SELECT keyword, category, performance_lift, avg_players_with_keyword,
               success_correlation, total_games
        FROM trending_keywords_archive 
        WHERE analysis_date::date = CURRENT_DATE AND performance_lift > 0.2
        ORDER BY performance_lift DESC
        LIMIT 8
        """
        
        keyword_recs = pd.read_sql(keyword_rec_query, conn)
        
        print(f"\n🔤 Keyword Recommendations:")
        for _, rec in keyword_recs.iterrows():
            lift_pct = rec['performance_lift'] * 100
            print(f"  📝 Include '{rec['keyword']}' in your game title/description")
            print(f"     • Category: {rec['category']} | +{lift_pct:.1f}% performance boost")
            print(f"     • Games using this: {rec['avg_players_with_keyword']:.0f} avg players")
        
        # Color recommendations
        color_rec_query = """
        SELECT asset_type, 
               AVG((dominant_colors->0->>'r')::float) as avg_red,
               AVG((dominant_colors->0->>'g')::float) as avg_green, 
               AVG((dominant_colors->0->>'b')::float) as avg_blue,
               AVG(performance_score) as avg_score
        FROM top_performing_assets 
        WHERE date_added::date = CURRENT_DATE AND performance_rank <= 20
        GROUP BY asset_type
        """
        
        color_recs = pd.read_sql(color_rec_query, conn)
        
        print(f"\n🎨 Color Palette Recommendations:")
        for _, rec in color_recs.iterrows():
            print(f"  🎯 {rec['asset_type'].title()} dominant color: RGB({rec['avg_red']:.0f}, {rec['avg_green']:.0f}, {rec['avg_blue']:.0f})")
            print(f"     • This color scheme averages {rec['avg_score']:.2f} performance score")

def query_trending_database():
    """Interactive function to query trending database"""
    analyzer = VisualTrendAnalyzer()
    
    print("🔍 Trending Database Query Tool")
    print("Available queries:")
    print("1. Top visual styles by performance")
    print("2. Most successful keywords")
    print("3. Best performing games with specific style")
    print("4. Color analysis of top performers")
    print("5. Trend comparison over time")
    
    while True:
        try:
            choice = input("\nSelect query (1-5) or 'quit': ").strip()
            
            if choice.lower() == 'quit':
                break
            elif choice == '1':
                query_top_visual_styles(analyzer)
            elif choice == '2':
                query_top_keywords(analyzer)
            elif choice == '3':
                query_games_by_style(analyzer)
            elif choice == '4':
                query_color_analysis(analyzer)
            elif choice == '5':
                query_trend_comparison(analyzer)
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            break

def query_top_visual_styles(analyzer):
    """Query top performing visual styles"""
    with analyzer.get_conn() as conn:
        query = """
        SELECT visual_style, asset_type, COUNT(*) as games,
               AVG(performance_score) as avg_score,
               AVG(avg_players) as avg_players,
               AVG(brightness) as avg_brightness,
               AVG(saturation) as avg_saturation
        FROM trending_visual_assets 
        WHERE analysis_date > NOW() - INTERVAL '7 days'
        GROUP BY visual_style, asset_type
        HAVING COUNT(*) >= 3
        ORDER BY avg_score DESC
        LIMIT 15
        """
        
        results = pd.read_sql(query, conn)
        
        print(f"\n📊 Top Visual Styles (Last 7 Days):")
        print("-" * 60)
        
        for _, row in results.iterrows():
            print(f"🎨 {row['visual_style'].replace('_', ' ').title()} ({row['asset_type']})")
            print(f"   Games: {row['games']} | Score: {row['avg_score']:.2f} | Players: {row['avg_players']:.0f}")
            print(f"   Brightness: {row['avg_brightness']:.0f} | Saturation: {row['avg_saturation']:.0f}")
            print()

def query_top_keywords(analyzer):
    """Query top performing keywords"""
    category_filter = input("Filter by category (genre/theme/setting/activity) or press Enter for all: ").strip()
    
    with analyzer.get_conn() as conn:
        if category_filter:
            query = """
            SELECT keyword, category, performance_lift, total_games,
                   avg_players_with_keyword, success_correlation
            FROM trending_keywords_archive 
            WHERE analysis_date > NOW() - INTERVAL '7 days'
            AND category = %s
            ORDER BY performance_lift DESC
            LIMIT 20
            """
            results = pd.read_sql(query, conn, params=(category_filter,))
        else:
            query = """
            SELECT keyword, category, performance_lift, total_games,
                   avg_players_with_keyword, success_correlation
            FROM trending_keywords_archive 
            WHERE analysis_date > NOW() - INTERVAL '7 days'
            ORDER BY performance_lift DESC
            LIMIT 20
            """
            results = pd.read_sql(query, conn)
        
        print(f"\n📈 Top Keywords{f' ({category_filter})' if category_filter else ''} (Last 7 Days):")
        print("-" * 60)
        
        for _, row in results.iterrows():
            lift_pct = row['performance_lift'] * 100
            print(f"🔤 '{row['keyword']}' ({row['category']})")
            print(f"   Performance Lift: +{lift_pct:.1f}% | Games: {row['total_games']} | Avg Players: {row['avg_players_with_keyword']:.0f}")
            print()

def query_games_by_style(analyzer):
    """Query games using specific visual style"""
    style = input("Enter visual style to search for: ").strip().lower().replace(' ', '_')
    asset_type = input("Asset type (icon/thumbnail) or press Enter for both: ").strip()
    
    with analyzer.get_conn() as conn:
        if asset_type:
            query = """
            SELECT game_name, visual_style, performance_score, avg_players, 
                   like_ratio, brightness, saturation
            FROM trending_visual_assets 
            WHERE LOWER(visual_style) LIKE %s AND asset_type = %s
            AND analysis_date > NOW() - INTERVAL '7 days'
            ORDER BY performance_score DESC
            LIMIT 15
            """
            results = pd.read_sql(query, conn, params=(f'%{style}%', asset_type))
        else:
            query = """
            SELECT game_name, visual_style, asset_type, performance_score, 
                   avg_players, like_ratio, brightness, saturation
            FROM trending_visual_assets 
            WHERE LOWER(visual_style) LIKE %s
            AND analysis_date > NOW() - INTERVAL '7 days'
            ORDER BY performance_score DESC
            LIMIT 15
            """
            results = pd.read_sql(query, conn, params=(f'%{style}%',))
        
        if len(results) == 0:
            print(f"No games found with style matching '{style}'")
            return
        
        print(f"\n🎮 Games Using '{style}' Style:")
        print("-" * 60)
        
        for _, row in results.iterrows():
            asset_info = f" ({row['asset_type']})" if 'asset_type' in row else ""
            print(f"🏆 {row['game_name']}{asset_info}")
            print(f"   Style: {row['visual_style']} | Score: {row['performance_score']:.2f}")
            print(f"   Players: {row['avg_players']:.0f} | Like Ratio: {row['like_ratio']:.1%}")
            print(f"   Visual: Brightness {row['brightness']:.0f}, Saturation {row['saturation']:.0f}")
            print()

def query_color_analysis(analyzer):
    """Analyze color trends in top performing assets"""
    with analyzer.get_conn() as conn:
        query = """
        SELECT asset_type,
               AVG((dominant_colors->0->>'r')::float) as dom1_red,
               AVG((dominant_colors->0->>'g')::float) as dom1_green,
               AVG((dominant_colors->0->>'b')::float) as dom1_blue,
               AVG((dominant_colors->1->>'r')::float) as dom2_red,
               AVG((dominant_colors->1->>'g')::float) as dom2_green,
               AVG((dominant_colors->1->>'b')::float) as dom2_blue,
               AVG(performance_score) as avg_performance,
               COUNT(*) as asset_count
        FROM top_performing_assets 
        WHERE date_added > NOW() - INTERVAL '7 days'
        AND performance_rank <= 25
        GROUP BY asset_type
        """
        
        results = pd.read_sql(query, conn)
        
        print(f"\n🎨 Color Analysis of Top Performing Assets:")
        print("-" * 60)
        
        for _, row in results.iterrows():
            print(f"📱 {row['asset_type'].title()}s (Top 25 performers, {row['asset_count']} analyzed)")
            print(f"   Average Performance Score: {row['avg_performance']:.2f}")
            print(f"   Primary Color: RGB({row['dom1_red']:.0f}, {row['dom1_green']:.0f}, {row['dom1_blue']:.0f})")
            print(f"   Secondary Color: RGB({row['dom2_red']:.0f}, {row['dom2_green']:.0f}, {row['dom2_blue']:.0f})")
            
            # Convert to hex for easier use
            def rgb_to_hex(r, g, b):
                return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            
            hex1 = rgb_to_hex(row['dom1_red'], row['dom1_green'], row['dom1_blue'])
            hex2 = rgb_to_hex(row['dom2_red'], row['dom2_green'], row['dom2_blue'])
            print(f"   Hex Colors: {hex1} (primary), {hex2} (secondary)")
            print()

def query_trend_comparison(analyzer):
    """Compare trends over different time periods"""
    print("Comparing trends: Last 7 days vs Previous 7 days")
    
    with analyzer.get_conn() as conn:
        # Current trends
        current_query = """
        SELECT visual_style, AVG(performance_score) as avg_score, COUNT(*) as games
        FROM trending_visual_assets 
        WHERE analysis_date > NOW() - INTERVAL '7 days'
        GROUP BY visual_style
        HAVING COUNT(*) >= 2
        """
        
        # Previous trends (if data exists)
        previous_query = """
        SELECT visual_style, AVG(performance_score) as avg_score, COUNT(*) as games
        FROM trending_visual_assets 
        WHERE analysis_date BETWEEN NOW() - INTERVAL '14 days' AND NOW() - INTERVAL '7 days'
        GROUP BY visual_style
        HAVING COUNT(*) >= 2
        """
        
        current_trends = pd.read_sql(current_query, conn)
        previous_trends = pd.read_sql(previous_query, conn)
        
        if len(previous_trends) == 0:
            print("Not enough historical data for comparison yet.")
            return
        
        print(f"\n📊 Trend Comparison:")
        print("-" * 60)
        
        # Merge dataframes for comparison
        comparison = current_trends.merge(
            previous_trends, 
            on='visual_style', 
            suffixes=('_current', '_previous'),
            how='outer'
        ).fillna(0)
        
        comparison['score_change'] = comparison['avg_score_current'] - comparison['avg_score_previous']
        comparison['games_change'] = comparison['games_current'] - comparison['games_previous']
        
        # Sort by score change
        comparison = comparison.sort_values('score_change', ascending=False)
        
        for _, row in comparison.iterrows():
            trend_arrow = "📈" if row['score_change'] > 0 else "📉" if row['score_change'] < 0 else "➡️"
            
            print(f"{trend_arrow} {row['visual_style'].replace('_', ' ').title()}")
            print(f"   Score: {row['avg_score_current']:.2f} (was {row['avg_score_previous']:.2f}) | Change: {row['score_change']:+.2f}")
            print(f"   Games: {int(row['games_current'])} (was {int(row['games_previous'])}) | Change: {int(row['games_change']):+d}")
            print()

def main():
    """Main function for visual trend analysis"""
    print("🎨 Visual Trend Analyzer - Icon & Thumbnail Performance")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "analyze":
            # Full trending analysis
            analyzer = VisualTrendAnalyzer()
            analyzer.generate_trending_report()
            
        elif mode == "recommendations":
            # Get actionable recommendations
            get_trending_recommendations()
            
        elif mode == "query":
            # Interactive database queries
            query_trending_database()
            
        elif mode == "update":
            # Just update the database with current trends
            analyzer = VisualTrendAnalyzer()
            analyzer.save_trending_assets_to_database()
            print("✅ Trending database updated successfully!")
            
        else:
            print("Usage: python visual_trend_analyzer.py [analyze|recommendations|query|update]")
    
    else:
        # Default: run full analysis
        analyzer = VisualTrendAnalyzer()
        analyzer.generate_trending_report()

if __name__ == "__main__":
    import sys
    main()#!/usr/bin/env python3
"""
Visual Trend Analyzer - Analyzes which icons and thumbnails perform best
- Generates visual trend reports with actual image examples
- Shows top-performing visual patterns with metrics
- Creates visual style guides for optimal game assets
- Tracks trending visual elements over time
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from collections import defaultdict, Counter
import base64
from io import BytesIO

# Image Analysis
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# ML for clustering visual patterns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

load_dotenv()

class VisualTrendAnalyzer:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.output_dir = "visual_trends"
        self.ensure_output_directory()
        
    def get_conn(self):
        return psycopg2.connect(self.db_url, sslmode="require")
    
    def ensure_output_directory(self):
        """Create output directories for trend analysis"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/top_icons", exist_ok=True)
        os.makedirs(f"{self.output_dir}/top_thumbnails", exist_ok=True)
        os.makedirs(f"{self.output_dir}/trend_samples", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
    
    def extract_enhanced_visual_features(self, image_data):
        """Extract comprehensive visual features for trend analysis"""
        if not image_data:
            return self._empty_visual_features()
        
        try:
            img = Image.open(BytesIO(image_data))
            img_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            features = {}
            
            # Basic properties
            features['width'], features['height'] = img.size
            features['aspect_ratio'] = features['width'] / features['height']
            features['total_pixels'] = features['width'] * features['height']
            features['file_size'] = len(image_data)
            
            # Color analysis
            colors = img.convert('RGB')
            pixels = np.array(colors).reshape(-1, 3)
            
            # RGB statistics
            features['red_mean'] = np.mean(pixels[:, 0])
            features['green_mean'] = np.mean(pixels[:, 1])
            features['blue_mean'] = np.mean(pixels[:, 2])
            features['red_std'] = np.std(pixels[:, 0])
            features['green_std'] = np.std(pixels[:, 1])
            features['blue_std'] = np.std(pixels[:, 2])
            
            # HSV analysis for better color understanding
            hsv_pixels = []
            for pixel in pixels[::50]:  # Sample for performance
                r, g, b = pixel / 255.0
                import colorsys
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_pixels.append([h * 360, s * 100, v * 100])
            
            hsv_array = np.array(hsv_pixels)
            features['hue_mean'] = np.mean(hsv_array[:, 0])
            features['saturation_mean'] = np.mean(hsv_array[:, 1])
            features['brightness_mean'] = np.mean(hsv_array[:, 2])
            features['hue_std'] = np.std(hsv_array[:, 0])
            features['saturation_std'] = np.std(hsv_array[:, 1])
            features['brightness_std'] = np.std(hsv_array[:, 2])
            
            # Color diversity (number of unique colors)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1])))))
            features['color_diversity'] = unique_colors / features['total_pixels']
            
            # Dominant color analysis
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Store dominant colors as features
            for i, color in enumerate(dominant_colors):
                features[f'dominant_color_{i}_r'] = color[0]
                features[f'dominant_color_{i}_g'] = color[1]
                features[f'dominant_color_{i}_b'] = color[2]
            
            # Visual complexity analysis
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis
            features['texture_variance'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast and brightness
            features['contrast'] = np.std(gray)
            features['overall_brightness'] = np.mean(gray)
            
            # Detect text in image
            features['text_detected'] = self.detect_text_regions(gray)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            features['face_count'] = len(faces)
            features['has_faces'] = len(faces) > 0
            
            # Character/logo detection (simplified)
            features['character_regions'] = self.detect_character_regions(img_cv)
            
            # Layout analysis
            features['central_focus'] = self.analyze_central_focus(gray)
            features['symmetry_score'] = self.calculate_symmetry(gray)
            
            # Style classification
            features['visual_style'] = self.classify_comprehensive_style(features, img_cv)
            
            # Trend-specific features
            features['is_minimalist'] = self.is_minimalist_design(features)
            features['has_gradient'] = self.detect_gradient(pixels)
            features['emoji_style'] = self.detect_emoji_style(features)
            features['neon_colors'] = self.detect_neon_colors(hsv_array)
            
            return features
            
        except Exception as e:
            print(f"[visual_analysis] Error: {e}")
            return self._empty_visual_features()
    
    def detect_text_regions(self, gray_image):
        """Detect if image contains text"""
        try:
            # Use morphological operations to detect text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            connected = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # Text regions typically have certain aspect ratios and sizes
                if 0.2 < aspect_ratio < 5 and area > 100:
                    text_regions += 1
            
            return text_regions > 2  # Likely has text if multiple text-like regions
        except:
            return False
    
    def detect_character_regions(self, img_cv):
        """Detect character-like regions in the image"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Use blob detection for character-like shapes
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 500
            params.maxArea = 50000
            params.filterByCircularity = False
            params.filterByConvexity = False
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            return len(keypoints)
        except:
            return 0
    
    def analyze_central_focus(self, gray_image):
        """Analyze how much visual focus is in the center"""
        try:
            h, w = gray_image.shape
            center_h, center_w = h // 2, w // 2
            
            # Define center region (middle 50% of image)
            center_region = gray_image[
                center_h - h//4:center_h + h//4,
                center_w - w//4:center_w + w//4
            ]
            
            center_variance = np.var(center_region)
            total_variance = np.var(gray_image)
            
            return center_variance / total_variance if total_variance > 0 else 0
        except:
            return 0
    
    def calculate_symmetry(self, gray_image):
        """Calculate horizontal symmetry score"""
        try:
            h, w = gray_image.shape
            left_half = gray_image[:, :w//2]
            right_half = cv2.flip(gray_image[:, w//2:], 1)
            
            # Resize to match if odd width
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate similarity
            diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
            symmetry_score = 1 - (np.mean(diff) / 255.0)
            
            return max(0, symmetry_score)
        except:
            return 0
    
    def is_minimalist_design(self, features):
        """Detect if design is minimalist"""
        return (
            features['color_diversity'] < 0.1 and
            features['edge_density'] < 0.1 and
            features['texture_variance'] < 1000
        )
    
    def detect_gradient(self, pixels):
        """Detect if image has gradient effects"""
        try:
            # Check for smooth color transitions (gradients)
            gradient_score = 0
            
            # Sample rows and columns to check for gradual color changes
            h, w = int(np.sqrt(len(pixels))), int(np.sqrt(len(pixels)))
            reshaped = pixels[:h*w].reshape(h, w, 3)
            
            # Check horizontal gradients
            for row in reshaped[::10]:  # Sample every 10th row
                color_changes = np.diff(row.mean(axis=1))
                smooth_changes = np.sum(np.abs(color_changes) < 5)  # Small, smooth changes
                gradient_score += smooth_changes / len(color_changes)
            
            return gradient_score > 3  # Threshold for gradient detection
        except:
            return False
    
    def detect_emoji_style(self, features):
        """Detect emoji-like style"""
        return (
            features['has_faces'] and
            features['saturation_mean'] > 60 and
            features['brightness_mean'] > 150 and
            features['symmetry_score'] > 0.7
        )
    
    def detect_neon_colors(self, hsv_array):
        """Detect neon/vibrant color schemes"""
        try:
            # Neon colors: high saturation, bright
            neon_pixels = np.sum(
                (hsv_array[:, 1] > 80) &  # High saturation
                (hsv_array[:, 2] > 70)    # High brightness
            )
            return neon_pixels / len(hsv_array) > 0.3
        except:
            return False
    
    def classify_comprehensive_style(self, features, img_cv):
        """Comprehensive style classification"""
        try:
            # More detailed style classification based on features
            if features['emoji_style']:
                return "emoji_character"
            elif features['is_minimalist']:
                return "minimalist_clean"
            elif features['neon_colors']:
                return "neon_vibrant"
            elif features['has_faces'] and features['character_regions'] > 3:
                return "character_rich"
            elif features['text_detected'] and features['edge_density'] > 0.2:
                return "text_heavy"
            elif features['has_gradient'] and features['saturation_mean'] > 50:
                return "gradient_colorful"
            elif features['brightness_mean'] > 180:
                if features['saturation_mean'] > 60:
                    return "bright_saturated"
                else:
                    return "bright_pastel"
            elif features['brightness_mean'] < 80:
                if features['saturation_mean'] > 40:
                    return "dark_vibrant"
                else:
                    return "dark_monochrome"
            elif features['texture_variance'] > 2000:
                return "detailed_complex"
            elif features['central_focus'] > 1.5:
                return "centered_focus"
            else:
                return "balanced_standard"
        except:
            return "unknown"
    
    def _empty_visual_features(self):
        """Default visual features for missing images"""
        empty = {
            'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'total_pixels': 0, 'file_size': 0,
            'red_mean': 128, 'green_mean': 128, 'blue_mean': 128,
            'red_std': 0, 'green_std': 0, 'blue_std': 0,
            'hue_mean': 0, 'saturation_mean': 0, 'brightness_mean': 50,
            'hue_std': 0, 'saturation_std': 0, 'brightness_std': 0,
            'color_diversity': 0, 'edge_density': 0, 'texture_variance': 0,
            'contrast': 0, 'overall_brightness': 128, 'text_detected': False,
            'face_count': 0, 'has_faces': False, 'character_regions': 0,
            'central_focus': 0, 'symmetry_score': 0, 'visual_style': 'none',
            'is_minimalist': False, 'has_gradient': False, 'emoji_style': False, 'neon_colors': False
        }
        
        # Add dominant color features
        for i in range(5):
            empty[f'dominant_color_{i}_r'] = 128
            empty[f'dominant_color_{i}_g'] = 128
            empty[f'dominant_color_{i}_b'] = 128
        
        return empty
    
    def analyze_trending_visuals(self, days_back=30, min_games=5):
        """Analyze which visual styles are trending and performing well"""
        print(f"[trends] Analyzing visual trends over last {days_back} days...")
        
        with self.get_conn() as conn:
            # Get recent high-performing games with images
            query = """
            SELECT DISTINCT ON (g.id)
                g.id, g.name,
                s.icon_data, s.thumbnail_data,
                AVG(s.playing) OVER (PARTITION BY g.id) as avg_playing,
                AVG(s.visits) OVER (PARTITION BY g.id) as avg_visits,
                AVG(s.likes::float / NULLIF(s.likes + s.dislikes, 0)) OVER (PARTITION BY g.id) as like_ratio,
                COUNT(s.game_id) OVER (PARTITION BY g.id) as snapshot_count
            FROM games g
            JOIN snapshots s ON g.id = s.game_id
            WHERE s.snapshot_time > NOW() - INTERVAL '%s days'
            AND (s.icon_data IS NOT NULL OR s.thumbnail_data IS NOT NULL)
            ORDER BY g.id, s.snapshot_time DESC
            """
            
            df = pd.read_sql(query, conn, params=(days_back,))
        
        # Filter for games with sufficient data
        df = df[df['snapshot_count'] >= min_games]
        
        print(f"[trends] Analyzing {len(df)} games with visual data...")
        
        # Extract visual features for all games
        icon_features = []
        thumbnail_features = []
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"[trends] Processing game {idx+1}/{len(df)}")
            
            game_data = {
                'game_id': row['id'],
                'game_name': row['name'],
                'avg_playing': row['avg_playing'],
                'avg_visits': row['avg_visits'],
                'like_ratio': row['like_ratio'] or 0.5,
                'performance_score': self.calculate_performance_score(row)
            }
            
            # Icon analysis
            if row['icon_data']:
                icon_visual = self.extract_enhanced_visual_features(row['icon_data'])
                icon_visual.update(game_data)
                icon_visual['image_type'] = 'icon'
                icon_features.append(icon_visual)
            
            # Thumbnail analysis
            if row['thumbnail_data']:
                thumb_visual = self.extract_enhanced_visual_features(row['thumbnail_data'])
                thumb_visual.update(game_data)
                thumb_visual['image_type'] = 'thumbnail'
                thumbnail_features.append(thumb_visual)
        
        # Analyze trends
        icon_trends = self.analyze_style_performance(icon_features, 'icon')
        thumbnail_trends = self.analyze_style_performance(thumbnail_features, 'thumbnail')
        
        # Generate trend reports
        trend_report = self.generate_trend_report(icon_trends, thumbnail_trends, days_back)
        
        # Create visual examples
        self.create_visual_examples(icon_features, thumbnail_features, icon_trends, thumbnail_trends)
        
        # Save comprehensive report
        self.save_trend_analysis(trend_report, icon_trends, thumbnail_trends)
        
        return trend_report
    
    def calculate_performance_score(self, game_row):
        """Calculate composite performance score"""
        playing_score = min(game_row['avg_playing'] / 1000, 10)  # Cap at 10
        visits_score = min(game_row['avg_visits'] / 1000000, 10)  # Cap at 10
        engagement_score = (game_row['like_ratio'] or 0.5) * 10
        
        return (playing_score + visits_score + engagement_score) / 3
    
    def analyze_style_performance(self, features_list, image_type):
        """Analyze performance by visual style"""
        if not features_list:
            return {}
        
        df = pd.DataFrame(features_list)
        
        # Group by visual style
        style_analysis = df.groupby('visual_style').agg({
            'performance_score': ['count', 'mean', 'std'],
            'avg_playing': ['mean', 'median'],
            'like_ratio': 'mean',
            'saturation_mean': 'mean',
            'brightness_mean': 'mean',
            'edge_density': 'mean',
            'has_faces': 'mean',
            'character_regions': 'mean'
        }).round(3)
        
        # Flatten column names
        style_analysis.columns = ['_'.join(col).strip() for col in style_analysis.columns]
        
        # Calculate trend scores
        style_analysis['trend_score'] = (
            style_analysis['performance_score_mean'] * 0.4 +
            style_analysis['avg_playing_mean'] / 100 * 0.3 +  # Normalize
            style_analysis['like_ratio_mean'] * 10 * 0.3
        )
        
        # Sort by trend score
        style_analysis = style_analysis.sort_values('trend_score', ascending=False)
        
        # Add color and visual characteristics
        for style in style_analysis.index:
            style_games = df[df['visual_style'] == style]
            
            # Calculate average color characteristics
            style_analysis.loc[style, 'avg_red'] = style_games['red_mean'].mean()
            style_analysis.loc[style, 'avg_green'] = style_games['green_mean'].mean()
            style_analysis.loc[style, 'avg_blue'] = style_games['blue_mean'].mean()
            
            # Dominant characteristics
            style_analysis.loc[style, 'minimalist_pct'] = style_games['is_minimalist'].mean() * 100
            style_analysis.loc[style, 'gradient_pct'] = style_games['has_gradient'].mean() * 100
            style_analysis.loc[style, 'neon_pct'] = style_games['neon_colors'].mean() * 100
        
        return style_analysis.to_dict('index')
    
    def generate_trend_report(self, icon_trends, thumbnail_trends, days_back):
        """Generate comprehensive trend report"""
        report = {
            'analysis_date': datetime.utcnow().isoformat(),
            'analysis_period_days': days_back,
            'summary': {},
            'top_icon_styles': [],
            'top_thumbnail_styles': [],
            'visual_recommendations': [],
            'trending_characteristics': {}
        }
        
        # Icon trend summary
        if icon_trends:
            top_icon_style = max(icon_trends.keys(), key=lambda x: icon_trends[x]['trend_score'])
            report['summary']['best_icon_style'] = top_icon_style
            report['summary']['best_icon_performance'] = icon_trends[top_icon_style]['performance_score_mean']
            
            # Top 5 icon styles
            sorted_icons = sorted(icon_trends.items(), key=lambda x: x[1]['trend_score'], reverse=True)
            for style_name, data in sorted_icons[:5]:
                report['top_icon_styles'].append({
                    'style': style_name,
                    'performance_score': round(data['performance_score_mean'], 2),
                    'game_count': int(data['performance_score_count']),
                    'avg_players': round(data['avg_playing_mean'], 0),
                    'like_ratio': round(data['like_ratio_mean'], 3),
                    'characteristics': {
                        'brightness': round(data['brightness_mean'], 1),
                        'saturation': round(data['saturation_mean'], 1),
                        'has_faces_pct': round(data['has_faces'] * 100, 1),
                        'minimalist_pct': round(data.get('minimalist_pct', 0), 1)
                    }
                })
        
        # Thumbnail trend summary
        if thumbnail_trends:
            top_thumb_style = max(thumbnail_trends.keys(), key=lambda x: thumbnail_trends[x]['trend_score'])
            report['summary']['best_thumbnail_style'] = top_thumb_style
            report['summary']['best_thumbnail_performance'] = thumbnail_trends[top_thumb_style]['performance_score_mean']
            
            # Top 5 thumbnail styles
            sorted_thumbs = sorted(thumbnail_trends.items(), key=lambda x: x[1]['trend_score'], reverse=True)
            for style_name, data in sorted_thumbs[:5]:
                report['top_thumbnail_styles'].append({
                    'style': style_name,
                    'performance_score': round(data['performance_score_mean'], 2),
                    'game_count': int(data['performance_score_count']),
                    'avg_players': round(data['avg_playing_mean'], 0),
                    'like_ratio': round(data['like_ratio_mean'], 3),
                    'characteristics': {
                        'brightness': round(data['brightness_mean'], 1),
                        'saturation': round(data['saturation_mean'], 1),
                        'has_faces_pct': round(data['has_faces'] * 100, 1),
                        'gradient_pct': round(data.get('gradient_pct', 0), 1)
                    }
                })
        
        # Generate recommendations
        report['visual_recommendations'] = self.generate_visual_recommendations(icon_trends, thumbnail_trends)
        
        # Trending characteristics
        report['trending_characteristics'] = self.analyze_trending_characteristics(icon_trends, thumbnail_trends)
        
        return report
    
    def generate_visual_recommendations(self, icon_trends, thumbnail_trends):
        """Generate actionable visual recommendations"""
        recommendations = []
        
        # Icon recommendations
        if icon_trends:
            best_icon = max(icon_trends.keys(), key=lambda x: icon_trends[x]['trend_score'])
            icon_data = icon_trends[best_icon]
            
            recommendations.append({
                'type': 'icon',
                'recommendation': f"Use '{best_icon}' style icons",
                'reason': f"Shows {icon_data['performance_score_mean']:.1f} performance score across {icon_data['performance_score_count']} games",
                'specific_tips': [
                    f"Target brightness: {icon_data['brightness_mean']:.0f}/255",
                    f"Target saturation: {icon_data['saturation_mean']:.0f}%",
                    f"Include faces: {'Yes' if icon_data['has_faces'] > 0.5 else 'No'}",
                    f"Minimalist approach: {'Yes' if icon_data.get('minimalist_pct', 0) > 50 else 'No'}"
                ]
            })
        
        # Thumbnail recommendations
        if thumbnail_trends:
            best_thumb = max(thumbnail_trends.keys(), key=lambda x: thumbnail_trends[x]['trend_score'])
            thumb_data = thumbnail_trends[best_thumb]
            
            recommendations.append({
                'type': 'thumbnail',
                'recommendation': f"Use '{best_thumb}' style thumbnails",
                'reason': f"Shows {thumb_data['performance_score_mean']:.1f} performance score across {thumb_data['performance_score_count']} games",
                'specific_tips': [
                    f"Target brightness: {thumb_data['brightness_mean']:.0f}/255",
                    f"Target saturation: {thumb_data['saturation_mean']:.0f}%",
                    f"Include characters: {'Yes' if thumb_data['character_regions'] > 2 else 'No'}",
                    f"Use gradients: {'Yes' if thumb_data.get('gradient_pct', 0) > 30 else 'No'}"
                ]
            })
        
        return recommendations
    
    def analyze_trending_characteristics(self, icon_trends, thumbnail_trends):
        """Analyze what visual characteristics are trending"""
        characteristics = {
            'color_trends': {},
            'style_trends': {},
            'composition_trends': {}
        }
        
        # Combine all trend data
        all_trends = {}
        all_trends.update({f"icon_{k}": v for k, v in (icon_trends or {}).items()})
        all_trends.update({f"thumb_{k}": v for k, v in (thumbnail_trends or {}).items()})
        
        if not all_trends:
            return characteristics
        
        # Analyze color trends
        high_performing = {k: v for k, v in all_trends.items() if v['trend_score'] > 5}
        
        if high_performing:
            avg_brightness = np.mean([data['brightness_mean'] for data in high_performing.values()])
            avg_saturation = np.mean([data['saturation_mean'] for data in high_performing.values()])
            
            characteristics['color_trends'] = {
                'trending_brightness': 'high' if avg_brightness > 150 else 'medium' if avg_brightness > 100 else 'low',
                'trending_saturation': 'high' if avg_saturation > 60 else 'medium' if avg_saturation > 30 else 'low',
                'avg_brightness_value': round(avg_brightness, 1),
                'avg_saturation_value': round(avg_saturation, 1)
            }
            
            # Style trends
            face_usage = np.mean([data['has_faces'] for data in high_performing.values()])
            minimalist_usage = np.mean([data.get('minimalist_pct', 0) for data in high_performing.values()])
            
            characteristics['style_trends'] = {
                'faces_trending': face_usage > 0.4,
                'minimalism_trending': minimalist_usage > 40,
                'face_usage_pct': round(face_usage * 100, 1),
                'minimalism_pct': round(minimalist_usage, 1)
            }
        
        return characteristics
    
    def create_visual_examples(self, icon_features, thumbnail_features, icon_trends, thumbnail_trends):
        """Create visual examples of top-performing styles"""
        print("[examples] Creating visual examples...")
        
        # Save top icon examples
        if icon_features and icon_trends:
            self.save_style_examples(icon_features, icon_trends, 'icon')
        
        # Save top thumbnail examples
        if thumbnail_features and thumbnail_trends:
            self.save_style_examples(thumbnail_features, thumbnail_trends, 'thumbnail')
        
    def save_style_examples(self, features_list, trends, image_type):
        """Save example images for each top-performing style"""
        df = pd.DataFrame(features_list)
        
        # Get top 5 styles
        top_styles = sorted(trends.items(), key=lambda x: x[1]['trend_score'], reverse=True)[:5]
        
        for style_name, style_data in top_styles:
            style_games = df[df['visual_style'] == style_name].nlargest(3, 'performance_score')
            
            if len(style_games) == 0:
                continue
            
            # Create style summary image
            self.create_style_summary_image(style_games, style_name, style_data, image_type)
    
    def create_style_summary_image(self, style_games, style_name, style_data, image_type):
        """Create a summary image showing examples of a style"""
        try:
            # Create a summary image with examples and stats
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Top Performing {image_type.title()} Style: {style_name.
