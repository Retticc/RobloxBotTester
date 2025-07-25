new_games = len(result) if len(result) > 0 else 0
                else:
                    # First time training
                    new_games = float('inf')
                
                # Check total available games
                total_query = """
                SELECT COUNT(DISTINCT s.game_id) as total_games
                FROM snapshots s
                GROUP BY s.game_id
                HAVING COUNT(s.game_id) >= 5
                """
                total_result = pd.read_sql(total_query, conn)
                total_games = len(total_result)
                
                print(f"[retrain_check] New games: {new_games}, Total games: {total_games}")
                
                # Retrain conditions
                should_retrain = (
                    new_games >= self.retrain_threshold or  # Enough new data
                    total_games >= self.min_training_games and not self.models or  # First training
                    (self.model_metadata['last_training'] and 
                     (datetime.utcnow() - self.model_metadata['last_training']).days >= 7)  # Weekly retrain
                )
                
                return should_retrain, new_games, total_games
                
        except Exception as e:
            print(f"[retrain_check] Error: {e}")
            return True, 0, 0  # Default to retrain on error
    
    def continuous_train(self):
        """Main function to run after each snapshot batch"""
        print("\n🔄 Starting Continuous Learning Cycle")
        print("=" * 50)
        
        # Set up tracking tables
        self.ensure_tracking_tables()
        
        # Update dynamic keywords
        self.update_dynamic_keywords()
        
        # Detect image changes
        image_changes = self.detect_image_changes()
        
        # Check if we should retrain
        should_retrain, new_games, total_games = self.should_retrain()
        
        if should_retrain:
            print(f"[training] Retraining triggered: {new_games} new games, {total_games} total")
            
            # Create training dataset
            df = self.create_enhanced_training_dataset()
            
            if len(df) >= self.min_training_games:
                # Train models
                results = self.train_models_enhanced(df)
                
                # Save training metadata
                self.save_training_metadata(results, df)
                
                # Analyze recent performance patterns
                self.analyze_recent_patterns()
                
                print(f"✅ Retraining complete! Accuracy: {max([r['accuracy'] for r in results.values()]):.1%}")
            else:
                print(f"⚠️  Not enough games for training ({len(df)} < {self.min_training_games})")
        else:
            print("📊 No retraining needed, analyzing existing patterns...")
            self.analyze_recent_patterns()
        
        # Generate insights report
        self.generate_insights_report()
        
        return {
            'retrained': should_retrain,
            'image_changes': image_changes,
            'total_games': total_games,
            'keywords_updated': len(self.dynamic_keywords)
        }
    
    def create_enhanced_training_dataset(self, min_snapshots=5):
        """Create training dataset with enhanced features"""
        print("[dataset] Creating enhanced training dataset...")
        
        with self.get_conn() as conn:
            # Get games with sufficient data
            query = """
            SELECT g.id, g.name, g.description,
                   COUNT(s.game_id) as snapshot_count,
                   MAX(s.snapshot_time) as latest_snapshot,
                   MIN(s.snapshot_time) as earliest_snapshot,
                   AVG(s.playing) as avg_playing,
                   MAX(s.playing) as max_playing,
                   AVG(s.visits) as avg_visits,
                   MAX(s.visits) as max_visits,
                   AVG(s.favorites) as avg_favorites,
                   AVG(s.likes) as avg_likes,
                   AVG(s.dislikes) as avg_dislikes
            FROM games g
            JOIN snapshots s ON g.id = s.game_id
            GROUP BY g.id, g.name, g.description
            HAVING COUNT(s.game_id) >= %s
            ORDER BY avg_playing DESC
            """
            
            df = pd.read_sql(query, conn, params=(min_snapshots,))
        
        print(f"[dataset] Found {len(df)} games with sufficient data")
        
        all_features = []
        
        for idx, row in df.iterrows():
            if idx % 25 == 0:
                print(f"[dataset] Processing game {idx+1}/{len(df)}: {row['name'][:50]}...")
            
            try:
                game_features = {}
                
                # Basic metrics
                game_features['avg_playing'] = row['avg_playing']
                game_features['max_playing'] = row['max_playing']
                game_features['avg_visits'] = row['avg_visits']
                game_features['max_visits'] = row['max_visits']
                game_features['avg_favorites'] = row['avg_favorites']
                game_features['avg_likes'] = row['avg_likes']
                game_features['avg_dislikes'] = row['avg_dislikes']
                game_features['snapshot_count'] = row['snapshot_count']
                
                # Data age and consistency
                data_span_days = (row['latest_snapshot'] - row['earliest_snapshot']).days
                game_features['data_span_days'] = data_span_days
                game_features['data_density'] = row['snapshot_count'] / max(data_span_days, 1)
                
                # Derived metrics
                total_votes = row['avg_likes'] + row['avg_dislikes']
                game_features['like_ratio'] = row['avg_likes'] / total_votes if total_votes > 0 else 0.5
                game_features['engagement_rate'] = total_votes / row['avg_visits'] if row['avg_visits'] > 0 else 0
                
                # Enhanced text features with dynamic keywords
                combined_text = f"{row['name']} {row['description'] or ''}"
                text_features = self.extract_text_features_enhanced(combined_text)
                game_features.update(text_features)
                
                # Get latest images for this game
                image_query = """
                SELECT icon_data, thumbnail_data
                FROM snapshots
                WHERE game_id = %s AND (icon_data IS NOT NULL OR thumbnail_data IS NOT NULL)
                ORDER BY snapshot_time DESC
                LIMIT 1
                """
                
                image_result = pd.read_sql(image_query, conn, params=(row['id'],))
                
                if len(image_result) > 0:
                    # Icon features
                    icon_features = self.extract_image_features(image_result.iloc[0]['icon_data'])
                    icon_features = {f"icon_{k}": v for k, v in icon_features.items()}
                    game_features.update(icon_features)
                    
                    # Thumbnail features
                    thumb_features = self.extract_image_features(image_result.iloc[0]['thumbnail_data'])
                    thumb_features = {f"thumb_{k}": v for k, v in thumb_features.items()}
                    game_features.update(thumb_features)
                else:
                    # Empty image features
                    empty_icon = {f"icon_{k}": v for k, v in self._empty_image_features().items()}
                    empty_thumb = {f"thumb_{k}": v for k, v in self._empty_image_features().items()}
                    game_features.update(empty_icon)
                    game_features.update(empty_thumb)
                
                # Time-series features
                time_features = self.calculate_time_series_features_enhanced(row['id'])
                game_features.update(time_features)
                
                # Image change impact features
                image_impact_features = self.get_image_change_features(row['id'])
                game_features.update(image_impact_features)
                
                # Success labels with adaptive thresholds
                adaptive_thresholds = self.calculate_adaptive_thresholds(df)
                
                game_features['is_popular'] = row['avg_playing'] >= adaptive_thresholds['playing']
                game_features['has_growth'] = time_features.get('playing_growth', 1.0) >= adaptive_thresholds['growth']
                game_features['high_engagement'] = game_features['like_ratio'] >= adaptive_thresholds['engagement']
                
                # Combined success score
                success_score = (
                    (1 if game_features['is_popular'] else 0) +
                    (1 if game_features['has_growth'] else 0) + 
                    (1 if game_features['high_engagement'] else 0)
                )
                game_features['success_score'] = success_score
                game_features['is_successful'] = success_score >= 2
                
                game_features['game_id'] = row['id']
                game_features['game_name'] = row['name']
                
                all_features.append(game_features)
                
            except Exception as e:
                print(f"[dataset] Error processing game {row['id']}: {e}")
                continue
        
        feature_df = pd.DataFrame(all_features)
        
        print(f"[dataset] Created dataset with {len(feature_df)} games and {len(feature_df.columns)} features")
        print(f"[dataset] Success distribution: {feature_df['is_successful'].value_counts().to_dict()}")
        
        return feature_df
    
    def calculate_adaptive_thresholds(self, df):
        """Calculate adaptive success thresholds based on current data distribution"""
        return {
            'playing': max(df['avg_playing'].quantile(0.6), 50),  # Top 40% or minimum 50
            'growth': 1.2,  # Keep fixed for now
            'engagement': max(df['avg_likes'].sum() / (df['avg_likes'].sum() + df['avg_dislikes'].sum()), 0.7)
        }
    
    def calculate_time_series_features_enhanced(self, game_id, days=30):
        """Enhanced time-series analysis"""
        features = {}
        
        try:
            with self.get_conn() as conn:
                query = """
                SELECT snapshot_time, playing, visits, favorites, likes, dislikes
                FROM snapshots 
                WHERE game_id = %s 
                AND snapshot_time > %s
                ORDER BY snapshot_time
                """
                
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                df = pd.read_sql(query, conn, params=(game_id, cutoff_date))
                
                if len(df) < 2:
                    return self._empty_time_features()
                
                df['days_ago'] = (datetime.utcnow() - pd.to_datetime(df['snapshot_time'])).dt.days
                
                # Weekly performance analysis
                recent_data = df[df['days_ago'] <= 7]
                older_data = df[df['days_ago'] > 7]
                
                if len(recent_data) > 0 and len(older_data) > 0:
                    features['playing_growth'] = (recent_data['playing'].mean() / older_data['playing'].mean()) if older_data['playing'].mean() > 0 else 1
                    features['visits_growth'] = (recent_data['visits'].mean() / older_data['visits'].mean()) if older_data['visits'].mean() > 0 else 1
                    features['favorites_growth'] = (recent_data['favorites'].mean() / older_data['favorites'].mean()) if older_data['favorites'].mean() > 0 else 1
                else:
                    features['playing_growth'] = 1
                    features['visits_growth'] = 1
                    features['favorites_growth'] = 1
                
                # Volatility and consistency
                features['playing_volatility'] = df['playing'].std() / df['playing'].mean() if df['playing'].mean() > 0 else 0
                features['visits_volatility'] = df['visits'].std() / df['visits'].mean() if df['visits'].mean() > 0 else 0
                
                # Enhanced trend analysis
                if len(df) >= 3:
                    x = np.arange(len(df))
                    features['playing_trend'] = np.polyfit(x, df['playing'], 1)[0]
                    features['visits_trend'] = np.polyfit(x, df['visits'], 1)[0]
                    features['favorites_trend'] = np.polyfit(x, df['favorites'], 1)[0]
                    
                    # Acceleration (second derivative)
                    if len(df) >= 5:
                        features['playing_acceleration'] = np.polyfit(x, df['playing'], 2)[0]
                    else:
                        features['playing_acceleration'] = 0
                else:
                    features['playing_trend'] = 0
                    features['visits_trend'] = 0
                    features['favorites_trend'] = 0
                    features['playing_acceleration'] = 0
                
                # Peak and consistency analysis
                features['max_playing'] = df['playing'].max()
                features['max_visits'] = df['visits'].max()
                features['avg_playing'] = df['playing'].mean()
                features['avg_visits'] = df['visits'].mean()
                features['playing_consistency'] = 1 - (df['playing'].std() / df['playing'].mean()) if df['playing'].mean() > 0 else 0
                
                # Recent performance indicators
                features['recent_peak'] = recent_data['playing'].max() if len(recent_data) > 0 else 0
                features['recent_avg'] = recent_data['playing'].mean() if len(recent_data) > 0 else 0
                
                # Engagement metrics
                total_votes = df['likes'] + df['dislikes']
                features['like_ratio'] = (df['likes'] / total_votes).mean() if total_votes.sum() > 0 else 0.5
                features['engagement_rate'] = (total_votes / df['visits']).mean() if df['visits'].sum() > 0 else 0
                features['engagement_trend'] = np.polyfit(np.arange(len(df)), total_votes, 1)[0] if len(df) >= 3 else 0
                
                return features
                
        except Exception as e:
            print(f"[time_analysis] Error for game {game_id}: {e}")
            return self._empty_time_features()
    
    def get_image_change_features(self, game_id):
        """Get features related to image changes and their impact"""
        features = {
            'total_image_changes': 0,
            'recent_image_changes': 0,
            'avg_image_impact': 0,
            'positive_image_changes': 0,
            'icon_changes': 0,
            'thumbnail_changes': 0,
            'days_since_last_change': 999
        }
        
        try:
            with self.get_conn() as conn:
                query = """
                SELECT change_type, change_date, performance_impact
                FROM image_change_tracking
                WHERE game_id = %s
                ORDER BY change_date DESC
                """
                
                changes_df = pd.read_sql(query, conn, params=(game_id,))
                
                if len(changes_df) > 0:
                    features['total_image_changes'] = len(changes_df)
                    features['recent_image_changes'] = len(changes_df[
                        pd.to_datetime(changes_df['change_date']) > datetime.utcnow() - timedelta(days=30)
                    ])
                    
                    # Impact analysis
                    impact_data = changes_df[changes_df['performance_impact'].notna()]
                    if len(impact_data) > 0:
                        features['avg_image_impact'] = impact_data['performance_impact'].mean()
                        features['positive_image_changes'] = len(impact_data[impact_data['performance_impact'] > 0])
                    
                    # Change type breakdown
                    features['icon_changes'] = len(changes_df[changes_df['change_type'] == 'icon'])
                    features['thumbnail_changes'] = len(changes_df[changes_df['change_type'] == 'thumbnail'])
                    
                    # Recency
                    last_change = pd.to_datetime(changes_df.iloc[0]['change_date'])
                    features['days_since_last_change'] = (datetime.utcnow() - last_change).days
                
        except Exception as e:
            print(f"[image_features] Error for game {game_id}: {e}")
        
        return features
    
    def _empty_time_features(self):
        """Enhanced empty time features"""
        return {
            'playing_growth': 1.0, 'visits_growth': 1.0, 'favorites_growth': 1.0,
            'playing_volatility': 0, 'visits_volatility': 0,
            'playing_trend': 0, 'visits_trend': 0, 'favorites_trend': 0,
            'playing_acceleration': 0, 'playing_consistency': 0,
            'max_playing': 0, 'max_visits': 0, 'avg_playing': 0, 'avg_visits': 0,
            'recent_peak': 0, 'recent_avg': 0,
            'like_ratio': 0.5, 'engagement_rate': 0, 'engagement_trend': 0
        }
    
    def train_models_enhanced(self, df):
        """Enhanced model training with better feature handling"""
        print("[training] Training enhanced prediction models...")
        
        # Prepare features
        exclude_cols = ['is_successful', 'is_popular', 'has_growth', 'high_engagement', 
                       'success_score', 'game_id', 'game_name']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_features = []
        for col in feature_cols:
            if df[col].dtype == 'object':
                categorical_features.append(col)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        X = df[feature_cols]
        y = df['is_successful']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.feature_names = feature_cols
        self.scalers['main'] = scaler
        
        # Enhanced models with better parameters
        models_to_train = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=12, min_samples_split=15, min_samples_leaf=5, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=10, 
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.1, 
                min_samples_split=10, random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"[training] Training {name}...")
            
            if name == 'decision_tree':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"[training] {name} accuracy: {accuracy:.3f}")
            
            self.models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'actual': y_test
            }
        
        # Update model metadata
        self.model_metadata['last_training'] = datetime.utcnow()
        self.model_metadata['training_game_count'] = len(df)
        self.model_metadata['accuracy_history'].append(max([r['accuracy'] for r in results.values()]))
        
        # Feature importance analysis
        self.analyze_feature_importance_enhanced()
        
        return results
    
    def analyze_feature_importance_enhanced(self):
        """Enhanced feature importance analysis"""
        if 'random_forest' not in self.models:
            return
        
        rf_model = self.models['random_forest']
        importances = rf_model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Store in metadata
        top_features = feature_importance.head(10)[['feature', 'importance']].to_dict('records')
        self.model_metadata['feature_importance_history'].append({
            'date': datetime.utcnow().isoformat(),
            'top_features': top_features
        })
        
        print("\n🔍 Top 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        return feature_importance
    
    def save_training_metadata(self, results, df):
        """Save training session metadata to database"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    best_accuracy = max([r['accuracy'] for r in results.values()])
                    successful_games = df['is_successful'].sum()
                    
                    # Get top features
                    if hasattr(self, 'feature_names') and 'random_forest' in self.models:
                        importances = self.models['random_forest'].feature_importances_
                        top_features = dict(zip(self.feature_names, importances.tolist()))
                    else:
                        top_features = {}
                    
                    cur.execute("""
                        INSERT INTO ml_training_history 
                        (total_games, successful_games, model_accuracy, top_features, model_version, notes)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        len(df), successful_games, best_accuracy, 
                        json.dumps(top_features), "continuous_v1.0",
                        f"Trained with {len(self.dynamic_keywords)} dynamic keywords"
                    ))
                conn.commit()
        except Exception as e:
            print(f"[metadata] Error saving training metadata: {e}")
    
    def analyze_recent_patterns(self):
        """Analyze recent success patterns and image change impacts"""
        print("\n📊 Analyzing Recent Patterns...")
        
        try:
            with self.get_conn() as conn:
                # Analyze recent image changes and their impact
                impact_query = """
                SELECT 
                    change_type,
                    new_art_style,
                    AVG(performance_impact) as avg_impact,
                    COUNT(*) as change_count
                FROM image_change_tracking
                WHERE change_date > NOW() - INTERVAL '30 days'
                AND performance_impact IS NOT NULL
                GROUP BY change_type, new_art_style
                HAVING COUNT(*) >= 2
                ORDER BY avg_impact DESC
                """
                
                impact_analysis = pd.read_sql(impact_query, conn)
                
                if len(impact_analysis) > 0:
                    print("\n🎨 Recent Image Change Performance:")
                    for _, row in impact_analysis.head(10).iterrows():
                        impact_pct = row['avg_impact'] * 100
                        print(f"  {row['change_type']} → {row['new_art_style']}: {impact_pct:+.1f}% ({row['change_count']} changes)")
                
                # Analyze trending keywords
                trending_query = """
                SELECT keyword, success_correlation, popularity_score, game_count
                FROM dynamic_keywords
                WHERE last_updated > NOW() - INTERVAL '7 days'
                ORDER BY success_correlation DESC
                LIMIT 10
                """
                
                trending_keywords = pd.read_sql(trending_query, conn)
                
                if len(trending_keywords) > 0:
                    print("\n🔥 Trending High-Performance Keywords:")
                    for _, row in trending_keywords.iterrows():
                        print(f"  '{row['keyword']}': {row['success_correlation']:.2f}x success rate ({row['game_count']} games)")
        
        except Exception as e:
            print(f"[patterns] Error analyzing recent patterns: {e}")
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n📋 Generating Insights Report...")
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_status': 'trained' if self.models else 'untrained',
            'total_keywords': len(self.dynamic_keywords),
            'insights': []
        }
        
        try:
            with self.get_conn() as conn:
                # Image change insights
                positive_changes_query = """
                SELECT COUNT(*) as positive_count, AVG(performance_impact) as avg_positive_impact
                FROM image_change_tracking
                WHERE performance_impact > 0.1
                AND change_date > NOW() - INTERVAL '60 days'
                """
                
                positive_result = pd.read_sql(positive_changes_query, conn)
                if len(positive_result) > 0 and positive_result.iloc[0]['positive_count'] > 0:
                    avg_impact = positive_result.iloc[0]['avg_positive_impact']
                    count = positive_result.iloc[0]['positive_count']
                    report['insights'].append(f"🎯 {count} recent image changes showed positive impact (+{avg_impact:.1%} average)")
                
                # Keyword performance insights
                high_perf_keywords_query = """
                SELECT COUNT(*) as keyword_count
                FROM dynamic_keywords  
                WHERE success_correlation > 1.5
                """
                
                keyword_result = pd.read_sql(high_perf_keywords_query, conn)
                if len(keyword_result) > 0:
                    high_perf_count = keyword_result.iloc[0]['keyword_count']
                    report['insights'].append(f"📈 {high_perf_count} keywords show 1.5x+ success correlation")
                
                # Save report
                with open('insights_report.json', 'w') as f:
                    json.dump(report, f, indent=2)
                
                print("✅ Insights report saved to insights_report.json")
                
        except Exception as e:
            print(f"[insights] Error generating report: {e}")
        
        return report
    
    def predict_game_success(self, game_id):
        """Predict success for a specific game"""
        print(f"[prediction] Analyzing game {game_id}...")
        
        if 'random_forest' not in self.models:
            print("[prediction] ERROR: Models not trained yet!")
            return None
        
        try:
            # Get game data
            with self.get_conn() as conn:
                query = """
                SELECT g.id, g.name, g.description,
                       AVG(s.playing) as avg_playing,
                       MAX(s.playing) as max_playing,
                       AVG(s.visits) as avg_visits,
                       MAX(s.visits) as max_visits,
                       AVG(s.favorites) as avg_favorites,
                       AVG(s.likes) as avg_likes,
                       AVG(s.dislikes) as avg_dislikes,
                       COUNT(s.game_id) as snapshot_count,
                       s.icon_data,
                       s.thumbnail_data
                FROM games g
                JOIN snapshots s ON g.id = s.game_id
                WHERE g.id = %s
                GROUP BY g.id, g.name, g.description, s.icon_data, s.thumbnail_data
                """
                
                result = pd.read_sql(query, conn, params=(game_id,))
                
                if len(result) == 0:
                    print(f"[prediction] Game {game_id} not found!")
                    return None
                
                row = result.iloc[0]
            
            # Extract all features (same as training)
            game_features = {}
            
            # Basic metrics
            game_features['avg_playing'] = row['avg_playing']
            game_features['max_playing'] = row['max_playing']
            game_features['avg_visits'] = row['avg_visits']
            game_features['max_visits'] = row['max_visits']
            game_features['avg_favorites'] = row['avg_favorites']
            game_features['avg_likes'] = row['avg_likes']
            game_features['avg_dislikes'] = row['avg_dislikes']
            game_features['snapshot_count'] = row['snapshot_count']
            
            # Derived metrics
            total_votes = row['avg_likes'] + row['avg_dislikes']
            game_features['like_ratio'] = row['avg_likes'] / total_votes if total_votes > 0 else 0.5
            game_features['engagement_rate'] = total_votes / row['avg_visits'] if row['avg_visits'] > 0 else 0
            
            # Text features
            combined_text = f"{row['name']} {row['description'] or ''}"
            text_features = self.extract_text_features_enhanced(combined_text)
            game_features.update(text_features)
            
            # Image features
            icon_features = self.extract_image_features(row['icon_data'])
            icon_features = {f"icon_{k}": v for k, v in icon_features.items()}
            game_features.update(icon_features)
            
            thumb_features = self.extract_image_features(row['thumbnail_data'])
            thumb_features = {f"thumb_{k}": v for k, v in thumb_features.items()}
            game_features.update(thumb_features)
            
            # Time-series features
            time_features = self.calculate_time_series_features_enhanced(game_id)
            game_features.update(time_features)
            
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in game_features:
                    value = game_features[feature_name]
                    # Handle categorical features
                    if isinstance(value, str):
                        # Simple hash for categorical values (not ideal but works)
                        value = hash(value) % 1000
                    feature_vector.append(value)
                else:
                    feature_vector.append(0)  # Default value for missing features
            
            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scalers['main'].transform(feature_vector)
            
            # Make predictions
            rf_pred = self.models['random_forest'].predict(feature_vector_scaled)[0]
            rf_prob = self.models['random_forest'].predict_proba(feature_vector_scaled)[0]
            
            dt_pred = self.models['decision_tree'].predict(feature_vector)[0]
            dt_prob = self.models['decision_tree'].predict_proba(feature_vector)[0]
            
            # Compile results
            prediction_result = {
                'game_id': game_id,
                'game_name': row['name'],
                'current_avg_playing': row['avg_playing'],
                'current_avg_visits': row['avg_visits'],
                'predictions': {
                    'random_forest': {
                        'success_predicted': bool(rf_pred),
                        'success_probability': float(rf_prob[1]),
                        'confidence': float(max(rf_prob))
                    },
                    'decision_tree': {
                        'success_predicted': bool(dt_pred),
                        'success_probability': float(dt_prob[1]),
                        'confidence': float(max(dt_prob))
                    }
                },
                'key_features': {
                    'like_ratio': game_features['like_ratio'],
                    'playing_growth': time_features['playing_growth'],
                    'engagement_rate': game_features['engagement_rate'],
                    'icon_art_style': icon_features.get('icon_art_style', 'unknown'),
                    'thumb_art_style': thumb_features.get('thumb_art_style', 'unknown'),
                    'gaming_keywords': text_features['dynamic_keyword_count']
                }
            }
            
            return prediction_result
            
        except Exception as e:
            print(f"[prediction] Error predicting game {game_id}: {e}")
            return None
    
    def save_models_enhanced(self, filepath='continuous_game_models.pkl'):
        """Save enhanced models with metadata"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'success_thresholds': self.success_thresholds,
            'dynamic_keywords': list(self.dynamic_keywords),
            'model_metadata': self.model_metadata,
            'last_keyword_update': self.last_keyword_update
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[save] Enhanced models saved to {filepath}")
    
    def load_models_enhanced(self, filepath='continuous_game_models.pkl'):
        """Load enhanced models with metadata"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_names = model_data.get('feature_names', [])
            self.success_thresholds = model_data.get('success_thresholds', self.success_thresholds)
            self.dynamic_keywords = set(model_data.get('dynamic_keywords', []))
            self.model_metadata = model_data.get('model_metadata', self.model_metadata)
            self.last_keyword_update = model_data.get('last_keyword_update')
            
            print(f"[load] Enhanced models loaded from {filepath}")
            print(f"[load] {len(self.dynamic_keywords)} keywords, last trained: {self.model_metadata.get('last_training')}")
            return True
            
        except FileNotFoundError:
            print(f"[load] Model file {filepath} not found")
            return False

def main():
    """Main function for continuous learning"""
    print("🧠 Continuous Learning Game Predictor")
    print("=" * 50)
    
    predictor = ContinuousGamePredictor()
    
    # Load existing models if available
    predictor.load_models_enhanced()
    
    # Run continuous learning cycle
    results = predictor.continuous_train()
    
    # Save updated models
    predictor.save_models_enhanced()
    
    print(f"\n🎉 Continuous Learning Complete!")
    print(f"📊 Results: {results}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
continuous_predictor.py - Continuous Learning Game Predictor
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pickle
import hashlib
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Image Analysis
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import colorsys

# Text Analysis
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
from collections import Counter

load_dotenv()

class ContinuousGamePredictor:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.dynamic_keywords = set()
        self.keyword_update_frequency = 7  # Update keywords every 7 days
        self.last_keyword_update = None
        self.min_training_games = 30
        self.retrain_threshold = 10  # Retrain after N new games
        
        # Enhanced success thresholds that adapt over time
        self.success_thresholds = {
            'playing_threshold': 100,
            'growth_threshold': 1.2,
            'engagement_threshold': 0.75
        }
        
        self.model_metadata = {
            'last_training': None,
            'training_game_count': 0,
            'accuracy_history': [],
            'feature_importance_history': []
        }
        
    def get_conn(self):
        return psycopg2.connect(self.db_url, sslmode="require")
    
    def ensure_tracking_tables(self):
        """Create tables for tracking image changes and continuous learning"""
        print("[setup] Creating tracking tables...")
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                # Table to track image changes and their performance impact
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS image_change_tracking (
                        id SERIAL PRIMARY KEY,
                        game_id BIGINT NOT NULL,
                        change_date TIMESTAMP NOT NULL,
                        change_type TEXT NOT NULL, -- 'icon' or 'thumbnail'
                        old_image_hash TEXT,
                        new_image_hash TEXT NOT NULL,
                        old_art_style TEXT,
                        new_art_style TEXT,
                        players_before_7d FLOAT DEFAULT 0,
                        players_after_7d FLOAT DEFAULT 0,
                        players_before_30d FLOAT DEFAULT 0,
                        players_after_30d FLOAT DEFAULT 0,
                        performance_impact FLOAT, -- calculated impact score
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Table for dynamic keyword tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dynamic_keywords (
                        id SERIAL PRIMARY KEY,
                        keyword TEXT UNIQUE NOT NULL,
                        first_seen TIMESTAMP DEFAULT NOW(),
                        popularity_score FLOAT DEFAULT 1.0,
                        success_correlation FLOAT DEFAULT 0.0,
                        game_count INTEGER DEFAULT 1,
                        last_updated TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Table for continuous learning metadata
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_training_history (
                        id SERIAL PRIMARY KEY,
                        training_date TIMESTAMP DEFAULT NOW(),
                        total_games INTEGER,
                        successful_games INTEGER,
                        model_accuracy FLOAT,
                        top_features JSONB,
                        model_version TEXT,
                        notes TEXT
                    );
                """)
                
                # Indexes for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_image_tracking_game_date 
                    ON image_change_tracking(game_id, change_date);
                    
                    CREATE INDEX IF NOT EXISTS idx_keywords_popularity 
                    ON dynamic_keywords(popularity_score DESC);
                    
                    CREATE INDEX IF NOT EXISTS idx_training_history_date 
                    ON ml_training_history(training_date DESC);
                """)
            
            conn.commit()
        print("[setup] Tracking tables ready.")
    
    def calculate_image_hash(self, image_data):
        """Calculate hash for image to detect changes"""
        if not image_data:
            return None
        return hashlib.md5(image_data).hexdigest()
    
    def detect_image_changes(self):
        """Detect when games change their icons or thumbnails"""
        print("[tracking] Detecting image changes...")
        
        with self.get_conn() as conn:
            # Get current snapshots with images
            query = """
            SELECT DISTINCT ON (s.game_id) 
                s.game_id, s.snapshot_time, s.icon_data, s.thumbnail_data,
                g.name
            FROM snapshots s
            JOIN games g ON s.game_id = g.id
            WHERE s.icon_data IS NOT NULL OR s.thumbnail_data IS NOT NULL
            ORDER BY s.game_id, s.snapshot_time DESC
            """
            
            current_snapshots = pd.read_sql(query, conn)
            
            # Get historical image hashes we've already tracked
            tracked_query = """
            SELECT game_id, change_type, new_image_hash, change_date
            FROM image_change_tracking
            ORDER BY game_id, change_date DESC
            """
            
            tracked_changes = pd.read_sql(tracked_query, conn)
            
            new_changes = []
            
            for _, row in current_snapshots.iterrows():
                game_id = row['game_id']
                
                # Check icon changes
                if row['icon_data']:
                    current_icon_hash = self.calculate_image_hash(row['icon_data'])
                    
                    # Get last tracked icon hash
                    last_icon = tracked_changes[
                        (tracked_changes['game_id'] == game_id) & 
                        (tracked_changes['change_type'] == 'icon')
                    ]
                    
                    if len(last_icon) == 0 or last_icon.iloc[0]['new_image_hash'] != current_icon_hash:
                        old_hash = last_icon.iloc[0]['new_image_hash'] if len(last_icon) > 0 else None
                        new_style = self.extract_image_features(row['icon_data']).get('art_style', 'unknown')
                        
                        new_changes.append({
                            'game_id': game_id,
                            'change_date': row['snapshot_time'],
                            'change_type': 'icon',
                            'old_image_hash': old_hash,
                            'new_image_hash': current_icon_hash,
                            'old_art_style': None,
                            'new_art_style': new_style
                        })
                
                # Check thumbnail changes
                if row['thumbnail_data']:
                    current_thumb_hash = self.calculate_image_hash(row['thumbnail_data'])
                    
                    last_thumb = tracked_changes[
                        (tracked_changes['game_id'] == game_id) & 
                        (tracked_changes['change_type'] == 'thumbnail')
                    ]
                    
                    if len(last_thumb) == 0 or last_thumb.iloc[0]['new_image_hash'] != current_thumb_hash:
                        old_hash = last_thumb.iloc[0]['new_image_hash'] if len(last_thumb) > 0 else None
                        new_style = self.extract_image_features(row['thumbnail_data']).get('art_style', 'unknown')
                        
                        new_changes.append({
                            'game_id': game_id,
                            'change_date': row['snapshot_time'],
                            'change_type': 'thumbnail',
                            'old_image_hash': old_hash,
                            'new_image_hash': current_thumb_hash,
                            'old_art_style': None,
                            'new_art_style': new_style
                        })
            
            # Insert new changes
            if new_changes:
                print(f"[tracking] Found {len(new_changes)} new image changes")
                
                for change in new_changes:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO image_change_tracking 
                                (game_id, change_date, change_type, old_image_hash, new_image_hash, 
                                 old_art_style, new_art_style)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                change['game_id'], change['change_date'], change['change_type'],
                                change['old_image_hash'], change['new_image_hash'],
                                change['old_art_style'], change['new_art_style']
                            ))
                        conn.commit()
                    except Exception as e:
                        print(f"[tracking] Error inserting change: {e}")
                
                # Calculate performance impact for changes
                self.calculate_image_change_impact()
            
            return len(new_changes)
    
    def calculate_image_change_impact(self):
        """Calculate the performance impact of image changes"""
        print("[analysis] Calculating image change impact...")
        
        with self.get_conn() as conn:
            # Get image changes that haven't had impact calculated yet
            query = """
            SELECT id, game_id, change_date, change_type, new_art_style
            FROM image_change_tracking
            WHERE performance_impact IS NULL
            AND change_date < NOW() - INTERVAL '7 days'
            ORDER BY change_date DESC
            LIMIT 50
            """
            
            pending_changes = pd.read_sql(query, conn)
            
            for _, change in pending_changes.iterrows():
                try:
                    # Get player counts before and after the change
                    before_query = """
                    SELECT AVG(playing) as avg_playing
                    FROM snapshots
                    WHERE game_id = %s
                    AND snapshot_time BETWEEN %s AND %s
                    """
                    
                    after_query = """
                    SELECT AVG(playing) as avg_playing
                    FROM snapshots
                    WHERE game_id = %s
                    AND snapshot_time BETWEEN %s AND %s
                    """
                    
                    change_date = change['change_date']
                    
                    # 7 days before vs 7 days after
                    before_7d = pd.read_sql(before_query, conn, params=(
                        change['game_id'],
                        change_date - timedelta(days=14),
                        change_date - timedelta(days=7)
                    ))
                    
                    after_7d = pd.read_sql(after_query, conn, params=(
                        change['game_id'],
                        change_date + timedelta(days=1),
                        change_date + timedelta(days=8)
                    ))
                    
                    # 30 days before vs 30 days after  
                    before_30d = pd.read_sql(before_query, conn, params=(
                        change['game_id'],
                        change_date - timedelta(days=60),
                        change_date - timedelta(days=30)
                    ))
                    
                    after_30d = pd.read_sql(after_query, conn, params=(
                        change['game_id'],
                        change_date + timedelta(days=1),
                        change_date + timedelta(days=31)
                    ))
                    
                    # Calculate impact
                    players_before_7d = before_7d.iloc[0]['avg_playing'] if len(before_7d) > 0 else 0
                    players_after_7d = after_7d.iloc[0]['avg_playing'] if len(after_7d) > 0 else 0
                    players_before_30d = before_30d.iloc[0]['avg_playing'] if len(before_30d) > 0 else 0
                    players_after_30d = after_30d.iloc[0]['avg_playing'] if len(after_30d) > 0 else 0
                    
                    # Calculate performance impact score
                    impact_7d = (players_after_7d / players_before_7d - 1) if players_before_7d > 0 else 0
                    impact_30d = (players_after_30d / players_before_30d - 1) if players_before_30d > 0 else 0
                    
                    # Weighted impact score (7d impact weighted more heavily)
                    performance_impact = (impact_7d * 0.7) + (impact_30d * 0.3)
                    
                    # Update the record
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE image_change_tracking
                            SET players_before_7d = %s, players_after_7d = %s,
                                players_before_30d = %s, players_after_30d = %s,
                                performance_impact = %s
                            WHERE id = %s
                        """, (
                            players_before_7d, players_after_7d,
                            players_before_30d, players_after_30d,
                            performance_impact, change['id']
                        ))
                    
                    conn.commit()
                    
                    print(f"[analysis] Game {change['game_id']} {change['change_type']} change: {performance_impact:.2%} impact")
                    
                except Exception as e:
                    print(f"[analysis] Error calculating impact for change {change['id']}: {e}")
    
    def update_dynamic_keywords(self):
        """Update keyword database from popular games"""
        print("[keywords] Updating dynamic keyword database...")
        
        # Check if it's time to update keywords
        if (self.last_keyword_update and 
            (datetime.utcnow() - self.last_keyword_update).days < self.keyword_update_frequency):
            print("[keywords] Keywords recently updated, skipping...")
            return
        
        with self.get_conn() as conn:
            # Get popular games (top 500 by average players)
            query = """
            SELECT g.id, g.name, g.description, AVG(s.playing) as avg_playing,
                   AVG(s.likes::float / NULLIF(s.likes + s.dislikes, 0)) as like_ratio
            FROM games g
            JOIN snapshots s ON g.id = s.game_id
            WHERE s.snapshot_time > NOW() - INTERVAL '30 days'
            GROUP BY g.id, g.name, g.description
            HAVING COUNT(s.game_id) >= 5
            ORDER BY avg_playing DESC
            LIMIT 500
            """
            
            popular_games = pd.read_sql(query, conn)
            
            # Extract keywords from game names and descriptions
            all_text = []
            successful_text = []
            
            # Define success threshold dynamically based on data
            success_threshold = popular_games['avg_playing'].quantile(0.7)  # Top 30%
            
            for _, game in popular_games.iterrows():
                text = f"{game['name']} {game['description'] or ''}".lower()
                all_text.append(text)
                
                if game['avg_playing'] >= success_threshold:
                    successful_text.append(text)
            
            # Extract keywords using multiple methods
            new_keywords = set()
            
            # Method 1: Common gaming terms extraction
            gaming_patterns = [
                r'\b(\w*simulator\w*)\b',
                r'\b(\w*tycoon\w*)\b', 
                r'\b(\w*obby\w*)\b',
                r'\b(\w*roleplay\w*)\b',
                r'\b(\w*rp\w*)\b',
                r'\b(\w*adventure\w*)\b',
                r'\b(\w*survival\w*)\b',
                r'\b(\w*racing\w*)\b',
                r'\b(\w*fighting\w*)\b',
                r'\b(\w*horror\w*)\b',
                r'\b(\w*anime\w*)\b',
                r'\b(\w*adopt\w*)\b',
                r'\b(\w*pet\w*)\b',
                r'\b(\w*story\w*)\b',
                r'\b(\w*world\w*)\b',
                r'\b(\w*life\w*)\b',
                r'\b(\w*battle\w*)\b',
                r'\b(\w*war\w*)\b',
                r'\b(\w*school\w*)\b',
                r'\b(\w*hospital\w*)\b',
                r'\b(\w*restaurant\w*)\b',
                r'\b(\w*hotel\w*)\b',
                r'\b(\w*city\w*)\b',
                r'\b(\w*island\w*)\b',
                r'\b(\w*prison\w*)\b',
                r'\b(\w*zombie\w*)\b',
                r'\b(\w*magic\w*)\b',
                r'\b(\w*fantasy\w*)\b',
                r'\b(\w*space\w*)\b',
                r'\b(\w*ninja\w*)\b',
                r'\b(\w*pirate\w*)\b',
                r'\b(\w*kingdom\w*)\b',
                r'\b(\w*empire\w*)\b'
            ]
            
            all_combined_text = ' '.join(all_text)
            
            for pattern in gaming_patterns:
                matches = re.findall(pattern, all_combined_text)
                for match in matches:
                    if len(match) >= 3 and len(match) <= 20:  # Reasonable length
                        new_keywords.add(match.strip())
            
            # Method 2: Frequent words in successful games
            successful_combined = ' '.join(successful_text)
            words = re.findall(r'\b[a-zA-Z]{3,15}\b', successful_combined)
            
            # Filter out common stop words
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 
                'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 
                'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'than', 'them', 'well', 'were', 'been', 
                'have', 'their', 'said', 'each', 'which', 'what', 'will', 'there', 'with', 'other', 'this', 'that'
            }
            
            word_counts = Counter(words)
            
            for word, count in word_counts.most_common(200):
                if (word.lower() not in stop_words and 
                    len(word) >= 4 and 
                    count >= 5 and  # Appears in at least 5 games
                    word.isalpha()):  # Only alphabetic characters
                    new_keywords.add(word.lower())
            
            # Method 3: Bigrams and trigrams from successful games
            successful_words = successful_combined.split()
            
            # Bigrams
            for i in range(len(successful_words) - 1):
                bigram = f"{successful_words[i]} {successful_words[i+1]}"
                if (len(bigram) <= 25 and 
                    bigram.count(' ') == 1 and
                    all(len(word) >= 3 for word in bigram.split())):
                    new_keywords.add(bigram)
            
            print(f"[keywords] Extracted {len(new_keywords)} potential keywords")
            
            # Update database with new keywords
            with conn.cursor() as cur:
                for keyword in new_keywords:
                    try:
                        # Calculate success correlation for this keyword
                        keyword_games = [game for game in popular_games.itertuples() 
                                       if keyword in f"{game.name} {game.description or ''}".lower()]
                        
                        if len(keyword_games) >= 3:  # Need at least 3 games for correlation
                            keyword_avg_playing = np.mean([game.avg_playing for game in keyword_games])
                            overall_avg_playing = popular_games['avg_playing'].mean()
                            
                            success_correlation = keyword_avg_playing / overall_avg_playing
                            popularity_score = len(keyword_games) / len(popular_games)
                            
                            # Insert or update keyword
                            cur.execute("""
                                INSERT INTO dynamic_keywords (keyword, popularity_score, success_correlation, game_count)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (keyword) DO UPDATE SET
                                    popularity_score = EXCLUDED.popularity_score,
                                    success_correlation = EXCLUDED.success_correlation,
                                    game_count = EXCLUDED.game_count,
                                    last_updated = NOW()
                            """, (keyword, popularity_score, success_correlation, len(keyword_games)))
                    
                    except Exception as e:
                        print(f"[keywords] Error processing keyword '{keyword}': {e}")
                
                conn.commit()
            
            # Update our in-memory keyword set
            keyword_query = """
            SELECT keyword FROM dynamic_keywords 
            WHERE popularity_score > 0.01 OR success_correlation > 1.1
            ORDER BY success_correlation DESC, popularity_score DESC
            """
            
            keywords_df = pd.read_sql(keyword_query, conn)
            self.dynamic_keywords = set(keywords_df['keyword'].tolist())
            self.last_keyword_update = datetime.utcnow()
            
            print(f"[keywords] Updated to {len(self.dynamic_keywords)} active keywords")
    
    def extract_text_features_enhanced(self, text):
        """Enhanced text analysis using dynamic keywords"""
        if not text or pd.isna(text):
            text = ""
        
        features = {}
        text_lower = text.lower()
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Readability metrics
        try:
            features['reading_ease'] = flesch_reading_ease(text) if text else 50
            features['reading_grade'] = flesch_kincaid_grade(text) if text else 5
        except:
            features['reading_ease'] = 50
            features['reading_grade'] = 5
        
        # Dynamic keyword analysis
        features['dynamic_keyword_count'] = sum(1 for keyword in self.dynamic_keywords if keyword in text_lower)
        
        # High-value keywords (those with high success correlation)
        if hasattr(self, 'high_value_keywords'):
            features['high_value_keyword_count'] = sum(1 for keyword in self.high_value_keywords if keyword in text_lower)
        else:
            features['high_value_keyword_count'] = 0
        
        # Excitement indicators
        excitement_words = ['amazing', 'epic', 'awesome', 'incredible', 'fantastic', 'ultimate', 'mega', 'super', 'best']
        features['excitement_words'] = sum(1 for word in excitement_words if word in text_lower)
        
        # Special characters and formatting
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Trend keywords (recently popular)
        current_trending = self.get_trending_keywords()
        features['trending_keyword_count'] = sum(1 for keyword in current_trending if keyword in text_lower)
        
        return features
    
    def get_trending_keywords(self):
        """Get currently trending keywords"""
        try:
            with self.get_conn() as conn:
                query = """
                SELECT keyword FROM dynamic_keywords
                WHERE last_updated > NOW() - INTERVAL '30 days'
                AND success_correlation > 1.2
                ORDER BY popularity_score DESC
                LIMIT 20
                """
                result = pd.read_sql(query, conn)
                return result['keyword'].tolist()
        except:
            return []
    
    def extract_image_features(self, image_data):
        """Enhanced image feature extraction"""
        if not image_data:
            return self._empty_image_features()
        
        try:
            img = Image.open(BytesIO(image_data))
            img_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            features = {}
            
            # Basic properties
            features['img_width'], features['img_height'] = img.size
            features['img_aspect_ratio'] = features['img_width'] / features['img_height']
            features['img_size_bytes'] = len(image_data)
            
            # Color analysis
            colors = img.convert('RGB')
            pixels = np.array(colors).reshape(-1, 3)
            
            # Basic color stats
            features['avg_red'] = np.mean(pixels[:, 0])
            features['avg_green'] = np.mean(pixels[:, 1])
            features['avg_blue'] = np.mean(pixels[:, 2])
            
            # HSV analysis
            hsv_pixels = []
            for pixel in pixels[::100]:
                r, g, b = pixel / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_pixels.append([h * 360, s * 100, v * 100])
            
            hsv_array = np.array(hsv_pixels)
            features['avg_hue'] = np.mean(hsv_array[:, 0])
            features['avg_saturation'] = np.mean(hsv_array[:, 1])
            features['avg_brightness'] = np.mean(hsv_array[:, 2])
            features['color_variance'] = np.var(hsv_array[:, 1])
            
            # Brightness and contrast
            from PIL import ImageStat
            stat = ImageStat.Stat(colors)
            features['brightness'] = sum(stat.mean) / 3
            features['contrast'] = sum(stat.stddev) / 3
            
            # Edge detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture complexity
            features['texture_complexity'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Art style classification
            features['art_style'] = self._classify_art_style_enhanced(features, img_cv)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            features['has_faces'] = len(faces) > 0
            features['face_count'] = len(faces)
            
            return features
            
        except Exception as e:
            print(f"[image_analysis] Error: {e}")
            return self._empty_image_features()
    
    def _classify_art_style_enhanced(self, features, img_cv):
        """Enhanced art style classification based on successful patterns"""
        try:
            brightness = features['brightness']
            saturation = features['avg_saturation']
            edge_density = features['edge_density']
            texture = features['texture_complexity']
            
            # Enhanced classification based on successful game patterns
            if brightness > 200 and saturation > 70:
                if edge_density > 0.2:
                    return "vibrant_detailed"
                else:
                    return "bright_clean"
            elif brightness > 160 and saturation > 50:
                if features['face_count'] > 0:
                    return "character_focused"
                elif edge_density > 0.15:
                    return "cartoon_dynamic"
                else:
                    return "cartoon_simple"
            elif brightness < 80:
                if saturation > 40:
                    return "dark_colorful"
                else:
                    return "dark_minimal"
            elif saturation < 25:
                if texture > 1500:
                    return "realistic_detailed"
                else:
                    return "realistic_simple"
            elif edge_density > 0.25:
                return "complex_busy"
            else:
                return "balanced_standard"
                
        except Exception:
            return "unknown"
    
    def _empty_image_features(self):
        """Default image features"""
        return {
            'img_width': 0, 'img_height': 0, 'img_aspect_ratio': 1.0,
            'img_size_bytes': 0, 'avg_red': 128, 'avg_green': 128, 'avg_blue': 128,
            'avg_hue': 0, 'avg_saturation': 0, 'avg_brightness': 50,
            'color_variance': 0, 'brightness': 128, 'contrast': 0,
            'edge_density': 0, 'texture_complexity': 0, 'art_style': 'none',
            'has_faces': False, 'face_count': 0
        }
    
    def should_retrain(self):
        """Determine if models should be retrained"""
        try:
            with self.get_conn() as conn:
                # Count games since last training
                if self.model_metadata['last_training']:
                    query = """
                    SELECT COUNT(DISTINCT s.game_id) as new_games
                    FROM snapshots s
                    JOIN games g ON s.game_id = g.id
                    WHERE s.snapshot_time > %s
                    GROUP BY s.game_id
                    HAVING COUNT(s.game_id) >= 5
                    """
                    result = pd.read_sql(query, conn, params=(self.model_metadata['last_training'],))
                    new
