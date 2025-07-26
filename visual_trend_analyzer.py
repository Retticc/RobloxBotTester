#!/usr/bin/env python3
"""
visual_trend_analyzer_complete.py - Complete Visual Trend Analysis System
Enhanced with error handling, configuration management, and additional features
"""

import os
import sys
import logging
import configparser
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from collections import defaultdict, Counter
import base64
from io import BytesIO
import hashlib
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Image Analysis
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Image processing libraries not available: {e}")
    PIL_AVAILABLE = False

# ML for clustering visual patterns
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visual_trend_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class AnalysisConfig:
    """Configuration class for analysis parameters"""
    days_back: int = 30
    min_games: int = 5
    max_games_to_process: int = 1000
    min_performance_score: float = 1.0
    top_styles_count: int = 10
    cluster_count: int = 8
    image_sample_rate: int = 50  # Sample every nth pixel for performance
    enable_face_detection: bool = True
    enable_text_detection: bool = True
    save_images: bool = False
    create_visualizations: bool = True

class DatabaseManager:
    """Centralized database management"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection_pool = []
    
    def get_connection(self):
        """Get database connection with error handling"""
        try:
            return psycopg2.connect(self.db_url, sslmode="require")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Optional[List[Tuple]]:
        """Execute query with proper error handling"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    if fetch:
                        return cur.fetchall()
                    conn.commit()
                    return None
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def create_tables(self):
        """Create all necessary tables"""
        tables = {
            'trending_visual_assets': """
                CREATE TABLE IF NOT EXISTS trending_visual_assets (
                    id SERIAL PRIMARY KEY,
                    game_id BIGINT NOT NULL,
                    game_name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
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
                    confidence_score FLOAT DEFAULT 0.0,
                    UNIQUE(game_id, asset_type, analysis_date::date)
                );
            """,
            'trending_keywords_archive': """
                CREATE TABLE IF NOT EXISTS trending_keywords_archive (
                    id SERIAL PRIMARY KEY,
                    keyword TEXT NOT NULL,
                    popularity_rank INTEGER NOT NULL,
                    success_correlation FLOAT NOT NULL,
                    total_games INTEGER NOT NULL,
                    successful_games INTEGER NOT NULL,
                    avg_players_with_keyword FLOAT NOT NULL,
                    avg_players_without_keyword FLOAT NOT NULL,
                    performance_lift FLOAT NOT NULL,
                    trend_direction TEXT,
                    analysis_date TIMESTAMP DEFAULT NOW(),
                    category TEXT,
                    confidence_score FLOAT DEFAULT 0.0,
                    UNIQUE(keyword, analysis_date::date)
                );
            """,
            'top_performing_assets': """
                CREATE TABLE IF NOT EXISTS top_performing_assets (
                    id SERIAL PRIMARY KEY,
                    game_id BIGINT NOT NULL,
                    game_name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    performance_rank INTEGER NOT NULL,
                    performance_score FLOAT NOT NULL,
                    visual_style TEXT,
                    dominant_colors JSONB,
                    visual_features JSONB,
                    success_metrics JSONB,
                    image_data BYTEA,
                    image_hash TEXT UNIQUE,
                    date_added TIMESTAMP DEFAULT NOW(),
                    last_performance_update TIMESTAMP DEFAULT NOW(),
                    analysis_version TEXT DEFAULT '1.0'
                );
            """,
            'visual_clusters': """
                CREATE TABLE IF NOT EXISTS visual_clusters (
                    id SERIAL PRIMARY KEY,
                    cluster_id INTEGER NOT NULL,
                    cluster_name TEXT,
                    asset_type TEXT NOT NULL,
                    centroid_features JSONB,
                    games_count INTEGER,
                    avg_performance FLOAT,
                    analysis_date TIMESTAMP DEFAULT NOW(),
                    description TEXT
                );
            """,
            'analysis_metadata': """
                CREATE TABLE IF NOT EXISTS analysis_metadata (
                    id SERIAL PRIMARY KEY,
                    analysis_date TIMESTAMP DEFAULT NOW(),
                    analysis_type TEXT NOT NULL,
                    parameters JSONB,
                    results_summary JSONB,
                    execution_time_seconds FLOAT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT
                );
            """
        }
        
        for table_name, create_sql in tables.items():
            try:
                self.execute_query(create_sql, fetch=False)
                logger.info(f"Table {table_name} created/verified")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise

class ImageProcessor:
    """Handles all image processing operations"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.face_cascade = None
        
        if PIL_AVAILABLE and self.config.enable_face_detection:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception as e:
                logger.warning(f"Face detection unavailable: {e}")
                self.config.enable_face_detection = False
    
    def extract_enhanced_visual_features(self, image_data: bytes) -> Dict[str, Any]:
        """Extract comprehensive visual features with error handling"""
        if not image_data or not PIL_AVAILABLE:
            return self._empty_visual_features()
        
        try:
            # Load image
            img = Image.open(BytesIO(image_data))
            img_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            if img_cv is None or img.size[0] == 0 or img.size[1] == 0:
                return self._empty_visual_features()
            
            features = {}
            
            # Basic properties with validation
            features['width'], features['height'] = img.size
            features['aspect_ratio'] = self._safe_divide(features['width'], features['height'])
            features['total_pixels'] = features['width'] * features['height']
            features['file_size'] = len(image_data)
            
            # Color analysis
            colors = img.convert('RGB')
            pixels = np.array(colors).reshape(-1, 3)
            
            # RGB statistics
            features.update(self._calculate_color_stats(pixels))
            
            # HSV analysis
            features.update(self._calculate_hsv_stats(pixels))
            
            # Color diversity and dominant colors
            features.update(self._analyze_color_diversity(pixels))
            
            # Visual complexity analysis
            features.update(self._analyze_visual_complexity(img_cv))
            
            # Advanced feature detection
            if self.config.enable_face_detection:
                features.update(self._detect_faces(img_cv))
            
            if self.config.enable_text_detection:
                features.update(self._detect_text_regions(img_cv))
            
            # Layout and composition analysis
            features.update(self._analyze_composition(img_cv))
            
            # Style classification
            features['visual_style'] = self._classify_comprehensive_style(features, img_cv)
            features['confidence_score'] = self._calculate_confidence_score(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return self._empty_visual_features()
    
    def _safe_divide(self, a: float, b: float, default: float = 1.0) -> float:
        """Safe division with default value"""
        return a / b if b != 0 else default
    
    def _calculate_color_stats(self, pixels: np.ndarray) -> Dict[str, float]:
        """Calculate RGB color statistics"""
        try:
            return {
                'red_mean': float(np.mean(pixels[:, 0])),
                'green_mean': float(np.mean(pixels[:, 1])),
                'blue_mean': float(np.mean(pixels[:, 2])),
                'red_std': float(np.std(pixels[:, 0])),
                'green_std': float(np.std(pixels[:, 1])),
                'blue_std': float(np.std(pixels[:, 2])),
            }
        except Exception:
            return {
                'red_mean': 128.0, 'green_mean': 128.0, 'blue_mean': 128.0,
                'red_std': 0.0, 'green_std': 0.0, 'blue_std': 0.0
            }
    
    def _calculate_hsv_stats(self, pixels: np.ndarray) -> Dict[str, float]:
        """Calculate HSV color statistics"""
        try:
            import colorsys
            # Sample pixels for performance
            sampled_pixels = pixels[::self.config.image_sample_rate]
            
            hsv_values = []
            for pixel in sampled_pixels:
                if len(pixel) >= 3:
                    r, g, b = pixel[:3] / 255.0
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_values.append([h * 360, s * 100, v * 100])
            
            if hsv_values:
                hsv_array = np.array(hsv_values)
                return {
                    'hue_mean': float(np.mean(hsv_array[:, 0])),
                    'saturation_mean': float(np.mean(hsv_array[:, 1])),
                    'brightness_mean': float(np.mean(hsv_array[:, 2])),
                    'hue_std': float(np.std(hsv_array[:, 0])),
                    'saturation_std': float(np.std(hsv_array[:, 1])),
                    'brightness_std': float(np.std(hsv_array[:, 2])),
                }
            else:
                raise ValueError("No valid HSV values calculated")
                
        except Exception:
            return {
                'hue_mean': 0.0, 'saturation_mean': 0.0, 'brightness_mean': 50.0,
                'hue_std': 0.0, 'saturation_std': 0.0, 'brightness_std': 0.0
            }
    
    def _analyze_color_diversity(self, pixels: np.ndarray) -> Dict[str, Any]:
        """Analyze color diversity and dominant colors"""
        try:
            features = {}
            
            # Color diversity
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))
            features['color_diversity'] = unique_colors / len(pixels) if len(pixels) > 0 else 0
            
            # Dominant colors using clustering
            if SKLEARN_AVAILABLE and len(pixels) > 10:
                try:
                    # Sample for performance
                    sample_size = min(1000, len(pixels))
                    sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
                    sample_pixels = pixels[sample_indices]
                    
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    kmeans.fit(sample_pixels)
                    dominant_colors = kmeans.cluster_centers_
                    
                    for i, color in enumerate(dominant_colors):
                        features[f'dominant_color_{i}_r'] = float(color[0])
                        features[f'dominant_color_{i}_g'] = float(color[1])
                        features[f'dominant_color_{i}_b'] = float(color[2])
                        
                except Exception:
                    # Fallback: use mean colors
                    for i in range(5):
                        features[f'dominant_color_{i}_r'] = float(np.mean(pixels[:, 0]))
                        features[f'dominant_color_{i}_g'] = float(np.mean(pixels[:, 1]))
                        features[f'dominant_color_{i}_b'] = float(np.mean(pixels[:, 2]))
            else:
                # Simple fallback
                for i in range(5):
                    features[f'dominant_color_{i}_r'] = 128.0
                    features[f'dominant_color_{i}_g'] = 128.0
                    features[f'dominant_color_{i}_b'] = 128.0
            
            return features
            
        except Exception:
            features = {'color_diversity': 0.0}
            for i in range(5):
                features[f'dominant_color_{i}_r'] = 128.0
                features[f'dominant_color_{i}_g'] = 128.0
                features[f'dominant_color_{i}_b'] = 128.0
            return features
    
    def _analyze_visual_complexity(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Analyze visual complexity"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture analysis
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast and brightness
            contrast = float(np.std(gray))
            overall_brightness = float(np.mean(gray))
            
            return {
                'edge_density': float(edge_density),
                'texture_variance': float(texture_variance),
                'contrast': contrast,
                'overall_brightness': overall_brightness
            }
            
        except Exception:
            return {
                'edge_density': 0.0,
                'texture_variance': 0.0,
                'contrast': 0.0,
                'overall_brightness': 128.0
            }
    
    def _detect_faces(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Detect faces in image"""
        try:
            if self.face_cascade is None:
                return {'face_count': 0, 'has_faces': False}
            
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return {
                'face_count': len(faces),
                'has_faces': len(faces) > 0
            }
            
        except Exception:
            return {'face_count': 0, 'has_faces': False}
    
    def _detect_text_regions(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Detect text regions in image"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Use morphological operations to detect text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            connected = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                if 0.2 < aspect_ratio < 5 and area > 100:
                    text_regions += 1
            
            return {
                'text_regions': text_regions,
                'text_detected': text_regions > 2
            }
            
        except Exception:
            return {'text_regions': 0, 'text_detected': False}
    
    def _analyze_composition(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Analyze image composition"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Central focus analysis
            center_h, center_w = h // 2, w // 2
            center_region = gray[
                max(0, center_h - h//4):min(h, center_h + h//4),
                max(0, center_w - w//4):min(w, center_w + w//4)
            ]
            
            center_variance = np.var(center_region) if center_region.size > 0 else 0
            total_variance = np.var(gray) if gray.size > 0 else 1
            central_focus = center_variance / total_variance if total_variance > 0 else 0
            
            # Symmetry analysis
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            min_width = min(left_half.shape[1], right_half.shape[1])
            if min_width > 0:
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                
                diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
                symmetry_score = max(0, 1 - (np.mean(diff) / 255.0))
            else:
                symmetry_score = 0
            
            return {
                'central_focus': float(central_focus),
                'symmetry_score': float(symmetry_score)
            }
            
        except Exception:
            return {'central_focus': 0.0, 'symmetry_score': 0.0}
    
    def _classify_comprehensive_style(self, features: Dict[str, Any], img_cv: np.ndarray) -> str:
        """Classify visual style based on extracted features"""
        try:
            # Extract key features for classification
            brightness = features.get('brightness_mean', 50)
            saturation = features.get('saturation_mean', 0)
            has_faces = features.get('has_faces', False)
            edge_density = features.get('edge_density', 0)
            color_diversity = features.get('color_diversity', 0)
            texture_variance = features.get('texture_variance', 0)
            symmetry = features.get('symmetry_score', 0)
            
            # Style classification logic
            if has_faces and saturation > 60 and brightness > 150 and symmetry > 0.7:
                return "emoji_character"
            elif color_diversity < 0.1 and edge_density < 0.1 and texture_variance < 1000:
                return "minimalist_clean"
            elif saturation > 80 and brightness > 70:
                return "neon_vibrant"
            elif has_faces and edge_density > 0.15:
                return "character_rich"
            elif features.get('text_detected', False) and edge_density > 0.2:
                return "text_heavy"
            elif brightness > 180:
                return "bright_saturated" if saturation > 60 else "bright_pastel"
            elif brightness < 80:
                return "dark_vibrant" if saturation > 40 else "dark_monochrome"
            elif texture_variance > 2000:
                return "detailed_complex"
            elif features.get('central_focus', 0) > 1.5:
                return "centered_focus"
            else:
                return "balanced_standard"
                
        except Exception:
            return "unknown"
    
    def _calculate_confidence_score(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Factors that increase confidence
            confidence = 0.5  # Base confidence
            
            # Image quality factors
            if features.get('file_size', 0) > 1000:  # Larger files generally better quality
                confidence += 0.1
            
            if features.get('total_pixels', 0) > 10000:  # Higher resolution
                confidence += 0.1
            
            # Feature detection success
            if features.get('color_diversity', 0) > 0:
                confidence += 0.1
            
            if features.get('texture_variance', 0) > 0:
                confidence += 0.1
            
            # Style classification confidence
            style = features.get('visual_style', 'unknown')
            if style != 'unknown':
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5
    
    def _empty_visual_features(self) -> Dict[str, Any]:
        """Return default empty visual features"""
        features = {
            'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'total_pixels': 0, 'file_size': 0,
            'red_mean': 128.0, 'green_mean': 128.0, 'blue_mean': 128.0,
            'red_std': 0.0, 'green_std': 0.0, 'blue_std': 0.0,
            'hue_mean': 0.0, 'saturation_mean': 0.0, 'brightness_mean': 50.0,
            'hue_std': 0.0, 'saturation_std': 0.0, 'brightness_std': 0.0,
            'color_diversity': 0.0, 'edge_density': 0.0, 'texture_variance': 0.0,
            'contrast': 0.0, 'overall_brightness': 128.0, 'text_detected': False,
            'face_count': 0, 'has_faces': False, 'text_regions': 0,
            'central_focus': 0.0, 'symmetry_score': 0.0, 'visual_style': 'none',
            'confidence_score': 0.0
        }
        
        # Add dominant color features
        for i in range(5):
            features[f'dominant_color_{i}_r'] = 128.0
            features[f'dominant_color_{i}_g'] = 128.0
            features[f'dominant_color_{i}_b'] = 128.0
        
        return features

class VisualTrendAnalyzer:
    """Main Visual Trend Analyzer class"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.db_manager = DatabaseManager(os.getenv("DATABASE_URL"))
        self.image_processor = ImageProcessor(self.config)
        self.output_dir = "visual_trends"
        self._ensure_output_directory()
        
        # Initialize database tables
        try:
            self.db_manager.create_tables()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    def _ensure_output_directory(self):
        """Create output directories"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/reports",
            f"{self.output_dir}/visualizations",
            f"{self.output_dir}/exports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_full_analysis(self, days_back: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Run complete visual trend analysis with comprehensive error handling"""
        start_time = datetime.now()
        days_back = days_back or self.config.days_back
        
        logger.info(f"Starting full visual trend analysis for {days_back} days")
        
        try:
            # Record analysis start
            analysis_id = self._record_analysis_start(days_back)
            
            # Main analysis
            trend_report = self.analyze_trending_visuals(days_back)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record successful completion
            self._record_analysis_completion(analysis_id, trend_report, execution_time, True)
            
            logger.info(f"Analysis completed successfully in {execution_time:.2f} seconds")
            return trend_report
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis failed after {execution_time:.2f} seconds: {e}")
            
            # Record failure
            try:
                self._record_analysis_completion(analysis_id, {}, execution_time, False, str(e))
            except:
                pass
            
            return None
    
    def analyze_trending_visuals(self, days_back: int) -> Dict[str, Any]:
        """Main analysis method"""
        logger.info(f"Analyzing visual trends over last {days_back} days...")
        
        # Get game data
        games_df = self._fetch_game_data(days_back)
        
        if games_df.empty:
            logger.warning("No games found for analysis")
            return self._create_empty_report(days_back)
        
        logger.info(f"Processing {len(games_df)} games")
        
        # Process visual features
        icon_features, thumbnail_features = self._process_visual_features(games_df)
        
        # Analyze trends
        icon_trends = self._analyze_style_performance(icon_features, 'icon')
        thumbnail_trends = self._analyze_style_performance(thumbnail_features, 'thumbnail')
        
        # Generate comprehensive report
        trend_report = self._generate_comprehensive_report(
            icon_trends, thumbnail_trends, days_back, len(games_df)
        )
        
        # Save results
        self._save_analysis_results(trend_report, icon_trends, thumbnail_trends)
        
        return trend_report
    
    def _fetch_game_data(self, days_back: int) -> pd.DataFrame:
        """Fetch game data with visual assets"""
        query = """
        SELECT DISTINCT ON (g.id)
            g.id, g.name, g.description,
            s.icon_data, s.thumbnail_data,
            AVG(s.playing) OVER (PARTITION BY g.id) as avg_playing,
            AVG(s.visits) OVER (PARTITION BY g.id) as avg_visits,
            AVG(s.likes::float / NULLIF(s.likes + s.dislikes, 0)) OVER (PARTITION BY g.id) as like_ratio,
            COUNT(s.game_id) OVER (PARTITION BY g.id) as snapshot_count,
            MAX(s.snapshot_time) OVER (PARTITION BY g.id) as last_snapshot
        FROM games g
        JOIN snapshots s ON g.id = s.game_id
        WHERE s.snapshot_time > NOW() - INTERVAL '%s days'
        AND (s.icon_data IS NOT NULL OR s.thumbnail_data IS NOT NULL)
        ORDER BY g.id, s.snapshot_time DESC
        LIMIT %s
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=(days_back, self.config.max_games_to_process))
            
            # Filter for games with sufficient data
            df = df[df['snapshot_count'] >= self.config.min_games]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch game data: {e}")
            return pd.DataFrame()
    
    def _process_visual_features(self, games_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Process visual features for all games"""
        icon_features = []
        thumbnail_features = []
        
        total_games = len(games_df)
        
        for idx, row in games_df.iterrows():
            if idx % 50 == 0:
                logger.info(f"Processing game {idx+1}/{total_games}")
            
            try:
                # Calculate performance score
                performance_score = self._calculate_performance_score(row)
                
                if performance_score < self.config.min_performance_score:
                    continue
                
                game_data = {
                    'game_id': row['id'],
                    'game_name': row['name'],
                    'avg_playing': row['avg_playing'],
                    'avg_visits': row['avg_visits'],
                    'like_ratio': row['like_ratio'] or 0.5,
                    'performance_score': performance_score
                }
                
                # Process icon
                if row['icon_data']:
                    icon_visual = self.image_processor.extract_enhanced_visual_features(row['icon_data'])
                    icon_visual.update(game_data)
                    icon_visual['image_type'] = 'icon'
                    icon_features.append(icon_visual)
                
                # Process thumbnail
                if row['thumbnail_data']:
                    thumb_visual = self.image_processor.extract_enhanced_visual_features(row['thumbnail_data'])
                    thumb_visual.update(game_data)
                    thumb_visual['image_type'] = 'thumbnail'
                    thumbnail_features.append(thumb_visual)
                    
            except Exception as e:
                logger.warning(f"Error processing game {row['id']}: {e}")
                continue
        
        logger.info(f"Processed {len(icon_features)} icons and {len(thumbnail_features)} thumbnails")
        return icon_features, thumbnail_features
    
    def _calculate_performance_score(self, game_row: pd.Series) -> float:
        """Calculate normalized performance score"""
        try:
            playing_score = min(game_row['avg_playing'] / 1000, 10)
            visits_score = min(game_row['avg_visits'] / 1000000, 10)
            engagement_score = (game_row['like_ratio'] or 0.5) * 10
            
            return (playing_score + visits_score + engagement_score) / 3
        except Exception:
            return 0.0
    
    def _analyze_style_performance(self, features_list: List[Dict], image_type: str) -> Dict[str, Dict]:
        """Analyze performance by visual style"""
        if not features_list:
            return {}
        
        try:
            df = pd.DataFrame(features_list)
            
            # Group by visual style and calculate statistics
            style_analysis = df.groupby('visual_style').agg({
                'performance_score': ['count', 'mean', 'std'],
                'avg_playing': ['mean', 'median'],
                'like_ratio': 'mean',
                'saturation_mean': 'mean',
                'brightness_mean': 'mean',
                'edge_density': 'mean',
                'has_faces': 'mean',
                'confidence_score': 'mean'
            }).round(3)
            
            # Flatten column names
            style_analysis.columns = ['_'.join(col).strip() for col in style_analysis.columns]
            
            # Calculate trend scores
            style_analysis['trend_score'] = (
                style_analysis['performance_score_mean'] * 0.4 +
                style_analysis['avg_playing_mean'] / 100 * 0.3 +
                style_analysis['like_ratio_mean'] * 10 * 0.3
            )
            
            # Sort by trend score
            style_analysis = style_analysis.sort_values('trend_score', ascending=False)
            
            return style_analysis.to_dict('index')
            
        except Exception as e:
            logger.error(f"Error analyzing style performance for {image_type}: {e}")
            return {}
    
    def _generate_comprehensive_report(self, icon_trends: Dict, thumbnail_trends: Dict, 
                                     days_back: int, total_games: int) -> Dict[str, Any]:
        """Generate comprehensive trend report"""
        report = {
            'analysis_date': datetime.utcnow().isoformat(),
            'analysis_period_days': days_back,
            'total_games_analyzed': total_games,
            'total_icons_analyzed': sum(data['performance_score_count'] for data in icon_trends.values()),
            'total_thumbnails_analyzed': sum(data['performance_score_count'] for data in thumbnail_trends.values()),
            'summary': {},
            'top_icon_styles': [],
            'top_thumbnail_styles': [],
            'visual_recommendations': [],
            'trending_characteristics': {},
            'confidence_metrics': {}
        }
        
        # Generate summaries
        if icon_trends:
            report['summary'].update(self._generate_style_summary(icon_trends, 'icon'))
            report['top_icon_styles'] = self._format_top_styles(icon_trends)
        
        if thumbnail_trends:
            report['summary'].update(self._generate_style_summary(thumbnail_trends, 'thumbnail'))
            report['top_thumbnail_styles'] = self._format_top_styles(thumbnail_trends)
        
        # Generate recommendations and analysis
        report['visual_recommendations'] = self._generate_recommendations(icon_trends, thumbnail_trends)
        report['trending_characteristics'] = self._analyze_trending_characteristics(icon_trends, thumbnail_trends)
        report['confidence_metrics'] = self._calculate_confidence_metrics(icon_trends, thumbnail_trends)
        
        return report
    
    def _generate_style_summary(self, trends: Dict, style_type: str) -> Dict[str, Any]:
        """Generate summary for a style type"""
        if not trends:
            return {}
        
        best_style = max(trends.keys(), key=lambda x: trends[x]['trend_score'])
        best_data = trends[best_style]
        
        return {
            f'best_{style_type}_style': best_style,
            f'best_{style_type}_performance': round(best_data['performance_score_mean'], 2),
            f'best_{style_type}_confidence': round(best_data.get('confidence_score_mean', 0.5), 2)
        }
    
    def _format_top_styles(self, trends: Dict) -> List[Dict[str, Any]]:
        """Format top styles for report"""
        sorted_trends = sorted(trends.items(), key=lambda x: x[1]['trend_score'], reverse=True)
        
        top_styles = []
        for style_name, data in sorted_trends[:self.config.top_styles_count]:
            top_styles.append({
                'style': style_name,
                'performance_score': round(data['performance_score_mean'], 2),
                'game_count': int(data['performance_score_count']),
                'avg_players': round(data['avg_playing_mean'], 0),
                'like_ratio': round(data['like_ratio_mean'], 3),
                'confidence': round(data.get('confidence_score_mean', 0.5), 2),
                'characteristics': {
                    'brightness': round(data['brightness_mean'], 1),
                    'saturation': round(data['saturation_mean'], 1),
                    'has_faces_pct': round(data['has_faces'] * 100, 1),
                    'edge_density': round(data['edge_density'], 3)
                }
            })
        
        return top_styles
    
    def _generate_recommendations(self, icon_trends: Dict, thumbnail_trends: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Icon recommendations
        if icon_trends:
            best_icon = max(icon_trends.keys(), key=lambda x: icon_trends[x]['trend_score'])
            icon_data = icon_trends[best_icon]
            
            recommendations.append({
                'type': 'icon',
                'priority': 'high',
                'recommendation': f"Adopt '{best_icon.replace('_', ' ').title()}' style for icons",
                'reason': f"Shows {icon_data['performance_score_mean']:.1f} performance score across {icon_data['performance_score_count']} games",
                'confidence': icon_data.get('confidence_score_mean', 0.5),
                'specific_tips': [
                    f"Target brightness: {icon_data['brightness_mean']:.0f}/255",
                    f"Target saturation: {icon_data['saturation_mean']:.0f}%",
                    f"Include faces: {'Recommended' if icon_data['has_faces'] > 0.5 else 'Not necessary'}",
                    f"Edge density: {icon_data['edge_density']:.3f} (complexity level)"
                ]
            })
        
        # Thumbnail recommendations
        if thumbnail_trends:
            best_thumb = max(thumbnail_trends.keys(), key=lambda x: thumbnail_trends[x]['trend_score'])
            thumb_data = thumbnail_trends[best_thumb]
            
            recommendations.append({
                'type': 'thumbnail',
                'priority': 'high',
                'recommendation': f"Adopt '{best_thumb.replace('_', ' ').title()}' style for thumbnails",
                'reason': f"Shows {thumb_data['performance_score_mean']:.1f} performance score across {thumb_data['performance_score_count']} games",
                'confidence': thumb_data.get('confidence_score_mean', 0.5),
                'specific_tips': [
                    f"Target brightness: {thumb_data['brightness_mean']:.0f}/255",
                    f"Target saturation: {thumb_data['saturation_mean']:.0f}%",
                    f"Include faces: {'Recommended' if thumb_data['has_faces'] > 0.5 else 'Not necessary'}",
                    f"Edge density: {thumb_data['edge_density']:.3f} (complexity level)"
                ]
            })
        
        return recommendations
    
    def _analyze_trending_characteristics(self, icon_trends: Dict, thumbnail_trends: Dict) -> Dict[str, Any]:
        """Analyze overall trending characteristics"""
        characteristics = {
            'color_trends': {},
            'style_trends': {},
            'composition_trends': {}
        }
        
        try:
            # Combine all high-performing trends
            all_trends = {}
            all_trends.update({f"icon_{k}": v for k, v in (icon_trends or {}).items()})
            all_trends.update({f"thumb_{k}": v for k, v in (thumbnail_trends or {}).items()})
            
            if not all_trends:
                return characteristics
            
            # Filter for high-performing styles
            high_performing = {k: v for k, v in all_trends.items() if v['trend_score'] > 5}
            
            if high_performing:
                # Color trends
                avg_brightness = np.mean([data['brightness_mean'] for data in high_performing.values()])
                avg_saturation = np.mean([data['saturation_mean'] for data in high_performing.values()])
                
                characteristics['color_trends'] = {
                    'trending_brightness': self._categorize_brightness(avg_brightness),
                    'trending_saturation': self._categorize_saturation(avg_saturation),
                    'avg_brightness_value': round(avg_brightness, 1),
                    'avg_saturation_value': round(avg_saturation, 1)
                }
                
                # Style trends
                face_usage = np.mean([data['has_faces'] for data in high_performing.values()])
                edge_density = np.mean([data['edge_density'] for data in high_performing.values()])
                
                characteristics['style_trends'] = {
                    'faces_trending': face_usage > 0.4,
                    'complexity_trending': edge_density > 0.15,
                    'face_usage_pct': round(face_usage * 100, 1),
                    'avg_complexity': round(edge_density, 3)
                }
        
        except Exception as e:
            logger.error(f"Error analyzing trending characteristics: {e}")
        
        return characteristics
    
    def _categorize_brightness(self, brightness: float) -> str:
        """Categorize brightness level"""
        if brightness > 150:
            return 'high'
        elif brightness > 100:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_saturation(self, saturation: float) -> str:
        """Categorize saturation level"""
        if saturation > 60:
            return 'high'
        elif saturation > 30:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_metrics(self, icon_trends: Dict, thumbnail_trends: Dict) -> Dict[str, float]:
        """Calculate overall confidence metrics for the analysis"""
        try:
            all_trends = list((icon_trends or {}).values()) + list((thumbnail_trends or {}).values())
            
            if not all_trends:
                return {'overall_confidence': 0.0}
            
            # Calculate average confidence
            confidences = [trend.get('confidence_score_mean', 0.5) for trend in all_trends]
            avg_confidence = np.mean(confidences)
            
            # Calculate sample size confidence
            total_samples = sum(trend['performance_score_count'] for trend in all_trends)
            sample_confidence = min(1.0, total_samples / 100)  # 100+ samples = full confidence
            
            # Combined confidence
            overall_confidence = (avg_confidence + sample_confidence) / 2
            
            return {
                'overall_confidence': round(overall_confidence, 3),
                'avg_feature_confidence': round(avg_confidence, 3),
                'sample_size_confidence': round(sample_confidence, 3),
                'total_samples': total_samples
            }
            
        except Exception:
            return {'overall_confidence': 0.5}
    
    def _save_analysis_results(self, trend_report: Dict, icon_trends: Dict, thumbnail_trends: Dict):
        """Save analysis results to files and database"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save main report
            report_path = f"{self.output_dir}/reports/trend_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(trend_report, f, indent=2, default=str)
            
            logger.info(f"Trend report saved to {report_path}")
            
            # Save detailed trend data
            if icon_trends:
                icon_path = f"{self.output_dir}/reports/icon_trends_{timestamp}.csv"
                pd.DataFrame.from_dict(icon_trends, orient='index').to_csv(icon_path)
                logger.info(f"Icon trends saved to {icon_path}")
            
            if thumbnail_trends:
                thumb_path = f"{self.output_dir}/reports/thumbnail_trends_{timestamp}.csv"
                pd.DataFrame.from_dict(thumbnail_trends, orient='index').to_csv(thumb_path)
                logger.info(f"Thumbnail trends saved to {thumb_path}")
            
            # Create visualizations if enabled
            if self.config.create_visualizations:
                self._create_visualizations(trend_report, icon_trends, thumbnail_trends, timestamp)
                
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def _create_visualizations(self, trend_report: Dict, icon_trends: Dict, 
                             thumbnail_trends: Dict, timestamp: str):
        """Create analysis visualizations"""
        try:
            if not (plt and sns):
                logger.warning("Matplotlib/Seaborn not available for visualizations")
                return
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create summary visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Visual Trend Analysis Summary', fontsize=16)
            
            # Icon performance chart
            if icon_trends:
                self._plot_style_performance(axes[0, 0], icon_trends, 'Icon Styles Performance')
            
            # Thumbnail performance chart
            if thumbnail_trends:
                self._plot_style_performance(axes[0, 1], thumbnail_trends, 'Thumbnail Styles Performance')
            
            # Color trends
            self._plot_color_trends(axes[1, 0], trend_report)
            
            # Confidence metrics
            self._plot_confidence_metrics(axes[1, 1], trend_report)
            
            plt.tight_layout()
            viz_path = f"{self.output_dir}/visualizations/trend_analysis_{timestamp}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _plot_style_performance(self, ax, trends: Dict, title: str):
        """Plot style performance chart"""
        try:
            df = pd.DataFrame.from_dict(trends, orient='index')
            top_styles = df.nlargest(8, 'performance_score_mean')
            
            y_pos = range(len(top_styles))
            performance = top_styles['performance_score_mean']
            
            bars = ax.barh(y_pos, performance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([style.replace('_', ' ').title() for style in top_styles.index])
            ax.set_xlabel('Performance Score')
            ax.set_title(title)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', ha='left', va='center')
                       
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, 
                   ha='center', va='center')
    
    def _plot_color_trends(self, ax, trend_report: Dict):
        """Plot color trends"""
        try:
            color_data = trend_report.get('trending_characteristics', {}).get('color_trends', {})
            
            if color_data:
                categories = ['Brightness', 'Saturation']
                values = [
                    color_data.get('avg_brightness_value', 0),
                    color_data.get('avg_saturation_value', 0)
                ]
                
                bars = ax.bar(categories, values, color=['orange', 'blue'])
                ax.set_title('Trending Color Characteristics')
                ax.set_ylabel('Average Value')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No color trend data available', 
                       transform=ax.transAxes, ha='center', va='center')
                       
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, 
                   ha='center', va='center')
    
    def _plot_confidence_metrics(self, ax, trend_report: Dict):
        """Plot confidence metrics"""
        try:
            confidence_data = trend_report.get('confidence_metrics', {})
            
            if confidence_data:
                metrics = ['Overall', 'Feature', 'Sample Size']
                values = [
                    confidence_data.get('overall_confidence', 0) * 100,
                    confidence_data.get('avg_feature_confidence', 0) * 100,
                    confidence_data.get('sample_size_confidence', 0) * 100
                ]
                
                bars = ax.bar(metrics, values, color=['green', 'blue', 'purple'])
                ax.set_title('Analysis Confidence Metrics')
                ax.set_ylabel('Confidence (%)')
                ax.set_ylim(0, 100)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No confidence data available', 
                       transform=ax.transAxes, ha='center', va='center')
                       
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', transform=ax.transAxes, 
                   ha='center', va='center')
    
    def _record_analysis_start(self, days_back: int) -> int:
        """Record analysis start in metadata table"""
        try:
            query = """
            INSERT INTO analysis_metadata (analysis_type, parameters)
            VALUES (%s, %s)
            RETURNING id
            """
            
            parameters = {
                'days_back': days_back,
                'config': {
                    'min_games': self.config.min_games,
                    'max_games_to_process': self.config.max_games_to_process,
                    'min_performance_score': self.config.min_performance_score
                }
            }
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, ('visual_trend_analysis', json.dumps(parameters)))
                    analysis_id = cur.fetchone()[0]
                    conn.commit()
                    return analysis_id
                    
        except Exception as e:
            logger.error(f"Failed to record analysis start: {e}")
            return 0
    
    def _record_analysis_completion(self, analysis_id: int, trend_report: Dict, 
                                  execution_time: float, success: bool, error_message: str = None):
        """Record analysis completion"""
        try:
            query = """
            UPDATE analysis_metadata 
            SET results_summary = %s, execution_time_seconds = %s, success = %s, error_message = %s
            WHERE id = %s
            """
            
            results_summary = {
                'total_games': trend_report.get('total_games_analyzed', 0),
                'total_icons': trend_report.get('total_icons_analyzed', 0),
                'total_thumbnails': trend_report.get('total_thumbnails_analyzed', 0),
                'confidence': trend_report.get('confidence_metrics', {}).get('overall_confidence', 0)
            }
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (
                        json.dumps(results_summary), execution_time, success, error_message, analysis_id
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to record analysis completion: {e}")
    
    def _create_empty_report(self, days_back: int) -> Dict[str, Any]:
        """Create empty report when no data is available"""
        return {
            'analysis_date': datetime.utcnow().isoformat(),
            'analysis_period_days': days_back,
            'total_games_analyzed': 0,
            'total_icons_analyzed': 0,
            'total_thumbnails_analyzed': 0,
            'summary': {'message': 'No sufficient data available for analysis'},
            'top_icon_styles': [],
            'top_thumbnail_styles': [],
            'visual_recommendations': ['Collect more visual data to enable trend analysis'],
            'trending_characteristics': {},
            'confidence_metrics': {'overall_confidence': 0.0}
        }

def create_analyzer_from_config(config_file: str = 'analyzer_config.ini') -> VisualTrendAnalyzer:
    """Create analyzer instance from configuration file"""
    config = AnalysisConfig()
    
    if os.path.exists(config_file):
        parser = configparser.ConfigParser()
        parser.read(config_file)
        
        if 'analysis' in parser:
            analysis_section = parser['analysis']
            config.days_back = analysis_section.getint('days_back', config.days_back)
            config.min_games = analysis_section.getint('min_games', config.min_games)
            config.max_games_to_process = analysis_section.getint('max_games_to_process', config.max_games_to_process)
            config.min_performance_score = analysis_section.getfloat('min_performance_score', config.min_performance_score)
            config.enable_face_detection = analysis_section.getboolean('enable_face_detection', config.enable_face_detection)
            config.enable_text_detection = analysis_section.getboolean('enable_text_detection', config.enable_text_detection)
            config.create_visualizations = analysis_section.getboolean('create_visualizations', config.create_visualizations)
    
    return VisualTrendAnalyzer(config)

# Example configuration file content
CONFIG_TEMPLATE = """
[analysis]
days_back = 30
min_games = 5
max_games_to_process = 1000
min_performance_score = 1.0
top_styles_count = 10
enable_face_detection = true
enable_text_detection = true
create_visualizations = true
save_images = false

[database]
# Database URL is read from environment variable DATABASE_URL

[output]
output_dir = visual_trends
"""

def main():
    """Main execution function"""
    try:
        # Create configuration file if it doesn't exist
        if not os.path.exists('analyzer_config.ini'):
            with open('analyzer_config.ini', 'w') as f:
                f.write(CONFIG_TEMPLATE)
            logger.info("Created default configuration file: analyzer_config.ini")
        
        # Create analyzer
        analyzer = create_analyzer_from_config()
        
        # Run analysis
        logger.info("Starting visual trend analysis...")
        report = analyzer.run_full_analysis()
        
        if report:
            logger.info("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
            logger.info(f"Games Analyzed: {report['total_games_analyzed']}")
            logger.info(f"Icons Analyzed: {report['total_icons_analyzed']}")
            logger.info(f"Thumbnails Analyzed: {report['total_thumbnails_analyzed']}")
            logger.info(f"Overall Confidence: {report['confidence_metrics'].get('overall_confidence', 0):.1%}")
            
            if report['summary'].get('best_icon_style'):
                logger.info(f"Best Icon Style: {report['summary']['best_icon_style']}")
            
            if report['summary'].get('best_thumbnail_style'):
                logger.info(f"Best Thumbnail Style: {report['summary']['best_thumbnail_style']}")
            
            logger.info(f"Reports saved to: {analyzer.output_dir}/reports/")
            
            return True
        else:
            logger.error("Analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
