#!/usr/bin/env python3
"""
Complete Integrated System
Combines: Data Scraping + Continuous ML + Visual Trend Analysis
"""

import os
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Import all components
from main import scrape_and_snapshot, ensure_tables as scraper_ensure_tables
from continuous_predictor import ContinuousGamePredictor
from visual_trend_analyzer import VisualTrendAnalyzer

load_dotenv()

class CompleteGameAnalysisSystem:
    def __init__(self):
        self.scraper_enabled = True
        self.ml_enabled = os.getenv("ENABLE_ML", "true").lower() == "true"
        self.visual_analysis_enabled = os.getenv("ENABLE_VISUAL_ANALYSIS", "true").lower() == "true"
        
        self.ml_frequency = int(os.getenv("ML_FREQUENCY", "1"))
        self.visual_frequency = int(os.getenv("VISUAL_ANALYSIS_FREQUENCY", "2"))  # Every 2 sessions
        
        self.session_counter = 0
        
        print(f"[system] Complete Analysis System Initialized")
        print(f"[system] ML: {self.ml_enabled} (every {self.ml_frequency} sessions)")
        print(f"[system] Visual Analysis: {self.visual_analysis_enabled} (every {self.visual_frequency} sessions)")
    
    def run_complete_analysis_cycle(self):
        """Run the complete analysis cycle: Scraping + ML + Visual Trends"""
        session_start = datetime.utcnow()
        results = {
            'session_start': session_start,
            'scraping_success': False,
            'ml_results': None,
            'visual_analysis_results': None,
            'errors': []
        }
        
        print(f"\n🚀 COMPLETE ANALYSIS CYCLE STARTING")
        print(f"📅 Session: {session_start}")
        print("=" * 60)
        
        # Phase 1: Data Collection
        if self.scraper_enabled:
            print(f"\n📡 PHASE 1: DATA COLLECTION")
            print("-" * 30)
            
            try:
                scraper_ensure_tables()
                scrape_and_snapshot()
                results['scraping_success'] = True
                print("✅ Data collection completed successfully")
                
            except Exception as e:
                error_msg = f"Data collection failed: {e}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                traceback.print_exc()
        
        # Phase 2: Machine Learning Analysis
        if self.ml_enabled and results['scraping_success']:
            self.session_counter += 1
            
            if self.session_counter % self.ml_frequency == 0:
                print(f"\n🧠 PHASE 2: MACHINE LEARNING ANALYSIS")
                print("-" * 30)
                
                try:
                    predictor = ContinuousGamePredictor()
                    ml_results = predictor.continuous_train()
                    results['ml_results'] = ml_results
                    print("✅ Machine learning analysis completed")
                    
                    # Quick predictions summary
                    self.show_ml_insights(predictor, ml_results)
                    
                except Exception as e:
                    error_msg = f"ML analysis failed: {e}"
                    print(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    traceback.print_exc()
            else:
                print(f"\n⏭️ PHASE 2: ML Skipped (Session {self.session_counter}/{self.ml_frequency})")
        
        # Phase 3: Visual Trend Analysis
        if (self.visual_analysis_enabled and 
            results['scraping_success'] and 
            self.session_counter % self.visual_frequency == 0):
            
            print(f"\n🎨 PHASE 3: VISUAL TREND ANALYSIS")
            print("-" * 30)
            
            try:
                visual_analyzer = VisualTrendAnalyzer()
                visual_results = visual_analyzer.generate_trending_report()
                results['visual_analysis_results'] = visual_results
                print("✅ Visual trend analysis completed")
                
                # Show visual insights
                self.show_visual_insights(visual_results)
                
            except Exception as e:
                error_msg = f"Visual analysis failed: {e}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                traceback.print_exc()
        elif self.visual_analysis_enabled:
            print(f"\n⏭️ PHASE 3: Visual Analysis Skipped (Session {self.session_counter}/{self.visual_frequency})")
        
        # Phase 4: Generate Combined Insights
        if results['scraping_success']:
            print(f"\n📊 PHASE 4: COMBINED INSIGHTS GENERATION")
            print("-" * 30)
            
            try:
                combined_insights = self.generate_combined_insights(results)
                results['combined_insights'] = combined_insights
                self.save_session_report(results)
                print("✅ Combined insights generated and saved")
                
            except Exception as e:
                error_msg = f"Insights generation failed: {e}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
        
        # Session Summary
        session_end = datetime.utcnow()
        duration = (session_end - session_start).total_seconds()
        results['session_end'] = session_end
        results['duration_seconds'] = duration
        
        print(f"\n🏁 SESSION COMPLETE")
        print("=" * 30)
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Components run: {self.count_completed_phases(results)}/4")
        
        if results['errors']:
            print(f"⚠️  Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   • {error}")
        else:
            print("✅ All phases completed successfully!")
        
        return results
    
    def show_ml_insights(self, predictor, ml_results):
        """Show key ML insights from the session"""
        try:
            if ml_results and ml_results.get('retrained'):
                print(f"\n🧠 ML Insights:")
                print(f"   • Models retrained with {ml_results.get('total_games', 0)} games")
                print(f"   • {ml_results.get('image_changes', 0)} new image changes detected")
                print(f"   • {ml_results.get('keywords_updated', 0)} keywords in database")
                
                # Quick top predictions
                self.show_quick_predictions(predictor)
        except Exception as e:
            print(f"[ml_insights] Error: {e}")
    
    def show_quick_predictions(self, predictor):
        """Show quick predictions for top recent games"""
        try:
            print(f"\n🔮 Recent High-Potential Games:")
            
            with predictor.get_conn() as conn:
                import pandas as pd
                query = """
                SELECT g.id, g.name, AVG(s.playing) as avg_playing
                FROM games g
                JOIN snapshots s ON g.id = s.game_id
                WHERE s.snapshot_time > NOW() - INTERVAL '12 hours'
                GROUP BY g.id, g.name
                HAVING COUNT(s.game_id) >= 2 AND AVG(s.playing) >= 100
                ORDER BY avg_playing DESC
                LIMIT 5
                """
                
                recent_games = pd.read_sql(query, conn)
            
            for _, game in recent_games.iterrows():
                try:
                    result = predictor.predict_game_success(game['id'])
                    if result:
                        rf_prob = result['predictions']['random_forest']['success_probability']
                        status = "🟢" if rf_prob > 0.7 else "🟡" if rf_prob > 0.4 else "🔴"
                        print(f"   {status} {game['name'][:35]:35} | {rf_prob:.1%} success prob | {game['avg_playing']:.0f} players")
                except:
                    continue
                    
        except Exception as e:
            print(f"[quick_predictions] Error: {e}")
    
    def show_visual_insights(self, visual_results):
        """Show key visual trend insights"""
        try:
            if not visual_results:
                return
                
            print(f"\n🎨 Visual Trend Insights:")
            
            # Top visual styles
            if visual_results.get('top_icon_styles'):
                top_icon = visual_results['top_icon_styles'][0]
                print(f"   • Best icon style: {top_icon['style']} ({top_icon['game_count']} games)")
            
            if visual_results.get('top_thumbnail_styles'):
                top_thumb = visual_results['top_thumbnail_styles'][0]
                print(f"   • Best thumbnail style: {top_thumb['style']} ({top_thumb['game_count']} games)")
            
            # Key recommendations
            if visual_results.get('visual_recommendations'):
                print(f"   • {len(visual_results['visual_recommendations'])} actionable recommendations generated")
                
        except Exception as e:
            print(f"[visual_insights] Error: {e}")
    
    def generate_combined_insights(self, results):
        """Generate insights combining ML and visual analysis"""
        insights = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_summary': {},
            'cross_analysis': {},
            'recommendations': []
        }
        
        try:
            # Session summary
            insights['session_summary'] = {
                'scraping_completed': results['scraping_success'],
                'ml_training_completed': bool(results.get('ml_results')),
                'visual_analysis_completed': bool(results.get('visual_analysis_results')),
                'total_errors': len(results['errors']),
                'session_duration_minutes': results.get('duration_seconds', 0) / 60
            }
            
            # Cross-analysis (if both ML and visual data available)
            if results.get('ml_results') and results.get('visual_analysis_results'):
                insights['cross_analysis'] = self.perform_cross_analysis(
                    results['ml_results'], 
                    results['visual_analysis_results']
                )
            
            # Combined recommendations
            insights['recommendations'] = self.generate_combined_recommendations(results)
            
        except Exception as e:
            print(f"[combined_insights] Error: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def perform_cross_analysis(self, ml_results, visual_results):
        """Perform analysis combining ML and visual insights"""
        cross_analysis = {
            'correlation_found': False,
            'insights': []
        }
        
        try:
            # Example: Correlate image changes with keyword trends
            if ml_results.get('image_changes', 0) > 0:
                cross_analysis['insights'].append(
                    f"Detected {ml_results['image_changes']} image changes this session"
                )
            
            # Example: Visual style performance vs ML predictions
            if visual_results.get('summary'):
                best_icon_style = visual_results['summary'].get('best_icon_style')
                if best_icon_style:
                    cross_analysis['insights'].append(
                        f"ML models should weight '{best_icon_style}' icons higher"
                    )
            
            cross_analysis['correlation_found'] = len(cross_analysis['insights']) > 0
            
        except Exception as e:
            cross_analysis['error'] = str(e)
        
        return cross_analysis
    
    def generate_combined_recommendations(self, results):
        """Generate recommendations combining all analysis types"""
        recommendations = []
        
        try:
            # ML-based recommendations
            if results.get('ml_results'):
                recommendations.append({
                    'type': 'ml_insight',
                    'title': 'Machine Learning Insights',
                    'recommendation': f"Models trained on {results['ml_results'].get('total_games', 0)} games with latest patterns"
                })
            
            # Visual-based recommendations
            if results.get('visual_analysis_results'):
                visual_recs = results['visual_analysis_results'].get('visual_recommendations', [])
                for rec in visual_recs[:3]:  # Top 3
                    recommendations.append({
                        'type': 'visual_trend',
                        'title': f"Visual Trend: {rec['type'].title()}",
                        'recommendation': rec['recommendation'],
                        'details': rec.get('specific_tips', [])
                    })
            
            # Combined strategic recommendation
            recommendations.append({
                'type': 'strategic',
                'title': 'Strategic Recommendation',
                'recommendation': 'Focus on games that combine trending visual styles with high-performing keywords for maximum success probability'
            })
            
        except Exception as e:
            recommendations.append({
                'type': 'error',
                'title': 'Recommendation Generation Error',
                'recommendation': f'Error generating recommendations: {e}'
            })
