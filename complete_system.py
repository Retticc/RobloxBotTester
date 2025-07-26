#!/usr/bin/env python3
"""
Complete Integrated System for Railway Cron
Combines: Data Scraping + Continuous ML + Visual Trend Analysis

Single execution - perfect for Railway cron scheduling

RAILWAY ENVIRONMENT VARIABLES:
-----------------------------
ENABLE_ML=true                    # Enable ML analysis (default: true)
ENABLE_VISUAL_ANALYSIS=true      # Enable visual trend analysis (default: true)  
DATABASE_URL=postgresql://...    # Your database connection string

USAGE:
------
Set up Railway cron to run: python integrated_system.py
The system will automatically run all enabled components once and exit.
"""

import os
import sys
import traceback
import json
from datetime import datetime
from dotenv import load_dotenv

# Import all components
try:
    from main import scrape_and_snapshot, ensure_tables as scraper_ensure_tables
    from continuous_predictor import ContinuousGamePredictor
    from visual_trend_analyzer import VisualTrendAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

load_dotenv()

class CompleteGameAnalysisSystem:
    def __init__(self):
        self.ml_enabled = os.getenv("ENABLE_ML", "true").lower() == "true"
        self.visual_analysis_enabled = os.getenv("ENABLE_VISUAL_ANALYSIS", "true").lower() == "true"
        
        # Create output directory for reports
        self.output_dir = "integrated_analysis_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🚀 Complete Game Analysis System - Railway Cron Mode")
        print(f"ML: {self.ml_enabled} | Visual Analysis: {self.visual_analysis_enabled}")
    
    def run_analysis_cycle(self):
        """Run single analysis cycle for Railway cron"""
        session_start = datetime.utcnow()
        results = {
            'session_start': session_start,
            'scraping_success': False,
            'ml_results': None,
            'visual_analysis_results': None,
            'errors': []
        }
        
        print(f"\n📡 PHASE 1: DATA COLLECTION")
        print("-" * 30)
        
        # Phase 1: Data Collection (Always runs)
        try:
            scraper_ensure_tables()
            scraping_results = scrape_and_snapshot()
            results['scraping_success'] = True
            results['scraping_results'] = scraping_results
            print("✅ Data collection completed")
            
        except Exception as e:
            error_msg = f"Data collection failed: {e}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            traceback.print_exc()
            return results  # Exit early if scraping fails
        
        # Phase 2: Machine Learning Analysis
        if self.ml_enabled:
            print(f"\n🧠 PHASE 2: MACHINE LEARNING ANALYSIS")
            print("-" * 30)
            
            try:
                predictor = ContinuousGamePredictor()
                predictor.load_models_enhanced()
                ml_results = predictor.continuous_train()
                predictor.save_models_enhanced()
                results['ml_results'] = ml_results
                print("✅ Machine learning analysis completed")
                
                # Show quick insights
                self.show_ml_insights(predictor, ml_results)
                
            except Exception as e:
                error_msg = f"ML analysis failed: {e}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                traceback.print_exc()
        
        # Phase 3: Visual Trend Analysis
        if self.visual_analysis_enabled:
            print(f"\n🎨 PHASE 3: VISUAL TREND ANALYSIS")
            print("-" * 30)
            
            try:
                visual_analyzer = VisualTrendAnalyzer()
                visual_results = visual_analyzer.run_full_analysis()
                results['visual_analysis_results'] = visual_results
                print("✅ Visual trend analysis completed")
                
                # Show visual insights
                self.show_visual_insights(visual_results)
                
            except Exception as e:
                error_msg = f"Visual analysis failed: {e}"
                print(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                traceback.print_exc()
        
        # Phase 4: Generate Combined Insights
        print(f"\n📊 PHASE 4: COMBINED INSIGHTS")
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
        print("-" * 30)
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Components: {self.count_completed_phases(results)}/4 completed")
        
        if results['errors']:
            print(f"⚠️  {len(results['errors'])} errors occurred")
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
            'session_summary': {
                'scraping_completed': results['scraping_success'],
                'ml_training_completed': bool(results.get('ml_results')),
                'visual_analysis_completed': bool(results.get('visual_analysis_results')),
                'total_errors': len(results['errors']),
                'session_duration_minutes': results.get('duration_seconds', 0) / 60
            },
            'recommendations': []
        }
        
        try:
            # ML-based recommendations
            if results.get('ml_results'):
                ml_results = results['ml_results']
                insights['recommendations'].append({
                    'type': 'ml_insight',
                    'title': 'Machine Learning Analysis',
                    'recommendation': f"Models updated with {ml_results.get('total_games', 0)} games",
                    'priority': 'high' if ml_results.get('retrained') else 'medium'
                })
            
            # Visual-based recommendations
            if results.get('visual_analysis_results'):
                visual_recs = results['visual_analysis_results'].get('visual_recommendations', [])
                for rec in visual_recs[:2]:  # Top 2
                    insights['recommendations'].append({
                        'type': 'visual_trend',
                        'title': f"Visual Trend: {rec['type'].title()}",
                        'recommendation': rec['recommendation'],
                        'priority': 'high'
                    })
            
            # Strategic recommendation
            insights['recommendations'].append({
                'type': 'strategic',
                'title': 'Strategic Recommendation',
                'recommendation': 'Combine trending visual styles with high-performing keywords for optimal success',
                'priority': 'high'
            })
            
        except Exception as e:
            print(f"[combined_insights] Error: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def count_completed_phases(self, results):
        """Count how many phases completed successfully"""
        completed = 0
        if results.get('scraping_success'):
            completed += 1
        if results.get('ml_results'):
            completed += 1
        if results.get('visual_analysis_results'):
            completed += 1
        if results.get('combined_insights'):
            completed += 1
        return completed
    
    def save_session_report(self, results):
        """Save session report to file"""
        try:
            timestamp = results['session_start'].strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/session_report_{timestamp}.json"
            
            # Prepare serializable results
            serializable_results = {}
            for key, value in results.items():
                if key in ['session_start', 'session_end']:
                    serializable_results[key] = value.isoformat() if value else None
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"📄 Report saved: {filename}")
            
        except Exception as e:
            print(f"[save_report] Error: {e}")

def main():
    """Main function for Railway cron execution"""
    print("🚀 RAILWAY CRON - GAME ANALYSIS SYSTEM")
    print("=" * 50)
    
    try:
        system = CompleteGameAnalysisSystem()
        results = system.run_analysis_cycle()
        
        # Exit with appropriate code for monitoring
        exit_code = 1 if results.get('errors') else 0
        
        if exit_code == 0:
            print(f"\n✅ SUCCESS: Analysis completed in {results.get('duration_seconds', 0)/60:.1f} minutes")
        else:
            print(f"\n⚠️  COMPLETED WITH ERRORS: {len(results.get('errors', []))} issues")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ SYSTEM FAILURE: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
