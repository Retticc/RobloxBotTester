#!/usr/bin/env python3
"""
Complete Integrated System
Combines: Data Scraping + Continuous ML + Visual Trend Analysis
"""

import os
import sys
import traceback
import json
import schedule
import time
from datetime import datetime, timedelta
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
        self.scraper_enabled = True
        self.ml_enabled = os.getenv("ENABLE_ML", "true").lower() == "true"
        self.visual_analysis_enabled = os.getenv("ENABLE_VISUAL_ANALYSIS", "true").lower() == "true"
        
        self.ml_frequency = int(os.getenv("ML_FREQUENCY", "1"))
        self.visual_frequency = int(os.getenv("VISUAL_ANALYSIS_FREQUENCY", "2"))  # Every 2 sessions
        
        self.session_counter = 0
        self.session_history = []
        
        # Create output directory for reports
        self.output_dir = "integrated_analysis_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[system] Complete Analysis System Initialized")
        print(f"[system] ML: {self.ml_enabled} (every {self.ml_frequency} sessions)")
        print(f"[system] Visual Analysis: {self.visual_analysis_enabled} (every {self.visual_frequency} sessions)")
        print(f"[system] Reports saved to: {self.output_dir}/")
    
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
                scraping_results = scrape_and_snapshot()
                results['scraping_success'] = True
                results['scraping_results'] = scraping_results
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
                    predictor.load_models_enhanced()  # Load existing models
                    ml_results = predictor.continuous_train()
                    predictor.save_models_enhanced()  # Save updated models
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
        
        # Store in history
        self.session_history.append(results)
        
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
            
            # Trending characteristics correlation
            if visual_results.get('trending_characteristics'):
                color_trends = visual_results['trending_characteristics'].get('color_trends', {})
                if color_trends:
                    brightness_trend = color_trends.get('trending_brightness', 'medium')
                    cross_analysis['insights'].append(
                        f"Current visual trend: {brightness_trend} brightness games performing well"
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
                ml_results = results['ml_results']
                recommendations.append({
                    'type': 'ml_insight',
                    'title': 'Machine Learning Insights',
                    'recommendation': f"Models trained on {ml_results.get('total_games', 0)} games with latest patterns",
                    'priority': 'high' if ml_results.get('retrained') else 'medium'
                })
                
                # Keyword recommendations
                if ml_results.get('keywords_updated', 0) > 0:
                    recommendations.append({
                        'type': 'keywords',
                        'title': 'Trending Keywords',
                        'recommendation': f"Incorporate trending keywords from {ml_results['keywords_updated']} keyword database",
                        'priority': 'medium'
                    })
            
            # Visual-based recommendations
            if results.get('visual_analysis_results'):
                visual_recs = results['visual_analysis_results'].get('visual_recommendations', [])
                for rec in visual_recs[:3]:  # Top 3
                    recommendations.append({
                        'type': 'visual_trend',
                        'title': f"Visual Trend: {rec['type'].title()}",
                        'recommendation': rec['recommendation'],
                        'details': rec.get('specific_tips', []),
                        'priority': 'high'
                    })
            
            # Combined strategic recommendation
            recommendations.append({
                'type': 'strategic',
                'title': 'Strategic Recommendation',
                'recommendation': 'Focus on games that combine trending visual styles with high-performing keywords for maximum success probability',
                'priority': 'high'
            })
            
            # Performance optimization recommendations
            if results.get('duration_seconds', 0) > 1800:  # > 30 minutes
                recommendations.append({
                    'type': 'performance',
                    'title': 'Performance Optimization',
                    'recommendation': 'Consider optimizing analysis frequency or parallel processing for faster execution',
                    'priority': 'low'
                })
            
        except Exception as e:
            recommendations.append({
                'type': 'error',
                'title': 'Recommendation Generation Error',
                'recommendation': f'Error generating recommendations: {e}',
                'priority': 'high'
            })
        
        return recommendations
    
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
        """Save detailed session report to file"""
        try:
            timestamp = results['session_start'].strftime('%Y%m%d_%H%M%S')
            filename = f"{self.output_dir}/session_report_{timestamp}.json"
            
            # Prepare serializable results
            serializable_results = self.prepare_serializable_results(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"📄 Session report saved: {filename}")
            
            # Also save a summary CSV for easy analysis
            self.save_session_summary_csv()
            
        except Exception as e:
            print(f"[save_report] Error: {e}")
    
    def prepare_serializable_results(self, results):
        """Prepare results for JSON serialization"""
        serializable = {}
        
        for key, value in results.items():
            if key in ['session_start', 'session_end']:
                serializable[key] = value.isoformat() if value else None
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable[key] = value
            else:
                serializable[key] = str(value)
        
        return serializable
    
    def save_session_summary_csv(self):
        """Save session summary as CSV for tracking over time"""
        try:
            import pandas as pd
            
            csv_file = f"{self.output_dir}/session_summary.csv"
            
            # Prepare summary data
            summary_data = []
            for session in self.session_history:
                summary_data.append({
                    'timestamp': session['session_start'],
                    'duration_minutes': session.get('duration_seconds', 0) / 60,
                    'scraping_success': session.get('scraping_success', False),
                    'ml_completed': bool(session.get('ml_results')),
                    'visual_completed': bool(session.get('visual_analysis_results')),
                    'errors_count': len(session.get('errors', [])),
                    'phases_completed': self.count_completed_phases(session)
                })
            
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            print(f"[save_csv] Error: {e}")
    
    def run_scheduled_analysis(self):
        """Run analysis on a schedule"""
        print("🕐 Setting up scheduled analysis...")
        
        # Get schedule from environment or use defaults
        scraping_schedule = os.getenv("SCRAPING_SCHEDULE", "*/30 * * * *")  # Every 30 minutes
        analysis_schedule = os.getenv("ANALYSIS_SCHEDULE", "0 */6 * * *")   # Every 6 hours
        
        print(f"📅 Scraping schedule: {scraping_schedule}")
        print(f"📅 Full analysis schedule: {analysis_schedule}")
        
        # Set up simple interval-based scheduling
        while True:
            try:
                print(f"\n⏰ {datetime.now()}: Running scheduled analysis...")
                self.run_complete_analysis_cycle()
                
                # Wait for next cycle (default: 30 minutes)
                wait_minutes = int(os.getenv("CYCLE_INTERVAL_MINUTES", "30"))
                print(f"⏳ Waiting {wait_minutes} minutes until next cycle...")
                time.sleep(wait_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Scheduled analysis stopped by user")
                break
            except Exception as e:
                print(f"❌ Scheduled analysis error: {e}")
                traceback.print_exc()
                # Wait before retrying
                time.sleep(300)  # 5 minutes
    
    def get_system_status(self):
        """Get current system status and health"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_count': self.session_counter,
            'components': {
                'scraper': self.scraper_enabled,
                'ml': self.ml_enabled,
                'visual_analysis': self.visual_analysis_enabled
            },
            'recent_sessions': len([s for s in self.session_history if 
                                  (datetime.utcnow() - s['session_start']).days < 1]),
            'last_session': self.session_history[-1] if self.session_history else None
        }
        
        return status
    
    def generate_weekly_report(self):
        """Generate comprehensive weekly performance report"""
        try:
            print("📊 Generating weekly report...")
            
            # Filter sessions from last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)
            weekly_sessions = [s for s in self.session_history if s['session_start'] > week_ago]
            
            if not weekly_sessions:
                print("No sessions in the last week")
                return
            
            report = {
                'week_ending': datetime.utcnow().isoformat(),
                'total_sessions': len(weekly_sessions),
                'successful_sessions': len([s for s in weekly_sessions if not s.get('errors')]),
                'average_duration_minutes': sum(s.get('duration_seconds', 0) for s in weekly_sessions) / len(weekly_sessions) / 60,
                'ml_sessions': len([s for s in weekly_sessions if s.get('ml_results')]),
                'visual_sessions': len([s for s in weekly_sessions if s.get('visual_analysis_results')]),
                'total_errors': sum(len(s.get('errors', [])) for s in weekly_sessions)
            }
            
            # Save weekly report
            filename = f"{self.output_dir}/weekly_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Weekly report saved: {filename}")
            print(f"📈 Summary: {report['successful_sessions']}/{report['total_sessions']} successful sessions")
            
        except Exception as e:
            print(f"[weekly_report] Error: {e}")

def main():
    """Main function to run the complete integrated system"""
    print("🚀 COMPLETE GAME ANALYSIS SYSTEM")
    print("=" * 50)
    
    system = CompleteGameAnalysisSystem()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            # Single run
            system.run_complete_analysis_cycle()
            
        elif command == "schedule":
            # Scheduled runs
            system.run_scheduled_analysis()
            
        elif command == "status":
            # Show system status
            status = system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif command == "weekly":
            # Generate weekly report
            system.generate_weekly_report()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: run, schedule, status, weekly")
    else:
        # Default: single run
        print("Running single analysis cycle...")
        print("Use 'python integrated_system.py schedule' for continuous monitoring")
        system.run_complete_analysis_cycle()

if __name__ == "__main__":
    main()
