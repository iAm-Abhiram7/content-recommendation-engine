"""
Demo script for Phase 3 Adaptive Learning System.
Demonstrates key features including real-time adaptation, drift detection, and user control.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from src.adaptive_learning import FeedbackType
from src.pipeline_integration import AdaptiveLearningPipeline
from src.monitoring.dashboard import AdaptiveLearningMonitor
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class AdaptiveLearningDemo:
    """
    Comprehensive demo of the adaptive learning system.
    Simulates realistic user behavior, preference drift, and system adaptation.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.pipeline = None
        self.monitor = None
        self.demo_users = []
        self.demo_items = []
        self.simulation_running = False
        
    async def setup(self):
        """Setup the demo environment."""
        print("🔧 Setting up Adaptive Learning Demo...")
        
        # Initialize pipeline with demo configuration - CONSERVATIVE SETTINGS
        demo_config = {
            'feedback_buffer_size': 100,
            'learning_rate': 0.002,  # VERY conservative for demo
            'drift_sensitivity': 0.0005,  # Much less sensitive
            'adaptation_rate': 0.02,  # Very conservative adaptation
            'stream_type': 'mock',  # Use mock streaming for demo
            'quality_validation_enabled': True,
            'min_quality_threshold': 0.35,
            'max_quality_drop': 0.03,  # Allow only 3% quality drop
            'rollback_on_quality_drop': True,
            'drift_window_size': 100,  # Larger window for stability
            'drift_confidence': 0.85,  # Higher confidence required
            'max_adaptations_per_day': 2  # Very limited adaptations
        }
        
        self.pipeline = AdaptiveLearningPipeline(demo_config)
        await self.pipeline.start()
        
        # Initialize monitoring
        self.monitor = AdaptiveLearningMonitor(self.pipeline)
        
        # Generate demo data
        self._generate_demo_data()
        
        print("✅ Demo setup complete!")
        print(f"📊 Generated {len(self.demo_users)} users and {len(self.demo_items)} items")
    
    def _generate_demo_data(self):
        """Generate demo users and items."""
        # Demo users with different preference profiles
        user_profiles = [
            {'user_id': 'user_action_lover', 'preferences': {'action': 0.9, 'drama': 0.3, 'comedy': 0.2}},
            {'user_id': 'user_comedy_fan', 'preferences': {'comedy': 0.8, 'romance': 0.6, 'action': 0.1}},
            {'user_id': 'user_drama_enthusiast', 'preferences': {'drama': 0.9, 'thriller': 0.7, 'romance': 0.5}},
            {'user_id': 'user_diverse_taste', 'preferences': {'action': 0.6, 'comedy': 0.6, 'drama': 0.6}},
            {'user_id': 'user_evolving', 'preferences': {'horror': 0.2, 'comedy': 0.8, 'action': 0.3}}
        ]
        
        # Add some variation to user preferences
        for profile in user_profiles:
            profile['stability'] = random.uniform(0.1, 0.9)  # How stable their preferences are
            profile['drift_probability'] = random.uniform(0.1, 0.3)  # Probability of preference drift
        
        self.demo_users = user_profiles
        
        # Demo items with different characteristics
        genres = ['action', 'comedy', 'drama', 'romance', 'thriller', 'horror', 'sci-fi']
        
        for i in range(50):
            item = {
                'item_id': f'item_{i:03d}',
                'genre': random.choice(genres),
                'quality': random.uniform(0.3, 1.0),
                'popularity': random.uniform(0.1, 1.0),
                'novelty': random.uniform(0.0, 1.0)
            }
            self.demo_items.append(item)
    
    async def run_demo(self):
        """Run the complete adaptive learning demo."""
        print("\n🚀 Starting Adaptive Learning Demo")
        print("=" * 50)
        
        # Phase 1: Initial Learning
        await self._demo_initial_learning()
        
        # Phase 2: Real-time Adaptation
        await self._demo_real_time_adaptation()
        
        # Phase 3: Drift Detection and Adaptation
        await self._demo_drift_detection()
        
        # Phase 4: User Control
        await self._demo_user_control()
        
        # Phase 5: Explanation Generation
        await self._demo_explanations()
        
        # Phase 6: System Monitoring
        await self._demo_monitoring()
        
        print("\n✨ Demo completed successfully!")
    
    async def _demo_initial_learning(self):
        """Demonstrate initial learning from user feedback."""
        print("\n📚 Phase 1: Initial Learning")
        print("-" * 30)
        
        # Simulate initial user interactions
        for user in self.demo_users:
            user_id = user['user_id']
            preferences = user['preferences']
            
            print(f"👤 Training initial model for {user_id}")
            
            # Generate feedback based on user preferences
            for _ in range(20):  # 20 initial interactions per user
                item = random.choice(self.demo_items)
                
                # Calculate feedback based on user preferences and item characteristics
                preference_match = preferences.get(item['genre'], 0.1)
                noise = random.uniform(-0.2, 0.2)
                feedback_value = max(0.1, min(1.0, preference_match * item['quality'] + noise))
                
                # Submit feedback
                await self.pipeline.process_feedback(
                    user_id=user_id,
                    item_id=item['item_id'],
                    feedback_type=FeedbackType.EXPLICIT,
                    value=feedback_value,
                    context={'genre': item['genre'], 'session': 'initial_training'}
                )
        
        print("✅ Initial learning phase completed")
        
        # Show learned preferences
        for user in self.demo_users[:2]:  # Show first 2 users
            user_id = user['user_id']
            learned_prefs = await self.pipeline.preference_tracker.get_user_preferences(user_id)
            print(f"📈 {user_id} learned preferences: {learned_prefs}")
    
    async def _demo_real_time_adaptation(self):
        """Demonstrate real-time adaptation to new feedback."""
        print("\n⚡ Phase 2: Real-time Adaptation")
        print("-" * 30)
        
        user_id = self.demo_users[0]['user_id']
        print(f"👤 Demonstrating real-time adaptation for {user_id}")
        
        # Get initial recommendations
        initial_recs = await self.pipeline.get_recommendations(
            user_id=user_id,
            num_items=5
        )
        print(f"📋 Initial recommendations: {[r.get('item_id', 'N/A') for r in initial_recs.get('recommendations', [])]}")
        
        # Simulate real-time feedback
        print("🔄 Simulating real-time feedback...")
        for i in range(10):
            item = random.choice(self.demo_items)
            feedback_value = random.uniform(0.1, 1.0)
            
            result = await self.pipeline.process_feedback(
                user_id=user_id,
                item_id=item['item_id'],
                feedback_type=FeedbackType.IMPLICIT,
                value=feedback_value,
                context={'timestamp': datetime.utcnow().isoformat()}
            )
            
            if i % 3 == 0:  # Show adaptation every 3 interactions
                current_recs = await self.pipeline.get_recommendations(
                    user_id=user_id,
                    num_items=5
                )
                print(f"📋 Updated recommendations: {[r.get('item_id', 'N/A') for r in current_recs.get('recommendations', [])]}")
            
            await asyncio.sleep(0.1)  # Small delay for realism
        
        print("✅ Real-time adaptation demo completed")
    
    async def _demo_drift_detection(self):
        """Demonstrate preference drift detection and adaptation."""
        print("\n🌊 Phase 3: Drift Detection and Adaptation")
        print("-" * 30)
        
        # Choose a user for drift simulation
        drift_user = self.demo_users[4]  # user_evolving
        user_id = drift_user['user_id']
        
        print(f"👤 Simulating preference drift for {user_id}")
        print("📊 Original preferences: Horror: 0.2, Comedy: 0.8, Action: 0.3")
        
        # Phase 1: Stable behavior
        print("⏳ Phase 1: Stable behavior...")
        for _ in range(15):
            # Generate feedback consistent with original preferences
            if random.random() < 0.7:  # 70% comedy preference
                genre = 'comedy'
                feedback_value = random.uniform(0.6, 1.0)
            else:
                genre = random.choice(['horror', 'action'])
                feedback_value = random.uniform(0.1, 0.5)
            
            item = next((item for item in self.demo_items if item['genre'] == genre), random.choice(self.demo_items))
            
            await self.pipeline.process_feedback(
                user_id=user_id,
                item_id=item['item_id'],
                feedback_type=FeedbackType.EXPLICIT,
                value=feedback_value
            )
        
        # Phase 2: Preference drift simulation
        print("🔄 Phase 2: Simulating preference shift (Comedy → Action)...")
        for i in range(20):
            # Gradually shift from comedy to action preference
            drift_progress = i / 20
            
            if random.random() < (0.7 - drift_progress * 0.5):  # Decreasing comedy preference
                genre = 'comedy'
                feedback_value = random.uniform(0.3 - drift_progress * 0.2, 0.8 - drift_progress * 0.3)
            else:  # Increasing action preference
                genre = 'action'
                feedback_value = random.uniform(0.3 + drift_progress * 0.4, 0.9)
            
            item = next((item for item in self.demo_items if item['genre'] == genre), random.choice(self.demo_items))
            
            result = await self.pipeline.process_feedback(
                user_id=user_id,
                item_id=item['item_id'],
                feedback_type=FeedbackType.EXPLICIT,
                value=feedback_value,
                context={'drift_simulation': True, 'progress': drift_progress}
            )
            
            if result.get('drift_detected'):
                print(f"🚨 Drift detected at step {i+1}!")
                break
        
        # Show drift analysis
        try:
            drift_analysis = await self.pipeline.drift_detector.analyze_user_drift(user_id)
            print(f"📈 Drift analysis: {drift_analysis}")
        except:
            print("📈 Drift analysis not available in demo mode")
        
        print("✅ Drift detection demo completed")
    
    async def _demo_user_control(self):
        """Demonstrate user control features."""
        print("\n🎛️ Phase 4: User Control")
        print("-" * 30)
        
        user_id = self.demo_users[1]['user_id']  # comedy fan
        print(f"👤 Demonstrating user control for {user_id}")
        
        # Show default control settings
        control_settings = await self.pipeline.adaptation_controller.get_control_settings(user_id)
        print(f"⚙️ Default control settings: {control_settings}")
        
        # Update user control settings
        from src.user_control import ControlLevel
        
        await self.pipeline.adaptation_controller.update_control_settings(
            user_id=user_id,
            control_level=ControlLevel.FULL,
            settings={
                'auto_adapt': False,
                'notify_adaptations': True,
                'blocked_categories': ['horror', 'thriller'],
                'novelty_preference': 0.3,
                'enforce_diversity': True
            }
        )
        
        print("✅ Updated control settings:")
        print("   - Auto-adaptation: DISABLED")
        print("   - Blocked categories: Horror, Thriller")
        print("   - Novelty preference: 30%")
        print("   - Diversity enforcement: ENABLED")
        
        # Get recommendations with user control applied
        controlled_recs = await self.pipeline.get_recommendations(
            user_id=user_id,
            num_items=8,
            context={'apply_user_control': True}
        )
        
        print(f"📋 Controlled recommendations: {len(controlled_recs.get('recommendations', []))} items")
        
        # Manual preference update
        await self.pipeline.preference_manager.update_preferences(
            user_id=user_id,
            preferences={'comedy': 0.9, 'romance': 0.7, 'action': 0.2},
            merge_strategy='update'
        )
        
        print("✅ User control demo completed")
    
    async def _demo_explanations(self):
        """Demonstrate explanation generation."""
        print("\n💡 Phase 5: Explanation Generation")
        print("-" * 30)
        
        user_id = self.demo_users[0]['user_id']
        
        # Get recommendations with explanations
        recs_with_explanations = await self.pipeline.get_recommendations(
            user_id=user_id,
            num_items=3,
            context={'include_explanations': True}
        )
        
        explanations = recs_with_explanations.get('explanations', {})
        
        print("📝 Generated explanations:")
        if 'adaptation' in explanations:
            print(f"   Adaptation: {explanations['adaptation']}")
        
        if 'natural_language' in explanations:
            print(f"   Natural Language: {explanations['natural_language']}")
        else:
            print("   Natural Language: Gemini explanations not available in demo")
        
        # Simulate adaptation explanation
        mock_adaptation = {
            'user_id': user_id,
            'adaptation_type': 'gradual',
            'trigger': 'preference_drift',
            'changes': {'action_weight': 0.15, 'comedy_weight': -0.1}
        }
        
        adaptation_explanation = await self.pipeline.adaptation_explainer.explain_adaptation(
            user_id=user_id,
            adaptation_result=mock_adaptation
        )
        
        print(f"🔧 Adaptation explanation: {adaptation_explanation}")
        print("✅ Explanation demo completed")
    
    async def _demo_monitoring(self):
        """Demonstrate system monitoring."""
        print("\n📊 Phase 6: System Monitoring")
        print("-" * 30)
        
        # Start monitoring briefly
        monitoring_task = asyncio.create_task(
            self.monitor.start_monitoring(interval=5)
        )
        
        # Let it collect some data
        await asyncio.sleep(10)
        
        # Get dashboard data
        dashboard_data = self.monitor.get_dashboard_data()
        
        print("📈 System Metrics:")
        summary = dashboard_data.get('summary', {})
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Get alerts summary
        alerts_summary = self.monitor.get_alerts_summary()
        print(f"\n🚨 Alerts: {alerts_summary.get('total', 0)} total")
        
        # System status
        status = self.pipeline.get_pipeline_status()
        print(f"\n⚡ Pipeline Status: {'✅ Running' if status.get('running') else '❌ Stopped'}")
        print(f"📊 Components: {len(status.get('components', {}))}")
        print(f"📈 Metrics: {status.get('metrics', {})}")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        monitoring_task.cancel()
        
        print("✅ Monitoring demo completed")
    
    async def simulate_realistic_usage(self, duration_minutes: int = 5):
        """Simulate realistic system usage for demo purposes."""
        print(f"\n🔄 Simulating realistic usage for {duration_minutes} minutes...")
        
        self.simulation_running = True
        start_time = time.time()
        
        async def user_simulation(user):
            """Simulate individual user behavior."""
            user_id = user['user_id']
            preferences = user['preferences']
            
            while self.simulation_running:
                try:
                    # Generate realistic feedback
                    item = random.choice(self.demo_items)
                    
                    # Calculate feedback based on preferences with some randomness
                    base_preference = preferences.get(item['genre'], 0.1)
                    quality_factor = item['quality']
                    noise = random.uniform(-0.3, 0.3)
                    
                    feedback_value = max(0.1, min(1.0, base_preference * quality_factor + noise))
                    
                    # Choose feedback type
                    feedback_type = random.choice([
                        FeedbackType.EXPLICIT,
                        FeedbackType.IMPLICIT,
                        FeedbackType.CONTEXTUAL
                    ])
                    
                    # Submit feedback
                    await self.pipeline.process_feedback(
                        user_id=user_id,
                        item_id=item['item_id'],
                        feedback_type=feedback_type,
                        value=feedback_value,
                        context={
                            'simulation': True,
                            'genre': item['genre'],
                            'quality': item['quality']
                        }
                    )
                    
                    # Occasionally get recommendations
                    if random.random() < 0.3:  # 30% chance
                        await self.pipeline.get_recommendations(user_id=user_id, num_items=5)
                    
                    # Wait before next interaction
                    await asyncio.sleep(random.uniform(1, 5))
                    
                except Exception as e:
                    logger.error(f"Error in user simulation: {e}")
                    await asyncio.sleep(1)
        
        # Start simulation for all users
        tasks = [asyncio.create_task(user_simulation(user)) for user in self.demo_users]
        
        # Run for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop simulation
        self.simulation_running = False
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        print("✅ Realistic usage simulation completed")
    
    async def cleanup(self):
        """Clean up demo resources."""
        print("\n🧹 Cleaning up demo...")
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.pipeline:
            await self.pipeline.stop()
        
        print("✅ Demo cleanup completed")


async def main():
    """Run the adaptive learning demo."""
    demo = AdaptiveLearningDemo()
    
    try:
        await demo.setup()
        await demo.run_demo()
        
        # Optional: Run realistic simulation
        print("\n❓ Would you like to run a realistic usage simulation?")
        print("   This will show the system handling continuous user interactions...")
        
        # For demo purposes, we'll run a short simulation
        await demo.simulate_realistic_usage(duration_minutes=1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.error(f"Demo error: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("🎬 Adaptive Learning System Demo")
    print("================================")
    print("This demo showcases the Phase 3 adaptive learning capabilities:")
    print("• Real-time feedback processing")
    print("• Online learning and model adaptation")
    print("• Preference drift detection")
    print("• User control and customization")
    print("• Explainable adaptations")
    print("• System monitoring and analytics")
    print("\nStarting demo...")
    
    asyncio.run(main())
