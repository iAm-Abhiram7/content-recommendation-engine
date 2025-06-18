"""
Comprehensive tests for Phase 3 adaptive learning components.
Tests online learning, drift detection, preference modeling, and user control.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.adaptive_learning import (
    FeedbackProcessor, OnlineLearner, DriftDetector, AdaptationEngine,
    FeedbackType, LearningAlgorithm, DriftDetectionMethod, AdaptationStrategy
)
from src.streaming import EventProcessor, StreamHandler, BatchStreamSynchronizer
from src.preference_modeling import PreferenceTracker, EvolutionModeler, ConfidenceScorer
from src.explanation import AdaptationExplainer, GeminiExplainer, VisualizationGenerator
from src.user_control import AdaptationController, PreferenceManager, FeedbackCollector
from src.pipeline_integration import AdaptiveLearningPipeline


class TestFeedbackProcessor:
    """Test feedback processing functionality."""
    
    @pytest.fixture
    def feedback_processor(self):
        return FeedbackProcessor(buffer_size=100, flush_interval=5)
    
    @pytest.mark.asyncio
    async def test_process_explicit_feedback(self, feedback_processor):
        """Test processing explicit feedback."""
        feedback_data = {
            'user_id': 'user123',
            'item_id': 'item456',
            'feedback_type': FeedbackType.EXPLICIT,
            'value': 4.5,
            'timestamp': datetime.utcnow(),
            'context': {'session_id': 'session789'}
        }
        
        result = await feedback_processor.process_feedback(feedback_data)
        
        assert result is not None
        assert result['user_id'] == 'user123'
        assert result['normalized_value'] is not None
        assert 'processing_time' in result
    
    @pytest.mark.asyncio
    async def test_process_implicit_feedback(self, feedback_processor):
        """Test processing implicit feedback."""
        feedback_data = {
            'user_id': 'user123',
            'item_id': 'item456',
            'feedback_type': FeedbackType.IMPLICIT,
            'value': 1.0,  # Click event
            'timestamp': datetime.utcnow(),
            'context': {'action': 'click', 'duration': 30}
        }
        
        result = await feedback_processor.process_feedback(feedback_data)
        
        assert result is not None
        assert result['confidence_score'] is not None
        assert result['inferred_preference'] is not None
    
    @pytest.mark.asyncio
    async def test_feedback_buffering(self, feedback_processor):
        """Test feedback buffering mechanism."""
        # Add multiple feedback items
        for i in range(5):
            feedback_data = {
                'user_id': f'user{i}',
                'item_id': f'item{i}',
                'feedback_type': FeedbackType.EXPLICIT,
                'value': float(i),
                'timestamp': datetime.utcnow()
            }
            await feedback_processor.process_feedback(feedback_data)
        
        # Check buffer size
        buffer_size = feedback_processor.get_buffer_size()
        assert buffer_size == 5
        
        # Flush buffer
        await feedback_processor.flush_buffer()
        buffer_size_after_flush = feedback_processor.get_buffer_size()
        assert buffer_size_after_flush == 0


class TestOnlineLearner:
    """Test online learning functionality."""
    
    @pytest.fixture
    def online_learner(self):
        return OnlineLearner(
            algorithm=LearningAlgorithm.INCREMENTAL,
            learning_rate=0.01
        )
    
    @pytest.mark.asyncio
    async def test_incremental_learning(self, online_learner):
        """Test incremental learning updates."""
        # Initial training data
        training_data = {
            'user_id': 'user123',
            'features': np.random.rand(10),
            'target': 1.0,
            'context': {}
        }
        
        # Update model
        result = await online_learner.update(training_data)
        
        assert result['status'] == 'success'
        assert 'model_version' in result
        assert result['learning_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_ensemble_learning(self):
        """Test ensemble learning approach."""
        online_learner = OnlineLearner(
            algorithm=LearningAlgorithm.ENSEMBLE,
            ensemble_size=3
        )
        
        # Train multiple models
        for i in range(10):
            training_data = {
                'user_id': 'user123',
                'features': np.random.rand(5),
                'target': float(i % 2),
                'context': {}
            }
            await online_learner.update(training_data)
        
        # Get prediction
        prediction = await online_learner.predict(
            user_id='user123',
            features=np.random.rand(5)
        )
        
        assert prediction is not None
        assert 'confidence' in prediction
        assert 'ensemble_agreement' in prediction
    
    @pytest.mark.asyncio
    async def test_bandit_learning(self):
        """Test multi-armed bandit learning."""
        online_learner = OnlineLearner(
            algorithm=LearningAlgorithm.BANDIT,
            num_arms=5
        )
        
        # Simulate bandit feedback
        for i in range(20):
            feedback_data = {
                'user_id': 'user123',
                'arm': i % 5,
                'reward': np.random.random(),
                'context': {}
            }
            await online_learner.update(feedback_data)
        
        # Get arm recommendation
        recommendation = await online_learner.recommend_arm('user123')
        
        assert recommendation is not None
        assert 'arm' in recommendation
        assert 'confidence' in recommendation


class TestDriftDetector:
    """Test drift detection functionality."""
    
    @pytest.fixture
    def drift_detector(self):
        return DriftDetector(
            method=DriftDetectionMethod.ADWIN,
            sensitivity=0.8,
            window_size=100
        )
    
    @pytest.mark.asyncio
    async def test_adwin_drift_detection(self, drift_detector):
        """Test ADWIN drift detection."""
        user_id = 'user123'
        
        # Simulate stable data
        for i in range(50):
            data_point = {
                'user_id': user_id,
                'value': np.random.normal(0.5, 0.1),
                'timestamp': datetime.utcnow()
            }
            drift_result = await drift_detector.detect_drift(user_id, data_point)
            assert not drift_result  # No drift expected
        
        # Simulate drift
        for i in range(30):
            data_point = {
                'user_id': user_id,
                'value': np.random.normal(0.8, 0.1),  # Distribution shift
                'timestamp': datetime.utcnow()
            }
            drift_result = await drift_detector.detect_drift(user_id, data_point)
        
        # Should detect drift eventually
        drift_score = await drift_detector.get_drift_score(user_id)
        assert drift_score > 0
    
    @pytest.mark.asyncio
    async def test_page_hinkley_detection(self):
        """Test Page-Hinkley drift detection."""
        drift_detector = DriftDetector(
            method=DriftDetectionMethod.PAGE_HINKLEY,
            sensitivity=0.05
        )
        
        user_id = 'user456'
        drift_detected = False
        
        # Generate data with gradual drift
        for i in range(100):
            value = 0.5 + (i / 100) * 0.3  # Gradual increase
            data_point = {
                'user_id': user_id,
                'value': value + np.random.normal(0, 0.05),
                'timestamp': datetime.utcnow()
            }
            
            result = await drift_detector.detect_drift(user_id, data_point)
            if result:
                drift_detected = True
                break
        
        assert drift_detected
    
    @pytest.mark.asyncio
    async def test_ensemble_drift_detection(self):
        """Test ensemble drift detection."""
        drift_detector = DriftDetector(
            method=DriftDetectionMethod.ENSEMBLE,
            sensitivity=0.7
        )
        
        user_id = 'user789'
        
        # Simulate concept drift
        for phase in range(3):
            mean = 0.3 + phase * 0.3
            for i in range(30):
                data_point = {
                    'user_id': user_id,
                    'value': np.random.normal(mean, 0.1),
                    'timestamp': datetime.utcnow()
                }
                await drift_detector.detect_drift(user_id, data_point)
        
        # Check drift statistics
        stats = await drift_detector.get_drift_statistics(user_id)
        assert 'total_drifts' in stats
        assert 'drift_rate' in stats


class TestAdaptationEngine:
    """Test adaptation engine functionality."""
    
    @pytest.fixture
    def adaptation_engine(self):
        return AdaptationEngine(
            strategy=AdaptationStrategy.GRADUAL,
            adaptation_rate=0.1
        )
    
    @pytest.mark.asyncio
    async def test_gradual_adaptation(self, adaptation_engine):
        """Test gradual adaptation strategy."""
        user_id = 'user123'
        drift_info = {
            'user_id': user_id,
            'drift_type': 'concept_drift',
            'magnitude': 0.5,
            'confidence': 0.8
        }
        
        result = await adaptation_engine.adapt(user_id, drift_info)
        
        assert result['status'] == 'success'
        assert result['adaptation_type'] == 'gradual'
        assert result['user_id'] == user_id
        assert 'adaptation_steps' in result
    
    @pytest.mark.asyncio
    async def test_rapid_adaptation(self):
        """Test rapid adaptation strategy."""
        adaptation_engine = AdaptationEngine(
            strategy=AdaptationStrategy.RAPID,
            adaptation_rate=0.5
        )
        
        user_id = 'user456'
        drift_info = {
            'user_id': user_id,
            'drift_type': 'sudden_drift',
            'magnitude': 0.9,
            'confidence': 0.95
        }
        
        result = await adaptation_engine.adapt(user_id, drift_info)
        
        assert result['adaptation_type'] == 'rapid'
        assert result['adaptation_magnitude'] > 0.3
    
    @pytest.mark.asyncio
    async def test_rollback_adaptation(self):
        """Test rollback adaptation strategy."""
        adaptation_engine = AdaptationEngine(
            strategy=AdaptationStrategy.ROLLBACK
        )
        
        user_id = 'user789'
        drift_info = {
            'user_id': user_id,
            'drift_type': 'false_positive',
            'rollback_to': 'previous_stable'
        }
        
        result = await adaptation_engine.adapt(user_id, drift_info)
        
        assert result['adaptation_type'] == 'rollback'
        assert 'rollback_target' in result


class TestPreferenceModeling:
    """Test preference modeling components."""
    
    @pytest.fixture
    def preference_tracker(self):
        return PreferenceTracker(time_decay=0.95)
    
    @pytest.mark.asyncio
    async def test_preference_tracking(self, preference_tracker):
        """Test basic preference tracking."""
        user_id = 'user123'
        feedback_data = {
            'user_id': user_id,
            'item_features': {'genre': 'action', 'rating': 4.5},
            'feedback_value': 1.0,
            'timestamp': datetime.utcnow()
        }
        
        await preference_tracker.update_preferences(user_id, feedback_data)
        preferences = await preference_tracker.get_user_preferences(user_id)
        
        assert preferences is not None
        assert user_id in preferences or len(preferences) > 0
    
    @pytest.mark.asyncio
    async def test_evolution_modeling(self):
        """Test preference evolution modeling."""
        evolution_modeler = EvolutionModeler(trend_window=7)
        user_id = 'user123'
        
        # Simulate preference evolution over time
        base_time = datetime.utcnow()
        for i in range(14):  # 2 weeks of data
            preference_data = {
                'user_id': user_id,
                'preferences': {'action': 0.5 + i * 0.02, 'comedy': 0.3},
                'timestamp': base_time + timedelta(days=i)
            }
            await evolution_modeler.update_evolution(user_id, preference_data)
        
        # Get evolution trends
        trends = await evolution_modeler.get_preference_trends(user_id)
        
        assert trends is not None
        assert 'trend_direction' in trends
        assert 'seasonality_detected' in trends
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence scoring."""
        confidence_scorer = ConfidenceScorer(min_interactions=5)
        user_id = 'user123'
        
        preferences = {
            'action': 0.8,
            'comedy': 0.3,
            'drama': 0.6
        }
        
        confidence = await confidence_scorer.calculate_confidence(user_id, preferences)
        
        assert confidence is not None
        assert 'overall_confidence' in confidence
        assert 'category_confidence' in confidence


class TestExplanationSystem:
    """Test explanation generation."""
    
    @pytest.fixture
    def adaptation_explainer(self):
        return AdaptationExplainer()
    
    @pytest.mark.asyncio
    async def test_adaptation_explanation(self, adaptation_explainer):
        """Test adaptation explanation generation."""
        adaptation_result = {
            'user_id': 'user123',
            'adaptation_type': 'gradual',
            'trigger': 'preference_drift',
            'changes': {'action_weight': 0.2, 'comedy_weight': -0.1}
        }
        
        explanation = await adaptation_explainer.explain_adaptation(
            user_id='user123',
            adaptation_result=adaptation_result
        )
        
        assert explanation is not None
        assert 'summary' in explanation
        assert 'detailed_explanation' in explanation
        assert 'user_impact' in explanation
    
    @pytest.mark.asyncio
    async def test_gemini_explanation(self):
        """Test Gemini-powered explanation generation."""
        with patch('src.explanation.gemini_explainer.GeminiExplainer.is_available', return_value=True):
            gemini_explainer = GeminiExplainer(api_key='test_key')
            
            with patch.object(gemini_explainer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
                mock_api.return_value = "Your preferences have shifted towards action movies."
                
                explanation = await gemini_explainer.explain_adaptation({
                    'user_id': 'user123',
                    'adaptation_type': 'gradual'
                })
                
                assert explanation is not None
                assert isinstance(explanation, str)
    
    @pytest.mark.asyncio
    async def test_visualization_generation(self):
        """Test visualization generation."""
        viz_generator = VisualizationGenerator()
        
        preferences = {
            'action': 0.8,
            'comedy': 0.3,
            'drama': 0.6
        }
        
        visualization = await viz_generator.generate_preference_viz(
            user_id='user123',
            preferences=preferences
        )
        
        assert visualization is not None
        assert 'preference_chart' in visualization


class TestUserControl:
    """Test user control functionality."""
    
    @pytest.fixture
    def adaptation_controller(self):
        return AdaptationController()
    
    @pytest.mark.asyncio
    async def test_control_settings_update(self, adaptation_controller):
        """Test updating user control settings."""
        from src.user_control import ControlLevel
        
        user_id = 'user123'
        settings = {
            'auto_adapt': False,
            'notify_adaptations': True,
            'blocked_categories': ['horror']
        }
        
        await adaptation_controller.update_control_settings(
            user_id=user_id,
            control_level=ControlLevel.MODERATE,
            settings=settings
        )
        
        retrieved_settings = await adaptation_controller.get_control_settings(user_id)
        
        assert retrieved_settings['auto_adapt'] == False
        assert retrieved_settings['notify_adaptations'] == True
        assert 'horror' in retrieved_settings['blocked_categories']
    
    @pytest.mark.asyncio
    async def test_preference_management(self):
        """Test preference management."""
        preference_manager = PreferenceManager()
        user_id = 'user123'
        
        preferences = {
            'action': 0.8,
            'comedy': 0.5,
            'drama': 0.3
        }
        
        await preference_manager.update_preferences(
            user_id=user_id,
            preferences=preferences,
            merge_strategy='update'
        )
        
        retrieved_prefs = await preference_manager.get_user_preferences(user_id)
        
        assert retrieved_prefs['action'] == 0.8
        assert retrieved_prefs['comedy'] == 0.5
    
    @pytest.mark.asyncio
    async def test_feedback_collection(self):
        """Test feedback collection."""
        feedback_collector = FeedbackCollector()
        
        feedback_data = await feedback_collector.collect_feedback(
            user_id='user123',
            item_id='item456',
            feedback_type=FeedbackType.EXPLICIT,
            value=4.5,
            context={'session_id': 'session789'}
        )
        
        assert feedback_data['user_id'] == 'user123'
        assert feedback_data['item_id'] == 'item456'
        assert feedback_data['feedback_type'] == FeedbackType.EXPLICIT
        assert 'feedback_id' in feedback_data


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    @pytest.fixture
    async def pipeline(self):
        """Create a test pipeline instance."""
        config = {
            'feedback_buffer_size': 100,
            'learning_rate': 0.01,
            'drift_sensitivity': 0.8
        }
        
        pipeline = AdaptiveLearningPipeline(config)
        yield pipeline
        
        # Cleanup
        if pipeline._running:
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_startup_shutdown(self, pipeline):
        """Test pipeline startup and shutdown."""
        # Start pipeline
        await pipeline.start()
        assert pipeline._running == True
        
        # Stop pipeline
        await pipeline.stop()
        assert pipeline._running == False
    
    @pytest.mark.asyncio
    async def test_end_to_end_feedback_processing(self, pipeline):
        """Test end-to-end feedback processing."""
        await pipeline.start()
        
        # Submit feedback
        result = await pipeline.process_feedback(
            user_id='user123',
            item_id='item456',
            feedback_type=FeedbackType.EXPLICIT,
            value=4.5,
            context={'session_id': 'session789'}
        )
        
        assert result['status'] == 'success'
        assert 'feedback_id' in result
        
        await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_adaptive_recommendations(self, pipeline):
        """Test adaptive recommendation generation."""
        await pipeline.start()
        
        # First, add some feedback to build preferences
        await pipeline.process_feedback(
            user_id='user123',
            item_id='item456',
            feedback_type=FeedbackType.EXPLICIT,
            value=4.5
        )
        
        # Get recommendations
        recommendations = await pipeline.get_recommendations(
            user_id='user123',
            num_items=5,
            context={'include_explanations': True}
        )
        
        assert 'recommendations' in recommendations
        assert 'confidence' in recommendations
        assert 'adaptation_info' in recommendations
        
        await pipeline.stop()


class TestStreamingComponents:
    """Test streaming components."""
    
    @pytest.mark.asyncio
    async def test_event_processing(self):
        """Test event processing."""
        event_processor = EventProcessor(max_workers=2)
        await event_processor.start()
        
        # Mock event handler
        processed_events = []
        
        async def mock_handler(event_data):
            processed_events.append(event_data)
        
        event_processor.register_handler('test_event', mock_handler)
        
        # Process events
        test_events = [
            {'type': 'test_event', 'data': f'event_{i}'}
            for i in range(5)
        ]
        
        for event in test_events:
            await event_processor.process_event(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        assert len(processed_events) == 5
        
        await event_processor.stop()
    
    @pytest.mark.asyncio
    async def test_batch_stream_sync(self):
        """Test batch-stream synchronization."""
        sync = BatchStreamSynchronizer(sync_interval=1, batch_size=10)
        sync.start()
        
        # Add some stream data
        for i in range(15):
            await sync.add_stream_data({
                'user_id': f'user{i}',
                'action': 'view',
                'timestamp': datetime.utcnow()
            })
        
        # Wait for sync
        await asyncio.sleep(1.5)
        
        # Check sync status
        status = sync.get_sync_status()
        assert 'last_sync' in status
        assert 'processed_items' in status
        
        sync.stop()


# Performance tests
class TestPerformance:
    """Test system performance under load."""
    
    @pytest.mark.asyncio
    async def test_feedback_processing_throughput(self):
        """Test feedback processing throughput."""
        feedback_processor = FeedbackProcessor(buffer_size=1000)
        
        start_time = datetime.utcnow()
        
        # Process 1000 feedback items
        tasks = []
        for i in range(1000):
            feedback_data = {
                'user_id': f'user{i % 100}',
                'item_id': f'item{i}',
                'feedback_type': FeedbackType.IMPLICIT,
                'value': 1.0,
                'timestamp': datetime.utcnow()
            }
            tasks.append(feedback_processor.process_feedback(feedback_data))
        
        await asyncio.gather(*tasks)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process 1000 items in reasonable time
        assert processing_time < 10.0  # Less than 10 seconds
        
        throughput = 1000 / processing_time
        print(f"Feedback processing throughput: {throughput:.2f} items/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_drift_detection(self):
        """Test concurrent drift detection for multiple users."""
        drift_detector = DriftDetector(method=DriftDetectionMethod.ADWIN)
        
        # Simulate concurrent drift detection for 50 users
        tasks = []
        for user_id in [f'user{i}' for i in range(50)]:
            for j in range(20):  # 20 data points per user
                data_point = {
                    'user_id': user_id,
                    'value': np.random.random(),
                    'timestamp': datetime.utcnow()
                }
                tasks.append(drift_detector.detect_drift(user_id, data_point))
        
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should handle concurrent requests efficiently
        assert processing_time < 30.0  # Less than 30 seconds for 1000 detections
        
        print(f"Drift detection processing time: {processing_time:.2f} seconds for {len(results)} detections")


# Integration test fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
