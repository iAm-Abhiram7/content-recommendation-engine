"""
Main pipeline integration for Phase 3 adaptive learning components.
Orchestrates the entire adaptive learning pipeline with all modules.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import random

from .adaptive_learning import (
    FeedbackProcessor, OnlineLearner, DriftDetector, AdaptationEngine,
    FeedbackType, LearningAlgorithm, DriftDetectionMethod, AdaptationStrategy,
    OnlineLearningConfig, DriftDetectionConfig, AdaptationConfig
)
from .streaming import EventProcessor, StreamHandler, BatchStreamSynchronizer, EventProcessingConfig, StreamConfig, StreamType, SyncConfig, SyncStrategy
from .preference_modeling import AdvancedPreferenceTracker, PreferenceEvolutionModeler, ConfidenceScorer
from .explanation import AdaptationExplainer, GeminiExplainer, VisualizationGenerator, VisualizationConfig
from .user_control import AdaptationController, PreferenceManager, FeedbackCollector
from .utils.config import get_config
from .utils.logging import setup_logger

logger = setup_logger(__name__)


class AdaptiveLearningPipeline:
    """
    Main pipeline that orchestrates all Phase 3 adaptive learning components.
    Provides a unified interface for real-time learning and adaptation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adaptive learning pipeline."""
        default_config = self._get_default_config()
        if config:
            default_config.update(config)  # Merge provided config with defaults
        self.config = default_config
        self._setup_components()
        self._setup_connections()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as a dictionary."""
        default_config = {
            'feedback_buffer_size': 1000,
            'feedback_processing_interval': 60,  # Increased from 30
            'learning_rate': 0.005,  # REDUCED from 0.01
            'ensemble_size': 5,
            'drift_sensitivity': 0.001,  # REDUCED from 0.002
            'drift_window_size': 50,  # INCREASED from 30
            'drift_confidence': 0.8,  # INCREASED from 0.7
            'adaptation_rate': 0.05,  # REDUCED from 0.1
            'event_workers': 5,
            'broker_url': 'localhost:9092',
            'stream_type': 'kafka',
            'stream_batch_size': 100,
            'sync_interval': 300,
            'batch_size': 10000,
            'preference_decay': 0.98,  # INCREASED from 0.95
            'preference_threshold': 0.15,  # INCREASED from 0.1
            'trend_window': 7,
            'seasonality_periods': [7, 30],
            'min_interactions': 20,  # INCREASED from 10
            'gemini_api_key': None,
            # NEW: Quality validation settings
            'quality_validation_enabled': True,
            'min_quality_threshold': 0.35,
            'max_quality_drop': 0.05,
            'rollback_on_quality_drop': True
        }
        try:
            system_config = get_config()
            if system_config:
                default_config['gemini_api_key'] = getattr(system_config, 'gemini_api_key', None)
        except Exception as e:
            logger.warning(f"Could not load system config: {e}")

        return default_config
        
    def _setup_components(self):
        """Initialize all pipeline components."""
        try:
            # Core adaptive learning components
            self.feedback_processor = FeedbackProcessor(
                buffer_size=self.config.get('feedback_buffer_size', 1000),
                processing_interval=self.config.get('feedback_processing_interval', 30)
            )
            
            online_config = OnlineLearningConfig()
            online_config.learning_rate = self.config.get('learning_rate', 0.005)  # More conservative
            online_config.batch_size = self.config.get('ensemble_size', 5)
            online_config.regularization = 0.02  # INCREASED regularization
            online_config.learning_rate_decay = 0.98  # More conservative decay
            online_config.min_learning_rate = 0.0005  # Lower minimum
            
            self.online_learner = OnlineLearner(config=online_config)
            
            drift_config = DriftDetectionConfig(
                adwin_delta=self.config.get('drift_sensitivity', 0.001),  # Less sensitive
                min_window_size=self.config.get('drift_window_size', 50),  # Larger window
                confidence_threshold=self.config.get('drift_confidence', 0.8)  # Higher confidence
            )
            self.drift_detector = DriftDetector(config=drift_config)
            
            adaptation_config = AdaptationConfig()
            adaptation_config.gradual_learning_rate = self.config.get('adaptation_rate', 0.05)  # More conservative
            adaptation_config.rapid_learning_rate = 0.05  # Reduced
            adaptation_config.emergency_learning_rate = 0.2  # Reduced
            adaptation_config.confidence_threshold = 0.8  # Higher
            adaptation_config.severity_threshold = 0.6  # Higher
            adaptation_config.max_adaptations_per_day = 3  # Fewer adaptations
            # NEW: Quality validation settings
            adaptation_config.quality_validation_enabled = self.config.get('quality_validation_enabled', True)
            adaptation_config.min_quality_threshold = self.config.get('min_quality_threshold', 0.35)
            adaptation_config.max_quality_drop = self.config.get('max_quality_drop', 0.05)
            adaptation_config.rollback_on_quality_drop = self.config.get('rollback_on_quality_drop', True)
            
            self.adaptation_engine = AdaptationEngine(
                config=adaptation_config,
                strategy=AdaptationStrategy.GRADUAL
            )
            
            # Streaming components
            event_config = EventProcessingConfig()
            event_config.max_concurrent_processors = self.config.get('event_workers', 5)
            
            self.event_processor = EventProcessor(config=event_config)
            
            stream_config = StreamConfig(
                stream_type=StreamType.KAFKA if self.config.get('stream_type', 'kafka') == 'kafka' else StreamType.REDIS,
                connection_params={'bootstrap_servers': self.config.get('broker_url', 'localhost:9092')},
                topics=['recommendations', 'feedback', 'interactions'],
                batch_size=self.config.get('stream_batch_size', 100)
            )
            
            self.stream_handler = StreamHandler(config={'main': stream_config})
            
            sync_config = SyncConfig(
                strategy=SyncStrategy.EVENTUAL_CONSISTENCY,
                batch_interval=timedelta(seconds=self.config.get('sync_interval', 300)),
                stream_buffer_size=self.config.get('batch_size', 10000)
            )
            
            self.batch_sync = BatchStreamSynchronizer(config=sync_config)
            
            # Preference modeling
            self.preference_tracker = AdvancedPreferenceTracker()
            
            # Note: PreferenceEvolutionModeler requires user_id, so we'll create instances per user
            self.evolution_modelers = {}  # Will be populated per user
            
            # Create a default evolution modeler for general use
            from .preference_modeling import PreferenceEvolutionModeler
            self.evolution_modeler = PreferenceEvolutionModeler("default_user")
            
            self.confidence_scorer = ConfidenceScorer()
            
            # Explanation and user control
            self.adaptation_explainer = AdaptationExplainer()
            self.gemini_explainer = GeminiExplainer()
            self.visualization_generator = VisualizationGenerator()
            
            self.adaptation_controller = AdaptationController()
            self.preference_manager = PreferenceManager()
            self.feedback_collector = FeedbackCollector()
            
            # Initialize components with test data for quality monitoring
            asyncio.create_task(self._initialize_test_data())
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def _setup_connections(self):
        """Setup connections between components."""
        try:
            # Note: Callback registration temporarily disabled for components that don't support it
            # TODO: Implement register_callback methods in components if needed
            
            # Connect feedback processing to online learning
            # self.feedback_processor.register_callback(
            #     self._on_feedback_processed
            # )
            
            # Connect drift detection to adaptation
            # self.drift_detector.register_callback(
            #     self._on_drift_detected
            # )
            
            # Connect stream events to processing
            # self.stream_handler.register_callback(
            #     self._on_stream_event
            # )
            
            # Connect adaptation to explanation
            # self.adaptation_engine.register_callback(
            #     self._on_adaptation_made
            # )
            
            logger.info("Component connections established")
            
        except Exception as e:
            logger.error(f"Failed to setup component connections: {e}")
            raise
    
    async def start(self):
        """Start the adaptive learning pipeline."""
        if self._running:
            logger.warning("Pipeline is already running")
            return
        
        try:
            logger.info("Starting adaptive learning pipeline...")
            
            # Start all components
            await self.stream_handler.start()
            self.event_processor.start_processing()
            self.batch_sync.start()
            
            # Start background tasks
            asyncio.create_task(self._feedback_processing_loop())
            asyncio.create_task(self._drift_monitoring_loop())
            asyncio.create_task(self._preference_modeling_loop())
            
            self._running = True
            logger.info("Adaptive learning pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            raise
    
    async def stop(self):
        """Stop the adaptive learning pipeline."""
        if not self._running:
            return
        
        try:
            logger.info("Stopping adaptive learning pipeline...")
            
            self._running = False
            
            # Stop all components that have stop methods
            if hasattr(self.stream_handler, 'stop'):
                await self.stream_handler.stop()
            
            if hasattr(self.event_processor, 'stop'):
                try:
                    await self.event_processor.stop()
                except AttributeError:
                    # EventProcessor might not have an async stop method
                    if hasattr(self.event_processor, 'shutdown'):
                        self.event_processor.shutdown()
            
            if hasattr(self.batch_sync, 'stop'):
                try:
                    self.batch_sync.stop()
                except Exception as e:
                    logger.warning(f"Error stopping batch sync: {e}")
            
            # Stop adaptation engine if it has background processes
            if hasattr(self.adaptation_engine, 'stop_background_adaptation'):
                self.adaptation_engine.stop_background_adaptation()
            
            # Shutdown executor
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            
            logger.info("Adaptive learning pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
            # Don't re-raise to avoid blocking shutdown
    
    async def process_feedback(self, user_id: str, item_id: str, 
                             feedback_type: FeedbackType, value: float,
                             context: Optional[Dict[str, Any]] = None):
        """Process user feedback through the pipeline."""
        try:
            # Collect feedback
            feedback_data = self.feedback_collector.collect_feedback(
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                value=value,
                context=context or {}
            )
            
            # Process through adaptive learning pipeline
            processed_feedback = self.feedback_processor.process_feedback(
                feedback_data
            )
            
            # Update preference tracking
            self.preference_tracker.update_preferences(
                user_id, processed_feedback
            )
            
            # Check for drift
            drift_detected = self.drift_detector.detect_preference_drift(
                user_id, processed_feedback
            )
            
            if drift_detected:
                logger.info(f"Drift detected for user {user_id}")
                self._handle_drift(user_id, processed_feedback)
            
            # Update online learning
            self.online_learner.update(processed_feedback)
            
            return {
                'status': 'success',
                'feedback_id': feedback_data.get('feedback_id'),
                'drift_detected': drift_detected
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_recommendations(self, user_id: str, num_items: int = 10,
                                context: Optional[Dict[str, Any]] = None):
        """Get adaptive recommendations for a user."""
        try:
            # Get user preferences and confidence
            preferences = self.preference_tracker.get_user_preferences(user_id)
            confidence = self.confidence_scorer.calculate_confidence(
                user_id, preferences
            )
            
            # Check adaptation control settings
            control_settings = self.adaptation_controller.get_control_settings(
                user_id
            )
            
            # Generate recommendations using online learner
            recommendations = self.online_learner.predict(
                user_id=user_id,
                preferences=preferences,
                context=context or {},
                num_items=num_items
            )
            
            # Apply user control constraints
            if control_settings.get('enabled', True):
                recommendations = await self._apply_user_control(
                    recommendations, control_settings
                )
            
            # Generate explanations if requested
            explanations = None
            if context and context.get('include_explanations', False):
                explanations = await self._generate_explanations(
                    user_id, recommendations, preferences
                )
            
            return {
                'recommendations': recommendations,
                'confidence': confidence,
                'explanations': explanations,
                'adaptation_info': {
                    'drift_score': await self.drift_detector.get_drift_score(user_id),
                    'learning_progress': await self.online_learner.get_learning_stats(user_id),
                    'preference_stability': await self.evolution_modeler.get_stability_score(user_id)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _on_feedback_processed(self, feedback_data: Dict[str, Any]):
        """Handle processed feedback."""
        try:
            user_id = feedback_data['user_id']
            
            # Update preference evolution model
            await self.evolution_modeler.update_evolution(user_id, feedback_data)
            
            # Update confidence scoring
            await self.confidence_scorer.update_confidence(user_id, feedback_data)
            
        except Exception as e:
            logger.error(f"Error handling processed feedback: {e}")
    
    async def _on_drift_detected(self, drift_info: Dict[str, Any]):
        """Handle detected preference drift."""
        try:
            user_id = drift_info['user_id']
            drift_type = drift_info['drift_type']
            
            logger.info(f"Handling {drift_type} drift for user {user_id}")
            
            # Trigger adaptation
            adaptation_result = await self.adaptation_engine.adapt(
                user_id=user_id,
                drift_info=drift_info
            )
            
            # Update preference tracker
            await self.preference_tracker.handle_drift(user_id, drift_info)
            
            # Generate explanation for the adaptation
            explanation = await self.adaptation_explainer.explain_adaptation(
                user_id=user_id,
                adaptation_result=adaptation_result,
                drift_info=drift_info
            )
            
            # Store adaptation record
            await self.preference_manager.record_adaptation(
                user_id=user_id,
                adaptation_result=adaptation_result,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error handling drift detection: {e}")
    
    async def _on_stream_event(self, event_data: Dict[str, Any]):
        """Handle streaming events."""
        try:
            event_type = event_data.get('type')
            
            if event_type == 'user_interaction':
                # Process as implicit feedback
                await self.process_feedback(
                    user_id=event_data['user_id'],
                    item_id=event_data['item_id'],
                    feedback_type=FeedbackType.IMPLICIT,
                    value=event_data.get('value', 1.0),
                    context=event_data.get('context', {})
                )
            
            elif event_type == 'user_preference_update':
                # Update preference manager
                await self.preference_manager.update_preferences(
                    event_data['user_id'],
                    event_data['preferences']
                )
            
        except Exception as e:
            logger.error(f"Error handling stream event: {e}")
    
    async def _on_adaptation_made(self, adaptation_info: Dict[str, Any]):
        """Handle completed adaptations."""
        try:
            # Generate natural language explanation
            if self.gemini_explainer.is_available():
                nl_explanation = await self.gemini_explainer.explain_adaptation(
                    adaptation_info
                )
                adaptation_info['natural_language_explanation'] = nl_explanation
            
            # Generate visualizations
            visualization = await self.visualization_generator.generate_adaptation_viz(
                adaptation_info
            )
            adaptation_info['visualization'] = visualization
            
            logger.info(f"Adaptation completed for user {adaptation_info.get('user_id')}")
            
        except Exception as e:
            logger.error(f"Error handling adaptation completion: {e}")
    
    async def _feedback_processing_loop(self):
        """Background loop for continuous feedback processing."""
        while self._running:
            try:
                await self.feedback_processor.flush_buffer()
                await asyncio.sleep(30)  # Process every 30 seconds
            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _drift_monitoring_loop(self):
        """Background loop for continuous drift monitoring."""
        while self._running:
            try:
                await self.drift_detector.monitor_all_users()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _preference_modeling_loop(self):
        """Background loop for preference evolution modeling."""
        while self._running:
            try:
                await self.evolution_modeler.update_all_models()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Error in preference modeling loop: {e}")
                await asyncio.sleep(30)
    
    async def _handle_drift(self, user_id: str, feedback_data: Dict[str, Any]):
        """Handle detected drift for a specific user."""
        try:
            # Check user's adaptation control settings
            control_settings = await self.adaptation_controller.get_control_settings(
                user_id
            )
            
            if not control_settings.get('auto_adapt', True):
                # User has disabled auto-adaptation, queue for manual review
                await self.preference_manager.queue_adaptation_review(
                    user_id=user_id,
                    drift_info=feedback_data
                )
                return
            
            # Proceed with automatic adaptation
            adaptation_result = await self.adaptation_engine.adapt(
                user_id=user_id,
                drift_info=feedback_data
            )
            
            # Generate explanation
            explanation = await self.adaptation_explainer.explain_adaptation(
                user_id=user_id,
                adaptation_result=adaptation_result,
                drift_info=feedback_data
            )
            
            # Notify user if they have notifications enabled
            if control_settings.get('notify_adaptations', False):
                await self._notify_user_adaptation(user_id, explanation)
            
        except Exception as e:
            logger.error(f"Error handling drift for user {user_id}: {e}")
    
    async def _apply_user_control(self, recommendations: List[Dict[str, Any]], 
                                control_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply user control constraints to recommendations."""
        try:
            filtered_recommendations = []
            
            for rec in recommendations:
                # Apply category filters
                if 'blocked_categories' in control_settings:
                    if rec.get('category') in control_settings['blocked_categories']:
                        continue
                
                # Apply diversity constraints
                if control_settings.get('enforce_diversity', False):
                    # Check if recommendation adds diversity
                    if not await self._check_diversity_constraint(
                        rec, filtered_recommendations, control_settings
                    ):
                        continue
                
                # Apply novelty constraints
                if control_settings.get('novelty_preference'):
                    novelty_score = rec.get('novelty_score', 0.5)
                    if novelty_score < control_settings['novelty_preference']:
                        continue
                
                filtered_recommendations.append(rec)
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error applying user control: {e}")
            return recommendations
    
    async def _generate_explanations(self, user_id: str, 
                                   recommendations: List[Dict[str, Any]],
                                   preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for recommendations."""
        try:
            explanations = {}
            
            # Generate adaptation explanations
            explanations['adaptation'] = await self.adaptation_explainer.explain_recommendations(
                user_id=user_id,
                recommendations=recommendations,
                preferences=preferences
            )
            
            # Generate natural language explanations if available
            if self.gemini_explainer.is_available():
                explanations['natural_language'] = await self.gemini_explainer.explain_recommendations(
                    user_id=user_id,
                    recommendations=recommendations,
                    preferences=preferences
                )
            
            # Generate visualizations
            explanations['visualizations'] = await self.visualization_generator.generate_recommendation_viz(
                user_id=user_id,
                recommendations=recommendations,
                preferences=preferences
            )
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return {}
    
    async def _check_diversity_constraint(self, recommendation: Dict[str, Any],
                                        existing_recs: List[Dict[str, Any]],
                                        control_settings: Dict[str, Any]) -> bool:
        """Check if recommendation meets diversity constraints."""
        if not existing_recs:
            return True
        
        # Simple category diversity check
        recommendation_category = recommendation.get('category')
        existing_categories = [rec.get('category') for rec in existing_recs]
        
        max_same_category = control_settings.get('max_same_category', 3)
        same_category_count = existing_categories.count(recommendation_category)
        
        return same_category_count < max_same_category
    
    async def _notify_user_adaptation(self, user_id: str, explanation: Dict[str, Any]):
        """Notify user about adaptation (placeholder for notification system)."""
        logger.info(f"Adaptation notification for user {user_id}: {explanation}")
        # In a real system, this would send notifications through the appropriate channels
    
    async def _initialize_test_data(self):
        """Initialize test data for quality monitoring and demo purposes"""
        try:
            # Add test users and items to the online learner
            test_users = ['user_001', 'user_002', 'user_003', 'user_004', 'user_005']
            test_items = [f'item_{i:03d}' for i in range(50)]
            
            # Initialize matrix factorization with test data
            if hasattr(self.online_learner, 'matrix_factorization'):
                mf = self.online_learner.matrix_factorization
                
                # Add test users
                for user_id in test_users:
                    mf.add_user(user_id)
                
                # Add test items with mock features
                for item_id in test_items:
                    features = {
                        'genre': np.random.choice(['action', 'comedy', 'drama', 'thriller']),
                        'year': np.random.randint(2000, 2024),
                        'rating': np.random.uniform(3.0, 5.0)
                    }
                    mf.add_item(item_id, features)
                
                # Add some test interactions
                for user_id in test_users:
                    for _ in range(10):  # 10 interactions per user
                        item_id = random.choice(test_items)
                        rating = np.random.uniform(1.0, 5.0)
                        mf.update_single_interaction(user_id, item_id, rating)
                
                logger.info(f"Initialized test data: {len(test_users)} users, {len(test_items)} items")
            
        except Exception as e:
            logger.warning(f"Could not initialize test data: {e}")
            # Don't fail pipeline initialization if test data setup fails
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            'running': self._running,
            'components': {
                'feedback_processor': self.feedback_processor.get_status(),
                'online_learner': self.online_learner.get_status(),
                'drift_detector': self.drift_detector.get_status(),
                'adaptation_engine': self.adaptation_engine.get_status(),
                'stream_handler': self.stream_handler.get_status(),
                'preference_tracker': self.preference_tracker.get_status()
            },
            'metrics': {
                'total_feedback_processed': self.feedback_processor.get_total_processed(),
                'total_adaptations': self.adaptation_engine.get_total_adaptations(),
                'drift_detections': self.drift_detector.get_total_detections(),
                'active_users': self.preference_tracker.get_active_user_count()
            }
        }
