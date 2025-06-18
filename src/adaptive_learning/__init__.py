"""
Adaptive Learning Module

This module provides comprehensive adaptive learning capabilities including:
- Real-time feedback processing
- Online learning algorithms
- Drift detection and adaptation
- Streaming data processing
- Preference evolution modeling
"""

from .feedback_processor import FeedbackProcessor, FeedbackType
from .online_learner import OnlineLearner, LearningAlgorithm, OnlineLearningConfig
from .drift_detector import DriftDetector, DriftDetectionMethod, DriftDetectionConfig
from .adaptation_engine import AdaptationEngine, AdaptationStrategy, AdaptationConfig

__all__ = [
    'FeedbackProcessor',
    'FeedbackType',
    'OnlineLearner',
    'LearningAlgorithm', 
    'OnlineLearningConfig',
    'DriftDetector',
    'DriftDetectionMethod',
    'DriftDetectionConfig',
    'AdaptationEngine',
    'AdaptationStrategy',
    'AdaptationConfig'
]
