"""
User Control Module

This module provides user control capabilities for recommendation adaptations:
- Adaptation control and customization
- User preference management
- Feedback collection and processing
- Transparency and control interfaces
"""

from .adaptation_controller import AdaptationController, ControlLevel, AdaptationPolicy
from .preference_manager import PreferenceManager, PreferenceCategory, PreferenceWeight
from .feedback_collector import FeedbackCollector, FeedbackValence, FeedbackResponse
from ..adaptive_learning.feedback_processor import FeedbackType

__all__ = [
    'AdaptationController',
    'ControlLevel',
    'AdaptationPolicy',
    'PreferenceManager',
    'PreferenceCategory',
    'PreferenceWeight',
    'FeedbackCollector',
    'FeedbackType',
    'FeedbackValence',
    'FeedbackResponse'
]
