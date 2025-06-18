"""
User Profiling Module

This module provides comprehensive user profiling capabilities including:
- Preference tracking and evolution
- Behavior analysis and pattern detection
- Profile evolution and lifecycle management
"""

from .preference_tracker import PreferenceTracker
from .behavior_analyzer import BehaviorAnalyzer
from .profile_evolution import ProfileEvolution

__all__ = [
    'PreferenceTracker',
    'BehaviorAnalyzer', 
    'ProfileEvolution'
]
