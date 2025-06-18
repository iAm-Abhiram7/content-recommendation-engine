"""
Explanation Module

This module provides explainable AI capabilities for recommendation adaptations:
- Adaptation explanation generation
- Gemini-powered natural language explanations
- Visualization generation for user interfaces
- Multi-modal explanation delivery
"""

from .adaptation_explainer import AdaptationExplainer, ExplanationType, ExplanationLevel
from .gemini_explainer import GeminiExplainer, ExplanationStyle
from .visualization_generator import VisualizationGenerator, VisualizationType, VisualizationConfig

__all__ = [
    'AdaptationExplainer',
    'ExplanationType',
    'ExplanationLevel',
    'GeminiExplainer',
    'ExplanationStyle',
    'VisualizationGenerator',
    'VisualizationType',
    'VisualizationConfig'
]
