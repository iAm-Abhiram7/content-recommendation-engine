"""
Recommendation Engine Modules

This package contains the core recommendation algorithms:
- Collaborative filtering (user-based and item-based)
- Content-based filtering
- Knowledge-based filtering
- Hybrid ensemble methods
- Group recommendations
"""

from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .knowledge_based import KnowledgeBasedRecommender
from .hybrid import HybridRecommender
from .group_recommender import GroupRecommender

__all__ = [
    'CollaborativeRecommender',
    'ContentBasedRecommender', 
    'KnowledgeBasedRecommender',
    'HybridRecommender',
    'GroupRecommender'
]
