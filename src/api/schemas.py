"""
API Schemas and Data Models

Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class RecommendationMethod(str, Enum):
    """Available recommendation methods"""
    COLLABORATIVE = "collaborative"
    CONTENT = "content"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"
    GROUP = "group"


class AggregationStrategy(str, Enum):
    """Group recommendation aggregation strategies"""
    AVERAGE = "average"
    LEAST_MISERY = "least_misery"
    MOST_PLEASURE = "most_pleasure"
    FAIRNESS = "fairness"
    APPROVAL_VOTING = "approval_voting"
    BORDA_COUNT = "borda_count"


class InteractionType(str, Enum):
    """Types of user interactions"""
    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    SHARE = "share"
    RATING = "rating"
    PURCHASE = "purchase"


# Request Models
class RecommendationRequest(BaseModel):
    """Standard recommendation request"""
    user_id: str = Field(..., description="User identifier")
    n_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    method: RecommendationMethod = Field(RecommendationMethod.HYBRID, description="Recommendation method")
    filters: Optional[Dict[str, Any]] = Field(None, description="Content filters")
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information")
    exclude_seen: bool = Field(True, description="Exclude previously seen items")


class GroupRecommendationRequest(BaseModel):
    """Group recommendation request"""
    user_ids: List[str] = Field(..., min_items=1, description="List of user IDs in group")
    n_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    aggregation_strategy: AggregationStrategy = Field(AggregationStrategy.AVERAGE, description="Aggregation strategy")
    user_weights: Optional[Dict[str, float]] = Field(None, description="Weights for each user")
    filters: Optional[Dict[str, Any]] = Field(None, description="Content filters")
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information")


class FeedbackRequest(BaseModel):
    """User feedback request"""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating score (1-5)")
    timestamp: Optional[datetime] = Field(None, description="Interaction timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information")


class NextItemPredictionRequest(BaseModel):
    """Next item prediction request"""
    user_id: str = Field(..., description="User identifier")
    current_session: List[str] = Field(..., description="Current session items")
    n_predictions: int = Field(5, ge=1, le=20, description="Number of predictions")


class SimilarItemsRequest(BaseModel):
    """Similar items request"""
    item_id: str = Field(..., description="Item identifier")
    n_similar: int = Field(10, ge=1, le=50, description="Number of similar items")
    use_metadata: bool = Field(False, description="Include metadata features")
    metadata_weight: float = Field(0.3, ge=0, le=1, description="Weight for metadata features")


class CrossDomainRequest(BaseModel):
    """Cross-domain recommendation request"""
    source_items: List[str] = Field(..., min_items=1, description="Source domain items")
    target_domain: str = Field(..., description="Target domain")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")


# Response Models
class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    item_id: str = Field(..., description="Item identifier")
    score: float = Field(..., description="Recommendation score")
    rank: int = Field(..., description="Recommendation rank")
    method: str = Field(..., description="Recommendation method used")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Explanation for recommendation")
    components: Optional[Dict[str, float]] = Field(None, description="Component scores")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Item metadata")


class RecommendationResponse(BaseModel):
    """Standard recommendation response"""
    user_id: str = Field(..., description="User identifier")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    method: str = Field(..., description="Recommendation method used")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    generated_at: datetime = Field(..., description="Generation timestamp")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")


class GroupRecommendationResponse(BaseModel):
    """Group recommendation response"""
    user_ids: List[str] = Field(..., description="Group user IDs")
    recommendations: List[RecommendationItem] = Field(..., description="Group recommendations")
    aggregation_strategy: str = Field(..., description="Aggregation strategy used")
    satisfaction_metrics: Dict[str, Any] = Field(..., description="Group satisfaction metrics")
    generated_at: datetime = Field(..., description="Generation timestamp")


class NextItemPrediction(BaseModel):
    """Next item prediction"""
    item_id: str = Field(..., description="Predicted item ID")
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class NextItemPredictionResponse(BaseModel):
    """Next item prediction response"""
    user_id: str = Field(..., description="User identifier")
    predictions: List[NextItemPrediction] = Field(..., description="Item predictions")
    session_context: List[str] = Field(..., description="Current session context")
    generated_at: datetime = Field(..., description="Generation timestamp")


class SimilarItem(BaseModel):
    """Similar item information"""
    item_id: str = Field(..., description="Similar item ID")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    embedding_similarity: Optional[float] = Field(None, description="Embedding similarity")
    metadata_similarity: Optional[float] = Field(None, description="Metadata similarity")


class SimilarItemsResponse(BaseModel):
    """Similar items response"""
    item_id: str = Field(..., description="Query item ID")
    similar_items: List[SimilarItem] = Field(..., description="Similar items")
    similarity_method: str = Field(..., description="Similarity method used")
    generated_at: datetime = Field(..., description="Generation timestamp")


class FeedbackResponse(BaseModel):
    """Feedback response"""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    processed: bool = Field(..., description="Whether feedback was processed")
    updated_preferences: bool = Field(..., description="Whether preferences were updated")
    message: str = Field(..., description="Response message")


class PerformanceMetrics(BaseModel):
    """System performance metrics"""
    ndcg_at_10: Optional[float] = Field(None, description="NDCG@10 score")
    intra_list_diversity: Optional[float] = Field(None, description="Intra-list diversity")
    coverage: Optional[float] = Field(None, description="Catalog coverage")
    novelty: Optional[float] = Field(None, description="Novelty score")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class SystemStatus(BaseModel):
    """System status information"""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    total_users: int = Field(..., description="Total number of users")
    total_items: int = Field(..., description="Total number of items")
    total_interactions: int = Field(..., description="Total number of interactions")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="System performance")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")


class UserProfile(BaseModel):
    """User profile information"""
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, float] = Field(..., description="User preferences")
    recent_interactions: List[Dict[str, Any]] = Field(..., description="Recent interactions")
    profile_evolution: Dict[str, Any] = Field(..., description="Profile evolution metrics")
    last_updated: datetime = Field(..., description="Last update timestamp")


class ContentItem(BaseModel):
    """Content item information"""
    item_id: str = Field(..., description="Item identifier")
    title: str = Field(..., description="Item title")
    genre: Optional[str] = Field(None, description="Item genre")
    domain: Optional[str] = Field(None, description="Content domain")
    release_date: Optional[datetime] = Field(None, description="Release date")
    quality_score: Optional[float] = Field(None, description="Quality score")
    popularity_score: Optional[float] = Field(None, description="Popularity score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
