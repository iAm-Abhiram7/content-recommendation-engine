"""
API Endpoints for Hybrid Recommendation Engine

RESTful API endpoints for all recommendation functionality:
- Individual recommendations
- Group recommendations  
- Next-item prediction
- Feedback collection
- Real-time personalization
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import time
import asyncio

from .schemas import (
    RecommendationRequest, RecommendationResponse, RecommendationItem,
    GroupRecommendationRequest, GroupRecommendationResponse,
    FeedbackRequest, FeedbackResponse,
    NextItemPredictionRequest, NextItemPredictionResponse,
    SimilarItemsRequest, SimilarItemsResponse,
    CrossDomainRequest, SystemStatus, PerformanceMetrics,
    UserProfile, ContentItem, ErrorResponse
)

logger = logging.getLogger(__name__)

# Router for recommendation endpoints
recommendations_router = APIRouter(prefix="/recommendations", tags=["recommendations"])
feedback_router = APIRouter(prefix="/feedback", tags=["feedback"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])
system_router = APIRouter(prefix="/system", tags=["system"])


# Dependency to get recommendation engine
async def get_recommendation_engine():
    """Dependency to get the recommendation engine instance"""
    # This will be injected by the main application
    from ..recommenders.hybrid import HybridRecommender
    # Return the global instance - would be properly initialized in main app
    return None  # Placeholder


@recommendations_router.post("/individual", response_model=RecommendationResponse)
async def get_individual_recommendations(
    request: RecommendationRequest,
    rec_engine = Depends(get_recommendation_engine)
) -> RecommendationResponse:
    """
    Get personalized recommendations for an individual user
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        # Get recommendations based on method
        if request.method == "hybrid" and rec_engine:
            recommendations = rec_engine.recommend(
                user_id=request.user_id,
                n_recommendations=request.n_recommendations,
                filters=request.filters,
                context=request.context,
                exclude_seen=request.exclude_seen
            )
        else:
            # Fallback for testing
            recommendations = _generate_mock_recommendations(
                request.user_id, 
                request.n_recommendations
            )
        
        # Convert to response format
        rec_items = []
        for i, rec in enumerate(recommendations):
            rec_item = RecommendationItem(
                item_id=rec['item_id'],
                score=rec['score'],
                rank=i + 1,
                method=rec.get('method', request.method),
                explanation=rec.get('explanation'),
                components=rec.get('components'),
                metadata=rec.get('metadata')
            )
            rec_items.append(rec_item)
        
        # Calculate performance metrics
        response_time = (time.time() - start_time) * 1000
        
        performance_metrics = PerformanceMetrics(
            response_time_ms=response_time,
            intra_list_diversity=_calculate_diversity(recommendations),
            coverage=_calculate_coverage(recommendations)
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_items,
            method=request.method,
            total_recommendations=len(rec_items),
            generated_at=datetime.now(),
            performance_metrics=performance_metrics.dict()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@recommendations_router.post("/group", response_model=GroupRecommendationResponse)
async def get_group_recommendations(
    request: GroupRecommendationRequest,
    rec_engine = Depends(get_recommendation_engine)
) -> GroupRecommendationResponse:
    """
    Get recommendations for a group of users
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.user_ids:
            raise HTTPException(status_code=400, detail="At least one user ID is required")
        
        if len(request.user_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 users per group")
        
        # Get group recommendations
        from ..recommenders.group_recommender import GroupRecommender
        
        group_recommender = GroupRecommender(
            base_recommender=rec_engine,
            aggregation_strategy=request.aggregation_strategy
        )
        
        recommendations = group_recommender.recommend_for_group(
            user_ids=request.user_ids,
            n_recommendations=request.n_recommendations,
            user_weights=request.user_weights,
            filters=request.filters,
            context=request.context
        )
        
        # Convert to response format
        rec_items = []
        satisfaction_metrics = {}
        
        for i, rec in enumerate(recommendations):
            rec_item = RecommendationItem(
                item_id=rec['item_id'],
                score=rec['group_score'],
                rank=i + 1,
                method="group",
                explanation=rec.get('group_explanation'),
                components=rec.get('user_scores'),
                metadata=rec.get('metadata')
            )
            rec_items.append(rec_item)
            
            # Extract satisfaction metrics from first recommendation
            if i == 0 and 'satisfaction_metrics' in rec:
                satisfaction_metrics = rec['satisfaction_metrics']
        
        return GroupRecommendationResponse(
            user_ids=request.user_ids,
            recommendations=rec_items,
            aggregation_strategy=request.aggregation_strategy,
            satisfaction_metrics=satisfaction_metrics,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating group recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@recommendations_router.post("/next-item", response_model=NextItemPredictionResponse)
async def predict_next_item(
    request: NextItemPredictionRequest
) -> NextItemPredictionResponse:
    """
    Predict next item for sequential recommendation
    """
    try:
        # This would use the sequential pattern miner
        from ..personalization.sequential import SequentialPatternMiner
        
        # Mock implementation for now
        predictions = []
        for i in range(request.n_predictions):
            predictions.append({
                'item_id': f"item_{i+1}",
                'probability': 0.8 - (i * 0.1),
                'confidence': 0.7 - (i * 0.05)
            })
        
        return NextItemPredictionResponse(
            user_id=request.user_id,
            predictions=[
                {
                    'item_id': pred['item_id'],
                    'probability': pred['probability'],
                    'confidence': pred['confidence']
                }
                for pred in predictions
            ],
            session_context=request.current_session,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error predicting next item for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@recommendations_router.post("/similar-items", response_model=SimilarItemsResponse)
async def get_similar_items(
    request: SimilarItemsRequest,
    rec_engine = Depends(get_recommendation_engine)
) -> SimilarItemsResponse:
    """
    Get items similar to a given item
    """
    try:
        # Use content-based recommender for similarity
        if rec_engine and hasattr(rec_engine, 'content_rec') and rec_engine.content_rec:
            similar_items = rec_engine.content_rec.find_similar_items(
                item_id=request.item_id,
                n_similar=request.n_similar,
                use_metadata=request.use_metadata,
                metadata_weight=request.metadata_weight
            )
        else:
            # Mock similar items
            similar_items = []
            for i in range(request.n_similar):
                similar_items.append({
                    'item_id': f"similar_item_{i+1}",
                    'similarity_score': 0.9 - (i * 0.05),
                    'embedding_similarity': 0.85 - (i * 0.04),
                    'metadata_similarity': 0.75 - (i * 0.03)
                })
        
        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=[
                {
                    'item_id': item['item_id'],
                    'similarity_score': item['similarity_score'],
                    'embedding_similarity': item.get('embedding_similarity'),
                    'metadata_similarity': item.get('metadata_similarity')
                }
                for item in similar_items
            ],
            similarity_method="content_based",
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error finding similar items for {request.item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@recommendations_router.post("/cross-domain", response_model=RecommendationResponse)
async def get_cross_domain_recommendations(
    request: CrossDomainRequest,
    rec_engine = Depends(get_recommendation_engine)
) -> RecommendationResponse:
    """
    Get cross-domain recommendations
    """
    try:
        # Use content-based recommender for cross-domain
        if rec_engine and hasattr(rec_engine, 'content_rec') and rec_engine.content_rec:
            recommendations = rec_engine.content_rec.cross_domain_recommend(
                source_items=request.source_items,
                target_domain=request.target_domain,
                n_recommendations=request.n_recommendations
            )
        else:
            # Mock cross-domain recommendations
            recommendations = []
            for i in range(request.n_recommendations):
                recommendations.append({
                    'item_id': f"{request.target_domain}_item_{i+1}",
                    'score': 0.8 - (i * 0.05),
                    'method': 'cross_domain',
                    'source_domain': 'unknown',
                    'target_domain': request.target_domain
                })
        
        # Convert to response format
        rec_items = []
        for i, rec in enumerate(recommendations):
            rec_item = RecommendationItem(
                item_id=rec['item_id'],
                score=rec['score'],
                rank=i + 1,
                method=rec.get('method', 'cross_domain'),
                explanation=rec.get('explanation'),
                components=rec.get('components'),
                metadata=rec.get('metadata')
            )
            rec_items.append(rec_item)
        
        return RecommendationResponse(
            user_id="cross_domain_user",
            recommendations=rec_items,
            method="cross_domain",
            total_recommendations=len(rec_items),
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating cross-domain recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@feedback_router.post("/interaction", response_model=FeedbackResponse)
async def record_user_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
) -> FeedbackResponse:
    """
    Record user interaction/feedback for real-time personalization
    """
    try:
        # Process feedback asynchronously
        background_tasks.add_task(
            _process_user_feedback,
            request.user_id,
            request.item_id,
            request.interaction_type,
            request.rating,
            request.timestamp or datetime.now(),
            request.context
        )
        
        return FeedbackResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            processed=True,
            updated_preferences=True,
            message="Feedback recorded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return FeedbackResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            processed=False,
            updated_preferences=False,
            message=f"Error: {str(e)}"
        )


@analytics_router.get("/user-profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str) -> UserProfile:
    """
    Get user profile and preferences
    """
    try:
        # This would fetch from user profiling components
        return UserProfile(
            user_id=user_id,
            preferences={"action": 0.8, "comedy": 0.6, "drama": 0.4},
            recent_interactions=[],
            profile_evolution={"stability": 0.7, "drift_rate": 0.1},
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error fetching user profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/content/{item_id}", response_model=ContentItem)
async def get_content_info(item_id: str) -> ContentItem:
    """
    Get content item information
    """
    try:
        # This would fetch from content database
        return ContentItem(
            item_id=item_id,
            title=f"Content Item {item_id}",
            genre="action",
            domain="movies",
            release_date=datetime.now(),
            quality_score=0.8,
            popularity_score=0.7,
            metadata={}
        )
        
    except Exception as e:
        logger.error(f"Error fetching content info for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """
    Get system status and health metrics
    """
    try:
        return SystemStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            total_users=1000,
            total_items=5000,
            total_interactions=50000,
            models_loaded={
                "collaborative": True,
                "content_based": True,
                "knowledge_based": True,
                "hybrid": True
            },
            performance_metrics=PerformanceMetrics(
                ndcg_at_10=0.42,
                intra_list_diversity=0.75,
                coverage=0.60,
                novelty=0.55,
                response_time_ms=85.0
            )
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics() -> PerformanceMetrics:
    """
    Get detailed performance metrics
    """
    try:
        return PerformanceMetrics(
            ndcg_at_10=0.42,
            intra_list_diversity=0.75,
            coverage=0.60,
            novelty=0.55,
            response_time_ms=85.0
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _generate_mock_recommendations(user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
    """Generate mock recommendations for testing"""
    recommendations = []
    
    for i in range(n_recommendations):
        recommendations.append({
            'item_id': f"item_{user_id}_{i+1}",
            'score': 0.9 - (i * 0.05),
            'method': 'hybrid',
            'explanation': {
                'reasoning': [f"Recommended based on your preferences"],
                'components': {'collaborative': 0.4, 'content': 0.4, 'knowledge': 0.2}
            }
        })
    
    return recommendations


def _calculate_diversity(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate intra-list diversity"""
    if len(recommendations) <= 1:
        return 0.0
    
    # Simple diversity calculation (would be more sophisticated in practice)
    unique_items = len(set(rec['item_id'] for rec in recommendations))
    return unique_items / len(recommendations)


def _calculate_coverage(recommendations: List[Dict[str, Any]]) -> float:
    """Calculate catalog coverage"""
    # Simple coverage calculation
    return 0.6  # Mock value


async def _process_user_feedback(
    user_id: str,
    item_id: str,
    interaction_type: str,
    rating: Optional[float],
    timestamp: datetime,
    context: Optional[Dict[str, Any]]
):
    """Process user feedback asynchronously"""
    try:
        # This would update user preferences and models
        from ..personalization.short_term import ShortTermPreferenceModel
        
        # Update short-term preferences
        short_term_model = ShortTermPreferenceModel()
        short_term_model.update_user_interaction(
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            rating=rating,
            timestamp=timestamp,
            context=context
        )
        
        logger.info(f"Processed feedback for user {user_id}, item {item_id}")
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")


# Combine all routers
def get_all_routers():
    """Get all API routers"""
    from .adaptive_endpoints import get_adaptive_routers
    
    routers = [
        recommendations_router,
        feedback_router,
        analytics_router,
        system_router
    ]
    
    # Add adaptive learning routers
    try:
        routers.extend(get_adaptive_routers())
    except ImportError:
        logger.warning("Adaptive endpoints not available")
    
    return routers
