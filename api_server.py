"""
API Server

Provides RESTful API endpoints for accessing processed data and features:
- User profile endpoints
- Content recommendation endpoints
- Data quality endpoints
- Feature store access
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import settings
from src.utils.logging import LoggingConfig
from src.user_profiling.preference_tracker import PreferenceTracker
from src.user_profiling.behavior_analyzer import BehaviorAnalyzer
from src.user_profiling.profile_evolution import ProfileEvolution
from src.content_understanding.quality_scorer import QualityScorer
from src.data_integration.schema_manager import get_session
from sqlalchemy import text

# Import new hybrid recommendation system
from src.recommenders.hybrid import HybridRecommender
from src.api.endpoints import get_all_routers
from src.api.schemas import RecommendationRequest, RecommendationResponse

# Import adaptive learning components
from src.api.adaptive_endpoints import adaptive_router, feedback_router
from src.pipeline_integration import AdaptiveLearningPipeline


# Pydantic models for API responses
class UserPreferenceResponse(BaseModel):
    user_id: str
    top_genres: List[Dict[str, Any]]
    top_content_types: List[Dict[str, Any]]
    rating_profile: Dict[str, float]
    diversity_score: float
    exploration_tendency: float
    last_updated: datetime


class UserBehaviorResponse(BaseModel):
    user_id: str
    activity_level: str
    engagement_score: float
    consistency_score: float
    total_sessions: int
    behavior_patterns: Dict[str, Any]
    last_updated: datetime


class UserEvolutionResponse(BaseModel):
    user_id: str
    current_lifecycle_stage: str
    profile_stability: float
    change_velocity: float
    predictability_score: float
    trends: Dict[str, Any]
    risk_summary: Dict[str, Any]
    last_updated: datetime


class ContentQualityResponse(BaseModel):
    content_id: str
    overall_score: float
    popularity_score: float
    acclaim_score: float
    rating_score: float
    freshness_score: float
    content_quality_score: float
    last_updated: datetime


class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    n_recommendations: int = 10  # Alternative field name for compatibility
    content_types: Optional[List[str]] = None
    exclude_seen: bool = True
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    recommendation_strategy: str
    confidence_score: float
    generated_at: datetime


# Initialize FastAPI app
app = FastAPI(
    title="Content Recommendation Engine API",
    description="API for accessing user profiles, content quality scores, and recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
config = None
preference_tracker = None
behavior_analyzer = None
profile_evolution = None
quality_scorer = None
hybrid_recommender = None

# Initialize adaptive learning pipeline
adaptive_pipeline = None

def initialize_adaptive_pipeline():
    """Initialize the adaptive learning pipeline."""
    global adaptive_pipeline
    if adaptive_pipeline is None:
        try:
            # Try to initialize the full pipeline
            adaptive_pipeline = AdaptiveLearningPipeline()
            adaptive_pipeline.start()
            logger.info("✅ Adaptive learning pipeline initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize full pipeline: {e}")
            # Use a mock pipeline for basic functionality
            adaptive_pipeline = MockAdaptivePipeline()
            logger.info("✅ Mock adaptive pipeline initialized")
    return adaptive_pipeline

class MockAdaptivePipeline:
    """Mock pipeline for basic adaptive functionality."""
    
    def __init__(self):
        self.running = True
    
    def get_recommendations(self, user_id, num_items=10, context=None, include_explanations=False):
        """Generate mock adaptive recommendations."""
        recommendations = []
        for i in range(num_items):
            rec = {
                "item_id": f"adaptive_item_{user_id}_{i+1}",
                "score": 0.9 - (i * 0.05),
                "rank": i + 1,
                "method": "adaptive_mock",
                "explanation": {
                    "reasoning": ["Mock adaptive recommendation"],
                    "confidence": 0.8
                } if include_explanations else None
            }
            recommendations.append(rec)
        return recommendations


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global preference_tracker, behavior_analyzer, profile_evolution, quality_scorer, logger, hybrid_recommender
    
    # Setup logging
    logging_config = LoggingConfig()
    logger = logging.getLogger(__name__)
    
    # Initialize user profiling components
    preference_tracker = PreferenceTracker()
    behavior_analyzer = BehaviorAnalyzer()
    profile_evolution = ProfileEvolution()
    quality_scorer = QualityScorer()
    
    # Initialize hybrid recommendation system
    try:
        hybrid_recommender = HybridRecommender(
            collaborative_weight=0.4,
            content_weight=0.4,
            knowledge_weight=0.2,
            diversity_factor=0.1,
            explanation_enabled=True,
            auto_tune_weights=True
        )
        logger.info("Hybrid recommender initialized")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid recommender: {e}")
        hybrid_recommender = None
    
    # Initialize adaptive learning pipeline
    initialize_adaptive_pipeline()
    
    logger.info("API server components initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global preference_tracker, behavior_analyzer, profile_evolution, quality_scorer, hybrid_recommender
    
    if preference_tracker:
        preference_tracker.close()
    if behavior_analyzer:
        behavior_analyzer.close()
    if profile_evolution:
        profile_evolution.close()
    if hybrid_recommender:
        # Save model if needed
        try:
            hybrid_recommender.save_model("models/hybrid_recommender.pkl")
        except Exception as e:
            logger.warning(f"Failed to save hybrid recommender: {e}")
    
    logger.info("API server resources cleaned up")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


# User profile endpoints
@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get comprehensive user profile from database"""
    try:
        session = get_session()
        
        try:
            # Get user's interaction statistics
            interaction_query = text("""
                SELECT 
                    COUNT(*) as total_interactions,
                    AVG(rating) as avg_rating,
                    COUNT(DISTINCT content_id) as unique_content,
                    MIN(timestamp) as first_interaction,
                    MAX(timestamp) as last_interaction
                FROM interaction_events 
                WHERE user_id = :user_id
            """)
            
            interaction_result = session.execute(interaction_query, {"user_id": user_id}).fetchone()
            
            if not interaction_result or interaction_result[0] == 0:
                # Check if user exists in user_profiles table
                user_check = session.execute(
                    text("SELECT COUNT(*) FROM user_profiles WHERE user_id = :user_id"),
                    {"user_id": user_id}
                ).scalar()
                
                if user_check == 0:
                    raise HTTPException(status_code=404, detail="User not found")
                else:
                    # User exists but no interactions
                    return {
                        "user_id": user_id,
                        "profile_status": "new_user",
                        "total_interactions": 0,
                        "avg_rating": 0.0,
                        "unique_content": 0,
                        "genre_preferences": [],
                        "content_type_preferences": [],
                        "activity_level": "inactive",
                        "last_updated": datetime.now()
                    }
            
            total_interactions, avg_rating, unique_content, first_interaction, last_interaction = interaction_result
            
            # Get genre preferences
            genre_query = text("""
                SELECT c.genres, COUNT(*) as count, AVG(i.rating) as avg_rating
                FROM interaction_events i
                JOIN content_metadata c ON i.content_id = c.content_id
                WHERE i.user_id = :user_id AND c.genres IS NOT NULL
                GROUP BY c.genres
                ORDER BY count DESC, avg_rating DESC
                LIMIT 5
            """)
            
            genre_result = session.execute(genre_query, {"user_id": user_id}).fetchall()
            
            # Process genres
            genre_preferences = []
            for genre_row in genre_result:
                genres_str, count, rating = genre_row
                # Extract individual genres from pipe-separated or comma-separated string
                if genres_str:
                    individual_genres = [g.strip() for g in genres_str.replace('|', ',').split(',')]
                    for genre in individual_genres[:3]:  # Top 3 genres from this group
                        if genre and genre not in [g['genre'] for g in genre_preferences]:
                            genre_preferences.append({
                                "genre": genre,
                                "interaction_count": count,
                                "avg_rating": round(rating, 2)
                            })
                if len(genre_preferences) >= 5:
                    break
            
            # Get content type preferences
            content_type_query = text("""
                SELECT c.content_type, COUNT(*) as count, AVG(i.rating) as avg_rating
                FROM interaction_events i
                JOIN content_metadata c ON i.content_id = c.content_id
                WHERE i.user_id = :user_id AND c.content_type IS NOT NULL
                GROUP BY c.content_type
                ORDER BY count DESC
            """)
            
            content_type_result = session.execute(content_type_query, {"user_id": user_id}).fetchall()
            content_type_preferences = [
                {
                    "content_type": row[0] or "unknown",
                    "interaction_count": row[1],
                    "avg_rating": round(row[2], 2)
                }
                for row in content_type_result
            ]
            
            # Determine activity level
            if total_interactions >= 50:
                activity_level = "high"
            elif total_interactions >= 10:
                activity_level = "medium"
            elif total_interactions > 0:
                activity_level = "low"
            else:
                activity_level = "inactive"
            
            # Calculate diversity score based on unique content vs total interactions
            diversity_score = unique_content / max(total_interactions, 1)
            
            return {
                "user_id": user_id,
                "profile_status": "active",
                "total_interactions": total_interactions,
                "avg_rating": round(avg_rating or 0.0, 2),
                "unique_content": unique_content,
                "diversity_score": round(diversity_score, 3),
                "activity_level": activity_level,
                "genre_preferences": genre_preferences,
                "content_type_preferences": content_type_preferences,
                "first_interaction": first_interaction,
                "last_interaction": last_interaction,
                "last_updated": datetime.now()
            }
            
        finally:
            session.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")


@app.get("/users/{user_id}/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences(user_id: str):
    """Get user preference profile"""
    try:
        summary = preference_tracker.get_preference_summary(user_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="User not found or no preferences available")
        
        return UserPreferenceResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting preferences for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/behavior", response_model=UserBehaviorResponse)
async def get_user_behavior(user_id: str):
    """Get user behavior profile"""
    try:
        summary = behavior_analyzer.get_behavior_summary(user_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="User not found or no behavior data available")
        
        return UserBehaviorResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting behavior for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/evolution", response_model=UserEvolutionResponse)
async def get_user_evolution(user_id: str):
    """Get user profile evolution"""
    try:
        summary = profile_evolution.get_evolution_summary(user_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="User not found or no evolution data available")
        
        return UserEvolutionResponse(**summary)
        
    except Exception as e:
        logger.error(f"Error getting evolution for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/profile")
async def get_complete_user_profile(user_id: str):
    """Get complete user profile combining all aspects"""
    try:
        # Get all profile components
        preferences = preference_tracker.get_preference_summary(user_id)
        behavior = behavior_analyzer.get_behavior_summary(user_id)
        evolution = profile_evolution.get_evolution_summary(user_id)
        
        return {
            "user_id": user_id,
            "preferences": preferences,
            "behavior": behavior,
            "evolution": evolution,
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting complete profile for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Content endpoints
@app.get("/content/{content_id}/quality", response_model=ContentQualityResponse)
async def get_content_quality(content_id: str):
    """Get content quality scores"""
    try:
        quality_scores = quality_scorer.get_quality_scores(content_id)
        
        if not quality_scores:
            raise HTTPException(status_code=404, detail="Content not found or no quality data available")
        
        return ContentQualityResponse(
            content_id=content_id,
            last_updated=datetime.now(),
            **quality_scores
        )
        
    except Exception as e:
        logger.error(f"Error getting quality for content {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content/top-quality")
async def get_top_quality_content(
    limit: int = Query(default=20, ge=1, le=100),
    content_type: Optional[str] = None,
    min_score: float = Query(default=0.7, ge=0.0, le=1.0)
):
    """Get top quality content"""
    try:
        session = get_session()
        
        try:
            # Build query based on parameters - use actual tables and calculate quality from ratings
            query = text("""
                SELECT c.content_id, c.title, c.content_type, c.genres, 
                       AVG(i.rating) as avg_rating, COUNT(i.rating) as rating_count
                FROM content_metadata c
                LEFT JOIN interaction_events i ON c.content_id = i.content_id
                WHERE i.rating > 0
                GROUP BY c.content_id, c.title, c.content_type, c.genres
                HAVING AVG(i.rating) >= :min_score AND COUNT(i.rating) >= 5
                ORDER BY AVG(i.rating) DESC, COUNT(i.rating) DESC
                LIMIT :limit
            """)
            
            params = {"min_score": min_score, "limit": limit}
            
            result = session.execute(query, params).fetchall()
            
            content_list = []
            for row in result:
                content_list.append({
                    "content_id": row[0],
                    "title": row[1],
                    "content_type": row[2],
                    "genres": row[3],
                    "quality_score": round(row[4], 2),  # avg_rating as quality score
                    "rating_count": row[5]
                })
            
            return {
                "content": content_list,
                "filters": {
                    "min_score": min_score,
                    "content_type": content_type,
                    "limit": limit
                },
                "total_found": len(content_list)
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting top quality content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Basic recommendation endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get content recommendations for a user"""
    try:
        session = get_session()
        
        try:
            # Check if user exists in database
            user_check = session.execute(
                text("SELECT COUNT(*) FROM interaction_events WHERE user_id = :user_id"),
                {"user_id": request.user_id}
            ).scalar()
            
            if user_check == 0:
                # New user - provide popular content recommendations
                strategy = "popular_content"
                query = text("""
                    SELECT c.content_id, c.title, c.content_type, c.genres,
                           COUNT(i.interaction_id) as popularity_count,
                           AVG(i.rating) as avg_rating
                    FROM content_metadata c
                    LEFT JOIN interaction_events i ON c.content_id = i.content_id
                    GROUP BY c.content_id, c.title, c.content_type, c.genres
                    HAVING COUNT(i.interaction_id) > 0
                    ORDER BY popularity_count DESC, avg_rating DESC
                    LIMIT :limit
                """)
                params = {"limit": request.num_recommendations}
                
            else:
                # Existing user - get user's genre preferences
                user_genre_query = text("""
                    SELECT c.genres, COUNT(*) as interaction_count, AVG(i.rating) as avg_rating
                    FROM interaction_events i
                    JOIN content_metadata c ON i.content_id = c.content_id
                    WHERE i.user_id = :user_id AND c.genres IS NOT NULL
                    GROUP BY c.genres
                    ORDER BY interaction_count DESC, avg_rating DESC
                    LIMIT 3
                """)
                
                user_genres_result = session.execute(user_genre_query, {"user_id": request.user_id}).fetchall()
                
                if user_genres_result:
                    # Genre-based recommendations
                    strategy = "genre_based"
                    
                    # Get content user hasn't seen in their preferred genres
                    if request.exclude_seen:
                        query = text("""
                            SELECT DISTINCT c.content_id, c.title, c.content_type, c.genres,
                                   COUNT(i2.interaction_id) as popularity_count,
                                   AVG(i2.rating) as avg_rating
                            FROM content_metadata c
                            LEFT JOIN interaction_events i2 ON c.content_id = i2.content_id
                            WHERE c.content_id NOT IN (
                                SELECT DISTINCT content_id 
                                FROM interaction_events 
                                WHERE user_id = :user_id
                            )
                            AND (c.genres LIKE :genre1 OR c.genres LIKE :genre2 OR c.genres LIKE :genre3)
                            GROUP BY c.content_id, c.title, c.content_type, c.genres
                            ORDER BY popularity_count DESC, avg_rating DESC
                            LIMIT :limit
                        """)
                        
                        # Extract genre keywords from user's preferences
                        top_genres = []
                        for genre_row in user_genres_result:
                            genre_str = genre_row[0] or ""
                            # Extract first genre from genre string
                            first_genre = genre_str.split('|')[0].split(',')[0].strip()
                            if first_genre:
                                top_genres.append(f"%{first_genre}%")
                        
                        # Pad with empty strings if needed
                        while len(top_genres) < 3:
                            top_genres.append("%")
                        
                        params = {
                            "user_id": request.user_id,
                            "genre1": top_genres[0],
                            "genre2": top_genres[1],
                            "genre3": top_genres[2],
                            "limit": request.num_recommendations
                        }
                    else:
                        # Include seen content
                        query = text("""
                            SELECT DISTINCT c.content_id, c.title, c.content_type, c.genres,
                                   COUNT(i2.interaction_id) as popularity_count,
                                   AVG(i2.rating) as avg_rating
                            FROM content_metadata c
                            LEFT JOIN interaction_events i2 ON c.content_id = i2.content_id
                            WHERE (c.genres LIKE :genre1 OR c.genres LIKE :genre2 OR c.genres LIKE :genre3)
                            GROUP BY c.content_id, c.title, c.content_type, c.genres
                            ORDER BY popularity_count DESC, avg_rating DESC
                            LIMIT :limit
                        """)
                        
                        top_genres = []
                        for genre_row in user_genres_result:
                            genre_str = genre_row[0] or ""
                            first_genre = genre_str.split('|')[0].split(',')[0].strip()
                            if first_genre:
                                top_genres.append(f"%{first_genre}%")
                        
                        while len(top_genres) < 3:
                            top_genres.append("%")
                        
                        params = {
                            "genre1": top_genres[0],
                            "genre2": top_genres[1],
                            "genre3": top_genres[2],
                            "limit": request.num_recommendations
                        }
                else:
                    # User exists but no clear preferences - use popular content
                    strategy = "popular_content"
                    if request.exclude_seen:
                        query = text("""
                            SELECT c.content_id, c.title, c.content_type, c.genres,
                                   COUNT(i.interaction_id) as popularity_count,
                                   AVG(i.rating) as avg_rating
                            FROM content_metadata c
                            LEFT JOIN interaction_events i ON c.content_id = i.content_id
                            WHERE c.content_id NOT IN (
                                SELECT DISTINCT content_id 
                                FROM interaction_events 
                                WHERE user_id = :user_id
                            )
                            GROUP BY c.content_id, c.title, c.content_type, c.genres
                            HAVING COUNT(i.interaction_id) > 0
                            ORDER BY popularity_count DESC, avg_rating DESC
                            LIMIT :limit
                        """)
                        params = {"user_id": request.user_id, "limit": request.num_recommendations}
                    else:
                        query = text("""
                            SELECT c.content_id, c.title, c.content_type, c.genres,
                                   COUNT(i.interaction_id) as popularity_count,
                                   AVG(i.rating) as avg_rating
                            FROM content_metadata c
                            LEFT JOIN interaction_events i ON c.content_id = i.content_id
                            GROUP BY c.content_id, c.title, c.content_type, c.genres
                            HAVING COUNT(i.interaction_id) > 0
                            ORDER BY popularity_count DESC, avg_rating DESC
                            LIMIT :limit
                        """)
                        params = {"limit": request.num_recommendations}
            
            # Execute the recommendation query
            result = session.execute(query, params).fetchall()
            
            recommendations = []
            for row in result:
                content_id, title, content_type, genres, popularity_count, avg_rating = row
                
                # Calculate recommendation score
                popularity_score = min(1.0, (popularity_count or 0) / 100.0)  # Normalize popularity
                rating_score = (avg_rating or 3.0) / 5.0  # Normalize rating
                recommendation_score = (popularity_score * 0.4 + rating_score * 0.6)
                
                recommendations.append({
                    "content_id": str(content_id),
                    "title": title or "Unknown Title",
                    "content_type": content_type or "unknown",
                    "genres": genres or "Unknown",
                    "popularity_count": popularity_count or 0,
                    "avg_rating": round(avg_rating or 0.0, 2),
                    "recommendation_score": round(recommendation_score, 3)
                })
            
            # Calculate confidence based on data availability
            if user_check == 0:
                confidence = 0.5  # Moderate confidence for new users
            else:
                # Higher confidence for users with preferences
                data_quality = min(1.0, user_check / 10.0)  # More interactions = higher confidence
                confidence = min(0.9, 0.3 + data_quality * 0.6)
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                recommendation_strategy=strategy,
                confidence_score=round(confidence, 3),
                generated_at=datetime.now()
            )
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


# Analytics endpoints
@app.get("/analytics/user-distribution")
async def get_user_distribution():
    """Get user distribution analytics"""
    try:
        session = get_session()
        
        try:
            # Activity level distribution
            activity_query = text("""
                SELECT 
                    CASE 
                        WHEN interaction_count >= 50 THEN 'high'
                        WHEN interaction_count >= 10 THEN 'medium'
                        WHEN interaction_count > 0 THEN 'low'
                        ELSE 'inactive'
                    END as activity_level,
                    COUNT(*) as user_count
                FROM (
                    SELECT user_id, COUNT(*) as interaction_count
                    FROM interaction_events
                    GROUP BY user_id
                ) user_stats
                GROUP BY activity_level
            """)
            
            activity_result = session.execute(activity_query).fetchall()
            activity_distribution = {row[0]: row[1] for row in activity_result}
            
            # Rating behavior distribution
            rating_query = text("""
                SELECT 
                    CASE 
                        WHEN avg_rating >= 4.0 THEN 'positive'
                        WHEN avg_rating >= 3.0 THEN 'neutral'
                        ELSE 'critical'
                    END as rating_behavior,
                    COUNT(*) as user_count
                FROM (
                    SELECT user_id, AVG(rating) as avg_rating
                    FROM interaction_events
                    WHERE rating > 0
                    GROUP BY user_id
                ) rating_stats
                GROUP BY rating_behavior
            """)
            
            rating_result = session.execute(rating_query).fetchall()
            rating_distribution = {row[0]: row[1] for row in rating_result}
            
            return {
                "activity_distribution": activity_distribution,
                "rating_distribution": rating_distribution,
                "generated_at": datetime.now()
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting user distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/content-stats")
async def get_content_statistics():
    """Get content statistics"""
    try:
        session = get_session()
        
        try:
            # Content type distribution
            type_query = text("SELECT content_type, COUNT(*) FROM content_metadata GROUP BY content_type")
            type_result = session.execute(type_query).fetchall()
            content_type_distribution = {row[0] or "unknown": row[1] for row in type_result}
            
            # Create a simple quality distribution based on average ratings
            quality_query = text("""
                SELECT 
                    CASE 
                        WHEN avg_rating >= 4.0 THEN 'excellent'
                        WHEN avg_rating >= 3.0 THEN 'good'
                        WHEN avg_rating >= 2.0 THEN 'average'
                        ELSE 'poor'
                    END as quality_tier,
                    COUNT(*) as content_count
                FROM (
                    SELECT content_id, AVG(rating) as avg_rating
                    FROM interaction_events
                    WHERE rating > 0
                    GROUP BY content_id
                ) content_ratings
                GROUP BY quality_tier
            """)
            
            quality_result = session.execute(quality_query).fetchall()
            quality_distribution = {row[0]: row[1] for row in quality_result}
            
            return {
                "content_type_distribution": content_type_distribution,
                "quality_distribution": quality_distribution,
                "generated_at": datetime.now()
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting content statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/data-quality")
async def get_data_quality_metrics():
    """Get overall data quality metrics"""
    try:
        session = get_session()
        
        try:
            metrics = {}
            
            # Data completeness
            total_users = session.execute(text("SELECT COUNT(*) FROM user_profiles")).scalar()
            active_users = session.execute(text("SELECT COUNT(DISTINCT user_id) FROM interaction_events")).scalar()
            metrics['user_engagement_rate'] = active_users / max(total_users, 1)
            
            total_content = session.execute(text("SELECT COUNT(*) FROM content_metadata")).scalar()
            rated_content = session.execute(text("SELECT COUNT(DISTINCT content_id) FROM interaction_events WHERE rating > 0")).scalar()
            metrics['content_rating_coverage'] = rated_content / max(total_content, 1)
            
            # Data freshness (interactions in last 30 days)
            recent_interactions = session.execute(text("""
                SELECT COUNT(*) FROM interaction_events 
                WHERE timestamp >= datetime('now', '-30 days')
            """)).scalar()
            total_interactions = session.execute(text("SELECT COUNT(*) FROM interaction_events")).scalar()
            metrics['data_freshness'] = recent_interactions / max(total_interactions, 1)
            
            # Rating quality
            rating_stats = session.execute(text("""
                SELECT AVG(rating), COUNT(*) 
                FROM interaction_events 
                WHERE rating > 0
            """)).fetchone()
            
            if rating_stats and rating_stats[0]:
                metrics['avg_rating'] = float(rating_stats[0])
                metrics['total_ratings'] = rating_stats[1]
            
            # Calculate overall quality score
            overall_quality = (
                metrics.get('user_engagement_rate', 0) * 0.3 +
                metrics.get('content_rating_coverage', 0) * 0.3 +
                metrics.get('data_freshness', 0) * 0.4
            )
            metrics['overall_quality_score'] = overall_quality
            
            return {
                "metrics": metrics,
                "quality_grade": (
                    "A" if overall_quality >= 0.8 else
                    "B" if overall_quality >= 0.6 else
                    "C" if overall_quality >= 0.4 else "D"
                ),
                "generated_at": datetime.now()
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting data quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data quality and system status endpoints
@app.get("/system/status")
async def get_system_status():
    """Get system status and data quality metrics"""
    try:
        session = get_session()
        
        try:
            # Get table counts
            interaction_count = session.execute(text("SELECT COUNT(*) FROM interaction_events")).scalar() or 0
            content_count = session.execute(text("SELECT COUNT(*) FROM content_metadata")).scalar() or 0
            user_count = session.execute(text("SELECT COUNT(*) FROM user_profiles")).scalar() or 0
            
            # Get unique counts
            unique_users_with_interactions = session.execute(text("SELECT COUNT(DISTINCT user_id) FROM interaction_events")).scalar() or 0
            unique_content_with_interactions = session.execute(text("SELECT COUNT(DISTINCT content_id) FROM interaction_events")).scalar() or 0
            
            # Calculate basic quality metrics
            user_coverage = unique_users_with_interactions / max(user_count, 1)
            content_coverage = unique_content_with_interactions / max(content_count, 1)
            
            # Rating statistics
            if interaction_count > 0:
                rating_stats = session.execute(text("""
                    SELECT AVG(rating), MIN(rating), MAX(rating), COUNT(rating)
                    FROM interaction_events WHERE rating > 0
                """)).fetchone()
                
                avg_rating, min_rating, max_rating, rating_count = rating_stats or (0, 0, 0, 0)
            else:
                avg_rating = min_rating = max_rating = rating_count = 0
            
            # Overall quality score
            volume_score = min(1.0, interaction_count / 1000)  # Normalize by 1000 interactions
            coverage_score = (user_coverage + content_coverage) / 2
            data_quality_score = (volume_score * 0.5 + coverage_score * 0.5)
            
            # Determine status
            if data_quality_score >= 0.7:
                status = "excellent"
            elif data_quality_score >= 0.5:
                status = "good"
            elif data_quality_score >= 0.3:
                status = "fair"
            else:
                status = "poor"
            
            return {
                "system_status": status,
                "data_quality_score": round(data_quality_score, 3),
                "database_statistics": {
                    "total_interactions": interaction_count,
                    "total_content": content_count,
                    "total_users": user_count,
                    "unique_users_with_interactions": unique_users_with_interactions,
                    "unique_content_with_interactions": unique_content_with_interactions
                },
                "coverage_metrics": {
                    "user_coverage": round(user_coverage, 3),
                    "content_coverage": round(content_coverage, 3)
                },
                "rating_statistics": {
                    "total_ratings": rating_count,
                    "average_rating": round(avg_rating or 0, 2),
                    "min_rating": min_rating or 0,
                    "max_rating": max_rating or 0
                },
                "recommendations_available": interaction_count > 0 and content_count > 0,
                "last_updated": datetime.now()
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            "system_status": "error",
            "error": str(e),
            "last_updated": datetime.now()
        }


# Include new recommendation API routes
for router in get_all_routers():
    app.include_router(router)

# Include adaptive learning routes
app.include_router(adaptive_router)
app.include_router(feedback_router)


# Dependency injection for recommendation engine
def get_recommendation_engine():
    """Get the global recommendation engine instance"""
    return hybrid_recommender


# Update the endpoints module to use the global instance
import src.api.endpoints as endpoints_module
endpoints_module.get_recommendation_engine = lambda: hybrid_recommender


# New hybrid recommendation endpoints
@app.post("/api/v1/recommendations/hybrid")
async def get_hybrid_recommendations(request: RecommendationRequest):
    """
    Get hybrid recommendations using all recommendation methods
    """
    try:
        if not hybrid_recommender:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")
        
        # Get user history from database
        user_history = []
        try:
            with get_session() as session:
                # Get recent user interactions
                query = text("""
                    SELECT DISTINCT content_id 
                    FROM user_content_interactions 
                    WHERE user_id = :user_id 
                    ORDER BY interaction_date DESC 
                    LIMIT 50
                """)
                result = session.execute(query, {"user_id": request.user_id})
                user_history = [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Could not fetch user history: {e}")
        
        # Generate recommendations
        recommendations = hybrid_recommender.recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
            user_history=user_history,
            filters=request.filters,
            context=request.context,
            exclude_seen=request.exclude_seen
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[
                {
                    "item_id": rec["item_id"],
                    "score": rec["score"],
                    "rank": i + 1,
                    "method": rec.get("method", "hybrid"),
                    "explanation": rec.get("explanation"),
                    "components": rec.get("components"),
                    "metadata": rec.get("metadata")
                }
                for i, rec in enumerate(recommendations)
            ],
            method="hybrid",
            total_recommendations=len(recommendations),
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feedback/implicit")
async def record_implicit_feedback(
    user_id: str,
    item_id: str, 
    interaction_type: str,
    context: dict = None
):
    """Record implicit feedback for real-time personalization"""
    try:
        # Store in database
        with get_session() as session:
            query = text("""
                INSERT INTO user_content_interactions 
                (user_id, content_id, interaction_type, interaction_date, context_data)
                VALUES (:user_id, :content_id, :interaction_type, :interaction_date, :context_data)
            """)
            session.execute(query, {
                "user_id": user_id,
                "content_id": item_id,
                "interaction_type": interaction_type,
                "interaction_date": datetime.now(),
                "context_data": str(context) if context else None
            })
            session.commit()
        
        # Update real-time preferences if hybrid recommender is available
        if hybrid_recommender:
            # This would update short-term preferences
            pass
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
