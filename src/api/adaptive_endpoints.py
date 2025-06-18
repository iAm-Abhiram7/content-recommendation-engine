"""
API endpoints for Phase 3 adaptive learning system.
Provides RESTful API for feedback processing, recommendations, and user control.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import json

from ..pipeline_integration import AdaptiveLearningPipeline
from ..adaptive_learning import FeedbackType
from ..user_control import ControlLevel, AdaptationPolicy
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# Create routers
adaptive_router = APIRouter(prefix="/adaptive", tags=["adaptive"])
feedback_router = APIRouter(prefix="/feedback", tags=["feedback"])

# Global pipeline instance
pipeline: Optional[AdaptiveLearningPipeline] = None

class MockAdaptivePipeline:
    """Mock pipeline for basic adaptive functionality when full pipeline fails."""
    
    def __init__(self):
        self.running = True
        logger.info("Mock adaptive pipeline initialized")
    
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
    
    def process_feedback(self, user_id, item_id, feedback_type, value, context=None):
        """Mock feedback processing."""
        return {
            "status": "processed",
            "user_id": user_id,
            "item_id": item_id,
            "feedback_type": feedback_type,
            "value": value,
            "processed_at": "2025-06-18T09:00:00Z"
        }

def get_pipeline() -> AdaptiveLearningPipeline:
    """Get the pipeline instance."""
    global pipeline
    if pipeline is None:
        try:
            # Try to initialize the full pipeline
            pipeline = AdaptiveLearningPipeline()
            logger.info("Full adaptive pipeline initialized")
        except Exception as e:
            logger.warning(f"Could not initialize full pipeline: {e}")
            # Fall back to mock pipeline
            pipeline = MockAdaptivePipeline()
            logger.info("Mock adaptive pipeline initialized as fallback")
    return pipeline
class FeedbackRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    feedback_type: str = Field(..., description="Type of feedback: explicit, implicit, or contextual")
    value: float = Field(..., description="Feedback value")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    num_items: int = Field(default=10, description="Number of recommendations")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Request context")
    include_explanations: bool = Field(default=False, description="Include explanations")

class UserControlUpdate(BaseModel):
    user_id: str = Field(..., description="User identifier")
    control_level: str = Field(..., description="Control level: minimal, moderate, or full")
    auto_adapt: bool = Field(default=True, description="Enable automatic adaptation")
    notify_adaptations: bool = Field(default=False, description="Notify about adaptations")
    blocked_categories: Optional[List[str]] = Field(default=[], description="Blocked categories")
    novelty_preference: Optional[float] = Field(default=0.5, description="Novelty preference (0-1)")
    enforce_diversity: bool = Field(default=False, description="Enforce diversity")

class PreferenceUpdate(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    merge_strategy: str = Field(default="update", description="Merge strategy: update, replace, or merge")

class DriftAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    time_range: Optional[str] = Field(default="7d", description="Time range for analysis")

class AdaptationHistoryRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    limit: int = Field(default=10, description="Number of adaptations to return")

# Global pipeline instance
pipeline: Optional[AdaptiveLearningPipeline] = None

def get_pipeline() -> AdaptiveLearningPipeline:
    """Get the pipeline instance."""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline

# Health check endpoint
@adaptive_router.get("/health")
async def adaptive_health_check():
    """Health check endpoint for adaptive learning."""
    try:
        pipeline_instance = get_pipeline()
        if hasattr(pipeline_instance, 'get_pipeline_status'):
            status = pipeline_instance.get_pipeline_status()
            return {
                "status": "healthy",
                "pipeline_running": status.get("running", False),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "healthy",
                "pipeline_initialized": True,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Feedback endpoints
@feedback_router.post("/submit")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit user feedback for processing."""
    try:
        pipeline_instance = get_pipeline()
        
        # Convert string feedback type to enum
        feedback_type_map = {
            "explicit": FeedbackType.EXPLICIT,
            "implicit": FeedbackType.IMPLICIT,
            "contextual": FeedbackType.CONTEXTUAL
        }
        
        feedback_type = feedback_type_map.get(request.feedback_type.lower())
        if not feedback_type:
            raise HTTPException(status_code=400, detail="Invalid feedback type")
        
        # Process feedback
        result = await pipeline_instance.process_feedback(
            user_id=request.user_id,
            item_id=request.item_id,
            feedback_type=feedback_type,
            value=request.value,
            context=request.context
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('message'))
        
        return {
            "status": "success",
            "feedback_id": result.get('feedback_id'),
            "drift_detected": result.get('drift_detected', False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.get("/{user_id}/history")
async def get_feedback_history(user_id: str, limit: int = 100):
    """Get feedback history for a user."""
    try:
        pipeline_instance = get_pipeline()
        
        history = await pipeline_instance.feedback_collector.get_feedback_history(
            user_id=user_id,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "feedback_history": history,
            "total_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoints
@adaptive_router.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get adaptive recommendations for a user."""
    try:
        pipeline_instance = get_pipeline()
        
        # Add explanation flag to context
        context = request.context.copy()
        context['include_explanations'] = request.include_explanations
        
        result = await pipeline_instance.get_recommendations(
            user_id=request.user_id,
            num_items=request.num_items,
            context=context
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('message'))
        
        return {
            "user_id": request.user_id,
            "recommendations": result.get('recommendations', []),
            "confidence": result.get('confidence', {}),
            "explanations": result.get('explanations'),
            "adaptation_info": result.get('adaptation_info', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/recommendations/{user_id}/quick")
async def get_quick_recommendations(user_id: str, num_items: int = 5):
    """Get quick recommendations without explanations."""
    try:
        request = RecommendationRequest(
            user_id=user_id,
            num_items=num_items,
            include_explanations=False
        )
        return await get_recommendations(request)
        
    except Exception as e:
        logger.error(f"Error getting quick recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User control endpoints
@adaptive_router.post("/users/{user_id}/control")
async def update_user_control(user_id: str, request: UserControlUpdate):
    """Update user control settings."""
    try:
        pipeline_instance = get_pipeline()
        
        # Convert string control level to enum
        control_level_map = {
            "minimal": ControlLevel.MINIMAL,
            "moderate": ControlLevel.MODERATE,
            "full": ControlLevel.FULL
        }
        
        control_level = control_level_map.get(request.control_level.lower())
        if not control_level:
            raise HTTPException(status_code=400, detail="Invalid control level")
        
        # Update control settings
        await pipeline_instance.adaptation_controller.update_control_settings(
            user_id=user_id,
            control_level=control_level,
            settings={
                'auto_adapt': request.auto_adapt,
                'notify_adaptations': request.notify_adaptations,
                'blocked_categories': request.blocked_categories,
                'novelty_preference': request.novelty_preference,
                'enforce_diversity': request.enforce_diversity
            }
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "control_level": request.control_level,
            "updated_settings": request.dict(exclude={'user_id', 'control_level'})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user control: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/users/{user_id}/control")
async def get_user_control(user_id: str):
    """Get current user control settings."""
    try:
        pipeline_instance = get_pipeline()
        
        settings = await pipeline_instance.adaptation_controller.get_control_settings(
            user_id
        )
        
        return {
            "user_id": user_id,
            "control_settings": settings
        }
        
    except Exception as e:
        logger.error(f"Error getting user control: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.post("/users/{user_id}/preferences")
async def update_user_preferences(user_id: str, request: PreferenceUpdate):
    """Update user preferences."""
    try:
        pipeline_instance = get_pipeline()
        
        await pipeline_instance.preference_manager.update_preferences(
            user_id=user_id,
            preferences=request.preferences,
            merge_strategy=request.merge_strategy
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "merge_strategy": request.merge_strategy
        }
        
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/users/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get current user preferences."""
    try:
        pipeline_instance = get_pipeline()
        
        preferences = await pipeline_instance.preference_tracker.get_user_preferences(
            user_id
        )
        
        confidence = await pipeline_instance.confidence_scorer.calculate_confidence(
            user_id, preferences
        )
        
        return {
            "user_id": user_id,
            "preferences": preferences,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Drift detection and adaptation endpoints
@adaptive_router.post("/users/{user_id}/drift/analyze")
async def analyze_drift(user_id: str, request: DriftAnalysisRequest):
    """Analyze preference drift for a user."""
    try:
        pipeline_instance = get_pipeline()
        
        drift_analysis = await pipeline_instance.drift_detector.analyze_user_drift(
            user_id=user_id,
            time_range=request.time_range
        )
        
        return {
            "user_id": user_id,
            "drift_analysis": drift_analysis,
            "analysis_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/users/{user_id}/adaptations/history")
async def get_adaptation_history(user_id: str, limit: int = 10):
    """Get adaptation history for a user."""
    try:
        pipeline_instance = get_pipeline()
        
        history = await pipeline_instance.preference_manager.get_adaptation_history(
            user_id=user_id,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "adaptation_history": history,
            "total_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting adaptation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.post("/users/{user_id}/adaptations/trigger")
async def trigger_manual_adaptation(user_id: str, background_tasks: BackgroundTasks):
    """Trigger manual adaptation for a user."""
    try:
        pipeline_instance = get_pipeline()
        
        # Add background task for adaptation
        background_tasks.add_task(
            pipeline_instance.adaptation_engine.manual_adapt,
            user_id
        )
        
        return {
            "status": "triggered",
            "user_id": user_id,
            "message": "Manual adaptation initiated"
        }
        
    except Exception as e:
        logger.error(f"Error triggering manual adaptation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Explanation endpoints
@adaptive_router.get("/users/{user_id}/explanations/adaptation")
async def get_adaptation_explanations(user_id: str):
    """Get explanations for recent adaptations."""
    try:
        pipeline_instance = get_pipeline()
        
        explanations = await pipeline_instance.adaptation_explainer.get_recent_explanations(
            user_id=user_id,
            limit=5
        )
        
        return {
            "user_id": user_id,
            "explanations": explanations
        }
        
    except Exception as e:
        logger.error(f"Error getting adaptation explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/users/{user_id}/visualizations/preferences")
async def get_preference_visualizations(user_id: str):
    """Get preference evolution visualizations."""
    try:
        pipeline_instance = get_pipeline()
        
        preferences = await pipeline_instance.preference_tracker.get_user_preferences(
            user_id
        )
        
        visualizations = await pipeline_instance.visualization_generator.generate_preference_viz(
            user_id=user_id,
            preferences=preferences
        )
        
        return {
            "user_id": user_id,
            "visualizations": visualizations
        }
        
    except Exception as e:
        logger.error(f"Error getting preference visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System monitoring endpoints
@adaptive_router.get("/system/status")
async def get_system_status():
    """Get system status and metrics."""
    try:
        pipeline_instance = get_pipeline()
        status = pipeline_instance.get_pipeline_status()
        
        return {
            "system_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@adaptive_router.get("/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics."""
    try:
        pipeline_instance = get_pipeline()
        status = pipeline_instance.get_pipeline_status()
        
        return {
            "metrics": status.get("metrics", {}),
            "component_status": status.get("components", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch operations
@adaptive_router.post("/batch/feedback")
async def submit_batch_feedback(feedback_batch: List[FeedbackRequest], 
                               background_tasks: BackgroundTasks):
    """Submit multiple feedback items in batch."""
    try:
        pipeline_instance = get_pipeline()
        
        # Process feedback in background
        background_tasks.add_task(
            _process_feedback_batch,
            pipeline_instance,
            feedback_batch
        )
        
        return {
            "status": "accepted",
            "batch_size": len(feedback_batch),
            "message": "Batch feedback processing initiated"
        }
        
    except Exception as e:
        logger.error(f"Error submitting batch feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_feedback_batch(pipeline_instance: AdaptiveLearningPipeline, 
                                 feedback_batch: List[FeedbackRequest]):
    """Process a batch of feedback items."""
    try:
        for feedback in feedback_batch:
            feedback_type_map = {
                "explicit": FeedbackType.EXPLICIT,
                "implicit": FeedbackType.IMPLICIT,
                "contextual": FeedbackType.CONTEXTUAL
            }
            
            feedback_type = feedback_type_map.get(feedback.feedback_type.lower())
            if feedback_type:
                await pipeline_instance.process_feedback(
                    user_id=feedback.user_id,
                    item_id=feedback.item_id,
                    feedback_type=feedback_type,
                    value=feedback.value,
                    context=feedback.context
                )
        
        logger.info(f"Processed batch of {len(feedback_batch)} feedback items")
        
    except Exception as e:
        logger.error(f"Error processing feedback batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


def get_adaptive_routers():
    """Get all adaptive learning routers"""
    return [adaptive_router, feedback_router]
