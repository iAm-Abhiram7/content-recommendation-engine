"""
Feedback Collector for Content Recommendation Engine

This module provides comprehensive feedback collection capabilities for user
interactions with recommendations and system adaptations.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from collections import defaultdict, deque

from ..utils.logging import setup_logger


from ..adaptive_learning.feedback_processor import FeedbackType


class FeedbackValence(Enum):
    """Valence of feedback"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class FeedbackContext(Enum):
    """Context in which feedback was given"""
    RECOMMENDATION_LIST = "recommendation_list"
    ITEM_DETAIL = "item_detail"
    POST_CONSUMPTION = "post_consumption"
    ADAPTATION_NOTIFICATION = "adaptation_notification"
    PREFERENCE_SETTINGS = "preference_settings"
    SEARCH_RESULTS = "search_results"
    DISCOVERY_INTERFACE = "discovery_interface"


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_id: str
    user_id: str
    item_id: Optional[str]
    feedback_type: FeedbackType
    value: Union[float, str, Dict[str, Any]]
    valence: FeedbackValence
    context: FeedbackContext
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    weight: float = 1.0


@dataclass
class FeedbackResponse:
    """Response to feedback collection"""
    success: bool
    feedback_id: str
    message: str
    suggestions: List[str] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)


@dataclass
class FeedbackSummary:
    """Summary of feedback for analysis"""
    user_id: str
    total_feedback_count: int
    feedback_by_type: Dict[str, int]
    feedback_by_valence: Dict[str, int]
    average_rating: float
    recent_trend: str  # 'improving', 'declining', 'stable'
    engagement_score: float
    last_feedback_time: datetime


class FeedbackCollector:
    """
    Comprehensive feedback collection system that captures and processes
    various types of user feedback for recommendation improvement
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Feedback storage
        self.feedback_store: Dict[str, List[FeedbackItem]] = defaultdict(list)
        self.recent_feedback: deque = deque(maxlen=1000)  # Last 1000 feedback items
        
        # Feedback processing callbacks
        self.feedback_processors: Dict[FeedbackType, List[Callable]] = defaultdict(list)
        
        # Configuration
        self.auto_process = self.config.get('auto_process', True)
        self.feedback_aggregation_window = self.config.get('aggregation_window', timedelta(hours=1))
        self.min_feedback_confidence = self.config.get('min_confidence', 0.3)
        
        # Feedback collection settings
        self.collection_strategies = {
            FeedbackType.EXPLICIT_RATING: self._collect_explicit_rating,
            FeedbackType.IMPLICIT_RATING: self._collect_implicit_rating,
            FeedbackType.EXPLANATION_FEEDBACK: self._collect_explanation_feedback,
            FeedbackType.ADAPTATION_FEEDBACK: self._collect_adaptation_feedback,
            FeedbackType.PREFERENCE_FEEDBACK: self._collect_preference_feedback,
            FeedbackType.NATURAL_LANGUAGE: self._collect_natural_language_feedback
        }
        
        # Feedback weights by type
        self.feedback_weights = {
            FeedbackType.EXPLICIT_RATING: 1.0,
            FeedbackType.IMPLICIT_RATING: 0.5,
            FeedbackType.EXPLANATION_FEEDBACK: 0.8,
            FeedbackType.ADAPTATION_FEEDBACK: 0.9,
            FeedbackType.PREFERENCE_FEEDBACK: 1.0,
            FeedbackType.NATURAL_LANGUAGE: 0.7
        }
        
        self.logger = setup_logger(__name__)
    
    async def collect_feedback(self, user_id: str, feedback_type: FeedbackType,
                             value: Union[float, str, Dict[str, Any]],
                             item_id: str = None,
                             context: FeedbackContext = FeedbackContext.RECOMMENDATION_LIST,
                             metadata: Dict[str, Any] = None) -> FeedbackResponse:
        """Collect feedback from user"""
        try:
            # Generate feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Determine valence
            valence = self._determine_valence(feedback_type, value)
            
            # Calculate confidence
            confidence = self._calculate_feedback_confidence(feedback_type, value, metadata or {})
            
            # Create feedback item
            feedback_item = FeedbackItem(
                feedback_id=feedback_id,
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                value=value,
                valence=valence,
                context=context,
                timestamp=datetime.now(),
                confidence=confidence,
                metadata=metadata or {},
                weight=self.feedback_weights.get(feedback_type, 1.0)
            )
            
            # Store feedback
            self.feedback_store[user_id].append(feedback_item)
            self.recent_feedback.append(feedback_item)
            
            # Process feedback if auto-processing is enabled
            if self.auto_process:
                await self._process_feedback(feedback_item)
            
            # Generate response
            response = await self._generate_feedback_response(feedback_item)
            
            self.logger.debug(f"Collected {feedback_type.value} feedback from user {user_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return FeedbackResponse(
                success=False,
                feedback_id="",
                message=f"Error collecting feedback: {str(e)}"
            )
    
    async def collect_explicit_rating(self, user_id: str, item_id: str, 
                                    rating: float, scale: str = "1-5",
                                    context: FeedbackContext = FeedbackContext.ITEM_DETAIL) -> FeedbackResponse:
        """Collect explicit rating feedback"""
        metadata = {'scale': scale, 'normalized_rating': self._normalize_rating(rating, scale)}
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            value=rating,
            item_id=item_id,
            context=context,
            metadata=metadata
        )
    
    async def collect_implicit_feedback(self, user_id: str, item_id: str,
                                      interaction_data: Dict[str, Any],
                                      context: FeedbackContext = FeedbackContext.RECOMMENDATION_LIST) -> FeedbackResponse:
        """Collect implicit feedback from user interactions"""
        # Calculate implicit rating from interaction data
        implicit_rating = self._calculate_implicit_rating(interaction_data)
        
        metadata = {
            'interaction_type': interaction_data.get('type', 'unknown'),
            'duration': interaction_data.get('duration', 0),
            'completion_rate': interaction_data.get('completion_rate', 0),
            'engagement_signals': interaction_data.get('engagement_signals', {})
        }
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.IMPLICIT_RATING,
            value=implicit_rating,
            item_id=item_id,
            context=context,
            metadata=metadata
        )
    
    async def collect_explanation_feedback(self, user_id: str, explanation_id: str,
                                         helpfulness: float, clarity: float,
                                         accuracy: float,
                                         context: FeedbackContext = FeedbackContext.RECOMMENDATION_LIST) -> FeedbackResponse:
        """Collect feedback on explanations"""
        feedback_value = {
            'helpfulness': helpfulness,
            'clarity': clarity,
            'accuracy': accuracy,
            'overall': (helpfulness + clarity + accuracy) / 3
        }
        
        metadata = {'explanation_id': explanation_id}
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.EXPLANATION_FEEDBACK,
            value=feedback_value,
            context=context,
            metadata=metadata
        )
    
    async def collect_adaptation_feedback(self, user_id: str, adaptation_id: str,
                                        satisfaction: float, usefulness: float,
                                        appropriateness: float,
                                        context: FeedbackContext = FeedbackContext.ADAPTATION_NOTIFICATION) -> FeedbackResponse:
        """Collect feedback on system adaptations"""
        feedback_value = {
            'satisfaction': satisfaction,
            'usefulness': usefulness,
            'appropriateness': appropriateness,
            'overall': (satisfaction + usefulness + appropriateness) / 3
        }
        
        metadata = {'adaptation_id': adaptation_id}
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.ADAPTATION_FEEDBACK,
            value=feedback_value,
            context=context,
            metadata=metadata
        )
    
    async def collect_preference_feedback(self, user_id: str, 
                                        preferences: Dict[str, float],
                                        context: FeedbackContext = FeedbackContext.PREFERENCE_SETTINGS) -> FeedbackResponse:
        """Collect direct preference feedback"""
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.PREFERENCE_FEEDBACK,
            value=preferences,
            context=context,
            metadata={'preference_count': len(preferences)}
        )
    
    async def collect_natural_language_feedback(self, user_id: str, text: str,
                                              sentiment: str = None,
                                              context: FeedbackContext = FeedbackContext.RECOMMENDATION_LIST) -> FeedbackResponse:
        """Collect natural language feedback"""
        # Analyze sentiment if not provided
        if sentiment is None:
            sentiment = self._analyze_sentiment(text)
        
        metadata = {
            'text_length': len(text),
            'sentiment': sentiment,
            'word_count': len(text.split())
        }
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.NATURAL_LANGUAGE,
            value=text,
            context=context,
            metadata=metadata
        )
    
    async def collect_comparative_feedback(self, user_id: str, item_a: str, item_b: str,
                                         preference: str, strength: float = 1.0,
                                         context: FeedbackContext = FeedbackContext.DISCOVERY_INTERFACE) -> FeedbackResponse:
        """Collect comparative feedback (A vs B)"""
        feedback_value = {
            'item_a': item_a,
            'item_b': item_b,
            'preferred': preference,
            'strength': strength
        }
        
        metadata = {'comparison_type': 'pairwise'}
        
        return await self.collect_feedback(
            user_id=user_id,
            feedback_type=FeedbackType.COMPARATIVE_FEEDBACK,
            value=feedback_value,
            context=context,
            metadata=metadata
        )
    
    def get_user_feedback_summary(self, user_id: str, 
                                 days: int = 30) -> FeedbackSummary:
        """Get feedback summary for a user"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_feedback = [
            f for f in self.feedback_store.get(user_id, [])
            if f.timestamp > cutoff_date
        ]
        
        if not user_feedback:
            return FeedbackSummary(
                user_id=user_id,
                total_feedback_count=0,
                feedback_by_type={},
                feedback_by_valence={},
                average_rating=0.0,
                recent_trend='stable',
                engagement_score=0.0,
                last_feedback_time=datetime.min
            )
        
        # Count by type
        type_counts = defaultdict(int)
        for feedback in user_feedback:
            type_counts[feedback.feedback_type.value] += 1
        
        # Count by valence
        valence_counts = defaultdict(int)
        for feedback in user_feedback:
            valence_counts[feedback.valence.value] += 1
        
        # Calculate average rating (for explicit ratings only)
        explicit_ratings = [
            f for f in user_feedback 
            if f.feedback_type == FeedbackType.EXPLICIT_RATING and isinstance(f.value, (int, float))
        ]
        
        average_rating = 0.0
        if explicit_ratings:
            total_rating = sum(f.value * f.weight for f in explicit_ratings)
            total_weight = sum(f.weight for f in explicit_ratings)
            average_rating = total_rating / total_weight if total_weight > 0 else 0.0
        
        # Calculate trend
        trend = self._calculate_feedback_trend(user_feedback)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(user_feedback, days)
        
        # Get last feedback time
        last_feedback_time = max(f.timestamp for f in user_feedback)
        
        return FeedbackSummary(
            user_id=user_id,
            total_feedback_count=len(user_feedback),
            feedback_by_type=dict(type_counts),
            feedback_by_valence=dict(valence_counts),
            average_rating=average_rating,
            recent_trend=trend,
            engagement_score=engagement_score,
            last_feedback_time=last_feedback_time
        )
    
    def get_feedback_insights(self, user_id: str = None, 
                            days: int = 7) -> Dict[str, Any]:
        """Get insights from feedback data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if user_id:
            feedback_items = [
                f for f in self.feedback_store.get(user_id, [])
                if f.timestamp > cutoff_date
            ]
        else:
            feedback_items = [
                f for f in self.recent_feedback
                if f.timestamp > cutoff_date
            ]
        
        if not feedback_items:
            return {}
        
        insights = {
            'total_feedback': len(feedback_items),
            'unique_users': len(set(f.user_id for f in feedback_items)),
            'most_common_type': self._get_most_common_feedback_type(feedback_items),
            'average_confidence': sum(f.confidence for f in feedback_items) / len(feedback_items),
            'positive_ratio': len([f for f in feedback_items if f.valence.value > 0]) / len(feedback_items),
            'engagement_patterns': self._analyze_engagement_patterns(feedback_items),
            'context_distribution': self._analyze_context_distribution(feedback_items)
        }
        
        return insights
    
    def add_feedback_processor(self, feedback_type: FeedbackType, 
                             processor: Callable[[FeedbackItem], None]):
        """Add a processor for specific feedback type"""
        self.feedback_processors[feedback_type].append(processor)
    
    async def _process_feedback(self, feedback_item: FeedbackItem):
        """Process feedback item"""
        try:
            # Mark as processed
            feedback_item.processed = True
            
            # Run type-specific processors
            for processor in self.feedback_processors.get(feedback_item.feedback_type, []):
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(feedback_item)
                    else:
                        processor(feedback_item)
                except Exception as e:
                    self.logger.error(f"Error in feedback processor: {e}")
            
            # Run general processors
            for processor in self.feedback_processors.get('all', []):
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(feedback_item)
                    else:
                        processor(feedback_item)
                except Exception as e:
                    self.logger.error(f"Error in general feedback processor: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
    
    def _determine_valence(self, feedback_type: FeedbackType, 
                          value: Union[float, str, Dict[str, Any]]) -> FeedbackValence:
        """Determine valence of feedback"""
        if feedback_type == FeedbackType.EXPLICIT_RATING:
            if isinstance(value, (int, float)):
                if value >= 0.8:
                    return FeedbackValence.VERY_POSITIVE
                elif value >= 0.6:
                    return FeedbackValence.POSITIVE
                elif value >= 0.4:
                    return FeedbackValence.NEUTRAL
                elif value >= 0.2:
                    return FeedbackValence.NEGATIVE
                else:
                    return FeedbackValence.VERY_NEGATIVE
        
        elif feedback_type == FeedbackType.NATURAL_LANGUAGE:
            sentiment = self._analyze_sentiment(str(value))
            if sentiment == 'very_positive':
                return FeedbackValence.VERY_POSITIVE
            elif sentiment == 'positive':
                return FeedbackValence.POSITIVE
            elif sentiment == 'negative':
                return FeedbackValence.NEGATIVE
            elif sentiment == 'very_negative':
                return FeedbackValence.VERY_NEGATIVE
            else:
                return FeedbackValence.NEUTRAL
        
        elif feedback_type in [FeedbackType.EXPLANATION_FEEDBACK, FeedbackType.ADAPTATION_FEEDBACK]:
            if isinstance(value, dict) and 'overall' in value:
                overall = value['overall']
                if overall >= 0.8:
                    return FeedbackValence.VERY_POSITIVE
                elif overall >= 0.6:
                    return FeedbackValence.POSITIVE
                elif overall >= 0.4:
                    return FeedbackValence.NEUTRAL
                elif overall >= 0.2:
                    return FeedbackValence.NEGATIVE
                else:
                    return FeedbackValence.VERY_NEGATIVE
        
        return FeedbackValence.NEUTRAL
    
    def _calculate_feedback_confidence(self, feedback_type: FeedbackType,
                                     value: Union[float, str, Dict[str, Any]],
                                     metadata: Dict[str, Any]) -> float:
        """Calculate confidence in the feedback"""
        base_confidence = 1.0
        
        # Adjust based on feedback type
        if feedback_type == FeedbackType.IMPLICIT_RATING:
            base_confidence = 0.6  # Lower confidence for implicit feedback
        elif feedback_type == FeedbackType.NATURAL_LANGUAGE:
            # Longer text generally more confident
            text_length = len(str(value))
            base_confidence = min(1.0, 0.5 + (text_length / 1000))
        
        # Adjust based on metadata
        if 'confidence_modifier' in metadata:
            base_confidence *= metadata['confidence_modifier']
        
        return max(0.1, min(1.0, base_confidence))
    
    def _normalize_rating(self, rating: float, scale: str) -> float:
        """Normalize rating to 0-1 scale"""
        if scale == "1-5":
            return (rating - 1) / 4
        elif scale == "1-10":
            return (rating - 1) / 9
        elif scale == "0-100":
            return rating / 100
        elif scale == "thumbs":
            return 1.0 if rating > 0 else 0.0
        else:
            # Assume already normalized
            return max(0.0, min(1.0, rating))
    
    def _calculate_implicit_rating(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate implicit rating from interaction data"""
        base_score = 0.5  # Neutral starting point
        
        # Completion rate
        completion_rate = interaction_data.get('completion_rate', 0)
        base_score += (completion_rate - 0.5) * 0.3
        
        # Duration relative to content length
        duration = interaction_data.get('duration', 0)
        expected_duration = interaction_data.get('expected_duration', duration)
        
        if expected_duration > 0:
            duration_ratio = min(2.0, duration / expected_duration)
            base_score += (duration_ratio - 1.0) * 0.2
        
        # Engagement signals
        engagement = interaction_data.get('engagement_signals', {})
        
        # Positive signals
        positive_signals = ['like', 'share', 'bookmark', 'comment', 'replay']
        for signal in positive_signals:
            if engagement.get(signal, False):
                base_score += 0.1
        
        # Negative signals
        negative_signals = ['skip', 'close_early', 'dislike', 'report']
        for signal in negative_signals:
            if engagement.get(signal, False):
                base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text (simplified implementation)"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count + 1:
            return 'very_positive' if positive_count > 2 else 'positive'
        elif negative_count > positive_count + 1:
            return 'very_negative' if negative_count > 2 else 'negative'
        else:
            return 'neutral'
    
    def _calculate_feedback_trend(self, feedback_items: List[FeedbackItem]) -> str:
        """Calculate trend in feedback over time"""
        if len(feedback_items) < 5:
            return 'stable'
        
        # Sort by timestamp
        sorted_feedback = sorted(feedback_items, key=lambda x: x.timestamp)
        
        # Split into early and recent halves
        mid_point = len(sorted_feedback) // 2
        early_half = sorted_feedback[:mid_point]
        recent_half = sorted_feedback[mid_point:]
        
        # Calculate average valence for each half
        early_avg = sum(f.valence.value for f in early_half) / len(early_half)
        recent_avg = sum(f.valence.value for f in recent_half) / len(recent_half)
        
        difference = recent_avg - early_avg
        
        if difference > 0.3:
            return 'improving'
        elif difference < -0.3:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_engagement_score(self, feedback_items: List[FeedbackItem], 
                                  days: int) -> float:
        """Calculate user engagement score based on feedback"""
        if not feedback_items:
            return 0.0
        
        # Frequency score (feedback per day)
        frequency_score = min(1.0, len(feedback_items) / days)
        
        # Diversity score (variety of feedback types)
        unique_types = len(set(f.feedback_type for f in feedback_items))
        max_types = len(FeedbackType)
        diversity_score = unique_types / max_types
        
        # Quality score (based on explicit feedback ratio)
        explicit_feedback = [f for f in feedback_items 
                           if f.feedback_type == FeedbackType.EXPLICIT_RATING]
        quality_score = min(1.0, len(explicit_feedback) / len(feedback_items) * 2)
        
        # Combined engagement score
        engagement_score = (frequency_score * 0.4 + diversity_score * 0.3 + quality_score * 0.3)
        
        return engagement_score
    
    def _get_most_common_feedback_type(self, feedback_items: List[FeedbackItem]) -> str:
        """Get most common feedback type"""
        type_counts = defaultdict(int)
        for feedback in feedback_items:
            type_counts[feedback.feedback_type.value] += 1
        
        if type_counts:
            return max(type_counts, key=type_counts.get)
        return 'none'
    
    def _analyze_engagement_patterns(self, feedback_items: List[FeedbackItem]) -> Dict[str, Any]:
        """Analyze engagement patterns in feedback"""
        if not feedback_items:
            return {}
        
        # Time distribution
        hours = [f.timestamp.hour for f in feedback_items]
        peak_hour = max(set(hours), key=hours.count)
        
        # Context distribution
        contexts = [f.context.value for f in feedback_items]
        most_common_context = max(set(contexts), key=contexts.count)
        
        return {
            'peak_feedback_hour': peak_hour,
            'most_common_context': most_common_context,
            'feedback_sessions': self._identify_feedback_sessions(feedback_items)
        }
    
    def _analyze_context_distribution(self, feedback_items: List[FeedbackItem]) -> Dict[str, int]:
        """Analyze distribution of feedback contexts"""
        context_counts = defaultdict(int)
        for feedback in feedback_items:
            context_counts[feedback.context.value] += 1
        
        return dict(context_counts)
    
    def _identify_feedback_sessions(self, feedback_items: List[FeedbackItem]) -> int:
        """Identify distinct feedback sessions"""
        if not feedback_items:
            return 0
        
        sorted_feedback = sorted(feedback_items, key=lambda x: x.timestamp)
        sessions = 1
        session_gap = timedelta(minutes=30)  # 30 minutes between sessions
        
        for i in range(1, len(sorted_feedback)):
            time_diff = sorted_feedback[i].timestamp - sorted_feedback[i-1].timestamp
            if time_diff > session_gap:
                sessions += 1
        
        return sessions
    
    async def _generate_feedback_response(self, feedback_item: FeedbackItem) -> FeedbackResponse:
        """Generate response to feedback"""
        success = True
        message = "Thank you for your feedback!"
        suggestions = []
        immediate_actions = []
        
        # Customize response based on feedback type and valence
        if feedback_item.feedback_type == FeedbackType.EXPLICIT_RATING:
            if feedback_item.valence.value >= 1:
                message = "Great! We're glad you enjoyed this recommendation."
                suggestions.append("Explore similar items in your recommendations")
            elif feedback_item.valence.value <= -1:
                message = "Thanks for letting us know. We'll improve your recommendations."
                immediate_actions.append("Reduce similar recommendations")
                suggestions.append("Try adjusting your preferences in settings")
        
        elif feedback_item.feedback_type == FeedbackType.ADAPTATION_FEEDBACK:
            if feedback_item.valence.value >= 1:
                message = "Excellent! The adaptation was helpful."
                immediate_actions.append("Continue similar adaptations")
            else:
                message = "We'll adjust our adaptation strategy based on your feedback."
                immediate_actions.append("Reduce adaptation aggressiveness")
        
        elif feedback_item.feedback_type == FeedbackType.NATURAL_LANGUAGE:
            message = "Thank you for the detailed feedback!"
            immediate_actions.append("Analyze text for specific improvements")
        
        return FeedbackResponse(
            success=success,
            feedback_id=feedback_item.feedback_id,
            message=message,
            suggestions=suggestions,
            immediate_actions=immediate_actions
        )
    
    def get_feedback_history(self, user_id: str, limit: int = 100, feedback_types: Optional[List[FeedbackType]] = None) -> List[Dict[str, Any]]:
        """
        Get feedback history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of feedback items to return
            feedback_types: Optional filter by feedback types
            
        Returns:
            List of feedback items for the user
        """
        try:
            user_feedback = []
            
            # Get all feedback for user
            for feedback_item in self.collected_feedback:
                if feedback_item.user_id == user_id:
                    if feedback_types is None or feedback_item.feedback_type in feedback_types:
                        user_feedback.append({
                            'feedback_id': feedback_item.feedback_id,
                            'item_id': feedback_item.item_id,
                            'feedback_type': feedback_item.feedback_type.value if hasattr(feedback_item.feedback_type, 'value') else str(feedback_item.feedback_type),
                            'valence': feedback_item.valence.value if hasattr(feedback_item.valence, 'value') else feedback_item.valence,
                            'value': feedback_item.value,
                            'context': feedback_item.context.value if hasattr(feedback_item.context, 'value') else str(feedback_item.context),
                            'metadata': feedback_item.metadata,
                            'timestamp': feedback_item.timestamp.isoformat() if isinstance(feedback_item.timestamp, datetime) else feedback_item.timestamp
                        })
            
            # Sort by timestamp (newest first) and limit
            user_feedback.sort(key=lambda x: x['timestamp'], reverse=True)
            return user_feedback[:limit]
            
        except Exception as e:
            logger.error(f"Error getting feedback history for user {user_id}: {e}")
            return []

    # Convenience methods for specific feedback collection strategies
    def _collect_explicit_rating(self, *args, **kwargs):
        """Collect explicit rating feedback"""
        return self.collect_explicit_rating(*args, **kwargs)
    
    def _collect_implicit_rating(self, *args, **kwargs):
        """Collect implicit rating feedback"""
        return self.collect_implicit_feedback(*args, **kwargs)
    
    def _collect_explanation_feedback(self, *args, **kwargs):
        """Collect explanation feedback"""
        return self.collect_explanation_feedback(*args, **kwargs)
    
    def _collect_adaptation_feedback(self, *args, **kwargs):
        """Collect adaptation feedback"""
        return self.collect_adaptation_feedback(*args, **kwargs)
    
    def _collect_preference_feedback(self, *args, **kwargs):
        """Collect preference feedback"""
        return self.collect_preference_feedback(*args, **kwargs)
    
    def _collect_natural_language_feedback(self, *args, **kwargs):
        """Collect natural language feedback"""
        return self.collect_natural_language_feedback(*args, **kwargs)
    
    def get_feedback_history(self, user_id: str, limit: int = 100, feedback_types: Optional[List[FeedbackType]] = None) -> List[Dict[str, Any]]:
        """Get feedback history for a user"""
        try:
            user_feedback = []
            for feedback_item in self.collected_feedback:
                if feedback_item.user_id == user_id:
                    if feedback_types is None or feedback_item.feedback_type in feedback_types:
                        user_feedback.append({
                            'feedback_id': feedback_item.feedback_id,
                            'item_id': feedback_item.item_id,
                            'feedback_type': feedback_item.feedback_type.value if hasattr(feedback_item.feedback_type, 'value') else str(feedback_item.feedback_type),
                            'valence': feedback_item.valence.value if hasattr(feedback_item.valence, 'value') else feedback_item.valence,
                            'value': feedback_item.value,
                            'context': feedback_item.context.value if hasattr(feedback_item.context, 'value') else str(feedback_item.context),
                            'metadata': feedback_item.metadata,
                            'timestamp': feedback_item.timestamp.isoformat() if isinstance(feedback_item.timestamp, datetime) else feedback_item.timestamp
                        })
            user_feedback.sort(key=lambda x: x['timestamp'], reverse=True)
            return user_feedback[:limit]
        except Exception as e:
            logger.error(f"Error getting feedback history for user {user_id}: {e}")
            return []
