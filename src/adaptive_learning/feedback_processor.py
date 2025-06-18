"""
Real-Time Feedback Collection and Processing System

This module implements comprehensive feedback capture mechanisms for adaptive learning:
- Explicit feedback processing (ratings, preferences, corrections)
- Implicit feedback processing (clicks, views, skips, searches)
- Contextual signal capture (session, device, temporal, location patterns)
- Multi-modal feedback integration and validation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import redis
from cachetools import TTLCache

from ..utils.redis_client import get_redis_client, RedisConfig

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback supported by the system"""
    # Explicit feedback types
    EXPLICIT = "explicit"  # General explicit feedback
    EXPLICIT_RATING = "explicit_rating"
    EXPLICIT_PREFERENCE = "explicit_preference"
    EXPLICIT_CORRECTION = "explicit_correction"
    
    # Implicit feedback types
    IMPLICIT = "implicit"  # General implicit feedback
    IMPLICIT_CLICK = "implicit_click"
    IMPLICIT_VIEW = "implicit_view"
    IMPLICIT_SKIP = "implicit_skip"
    IMPLICIT_SEARCH = "implicit_search"
    IMPLICIT_RATING = "implicit_rating"        # Watch time, completion rate
    
    # Contextual feedback types
    CONTEXTUAL = "contextual"  # General contextual feedback
    CONTEXTUAL_SESSION = "contextual_session"
    CONTEXTUAL_DEVICE = "contextual_device"
    CONTEXTUAL_FEEDBACK = "contextual_feedback"    # Context-specific feedback
    
    # Special feedback types
    EXPLANATION_FEEDBACK = "explanation_feedback"  # Feedback on explanations
    ADAPTATION_FEEDBACK = "adaptation_feedback"    # Feedback on adaptations
    PREFERENCE_FEEDBACK = "preference_feedback"    # Direct preference input
    COMPARATIVE_FEEDBACK = "comparative_feedback"   # A vs B comparisons
    NATURAL_LANGUAGE = "natural_language"          # Text feedback
    EMOTIONAL_FEEDBACK = "emotional_feedback"      # Emotion-based feedback
    CONTEXTUAL_TEMPORAL = "contextual_temporal"
    CONTEXTUAL_LOCATION = "contextual_location"


@dataclass
class FeedbackSignal:
    """Standardized feedback signal structure"""
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    value: Union[float, str, Dict[str, Any]]
    timestamp: datetime
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExplicitFeedback:
    """Explicit user feedback structure"""
    rating: Optional[float] = None
    like_dislike: Optional[bool] = None
    thumbs_up_down: Optional[bool] = None
    not_interested: bool = False
    already_seen: bool = False
    inappropriate: bool = False
    wishlist_add: bool = False
    wishlist_remove: bool = False
    preference_update: Optional[Dict[str, Any]] = None


@dataclass
class ImplicitFeedback:
    """Implicit user behavior feedback structure"""
    click_through: bool = False
    view_duration_seconds: float = 0.0
    completion_rate: float = 0.0
    skip_action: bool = False
    skip_position: float = 0.0
    search_query: Optional[str] = None
    search_refinement: bool = False
    interaction_sequence: Optional[List[str]] = None
    bounce_rate: float = 0.0
    scroll_depth: float = 0.0


@dataclass
class ContextualSignals:
    """Contextual signals for feedback interpretation"""
    session_duration: float = 0.0
    session_interaction_count: int = 0
    device_type: Optional[str] = None
    platform: Optional[str] = None
    time_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    location_context: Optional[str] = None
    social_context: Optional[str] = None
    previous_interactions: Optional[List[str]] = None
    user_mood_indicator: Optional[str] = None


class FeedbackProcessor:
    """
    Real-time feedback processing engine with comprehensive signal capture
    """
    
    def __init__(self,
                 redis_client: Optional[Any] = None,
                 buffer_size: int = 10000,
                 processing_interval: float = 1.0,
                 validation_threshold: float = 0.7,
                 context_window_hours: int = 24,
                 enable_real_time: bool = True):
        """
        Initialize feedback processor
        
        Args:
            redis_client: Redis client for caching and pub/sub
            buffer_size: Size of feedback buffer
            processing_interval: Interval for batch processing (seconds)
            validation_threshold: Threshold for feedback validation
            context_window_hours: Hours of context to maintain
            enable_real_time: Whether to enable real-time processing
        """
        # Initialize Redis client
        if redis_client is not None:
            self.redis_client = redis_client
        else:
            redis_conn = get_redis_client()
            self.redis_client = redis_conn.client if redis_conn and redis_conn.is_healthy() else None
        
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        self.validation_threshold = validation_threshold
        self.context_window_hours = context_window_hours
        self.enable_real_time = enable_real_time
        
        # Feedback buffers
        self.feedback_buffer = deque(maxlen=buffer_size)
        self.processed_feedback = []
        
        # Context tracking
        self.user_sessions = TTLCache(maxsize=100000, ttl=3600 * context_window_hours)
        self.user_context = TTLCache(maxsize=50000, ttl=3600 * 6)
        
        # Statistics and monitoring
        self.feedback_stats = defaultdict(int)
        self.processing_stats = {
            'total_processed': 0,
            'validation_failures': 0,
            'context_enrichments': 0,
            'real_time_processed': 0
        }
        
        # Threading for background processing
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if enable_real_time:
            self.start_background_processing()
    
    def start_background_processing(self):
        """Start background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._background_processor)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started background feedback processing")
    
    def stop_background_processing(self):
        """Stop background processing thread"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Stopped background feedback processing")
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Process feedback data - main entry point for feedback processing
        
        Args:
            feedback_data: Dictionary containing feedback information
            
        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            # Create feedback signal from data
            signal = FeedbackSignal(
                user_id=feedback_data.get('user_id', ''),
                item_id=feedback_data.get('item_id', ''),
                feedback_type=FeedbackType(feedback_data.get('feedback_type', 'implicit')),
                value=feedback_data.get('value', 0.0),
                timestamp=datetime.now(),
                session_id=feedback_data.get('session_id'),
                device_id=feedback_data.get('device_id'),
                context=feedback_data.get('context'),
                confidence=feedback_data.get('confidence', 1.0),
                metadata=feedback_data.get('metadata')
            )
            
            # Validate feedback
            if not self._validate_feedback(signal):
                return False
            
            # Add to buffer for processing
            self.feedback_buffer.append(signal)
            self.feedback_stats['total_received'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def _background_processor(self):
        """Background thread for processing feedback buffer"""
        while not self.stop_processing.is_set():
            try:
                if self.feedback_buffer:
                    batch_size = min(100, len(self.feedback_buffer))
                    batch = []
                    
                    for _ in range(batch_size):
                        if self.feedback_buffer:
                            batch.append(self.feedback_buffer.popleft())
                    
                    if batch:
                        self._process_feedback_batch(batch)
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                time.sleep(5)
    
    async def process_explicit_feedback(self,
                                      user_id: str,
                                      item_id: str,
                                      feedback: ExplicitFeedback,
                                      session_id: Optional[str] = None,
                                      context: Optional[Dict[str, Any]] = None) -> FeedbackSignal:
        """
        Process explicit user feedback
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            feedback: Explicit feedback data
            session_id: Session identifier
            context: Additional context
            
        Returns:
            Processed feedback signal
        """
        try:
            # Create feedback signal
            signal = FeedbackSignal(
                user_id=user_id,
                item_id=item_id,
                feedback_type=FeedbackType.EXPLICIT_RATING,
                value=asdict(feedback),
                timestamp=datetime.now(),
                session_id=session_id,
                context=context or {}
            )
            
            # Validate feedback
            if not self._validate_feedback(signal):
                logger.warning(f"Invalid explicit feedback from user {user_id}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            # Enrich with context
            signal = await self._enrich_with_context(signal)
            
            # Add to buffer for processing
            if self.enable_real_time:
                self.feedback_buffer.append(signal)
                self.processing_stats['real_time_processed'] += 1
            
            # Update statistics
            self.feedback_stats[FeedbackType.EXPLICIT_RATING] += 1
            
            # Cache for Redis pub/sub
            await self._publish_feedback_signal(signal)
            
            logger.debug(f"Processed explicit feedback: {user_id} -> {item_id}")
            return signal
            
        except Exception as e:
            logger.error(f"Error processing explicit feedback: {e}")
            return None
    
    async def process_implicit_feedback(self,
                                      user_id: str,
                                      item_id: str,
                                      feedback: ImplicitFeedback,
                                      session_id: Optional[str] = None,
                                      context: Optional[Dict[str, Any]] = None) -> FeedbackSignal:
        """
        Process implicit user behavior feedback
        
        Args:
            user_id: User identifier
            item_id: Item identifier  
            feedback: Implicit feedback data
            session_id: Session identifier
            context: Additional context
            
        Returns:
            Processed feedback signal
        """
        try:
            # Determine feedback type based on dominant signal
            feedback_type = self._classify_implicit_feedback(feedback)
            
            # Create feedback signal
            signal = FeedbackSignal(
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                value=asdict(feedback),
                timestamp=datetime.now(),
                session_id=session_id,
                context=context or {}
            )
            
            # Calculate confidence based on signal strength
            signal.confidence = self._calculate_implicit_confidence(feedback)
            
            # Validate feedback
            if not self._validate_feedback(signal):
                logger.warning(f"Invalid implicit feedback from user {user_id}")
                self.processing_stats['validation_failures'] += 1
                return None
            
            # Enrich with context
            signal = await self._enrich_with_context(signal)
            
            # Add to buffer for processing
            if self.enable_real_time:
                self.feedback_buffer.append(signal)
                self.processing_stats['real_time_processed'] += 1
            
            # Update statistics
            self.feedback_stats[feedback_type] += 1
            
            # Cache for Redis pub/sub
            await self._publish_feedback_signal(signal)
            
            logger.debug(f"Processed implicit feedback: {user_id} -> {item_id} ({feedback_type.value})")
            return signal
            
        except Exception as e:
            logger.error(f"Error processing implicit feedback: {e}")
            return None
    
    async def process_contextual_signals(self,
                                       user_id: str,
                                       signals: ContextualSignals,
                                       session_id: Optional[str] = None) -> List[FeedbackSignal]:
        """
        Process contextual signals for enhanced understanding
        
        Args:
            user_id: User identifier
            signals: Contextual signals
            session_id: Session identifier
            
        Returns:
            List of processed contextual feedback signals
        """
        try:
            processed_signals = []
            
            # Process each type of contextual signal
            signal_types = [
                (FeedbackType.CONTEXTUAL_SESSION, {
                    'session_duration': signals.session_duration,
                    'interaction_count': signals.session_interaction_count
                }),
                (FeedbackType.CONTEXTUAL_DEVICE, {
                    'device_type': signals.device_type,
                    'platform': signals.platform
                }),
                (FeedbackType.CONTEXTUAL_TEMPORAL, {
                    'time_of_day': signals.time_of_day,
                    'day_of_week': signals.day_of_week
                }),
                (FeedbackType.CONTEXTUAL_LOCATION, {
                    'location_context': signals.location_context,
                    'social_context': signals.social_context
                })
            ]
            
            for signal_type, signal_data in signal_types:
                if any(v is not None for v in signal_data.values()):
                    signal = FeedbackSignal(
                        user_id=user_id,
                        item_id="contextual",  # No specific item
                        feedback_type=signal_type,
                        value=signal_data,
                        timestamp=datetime.now(),
                        session_id=session_id,
                        confidence=0.8  # Contextual signals have moderate confidence
                    )
                    
                    processed_signals.append(signal)
                    
                    if self.enable_real_time:
                        self.feedback_buffer.append(signal)
                    
                    self.feedback_stats[signal_type] += 1
            
            # Update user context cache
            self.user_context[user_id] = asdict(signals)
            
            logger.debug(f"Processed {len(processed_signals)} contextual signals for user {user_id}")
            return processed_signals
            
        except Exception as e:
            logger.error(f"Error processing contextual signals: {e}")
            return []
    
    def _classify_implicit_feedback(self, feedback: ImplicitFeedback) -> FeedbackType:
        """Classify implicit feedback based on dominant signal"""
        if feedback.click_through:
            return FeedbackType.IMPLICIT_CLICK
        elif feedback.view_duration_seconds > 0:
            return FeedbackType.IMPLICIT_VIEW
        elif feedback.skip_action:
            return FeedbackType.IMPLICIT_SKIP
        elif feedback.search_query:
            return FeedbackType.IMPLICIT_SEARCH
        else:
            return FeedbackType.IMPLICIT_VIEW  # Default
    
    def _calculate_implicit_confidence(self, feedback: ImplicitFeedback) -> float:
        """Calculate confidence score for implicit feedback"""
        confidence_factors = []
        
        # View duration confidence
        if feedback.view_duration_seconds > 0:
            # Higher confidence for longer views
            view_confidence = min(1.0, feedback.view_duration_seconds / 300)  # 5 min = full confidence
            confidence_factors.append(view_confidence)
        
        # Completion rate confidence
        if feedback.completion_rate > 0:
            confidence_factors.append(feedback.completion_rate)
        
        # Click-through confidence
        if feedback.click_through:
            confidence_factors.append(0.9)
        
        # Skip penalty
        if feedback.skip_action:
            skip_confidence = max(0.1, 1.0 - feedback.skip_position)
            confidence_factors.append(skip_confidence)
        
        return float(np.mean(confidence_factors)) if confidence_factors else 0.5
    
    def _validate_feedback(self, signal: FeedbackSignal) -> bool:
        """Validate feedback signal quality and consistency"""
        try:
            # Basic validation
            if not signal.user_id or not signal.item_id:
                return False
            
            # Timestamp validation
            now = datetime.now()
            if signal.timestamp > now + timedelta(minutes=5):  # Future timestamp
                return False
            
            if signal.timestamp < now - timedelta(days=30):  # Too old
                return False
            
            # Confidence validation
            if signal.confidence < 0.0 or signal.confidence > 1.0:
                return False
            
            # Type-specific validation
            if signal.feedback_type in [FeedbackType.EXPLICIT_RATING]:
                return self._validate_explicit_feedback(signal)
            elif signal.feedback_type in [FeedbackType.IMPLICIT_CLICK, FeedbackType.IMPLICIT_VIEW]:
                return self._validate_implicit_feedback(signal)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating feedback: {e}")
            return False
    
    def _validate_explicit_feedback(self, signal: FeedbackSignal) -> bool:
        """Validate explicit feedback"""
        try:
            value = signal.value
            if isinstance(value, dict):
                # Check rating range
                if 'rating' in value and value['rating'] is not None:
                    if not (0.0 <= value['rating'] <= 5.0):
                        return False
                return True
            return False
        except Exception:
            return False
    
    def _validate_implicit_feedback(self, signal: FeedbackSignal) -> bool:
        """Validate implicit feedback"""
        try:
            value = signal.value
            if isinstance(value, dict):
                # Check view duration is reasonable
                if 'view_duration_seconds' in value:
                    if value['view_duration_seconds'] < 0 or value['view_duration_seconds'] > 86400:  # 24 hours max
                        return False
                
                # Check completion rate
                if 'completion_rate' in value:
                    if not (0.0 <= value['completion_rate'] <= 1.0):
                        return False
                
                return True
            return False
        except Exception:
            return False
    
    async def _enrich_with_context(self, signal: FeedbackSignal) -> FeedbackSignal:
        """Enrich feedback signal with contextual information"""
        try:
            # Get user context
            user_context = self.user_context.get(signal.user_id, {})
            
            # Get session context
            session_context = {}
            if signal.session_id:
                session_key = f"{signal.user_id}:{signal.session_id}"
                session_context = self.user_sessions.get(session_key, {})
            
            # Merge contexts
            enriched_context = {
                **signal.context,
                **user_context,
                **session_context
            }
            
            signal.context = enriched_context
            signal.metadata = signal.metadata or {}
            signal.metadata['enriched_at'] = datetime.now().isoformat()
            
            self.processing_stats['context_enrichments'] += 1
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enriching context: {e}")
            return signal
    
    async def _publish_feedback_signal(self, signal: FeedbackSignal):
        """Publish feedback signal to Redis for real-time consumption"""
        try:
            signal_data = {
                'user_id': signal.user_id,
                'item_id': signal.item_id,
                'feedback_type': signal.feedback_type.value,
                'value': signal.value,
                'timestamp': signal.timestamp.isoformat(),
                'confidence': signal.confidence,
                'context': signal.context or {}
            }
            
            # Publish to Redis channel
            channel = f"feedback:{signal.feedback_type.value}"
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.redis_client.publish,
                channel,
                json.dumps(signal_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error publishing feedback signal: {e}")
    
    def _process_feedback_batch(self, batch: List[FeedbackSignal]):
        """Process a batch of feedback signals"""
        try:
            # Group by user for efficient processing
            user_batches = defaultdict(list)
            for signal in batch:
                user_batches[signal.user_id].append(signal)
            
            # Process each user's batch
            for user_id, user_signals in user_batches.items():
                self._process_user_feedback_batch(user_id, user_signals)
            
            self.processing_stats['total_processed'] += len(batch)
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
    
    def _process_user_feedback_batch(self, user_id: str, signals: List[FeedbackSignal]):
        """Process feedback batch for a specific user"""
        try:
            # Sort by timestamp
            signals.sort(key=lambda x: x.timestamp)
            
            # Update user session tracking
            for signal in signals:
                if signal.session_id:
                    session_key = f"{user_id}:{signal.session_id}"
                    session_data = self.user_sessions.get(session_key, {
                        'start_time': signal.timestamp,
                        'interactions': []
                    })
                    
                    session_data['interactions'].append({
                        'item_id': signal.item_id,
                        'feedback_type': signal.feedback_type.value,
                        'timestamp': signal.timestamp.isoformat(),
                        'confidence': signal.confidence
                    })
                    
                    session_data['last_interaction'] = signal.timestamp
                    self.user_sessions[session_key] = session_data
            
            # Store processed feedback
            self.processed_feedback.extend(signals)
            
            # Keep only recent processed feedback
            cutoff_time = datetime.now() - timedelta(hours=self.context_window_hours)
            self.processed_feedback = [
                s for s in self.processed_feedback 
                if s.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error processing user feedback batch: {e}")
    
    async def get_user_feedback_history(self,
                                      user_id: str,
                                      hours_back: int = 24,
                                      feedback_types: Optional[List[FeedbackType]] = None) -> List[FeedbackSignal]:
        """
        Get user's recent feedback history
        
        Args:
            user_id: User identifier
            hours_back: Hours of history to retrieve
            feedback_types: Filter by specific feedback types
            
        Returns:
            List of feedback signals
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter feedback by user and time
            user_feedback = [
                signal for signal in self.processed_feedback
                if signal.user_id == user_id and signal.timestamp >= cutoff_time
            ]
            
            # Filter by feedback types if specified
            if feedback_types:
                user_feedback = [
                    signal for signal in user_feedback
                    if signal.feedback_type in feedback_types
                ]
            
            # Sort by timestamp (most recent first)
            user_feedback.sort(key=lambda x: x.timestamp, reverse=True)
            
            return user_feedback
            
        except Exception as e:
            logger.error(f"Error getting user feedback history: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get feedback processing statistics"""
        return {
            'processing_stats': dict(self.processing_stats),
            'feedback_stats': dict(self.feedback_stats),
            'buffer_size': len(self.feedback_buffer),
            'processed_count': len(self.processed_feedback),
            'user_sessions_count': len(self.user_sessions),
            'user_context_count': len(self.user_context)
        }
    
    def reset_user_feedback(self, user_id: str):
        """Reset feedback history for a user"""
        try:
            # Remove from processed feedback
            self.processed_feedback = [
                signal for signal in self.processed_feedback
                if signal.user_id != user_id
            ]
            
            # Remove from user context
            if user_id in self.user_context:
                del self.user_context[user_id]
            
            # Remove sessions
            sessions_to_remove = [
                key for key in self.user_sessions.keys()
                if key.startswith(f"{user_id}:")
            ]
            for session_key in sessions_to_remove:
                del self.user_sessions[session_key]
            
            logger.info(f"Reset feedback for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error resetting user feedback: {e}")
    
    def flush_buffer(self) -> List[Dict[str, Any]]:
        """
        Flush the feedback buffer and return all pending feedback
        
        Returns:
            List of feedback items that were in the buffer
        """
        try:
            flushed_items = []
            while self.feedback_buffer:
                item = self.feedback_buffer.popleft()
                if hasattr(item, '__dict__'):
                    flushed_items.append(asdict(item))
                else:
                    flushed_items.append(item)
            
            logger.info(f"Flushed {len(flushed_items)} items from feedback buffer")
            return flushed_items
            
        except Exception as e:
            logger.error(f"Error flushing feedback buffer: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the feedback processor
        
        Returns:
            Dictionary containing processor status information
        """
        try:
            return {
                'status': 'active' if self.enable_real_time else 'inactive',
                'buffer_size': len(self.feedback_buffer),
                'processed_count': len(self.processed_feedback),
                'processing_stats': dict(self.processing_stats),
                'feedback_stats': dict(self.feedback_stats),
                'user_sessions_count': len(self.user_sessions),
                'user_context_count': len(self.user_context),
                'background_processing': self.processing_thread.is_alive() if self.processing_thread else False,
                'last_processing_time': getattr(self, '_last_processing_time', None)
            }
        except Exception as e:
            logger.error(f"Error getting feedback processor status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    # ...existing code...
