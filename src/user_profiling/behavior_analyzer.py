"""
Behavior Analyzer

Analyzes user behavior patterns and engagement metrics:
- Session analysis and engagement patterns
- Content consumption behavior
- Rating behavior analysis
- Interaction patterns and frequency
- Anomaly detection in behavior
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..utils.config import settings
from ..utils.validation import DataValidator
from ..data_integration.schema_manager import UserProfile as User, ContentMetadata as Content, InteractionEvents as Interaction, get_session

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Metrics for a user session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    interaction_count: int
    unique_content_count: int
    avg_rating: float
    content_types: Set[str]
    genres_explored: Set[str]
    session_quality: float  # 0-1 score


@dataclass
class BehaviorPattern:
    """Identified behavior pattern"""
    pattern_id: str
    pattern_type: str  # 'binge', 'explorer', 'critic', 'casual', etc.
    confidence: float
    frequency: float  # How often this pattern occurs
    characteristics: Dict[str, Any]
    examples: List[str]  # Example session/interaction IDs


@dataclass
class BehaviorProfile:
    """Complete behavior profile for a user"""
    user_id: str
    session_metrics: List[SessionMetrics] = field(default_factory=list)
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    engagement_score: float = 0.0
    consistency_score: float = 0.0
    activity_level: str = 'unknown'  # 'low', 'medium', 'high'
    behavioral_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class BehaviorAnalyzer:
    """
    Advanced user behavior analysis system
    """
    
    def __init__(self, config=None):
        self.session = get_session()
        self.behavior_cache: Dict[str, BehaviorProfile] = {}
        
        # Analysis parameters
        self.session_timeout_minutes = 30  # Session ends after 30min inactivity
        self.min_session_interactions = 2
        self.anomaly_detection_window_days = 30
        
    def analyze_user_behavior(self, user_id: str, force_refresh: bool = False) -> BehaviorProfile:
        """
        Analyze comprehensive user behavior patterns
        
        Args:
            user_id: User identifier
            force_refresh: Force rebuild even if cached
            
        Returns:
            Complete behavior profile
        """
        try:
            # Check cache first
            if not force_refresh and user_id in self.behavior_cache:
                cached_profile = self.behavior_cache[user_id]
                if (datetime.now() - cached_profile.last_updated).days < 1:
                    return cached_profile
            
            logger.info(f"Analyzing behavior for user {user_id}")
            
            # Get user interactions
            interactions = self._get_user_interactions(user_id)
            if not interactions:
                logger.warning(f"No interactions found for user {user_id}")
                return BehaviorProfile(user_id=user_id)
            
            # Build behavior profile
            profile = BehaviorProfile(user_id=user_id)
            
            # Extract sessions
            profile.session_metrics = self._extract_sessions(interactions)
            
            # Identify behavior patterns
            profile.behavior_patterns = self._identify_behavior_patterns(profile.session_metrics, interactions)
            
            # Calculate engagement metrics
            profile.engagement_score = self._calculate_engagement_score(profile.session_metrics, interactions)
            
            # Calculate consistency
            profile.consistency_score = self._calculate_consistency_score(profile.session_metrics)
            
            # Determine activity level
            profile.activity_level = self._determine_activity_level(profile.session_metrics)
            
            # Detect anomalies
            profile.behavioral_anomalies = self._detect_behavioral_anomalies(profile.session_metrics, interactions)
            
            # Cache the profile
            self.behavior_cache[user_id] = profile
            
            logger.info(f"Analyzed behavior for user {user_id} with {len(profile.session_metrics)} sessions")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing behavior for user {user_id}: {str(e)}")
            return BehaviorProfile(user_id=user_id)
    
    def _get_user_interactions(self, user_id: str) -> List[Dict]:
        """Get all interactions for a user with content details"""
        try:
            query = text("""
                SELECT i.*, c.title, c.genres, c.content_type, c.metadata
                FROM interactions i
                JOIN content c ON i.content_id = c.content_id
                WHERE i.user_id = :user_id
                ORDER BY i.timestamp ASC
            """)
            
            result = self.session.execute(query, {'user_id': user_id}).fetchall()
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Error fetching interactions for user {user_id}: {str(e)}")
            return []
    
    def _extract_sessions(self, interactions: List[Dict]) -> List[SessionMetrics]:
        """Extract user sessions from interactions"""
        if not interactions:
            return []
        
        sessions = []
        current_session = []
        session_id = 0
        
        for i, interaction in enumerate(interactions):
            timestamp = interaction.get('timestamp', datetime.now())
            
            # Check if this interaction belongs to current session
            if current_session:
                last_timestamp = current_session[-1].get('timestamp', datetime.now())
                time_diff = (timestamp - last_timestamp).total_seconds() / 60  # minutes
                
                if time_diff > self.session_timeout_minutes:
                    # End current session and start new one
                    if len(current_session) >= self.min_session_interactions:
                        session_metrics = self._calculate_session_metrics(current_session, session_id)
                        if session_metrics:
                            sessions.append(session_metrics)
                    
                    current_session = [interaction]
                    session_id += 1
                else:
                    current_session.append(interaction)
            else:
                current_session.append(interaction)
        
        # Handle last session
        if len(current_session) >= self.min_session_interactions:
            session_metrics = self._calculate_session_metrics(current_session, session_id)
            if session_metrics:
                sessions.append(session_metrics)
        
        return sessions
    
    def _calculate_session_metrics(self, session_interactions: List[Dict], session_id: int) -> Optional[SessionMetrics]:
        """Calculate metrics for a single session"""
        if not session_interactions:
            return None
        
        try:
            # Basic metrics
            start_time = min(i.get('timestamp', datetime.now()) for i in session_interactions)
            end_time = max(i.get('timestamp', datetime.now()) for i in session_interactions)
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Content metrics
            unique_content = set(i.get('content_id') for i in session_interactions if i.get('content_id'))
            content_types = set(i.get('content_type', 'unknown') for i in session_interactions)
            
            # Genre analysis
            all_genres = set()
            for interaction in session_interactions:
                if interaction.get('genres'):
                    genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                    all_genres.update(g.strip() for g in genres if g.strip())
            
            # Rating analysis
            ratings = [i.get('rating', 0) for i in session_interactions if i.get('rating', 0) > 0]
            avg_rating = np.mean(ratings) if ratings else 0.0
            
            # Session quality (engagement indicator)
            session_quality = self._calculate_session_quality(session_interactions, duration_minutes)
            
            return SessionMetrics(
                session_id=f"{session_interactions[0].get('user_id', 'unknown')}_{session_id}",
                user_id=session_interactions[0].get('user_id', 'unknown'),
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration_minutes,
                interaction_count=len(session_interactions),
                unique_content_count=len(unique_content),
                avg_rating=avg_rating,
                content_types=content_types,
                genres_explored=all_genres,
                session_quality=session_quality
            )
            
        except Exception as e:
            logger.error(f"Error calculating session metrics: {str(e)}")
            return None
    
    def _calculate_session_quality(self, interactions: List[Dict], duration_minutes: float) -> float:
        """Calculate session quality score (0-1)"""
        try:
            # Factors that indicate quality
            factors = []
            
            # 1. Interaction density (interactions per minute)
            if duration_minutes > 0:
                density = len(interactions) / duration_minutes
                density_score = min(1.0, density / 2.0)  # Normalize assuming 2/min is high
                factors.append(density_score)
            
            # 2. Content diversity
            unique_content = len(set(i.get('content_id') for i in interactions if i.get('content_id')))
            diversity_score = min(1.0, unique_content / len(interactions))
            factors.append(diversity_score)
            
            # 3. Rating engagement
            ratings = [i.get('rating', 0) for i in interactions if i.get('rating', 0) > 0]
            rating_engagement = len(ratings) / len(interactions) if interactions else 0
            factors.append(rating_engagement)
            
            # 4. Session length (moderate length is better)
            if duration_minutes > 0:
                # Optimal around 20-60 minutes
                if 20 <= duration_minutes <= 60:
                    length_score = 1.0
                elif duration_minutes < 20:
                    length_score = duration_minutes / 20
                else:
                    length_score = max(0.2, 1.0 - (duration_minutes - 60) / 120)
                factors.append(length_score)
            
            return float(np.mean(factors)) if factors else 0.0
            
        except Exception:
            return 0.0
    
    def _identify_behavior_patterns(self, sessions: List[SessionMetrics], 
                                  interactions: List[Dict]) -> List[BehaviorPattern]:
        """Identify user behavior patterns"""
        patterns = []
        
        if not sessions:
            return patterns
        
        try:
            # 1. Binge pattern - long sessions with many interactions
            binge_pattern = self._detect_binge_pattern(sessions)
            if binge_pattern:
                patterns.append(binge_pattern)
            
            # 2. Explorer pattern - diverse content consumption
            explorer_pattern = self._detect_explorer_pattern(sessions)
            if explorer_pattern:
                patterns.append(explorer_pattern)
            
            # 3. Critic pattern - high rating activity with detailed feedback
            critic_pattern = self._detect_critic_pattern(sessions, interactions)
            if critic_pattern:
                patterns.append(critic_pattern)
            
            # 4. Casual pattern - short, infrequent sessions
            casual_pattern = self._detect_casual_pattern(sessions)
            if casual_pattern:
                patterns.append(casual_pattern)
            
            # 5. Scheduled pattern - regular timing patterns
            scheduled_pattern = self._detect_scheduled_pattern(sessions)
            if scheduled_pattern:
                patterns.append(scheduled_pattern)
            
        except Exception as e:
            logger.error(f"Error identifying behavior patterns: {str(e)}")
        
        return patterns
    
    def _detect_binge_pattern(self, sessions: List[SessionMetrics]) -> Optional[BehaviorPattern]:
        """Detect binge watching/reading patterns"""
        if not sessions:
            return None
        
        # Criteria for binge pattern
        long_sessions = [s for s in sessions if s.duration_minutes >= 60]
        high_interaction_sessions = [s for s in sessions if s.interaction_count >= 10]
        
        binge_sessions = list(set(long_sessions) & set(high_interaction_sessions))
        
        if len(binge_sessions) >= max(2, len(sessions) * 0.2):  # At least 20% or 2 sessions
            frequency = len(binge_sessions) / len(sessions)
            confidence = min(1.0, frequency * 2)  # Higher frequency = higher confidence
            
            avg_duration = np.mean([s.duration_minutes for s in binge_sessions])
            avg_interactions = np.mean([s.interaction_count for s in binge_sessions])
            
            return BehaviorPattern(
                pattern_id="binge",
                pattern_type="binge",
                confidence=confidence,
                frequency=frequency,
                characteristics={
                    'avg_session_duration_minutes': avg_duration,
                    'avg_interactions_per_session': avg_interactions,
                    'binge_session_count': len(binge_sessions),
                    'total_sessions': len(sessions)
                },
                examples=[s.session_id for s in binge_sessions[:3]]
            )
        
        return None
    
    def _detect_explorer_pattern(self, sessions: List[SessionMetrics]) -> Optional[BehaviorPattern]:
        """Detect content exploration patterns"""
        if not sessions:
            return None
        
        # Calculate diversity metrics
        total_content_types = set()
        total_genres = set()
        diverse_sessions = []
        
        for session in sessions:
            total_content_types.update(session.content_types)
            total_genres.update(session.genres_explored)
            
            # Session is diverse if it has multiple content types or many genres
            if len(session.content_types) > 1 or len(session.genres_explored) >= 3:
                diverse_sessions.append(session)
        
        if len(diverse_sessions) >= max(2, len(sessions) * 0.3):
            frequency = len(diverse_sessions) / len(sessions)
            confidence = min(1.0, frequency * 1.5)
            
            avg_genres_per_session = np.mean([len(s.genres_explored) for s in diverse_sessions])
            avg_content_types = np.mean([len(s.content_types) for s in diverse_sessions])
            
            return BehaviorPattern(
                pattern_id="explorer",
                pattern_type="explorer",
                confidence=confidence,
                frequency=frequency,
                characteristics={
                    'total_content_types_explored': len(total_content_types),
                    'total_genres_explored': len(total_genres),
                    'avg_genres_per_session': avg_genres_per_session,
                    'avg_content_types_per_session': avg_content_types,
                    'diverse_session_count': len(diverse_sessions)
                },
                examples=[s.session_id for s in diverse_sessions[:3]]
            )
        
        return None
    
    def _detect_critic_pattern(self, sessions: List[SessionMetrics], 
                             interactions: List[Dict]) -> Optional[BehaviorPattern]:
        """Detect critical/rating-focused behavior"""
        if not sessions or not interactions:
            return None
        
        # Calculate rating activity
        rated_interactions = [i for i in interactions if i.get('rating', 0) > 0]
        rating_frequency = len(rated_interactions) / len(interactions) if interactions else 0
        
        # Look for detailed rating patterns
        rating_sessions = []
        for session in sessions:
            # Sessions with high rating activity
            if session.avg_rating > 0:  # Has ratings
                session_interactions = [i for i in interactions 
                                      if session.start_time <= i.get('timestamp', datetime.now()) <= session.end_time]
                session_rated = [i for i in session_interactions if i.get('rating', 0) > 0]
                
                if len(session_rated) / len(session_interactions) >= 0.5:  # 50%+ rated
                    rating_sessions.append(session)
        
        if rating_frequency >= 0.4 and len(rating_sessions) >= max(2, len(sessions) * 0.3):
            # Calculate rating statistics
            ratings = [i.get('rating', 0) for i in rated_interactions]
            rating_std = np.std(ratings) if len(ratings) > 1 else 0
            
            frequency = len(rating_sessions) / len(sessions)
            confidence = min(1.0, rating_frequency * 2)
            
            return BehaviorPattern(
                pattern_id="critic",
                pattern_type="critic",
                confidence=confidence,
                frequency=frequency,
                characteristics={
                    'rating_frequency': rating_frequency,
                    'avg_rating': np.mean(ratings),
                    'rating_std': rating_std,
                    'total_ratings': len(rated_interactions),
                    'critic_sessions': len(rating_sessions)
                },
                examples=[s.session_id for s in rating_sessions[:3]]
            )
        
        return None
    
    def _detect_casual_pattern(self, sessions: List[SessionMetrics]) -> Optional[BehaviorPattern]:
        """Detect casual usage patterns"""
        if not sessions:
            return None
        
        # Criteria for casual pattern
        short_sessions = [s for s in sessions if s.duration_minutes <= 15]
        low_interaction_sessions = [s for s in sessions if s.interaction_count <= 5]
        
        casual_sessions = list(set(short_sessions) & set(low_interaction_sessions))
        
        if len(casual_sessions) >= max(2, len(sessions) * 0.4):  # At least 40% or 2 sessions
            frequency = len(casual_sessions) / len(sessions)
            confidence = min(1.0, frequency * 1.5)
            
            avg_duration = np.mean([s.duration_minutes for s in casual_sessions])
            avg_interactions = np.mean([s.interaction_count for s in casual_sessions])
            
            return BehaviorPattern(
                pattern_id="casual",
                pattern_type="casual",
                confidence=confidence,
                frequency=frequency,
                characteristics={
                    'avg_session_duration_minutes': avg_duration,
                    'avg_interactions_per_session': avg_interactions,
                    'casual_session_count': len(casual_sessions),
                    'session_frequency': self._calculate_session_frequency(sessions)
                },
                examples=[s.session_id for s in casual_sessions[:3]]
            )
        
        return None
    
    def _detect_scheduled_pattern(self, sessions: List[SessionMetrics]) -> Optional[BehaviorPattern]:
        """Detect regular timing patterns"""
        if len(sessions) < 5:
            return None
        
        # Extract timing features
        hours = [s.start_time.hour for s in sessions]
        days_of_week = [s.start_time.weekday() for s in sessions]
        
        # Check for consistency in timing
        hour_std = np.std(hours)
        day_variance = len(set(days_of_week)) / 7  # 0-1 scale
        
        # Regular if low hour variance and/or concentrated days
        is_regular = hour_std < 3 or day_variance < 0.5
        
        if is_regular:
            # Calculate frequency of most common patterns
            hour_counter = Counter(hours)
            day_counter = Counter(days_of_week)
            
            most_common_hour = hour_counter.most_common(1)[0][0]
            most_common_day = day_counter.most_common(1)[0][0]
            
            hour_frequency = hour_counter[most_common_hour] / len(sessions)
            day_frequency = day_counter[most_common_day] / len(sessions)
            
            frequency = max(hour_frequency, day_frequency)
            confidence = min(1.0, frequency * 2)
            
            return BehaviorPattern(
                pattern_id="scheduled",
                pattern_type="scheduled",
                confidence=confidence,
                frequency=frequency,
                characteristics={
                    'preferred_hour': most_common_hour,
                    'preferred_day': most_common_day,
                    'hour_consistency': 1.0 - (hour_std / 12),  # Normalized
                    'day_consistency': 1.0 - day_variance,
                    'timing_predictability': confidence
                },
                examples=[s.session_id for s in sessions 
                         if s.start_time.hour == most_common_hour][:3]
            )
        
        return None
    
    def _calculate_session_frequency(self, sessions: List[SessionMetrics]) -> float:
        """Calculate how frequently user has sessions"""
        if len(sessions) < 2:
            return 0.0
        
        # Calculate time span
        start_date = min(s.start_time for s in sessions).date()
        end_date = max(s.end_time for s in sessions).date()
        days_span = (end_date - start_date).days + 1
        
        return len(sessions) / days_span if days_span > 0 else 0.0
    
    def _calculate_engagement_score(self, sessions: List[SessionMetrics], 
                                  interactions: List[Dict]) -> float:
        """Calculate overall user engagement score"""
        if not sessions or not interactions:
            return 0.0
        
        factors = []
        
        try:
            # 1. Session quality
            avg_session_quality = np.mean([s.session_quality for s in sessions])
            factors.append(avg_session_quality)
            
            # 2. Session frequency
            session_frequency = self._calculate_session_frequency(sessions)
            frequency_score = min(1.0, session_frequency * 7)  # Normalize to weekly
            factors.append(frequency_score)
            
            # 3. Content diversity
            all_content = set(i.get('content_id') for i in interactions if i.get('content_id'))
            diversity_score = min(1.0, len(all_content) / len(interactions))
            factors.append(diversity_score)
            
            # 4. Rating engagement
            rated_interactions = [i for i in interactions if i.get('rating', 0) > 0]
            rating_engagement = len(rated_interactions) / len(interactions)
            factors.append(rating_engagement)
            
            # 5. Consistency (regular usage)
            time_gaps = []
            for i in range(1, len(sessions)):
                gap = (sessions[i].start_time - sessions[i-1].end_time).days
                time_gaps.append(gap)
            
            if time_gaps:
                gap_consistency = 1.0 / (1.0 + np.std(time_gaps) / 7)  # Penalize irregular gaps
                factors.append(gap_consistency)
            
            return float(np.mean(factors))
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {str(e)}")
            return 0.0
    
    def _calculate_consistency_score(self, sessions: List[SessionMetrics]) -> float:
        """Calculate behavioral consistency score"""
        if len(sessions) < 3:
            return 0.5  # Neutral for insufficient data
        
        try:
            factors = []
            
            # 1. Session duration consistency
            durations = [s.duration_minutes for s in sessions]
            duration_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 1
            duration_consistency = max(0.0, 1.0 - duration_cv)
            factors.append(duration_consistency)
            
            # 2. Interaction count consistency
            interactions = [s.interaction_count for s in sessions]
            interaction_cv = np.std(interactions) / np.mean(interactions) if np.mean(interactions) > 0 else 1
            interaction_consistency = max(0.0, 1.0 - interaction_cv)
            factors.append(interaction_consistency)
            
            # 3. Timing consistency
            hours = [s.start_time.hour for s in sessions]
            hour_std = np.std(hours)
            timing_consistency = max(0.0, 1.0 - hour_std / 12)
            factors.append(timing_consistency)
            
            # 4. Content type consistency
            all_types = set()
            session_type_counts = []
            for session in sessions:
                all_types.update(session.content_types)
                session_type_counts.append(len(session.content_types))
            
            type_consistency = 1.0 - (np.std(session_type_counts) / max(1, len(all_types)))
            factors.append(max(0.0, type_consistency))
            
            return float(np.mean(factors))
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.5
    
    def _determine_activity_level(self, sessions: List[SessionMetrics]) -> str:
        """Determine user activity level"""
        if not sessions:
            return 'inactive'
        
        # Calculate activity metrics
        session_frequency = self._calculate_session_frequency(sessions)
        avg_session_duration = np.mean([s.duration_minutes for s in sessions])
        avg_interactions = np.mean([s.interaction_count for s in sessions])
        
        # Define thresholds (could be made configurable)
        if session_frequency >= 1.0 and avg_session_duration >= 30 and avg_interactions >= 8:
            return 'high'
        elif session_frequency >= 0.3 and avg_session_duration >= 15 and avg_interactions >= 4:
            return 'medium'
        elif session_frequency >= 0.1:
            return 'low'
        else:
            return 'inactive'
    
    def _detect_behavioral_anomalies(self, sessions: List[SessionMetrics], 
                                   interactions: List[Dict]) -> List[Dict[str, Any]]:
        """Detect anomalies in user behavior"""
        anomalies = []
        
        if len(sessions) < 10:  # Need sufficient data
            return anomalies
        
        try:
            # 1. Session duration anomalies
            durations = [s.duration_minutes for s in sessions]
            duration_anomalies = self._detect_outliers(durations, 'session_duration')
            anomalies.extend(duration_anomalies)
            
            # 2. Interaction count anomalies
            interaction_counts = [s.interaction_count for s in sessions]
            interaction_anomalies = self._detect_outliers(interaction_counts, 'interaction_count')
            anomalies.extend(interaction_anomalies)
            
            # 3. Rating behavior anomalies
            session_ratings = []
            for session in sessions:
                if session.avg_rating > 0:
                    session_ratings.append(session.avg_rating)
            
            if len(session_ratings) >= 5:
                rating_anomalies = self._detect_outliers(session_ratings, 'avg_rating')
                anomalies.extend(rating_anomalies)
            
            # 4. Temporal anomalies (unusual timing)
            hours = [s.start_time.hour for s in sessions]
            temporal_anomalies = self._detect_temporal_anomalies(sessions)
            anomalies.extend(temporal_anomalies)
            
        except Exception as e:
            logger.error(f"Error detecting behavioral anomalies: {str(e)}")
        
        return anomalies
    
    def _detect_outliers(self, values: List[float], metric_name: str) -> List[Dict[str, Any]]:
        """Detect statistical outliers in a metric"""
        if len(values) < 5:
            return []
        
        anomalies = []
        values_array = np.array(values)
        
        # Use IQR method for outlier detection
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                severity = 'high' if (value < q1 - 3 * iqr or value > q3 + 3 * iqr) else 'medium'
                
                anomalies.append({
                    'type': 'statistical_outlier',
                    'metric': metric_name,
                    'value': value,
                    'expected_range': [lower_bound, upper_bound],
                    'severity': severity,
                    'timestamp': datetime.now(),
                    'session_index': i
                })
        
        return anomalies
    
    def _detect_temporal_anomalies(self, sessions: List[SessionMetrics]) -> List[Dict[str, Any]]:
        """Detect unusual timing patterns"""
        anomalies = []
        
        # Calculate typical usage patterns
        hours = [s.start_time.hour for s in sessions]
        hour_counter = Counter(hours)
        
        # Find sessions at unusual hours (bottom 10% of frequency)
        total_sessions = len(sessions)
        rare_threshold = max(1, total_sessions * 0.1)
        
        for session in sessions:
            hour_frequency = hour_counter[session.start_time.hour]
            
            if hour_frequency <= rare_threshold:
                anomalies.append({
                    'type': 'temporal_anomaly',
                    'metric': 'unusual_timing',
                    'session_id': session.session_id,
                    'hour': session.start_time.hour,
                    'frequency': hour_frequency,
                    'severity': 'low',
                    'timestamp': session.start_time
                })
        
        return anomalies
    
    def get_behavior_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user behavior for quick analysis"""
        profile = self.analyze_user_behavior(user_id)
        
        # Extract key patterns
        pattern_summary = {}
        for pattern in profile.behavior_patterns:
            pattern_summary[pattern.pattern_type] = {
                'confidence': pattern.confidence,
                'frequency': pattern.frequency,
                'characteristics': pattern.characteristics
            }
        
        # Recent activity
        recent_sessions = [s for s in profile.session_metrics 
                          if (datetime.now() - s.start_time).days <= 30]
        
        return {
            'user_id': user_id,
            'activity_level': profile.activity_level,
            'engagement_score': profile.engagement_score,
            'consistency_score': profile.consistency_score,
            'total_sessions': len(profile.session_metrics),
            'recent_sessions': len(recent_sessions),
            'behavior_patterns': pattern_summary,
            'anomaly_count': len(profile.behavioral_anomalies),
            'last_activity': max([s.end_time for s in profile.session_metrics]) if profile.session_metrics else None,
            'avg_session_duration': np.mean([s.duration_minutes for s in profile.session_metrics]) if profile.session_metrics else 0,
            'last_updated': profile.last_updated
        }
    
    def compare_behavior(self, user_id1: str, user_id2: str) -> Dict[str, Any]:
        """Compare behavior patterns between two users"""
        try:
            profile1 = self.analyze_user_behavior(user_id1)
            profile2 = self.analyze_user_behavior(user_id2)
            
            # Compare key metrics
            engagement_similarity = 1.0 - abs(profile1.engagement_score - profile2.engagement_score)
            consistency_similarity = 1.0 - abs(profile1.consistency_score - profile2.consistency_score)
            
            # Compare patterns
            patterns1 = set(p.pattern_type for p in profile1.behavior_patterns)
            patterns2 = set(p.pattern_type for p in profile2.behavior_patterns)
            
            pattern_overlap = len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)
            
            # Compare activity levels
            activity_similarity = 1.0 if profile1.activity_level == profile2.activity_level else 0.5
            
            overall_similarity = np.mean([
                engagement_similarity,
                consistency_similarity, 
                pattern_overlap,
                activity_similarity
            ])
            
            return {
                'engagement_similarity': engagement_similarity,
                'consistency_similarity': consistency_similarity,
                'pattern_overlap': pattern_overlap,
                'activity_similarity': activity_similarity,
                'overall_similarity': overall_similarity,
                'shared_patterns': list(patterns1 & patterns2),
                'unique_patterns_user1': list(patterns1 - patterns2),
                'unique_patterns_user2': list(patterns2 - patterns1)
            }
            
        except Exception as e:
            logger.error(f"Error comparing behavior for users {user_id1} and {user_id2}: {str(e)}")
            return {
                'engagement_similarity': 0.0,
                'consistency_similarity': 0.0,
                'pattern_overlap': 0.0,
                'activity_similarity': 0.0,
                'overall_similarity': 0.0,
                'shared_patterns': [],
                'unique_patterns_user1': [],
                'unique_patterns_user2': []
            }
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
