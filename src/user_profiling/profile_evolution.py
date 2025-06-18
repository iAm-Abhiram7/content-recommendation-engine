"""
Profile Evolution

Tracks and analyzes how user profiles evolve over time:
- Preference drift detection and analysis
- Lifecycle stage identification
- Profile stability metrics
- Evolution pattern recognition
- Predictive profile modeling
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..utils.config import settings
from ..utils.validation import DataValidator
from ..data_integration.schema_manager import UserProfile as User, ContentMetadata as Content, InteractionEvents as Interaction, get_session
from .preference_tracker import PreferenceTracker, PreferenceProfile
from .behavior_analyzer import BehaviorAnalyzer, BehaviorProfile

logger = logging.getLogger(__name__)


@dataclass
class EvolutionPeriod:
    """Represents a period in user profile evolution"""
    period_id: str
    start_date: datetime
    end_date: datetime
    preference_profile: PreferenceProfile
    behavior_profile: BehaviorProfile
    characteristics: Dict[str, Any]
    stability_score: float
    dominant_changes: List[str]


@dataclass
class EvolutionTrend:
    """Represents a trend in profile evolution"""
    trend_type: str  # 'preference_drift', 'behavior_shift', 'engagement_change', etc.
    direction: str  # 'increasing', 'decreasing', 'stable', 'oscillating'
    strength: float  # 0-1, strength of the trend
    confidence: float  # 0-1, confidence in the trend
    time_span_days: int
    key_metrics: Dict[str, float]
    inflection_points: List[datetime]


@dataclass
class LifecycleStage:
    """User lifecycle stage"""
    stage_name: str  # 'onboarding', 'exploration', 'established', 'mature', 'declining'
    confidence: float
    entry_date: datetime
    characteristics: Dict[str, Any]
    expected_behaviors: List[str]
    risk_factors: List[str]


@dataclass
class EvolutionProfile:
    """Complete evolution profile for a user"""
    user_id: str
    evolution_periods: List[EvolutionPeriod] = field(default_factory=list)
    evolution_trends: List[EvolutionTrend] = field(default_factory=list)
    current_lifecycle_stage: Optional[LifecycleStage] = None
    profile_stability: float = 0.0
    change_velocity: float = 0.0
    predictability_score: float = 0.0
    risk_indicators: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class ProfileEvolution:
    """
    Advanced profile evolution tracking and analysis system
    """
    
    def __init__(self, config=None):
        self.session = get_session()
        self.preference_tracker = PreferenceTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.evolution_cache: Dict[str, EvolutionProfile] = {}
        
        # Evolution analysis parameters
        self.min_period_days = 30  # Minimum period length for analysis
        self.evolution_window_days = 90  # Window for trend analysis
        self.stability_threshold = 0.7  # Threshold for considering profile stable
        
    def analyze_profile_evolution(self, user_id: str, force_refresh: bool = False) -> EvolutionProfile:
        """
        Analyze complete profile evolution for a user
        
        Args:
            user_id: User identifier
            force_refresh: Force rebuild even if cached
            
        Returns:
            Complete evolution profile
        """
        try:
            # Check cache first
            if not force_refresh and user_id in self.evolution_cache:
                cached_profile = self.evolution_cache[user_id]
                if (datetime.now() - cached_profile.last_updated).days < 1:
                    return cached_profile
            
            logger.info(f"Analyzing profile evolution for user {user_id}")
            
            # Get user interactions with time boundaries
            interactions = self._get_user_interactions_temporal(user_id)
            if not interactions:
                logger.warning(f"No interactions found for user {user_id}")
                return EvolutionProfile(user_id=user_id)
            
            # Build evolution profile
            profile = EvolutionProfile(user_id=user_id)
            
            # Extract evolution periods
            profile.evolution_periods = self._extract_evolution_periods(user_id, interactions)
            
            # Analyze evolution trends
            profile.evolution_trends = self._analyze_evolution_trends(profile.evolution_periods)
            
            # Determine current lifecycle stage
            profile.current_lifecycle_stage = self._determine_lifecycle_stage(profile.evolution_periods, interactions)
            
            # Calculate stability metrics
            profile.profile_stability = self._calculate_profile_stability(profile.evolution_periods)
            
            # Calculate change velocity
            profile.change_velocity = self._calculate_change_velocity(profile.evolution_periods)
            
            # Calculate predictability
            profile.predictability_score = self._calculate_predictability_score(profile.evolution_trends)
            
            # Identify risk indicators
            profile.risk_indicators = self._identify_risk_indicators(profile)
            
            # Cache the profile
            self.evolution_cache[user_id] = profile
            
            logger.info(f"Analyzed evolution for user {user_id} with {len(profile.evolution_periods)} periods")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing profile evolution for user {user_id}: {str(e)}")
            return EvolutionProfile(user_id=user_id)
    
    def _get_user_interactions_temporal(self, user_id: str) -> List[Dict]:
        """Get user interactions ordered by time with temporal grouping"""
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
            logger.error(f"Error fetching temporal interactions for user {user_id}: {str(e)}")
            return []
    
    def _extract_evolution_periods(self, user_id: str, interactions: List[Dict]) -> List[EvolutionPeriod]:
        """Extract distinct evolution periods based on behavior/preference changes"""
        if not interactions:
            return []
        
        periods = []
        
        # Group interactions by time periods
        time_periods = self._create_time_periods(interactions)
        
        for i, (start_date, end_date, period_interactions) in enumerate(time_periods):
            if len(period_interactions) < 5:  # Skip periods with too few interactions
                continue
            
            try:
                # Build preference profile for this period
                preference_profile = self._build_period_preference_profile(user_id, period_interactions)
                
                # Build behavior profile for this period
                behavior_profile = self._build_period_behavior_profile(user_id, period_interactions)
                
                # Calculate period characteristics
                characteristics = self._calculate_period_characteristics(period_interactions)
                
                # Calculate stability score for this period
                stability_score = self._calculate_period_stability(period_interactions)
                
                # Identify dominant changes
                dominant_changes = self._identify_period_changes(periods, preference_profile, behavior_profile) if periods else []
                
                period = EvolutionPeriod(
                    period_id=f"{user_id}_period_{i}",
                    start_date=start_date,
                    end_date=end_date,
                    preference_profile=preference_profile,
                    behavior_profile=behavior_profile,
                    characteristics=characteristics,
                    stability_score=stability_score,
                    dominant_changes=dominant_changes
                )
                
                periods.append(period)
                
            except Exception as e:
                logger.error(f"Error creating evolution period {i} for user {user_id}: {str(e)}")
        
        return periods
    
    def _create_time_periods(self, interactions: List[Dict]) -> List[Tuple[datetime, datetime, List[Dict]]]:
        """Create meaningful time periods from interactions"""
        if not interactions:
            return []
        
        # Sort interactions by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.get('timestamp', datetime.min))
        
        first_date = sorted_interactions[0].get('timestamp', datetime.now())
        last_date = sorted_interactions[-1].get('timestamp', datetime.now())
        
        total_days = (last_date - first_date).days
        
        # Determine period strategy based on data span
        if total_days <= 90:
            # Short history: monthly periods
            period_days = 30
        elif total_days <= 365:
            # Medium history: bi-monthly periods
            period_days = 60
        else:
            # Long history: quarterly periods
            period_days = 90
        
        periods = []
        current_start = first_date
        
        while current_start < last_date:
            current_end = min(current_start + timedelta(days=period_days), last_date)
            
            # Get interactions for this period
            period_interactions = [
                interaction for interaction in sorted_interactions
                if current_start <= interaction.get('timestamp', datetime.now()) <= current_end
            ]
            
            if period_interactions:  # Only add periods with interactions
                periods.append((current_start, current_end, period_interactions))
            
            current_start = current_end + timedelta(days=1)
        
        return periods
    
    def _build_period_preference_profile(self, user_id: str, interactions: List[Dict]) -> PreferenceProfile:
        """Build preference profile for a specific time period"""
        try:
            # Create a temporary preference tracker with period data
            # For simplicity, we'll use the main tracker and filter data
            # In a production system, this would be more sophisticated
            
            from .preference_tracker import PreferenceProfile
            
            # Analyze genre preferences for this period
            genre_data = defaultdict(list)
            for interaction in interactions:
                if interaction.get('genres'):
                    genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                    rating = interaction.get('rating', 0)
                    
                    for genre in genres:
                        genre = genre.strip()
                        if genre and rating > 0:
                            genre_data[genre].append(rating)
            
            # Calculate genre preferences
            genre_preferences = {}
            for genre, ratings in genre_data.items():
                if len(ratings) >= 2:
                    avg_rating = np.mean(ratings)
                    normalized_score = (avg_rating - 1) / 4.0  # Normalize to 0-1
                    genre_preferences[genre] = normalized_score
            
            # Analyze rating tendencies
            ratings = [i.get('rating', 0) for i in interactions if i.get('rating', 0) > 0]
            rating_tendencies = {}
            if ratings:
                ratings_array = np.array(ratings)
                rating_tendencies = {
                    'avg_rating': float(np.mean(ratings_array)),
                    'rating_std': float(np.std(ratings_array)),
                    'positive_bias': float(np.sum(ratings_array >= 4) / len(ratings_array))
                }
            
            # Analyze content type preferences
            type_ratings = defaultdict(list)
            for interaction in interactions:
                content_type = interaction.get('content_type', 'unknown')
                rating = interaction.get('rating', 0)
                if rating > 0:
                    type_ratings[content_type].append(rating)
            
            content_type_preferences = {}
            for content_type, ratings in type_ratings.items():
                if len(ratings) >= 2:
                    avg_rating = np.mean(ratings)
                    content_type_preferences[content_type] = (avg_rating - 1) / 4.0
            
            # Create simplified preference profile
            profile = PreferenceProfile(user_id=user_id)
            # Note: We're creating a simplified version here
            # The full implementation would use proper GenrePreference objects
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building period preference profile: {str(e)}")
            return PreferenceProfile(user_id=user_id)
    
    def _build_period_behavior_profile(self, user_id: str, interactions: List[Dict]) -> BehaviorProfile:
        """Build behavior profile for a specific time period"""
        try:
            # For simplicity, create a basic behavior profile
            # In production, this would use the full BehaviorAnalyzer
            
            from .behavior_analyzer import BehaviorProfile
            
            profile = BehaviorProfile(user_id=user_id)
            
            # Calculate basic metrics
            if interactions:
                # Activity level based on interaction frequency
                days_span = (max(i.get('timestamp', datetime.now()) for i in interactions) - 
                           min(i.get('timestamp', datetime.now()) for i in interactions)).days + 1
                
                interactions_per_day = len(interactions) / days_span if days_span > 0 else 0
                
                if interactions_per_day >= 2:
                    profile.activity_level = 'high'
                elif interactions_per_day >= 0.5:
                    profile.activity_level = 'medium'
                else:
                    profile.activity_level = 'low'
                
                # Basic engagement score
                rated_interactions = [i for i in interactions if i.get('rating', 0) > 0]
                rating_engagement = len(rated_interactions) / len(interactions)
                
                unique_content = len(set(i.get('content_id') for i in interactions if i.get('content_id')))
                diversity_score = unique_content / len(interactions)
                
                profile.engagement_score = (rating_engagement + diversity_score) / 2
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building period behavior profile: {str(e)}")
            return BehaviorProfile(user_id=user_id)
    
    def _calculate_period_characteristics(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Calculate characteristics for a time period"""
        if not interactions:
            return {}
        
        try:
            characteristics = {}
            
            # Basic metrics
            characteristics['interaction_count'] = len(interactions)
            characteristics['unique_content_count'] = len(set(i.get('content_id') for i in interactions if i.get('content_id')))
            
            # Rating metrics
            ratings = [i.get('rating', 0) for i in interactions if i.get('rating', 0) > 0]
            if ratings:
                characteristics['avg_rating'] = float(np.mean(ratings))
                characteristics['rating_std'] = float(np.std(ratings))
                characteristics['rating_count'] = len(ratings)
            
            # Content diversity
            content_types = set(i.get('content_type', 'unknown') for i in interactions)
            characteristics['content_type_count'] = len(content_types)
            
            # Genre diversity
            all_genres = set()
            for interaction in interactions:
                if interaction.get('genres'):
                    genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                    all_genres.update(g.strip() for g in genres if g.strip())
            characteristics['genre_count'] = len(all_genres)
            
            # Temporal patterns
            timestamps = [i.get('timestamp') for i in interactions if i.get('timestamp')]
            if timestamps:
                hours = [t.hour for t in timestamps]
                characteristics['activity_hour_std'] = float(np.std(hours))
                
                days = [(t - timestamps[0]).days for t in timestamps]
                characteristics['activity_span_days'] = max(days) if days else 0
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error calculating period characteristics: {str(e)}")
            return {}
    
    def _calculate_period_stability(self, interactions: List[Dict]) -> float:
        """Calculate stability score for a period"""
        if len(interactions) < 5:
            return 0.5  # Neutral for insufficient data
        
        try:
            # Calculate various stability metrics
            stability_factors = []
            
            # 1. Rating consistency
            ratings = [i.get('rating', 0) for i in interactions if i.get('rating', 0) > 0]
            if len(ratings) >= 3:
                rating_cv = np.std(ratings) / np.mean(ratings) if np.mean(ratings) > 0 else 1
                rating_stability = max(0.0, 1.0 - rating_cv)
                stability_factors.append(rating_stability)
            
            # 2. Content type consistency
            content_types = [i.get('content_type', 'unknown') for i in interactions]
            type_counter = Counter(content_types)
            type_entropy = stats.entropy(list(type_counter.values()))
            max_entropy = np.log(len(type_counter)) if len(type_counter) > 1 else 1
            type_stability = 1.0 - (type_entropy / max_entropy) if max_entropy > 0 else 1.0
            stability_factors.append(type_stability)
            
            # 3. Temporal consistency
            timestamps = [i.get('timestamp') for i in interactions if i.get('timestamp')]
            if len(timestamps) >= 3:
                hours = [t.hour for t in timestamps]
                hour_std = np.std(hours)
                temporal_stability = max(0.0, 1.0 - hour_std / 12)
                stability_factors.append(temporal_stability)
            
            return float(np.mean(stability_factors)) if stability_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating period stability: {str(e)}")
            return 0.5
    
    def _identify_period_changes(self, previous_periods: List[EvolutionPeriod], 
                               current_prefs: PreferenceProfile, 
                               current_behavior: BehaviorProfile) -> List[str]:
        """Identify significant changes from previous period"""
        if not previous_periods:
            return ['initial_period']
        
        changes = []
        prev_period = previous_periods[-1]
        
        try:
            # Compare engagement
            if hasattr(prev_period.behavior_profile, 'engagement_score') and hasattr(current_behavior, 'engagement_score'):
                engagement_change = current_behavior.engagement_score - prev_period.behavior_profile.engagement_score
                if abs(engagement_change) > 0.2:
                    changes.append('engagement_shift' if engagement_change > 0 else 'engagement_decline')
            
            # Compare activity level
            if hasattr(prev_period.behavior_profile, 'activity_level') and hasattr(current_behavior, 'activity_level'):
                activity_levels = {'low': 1, 'medium': 2, 'high': 3}
                prev_level = activity_levels.get(prev_period.behavior_profile.activity_level, 2)
                curr_level = activity_levels.get(current_behavior.activity_level, 2)
                
                if curr_level > prev_level:
                    changes.append('activity_increase')
                elif curr_level < prev_level:
                    changes.append('activity_decrease')
            
            # Add more sophisticated change detection here
            # For now, we'll use basic heuristics
            
        except Exception as e:
            logger.error(f"Error identifying period changes: {str(e)}")
        
        return changes if changes else ['stable_period']
    
    def _analyze_evolution_trends(self, periods: List[EvolutionPeriod]) -> List[EvolutionTrend]:
        """Analyze trends across evolution periods"""
        trends = []
        
        if len(periods) < 3:  # Need at least 3 periods for trend analysis
            return trends
        
        try:
            # 1. Engagement trend
            engagement_trend = self._analyze_engagement_trend(periods)
            if engagement_trend:
                trends.append(engagement_trend)
            
            # 2. Activity level trend
            activity_trend = self._analyze_activity_trend(periods)
            if activity_trend:
                trends.append(activity_trend)
            
            # 3. Stability trend
            stability_trend = self._analyze_stability_trend(periods)
            if stability_trend:
                trends.append(stability_trend)
            
            # 4. Diversity trend
            diversity_trend = self._analyze_diversity_trend(periods)
            if diversity_trend:
                trends.append(diversity_trend)
            
        except Exception as e:
            logger.error(f"Error analyzing evolution trends: {str(e)}")
        
        return trends
    
    def _analyze_engagement_trend(self, periods: List[EvolutionPeriod]) -> Optional[EvolutionTrend]:
        """Analyze engagement score trend"""
        engagement_scores = []
        timestamps = []
        
        for period in periods:
            if hasattr(period.behavior_profile, 'engagement_score'):
                engagement_scores.append(period.behavior_profile.engagement_score)
                timestamps.append(period.start_date)
        
        if len(engagement_scores) < 3:
            return None
        
        return self._create_trend_from_values(
            'engagement_change',
            engagement_scores,
            timestamps,
            'engagement_score'
        )
    
    def _analyze_activity_trend(self, periods: List[EvolutionPeriod]) -> Optional[EvolutionTrend]:
        """Analyze activity level trend"""
        activity_scores = []
        timestamps = []
        
        activity_levels = {'low': 1, 'medium': 2, 'high': 3, 'inactive': 0}
        
        for period in periods:
            if hasattr(period.behavior_profile, 'activity_level'):
                score = activity_levels.get(period.behavior_profile.activity_level, 1)
                activity_scores.append(score)
                timestamps.append(period.start_date)
        
        if len(activity_scores) < 3:
            return None
        
        return self._create_trend_from_values(
            'activity_change',
            activity_scores,
            timestamps,
            'activity_level'
        )
    
    def _analyze_stability_trend(self, periods: List[EvolutionPeriod]) -> Optional[EvolutionTrend]:
        """Analyze stability score trend"""
        stability_scores = [period.stability_score for period in periods]
        timestamps = [period.start_date for period in periods]
        
        return self._create_trend_from_values(
            'stability_change',
            stability_scores,
            timestamps,
            'stability_score'
        )
    
    def _analyze_diversity_trend(self, periods: List[EvolutionPeriod]) -> Optional[EvolutionTrend]:
        """Analyze content diversity trend"""
        diversity_scores = []
        timestamps = []
        
        for period in periods:
            if 'genre_count' in period.characteristics and 'interaction_count' in period.characteristics:
                diversity = period.characteristics['genre_count'] / max(period.characteristics['interaction_count'], 1)
                diversity_scores.append(min(1.0, diversity))
                timestamps.append(period.start_date)
        
        if len(diversity_scores) < 3:
            return None
        
        return self._create_trend_from_values(
            'diversity_change',
            diversity_scores,
            timestamps,
            'diversity_score'
        )
    
    def _create_trend_from_values(self, trend_type: str, values: List[float], 
                                timestamps: List[datetime], metric_name: str) -> Optional[EvolutionTrend]:
        """Create trend object from value series"""
        if len(values) < 3:
            return None
        
        try:
            # Calculate trend direction and strength using linear regression
            x = np.array([(t - timestamps[0]).days for t in timestamps])
            y = np.array(values)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine direction
            if abs(slope) < 0.001:  # Threshold for "stable"
                direction = 'stable'
            elif slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
            
            # Calculate strength (normalized absolute slope)
            value_range = max(values) - min(values)
            time_range = max(x) - min(x) if max(x) > min(x) else 1
            strength = min(1.0, abs(slope) * time_range / max(value_range, 0.001))
            
            # Confidence based on R-squared and p-value
            confidence = max(0.0, min(1.0, r_value**2)) if p_value < 0.05 else 0.0
            
            # Find inflection points (simplified)
            inflection_points = []
            for i in range(1, len(values) - 1):
                if ((values[i] > values[i-1] and values[i] > values[i+1]) or 
                    (values[i] < values[i-1] and values[i] < values[i+1])):
                    inflection_points.append(timestamps[i])
            
            return EvolutionTrend(
                trend_type=trend_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                time_span_days=(timestamps[-1] - timestamps[0]).days,
                key_metrics={
                    metric_name: {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'start_value': values[0],
                        'end_value': values[-1]
                    }
                },
                inflection_points=inflection_points
            )
            
        except Exception as e:
            logger.error(f"Error creating trend from values: {str(e)}")
            return None
    
    def _determine_lifecycle_stage(self, periods: List[EvolutionPeriod], 
                                 interactions: List[Dict]) -> Optional[LifecycleStage]:
        """Determine current user lifecycle stage"""
        if not periods or not interactions:
            return None
        
        try:
            # Calculate user metrics
            total_interactions = len(interactions)
            first_interaction = min(i.get('timestamp', datetime.now()) for i in interactions)
            last_interaction = max(i.get('timestamp', datetime.now()) for i in interactions)
            days_active = (last_interaction - first_interaction).days + 1
            
            # Recent activity (last 30 days)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_interactions = [i for i in interactions 
                                 if i.get('timestamp', datetime.min) >= recent_cutoff]
            
            # Determine stage based on metrics
            if days_active <= 30 and total_interactions <= 20:
                stage_name = 'onboarding'
                characteristics = {
                    'typical_behavior': 'learning platform, exploring content types',
                    'interaction_frequency': 'irregular',
                    'content_diversity': 'high exploration'
                }
                expected_behaviors = ['high_exploration', 'irregular_timing', 'rating_experimentation']
                risk_factors = ['early_abandonment', 'overwhelming_choice']
                
            elif days_active <= 90 and len(set(i.get('content_type') for i in interactions)) >= 2:
                stage_name = 'exploration'
                characteristics = {
                    'typical_behavior': 'trying different content types and genres',
                    'interaction_frequency': 'increasing',
                    'content_diversity': 'medium to high'
                }
                expected_behaviors = ['genre_experimentation', 'rating_activity', 'session_lengthening']
                risk_factors = ['preference_confusion', 'decision_fatigue']
                
            elif (days_active > 90 and total_interactions >= 50 and 
                  len(recent_interactions) >= 5):
                
                # Check if preferences are stabilizing
                if periods and len(periods) >= 2:
                    recent_stability = np.mean([p.stability_score for p in periods[-2:]])
                    if recent_stability >= self.stability_threshold:
                        stage_name = 'established'
                    else:
                        stage_name = 'exploration'  # Still exploring
                else:
                    stage_name = 'established'
                
                characteristics = {
                    'typical_behavior': 'consistent preferences, regular usage',
                    'interaction_frequency': 'regular',
                    'content_diversity': 'focused but varied'
                }
                expected_behaviors = ['consistent_timing', 'preference_stability', 'quality_focus']
                risk_factors = ['boredom', 'routine_staleness']
                
            elif days_active > 180 and total_interactions >= 100:
                stage_name = 'mature'
                characteristics = {
                    'typical_behavior': 'well-defined preferences, efficient content discovery',
                    'interaction_frequency': 'stable',
                    'content_diversity': 'selective'
                }
                expected_behaviors = ['efficient_discovery', 'quality_over_quantity', 'niche_preferences']
                risk_factors = ['platform_fatigue', 'reduced_exploration']
                
            elif len(recent_interactions) == 0 and days_active > 60:
                stage_name = 'declining'
                characteristics = {
                    'typical_behavior': 'reduced engagement, possible churn risk',
                    'interaction_frequency': 'decreasing',
                    'content_diversity': 'unknown'
                }
                expected_behaviors = ['sporadic_usage', 'reduced_rating_activity']
                risk_factors = ['churn_risk', 'platform_abandonment']
                
            else:
                stage_name = 'undefined'
                characteristics = {}
                expected_behaviors = []
                risk_factors = []
            
            # Calculate confidence based on data quality
            confidence = min(1.0, total_interactions / 50)  # Higher confidence with more data
            if days_active >= 30:
                confidence *= 0.8 + 0.2 * min(1.0, days_active / 90)
            
            return LifecycleStage(
                stage_name=stage_name,
                confidence=confidence,
                entry_date=first_interaction,
                characteristics=characteristics,
                expected_behaviors=expected_behaviors,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error determining lifecycle stage: {str(e)}")
            return None
    
    def _calculate_profile_stability(self, periods: List[EvolutionPeriod]) -> float:
        """Calculate overall profile stability across periods"""
        if len(periods) < 2:
            return 0.5  # Neutral for insufficient data
        
        try:
            stability_scores = [period.stability_score for period in periods]
            
            # Overall stability is the average individual stability minus variance penalty
            avg_stability = np.mean(stability_scores)
            stability_variance = np.var(stability_scores)
            
            # Penalize high variance (inconsistent stability)
            variance_penalty = min(0.3, stability_variance)
            
            return max(0.0, avg_stability - variance_penalty)
            
        except Exception as e:
            logger.error(f"Error calculating profile stability: {str(e)}")
            return 0.5
    
    def _calculate_change_velocity(self, periods: List[EvolutionPeriod]) -> float:
        """Calculate how quickly the user's profile changes"""
        if len(periods) < 2:
            return 0.0
        
        try:
            # Calculate change between consecutive periods
            changes = []
            
            for i in range(1, len(periods)):
                prev_period = periods[i-1]
                curr_period = periods[i]
                
                # Compare key characteristics
                change_score = 0.0
                comparisons = 0
                
                # Engagement change
                if (hasattr(prev_period.behavior_profile, 'engagement_score') and 
                    hasattr(curr_period.behavior_profile, 'engagement_score')):
                    engagement_change = abs(curr_period.behavior_profile.engagement_score - 
                                          prev_period.behavior_profile.engagement_score)
                    change_score += engagement_change
                    comparisons += 1
                
                # Stability change
                stability_change = abs(curr_period.stability_score - prev_period.stability_score)
                change_score += stability_change
                comparisons += 1
                
                # Average change for this transition
                if comparisons > 0:
                    changes.append(change_score / comparisons)
            
            # Return average change velocity
            return float(np.mean(changes)) if changes else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating change velocity: {str(e)}")
            return 0.0
    
    def _calculate_predictability_score(self, trends: List[EvolutionTrend]) -> float:
        """Calculate how predictable the user's evolution is"""
        if not trends:
            return 0.5  # Neutral for no trends
        
        try:
            # Predictability based on trend consistency and confidence
            trend_confidences = [trend.confidence for trend in trends]
            avg_confidence = np.mean(trend_confidences)
            
            # Bonus for having clear, consistent trends
            stable_trends = [t for t in trends if t.direction != 'stable']
            strong_trends = [t for t in trends if t.strength >= 0.5]
            
            consistency_bonus = len(strong_trends) / max(len(trends), 1) * 0.2
            
            return min(1.0, avg_confidence + consistency_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating predictability score: {str(e)}")
            return 0.5
    
    def _identify_risk_indicators(self, profile: EvolutionProfile) -> List[Dict[str, Any]]:
        """Identify potential risk indicators in profile evolution"""
        risks = []
        
        try:
            # 1. Declining engagement
            engagement_trends = [t for t in profile.evolution_trends if t.trend_type == 'engagement_change']
            for trend in engagement_trends:
                if trend.direction == 'decreasing' and trend.strength >= 0.3:
                    risks.append({
                        'type': 'engagement_decline',
                        'severity': 'high' if trend.strength >= 0.7 else 'medium',
                        'description': 'User engagement is significantly declining',
                        'trend_strength': trend.strength,
                        'confidence': trend.confidence
                    })
            
            # 2. Activity decline
            activity_trends = [t for t in profile.evolution_trends if t.trend_type == 'activity_change']
            for trend in activity_trends:
                if trend.direction == 'decreasing' and trend.strength >= 0.4:
                    risks.append({
                        'type': 'activity_decline',
                        'severity': 'high' if trend.strength >= 0.8 else 'medium',
                        'description': 'User activity level is declining',
                        'trend_strength': trend.strength,
                        'confidence': trend.confidence
                    })
            
            # 3. Low stability with high change velocity
            if profile.profile_stability < 0.3 and profile.change_velocity > 0.7:
                risks.append({
                    'type': 'profile_instability',
                    'severity': 'medium',
                    'description': 'User profile is highly unstable and changing rapidly',
                    'stability_score': profile.profile_stability,
                    'change_velocity': profile.change_velocity
                })
            
            # 4. Lifecycle stage risks
            if profile.current_lifecycle_stage:
                for risk_factor in profile.current_lifecycle_stage.risk_factors:
                    risks.append({
                        'type': 'lifecycle_risk',
                        'severity': 'low',
                        'description': f'Lifecycle stage risk: {risk_factor}',
                        'lifecycle_stage': profile.current_lifecycle_stage.stage_name,
                        'risk_factor': risk_factor
                    })
            
            # 5. No recent trends (stagnation)
            recent_trends = [t for t in profile.evolution_trends 
                           if t.time_span_days <= 60 and t.direction != 'stable']
            if not recent_trends and len(profile.evolution_periods) >= 3:
                risks.append({
                    'type': 'profile_stagnation',
                    'severity': 'low',
                    'description': 'No significant profile evolution detected recently',
                    'periods_analyzed': len(profile.evolution_periods)
                })
            
        except Exception as e:
            logger.error(f"Error identifying risk indicators: {str(e)}")
        
        return risks
    
    def get_evolution_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of profile evolution for quick analysis"""
        profile = self.analyze_profile_evolution(user_id)
        
        # Extract key trends
        trend_summary = {}
        for trend in profile.evolution_trends:
            trend_summary[trend.trend_type] = {
                'direction': trend.direction,
                'strength': trend.strength,
                'confidence': trend.confidence
            }
        
        # Risk summary
        risk_summary = {
            'high_risks': [r for r in profile.risk_indicators if r.get('severity') == 'high'],
            'medium_risks': [r for r in profile.risk_indicators if r.get('severity') == 'medium'],
            'low_risks': [r for r in profile.risk_indicators if r.get('severity') == 'low']
        }
        
        return {
            'user_id': user_id,
            'current_lifecycle_stage': profile.current_lifecycle_stage.stage_name if profile.current_lifecycle_stage else 'unknown',
            'lifecycle_confidence': profile.current_lifecycle_stage.confidence if profile.current_lifecycle_stage else 0.0,
            'profile_stability': profile.profile_stability,
            'change_velocity': profile.change_velocity,
            'predictability_score': profile.predictability_score,
            'evolution_periods_count': len(profile.evolution_periods),
            'trends': trend_summary,
            'risk_summary': risk_summary,
            'last_updated': profile.last_updated
        }
    
    def predict_future_behavior(self, user_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future user behavior based on evolution patterns"""
        try:
            profile = self.analyze_profile_evolution(user_id)
            
            predictions = {
                'confidence': profile.predictability_score,
                'time_horizon_days': days_ahead,
                'predictions': {}
            }
            
            # Predict based on trends
            for trend in profile.evolution_trends:
                if trend.confidence >= 0.5 and trend.direction != 'stable':
                    
                    # Get current value and trend
                    if trend.trend_type == 'engagement_change' and 'engagement_score' in trend.key_metrics:
                        current_value = trend.key_metrics['engagement_score']['end_value']
                        slope = trend.key_metrics['engagement_score']['slope']
                        
                        # Project forward
                        predicted_value = current_value + (slope * days_ahead)
                        predicted_value = max(0.0, min(1.0, predicted_value))  # Clamp to valid range
                        
                        predictions['predictions']['engagement_score'] = {
                            'current': current_value,
                            'predicted': predicted_value,
                            'change': predicted_value - current_value,
                            'confidence': trend.confidence
                        }
                    
                    # Similar predictions for other metrics...
            
            # Lifecycle stage prediction
            if profile.current_lifecycle_stage:
                stage_transitions = {
                    'onboarding': ['exploration', 'declining'],
                    'exploration': ['established', 'declining'],
                    'established': ['mature', 'declining'],
                    'mature': ['declining'],
                    'declining': ['mature']  # Possible recovery
                }
                
                possible_next_stages = stage_transitions.get(
                    profile.current_lifecycle_stage.stage_name, 
                    []
                )
                
                predictions['predictions']['lifecycle_stage'] = {
                    'current': profile.current_lifecycle_stage.stage_name,
                    'possible_transitions': possible_next_stages,
                    'confidence': profile.current_lifecycle_stage.confidence
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future behavior for user {user_id}: {str(e)}")
            return {
                'confidence': 0.0,
                'time_horizon_days': days_ahead,
                'predictions': {},
                'error': str(e)
            }
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        if self.preference_tracker:
            self.preference_tracker.close()
        if self.behavior_analyzer:
            self.behavior_analyzer.close()
