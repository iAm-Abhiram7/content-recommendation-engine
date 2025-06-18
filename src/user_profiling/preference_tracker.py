"""
Preference Tracker

Tracks and analyzes user preferences across different dimensions:
- Genre preferences and evolution
- Rating patterns and standards
- Content type preferences
- Temporal preference changes
- Cross-domain preference mapping
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

from ..utils.config import settings
from ..utils.validation import DataValidator
from ..data_integration.schema_manager import UserProfile as User, ContentMetadata as Content, InteractionEvents as Interaction, get_session

logger = logging.getLogger(__name__)


@dataclass
class GenrePreference:
    """Genre preference with confidence and evolution tracking"""
    genre: str
    score: float
    confidence: float
    interaction_count: int
    avg_rating: float
    recent_trend: str  # 'increasing', 'decreasing', 'stable'
    first_seen: datetime
    last_seen: datetime


@dataclass
class PreferenceProfile:
    """Complete preference profile for a user"""
    user_id: str
    genre_preferences: Dict[str, GenrePreference] = field(default_factory=dict)
    rating_tendencies: Dict[str, float] = field(default_factory=dict)
    content_type_preferences: Dict[str, float] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    diversity_score: float = 0.0
    exploration_tendency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class PreferenceTracker:
    """
    Advanced preference tracking and analysis system
    """
    
    def __init__(self, config=None):
        self.session = get_session()
        self.preference_cache: Dict[str, PreferenceProfile] = {}
        
        # Preference analysis parameters
        self.min_interactions_for_confidence = 5
        self.genre_decay_rate = 0.1  # How quickly old preferences decay
        self.trend_window_days = 90
        
    def build_user_profile(self, user_id: str, force_refresh: bool = False) -> PreferenceProfile:
        """
        Build comprehensive preference profile for a user
        
        Args:
            user_id: User identifier
            force_refresh: Force rebuild even if cached
            
        Returns:
            Complete preference profile
        """
        try:
            # Check cache first
            if not force_refresh and user_id in self.preference_cache:
                cached_profile = self.preference_cache[user_id]
                if (datetime.now() - cached_profile.last_updated).days < 1:
                    return cached_profile
            
            logger.info(f"Building preference profile for user {user_id}")
            
            # Get user interactions
            interactions = self._get_user_interactions(user_id)
            if not interactions:
                logger.warning(f"No interactions found for user {user_id}")
                return PreferenceProfile(user_id=user_id)
            
            # Build profile components
            profile = PreferenceProfile(user_id=user_id)
            
            # Analyze genre preferences
            profile.genre_preferences = self._analyze_genre_preferences(interactions)
            
            # Analyze rating tendencies
            profile.rating_tendencies = self._analyze_rating_tendencies(interactions)
            
            # Analyze content type preferences
            profile.content_type_preferences = self._analyze_content_type_preferences(interactions)
            
            # Analyze temporal patterns
            profile.temporal_patterns = self._analyze_temporal_patterns(interactions)
            
            # Calculate diversity metrics
            profile.diversity_score = self._calculate_diversity_score(interactions)
            profile.exploration_tendency = self._calculate_exploration_tendency(interactions)
            
            # Cache the profile
            self.preference_cache[user_id] = profile
            
            logger.info(f"Built preference profile for user {user_id} with {len(interactions)} interactions")
            return profile
            
        except Exception as e:
            logger.error(f"Error building preference profile for user {user_id}: {str(e)}")
            return PreferenceProfile(user_id=user_id)
    
    def _get_user_interactions(self, user_id: str) -> List[Dict]:
        """Get all interactions for a user with content details"""
        try:
            query = text("""
                SELECT i.*, c.title, c.genres, c.content_type, c.metadata
                FROM interactions i
                JOIN content c ON i.content_id = c.content_id
                WHERE i.user_id = :user_id
                ORDER BY i.timestamp DESC
            """)
            
            result = self.session.execute(query, {'user_id': user_id}).fetchall()
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Error fetching interactions for user {user_id}: {str(e)}")
            return []
    
    def _analyze_genre_preferences(self, interactions: List[Dict]) -> Dict[str, GenrePreference]:
        """Analyze user's genre preferences with confidence and trends"""
        genre_data = defaultdict(list)
        genre_timestamps = defaultdict(list)
        
        # Collect genre interaction data
        for interaction in interactions:
            if interaction.get('genres'):
                genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                timestamp = interaction.get('timestamp', datetime.now())
                rating = interaction.get('rating', 0)
                
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        genre_data[genre].append({
                            'rating': rating,
                            'timestamp': timestamp,
                            'interaction_type': interaction.get('interaction_type', 'rating')
                        })
                        genre_timestamps[genre].append(timestamp)
        
        # Build genre preferences
        preferences = {}
        current_time = datetime.now()
        
        for genre, data in genre_data.items():
            if len(data) < 2:  # Need minimum interactions
                continue
                
            ratings = [d['rating'] for d in data if d['rating'] > 0]
            if not ratings:
                continue
                
            # Calculate base metrics
            avg_rating = np.mean(ratings)
            interaction_count = len(data)
            
            # Calculate time-weighted score (recent interactions matter more)
            weighted_score = self._calculate_time_weighted_score(data, current_time)
            
            # Calculate confidence based on interaction count and consistency
            confidence = min(1.0, interaction_count / self.min_interactions_for_confidence)
            rating_std = np.std(ratings) if len(ratings) > 1 else 0
            confidence *= max(0.1, 1.0 - (rating_std / 5.0))  # Penalize inconsistency
            
            # Analyze trend
            trend = self._analyze_preference_trend(data, self.trend_window_days)
            
            # Get time range
            timestamps = genre_timestamps[genre]
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            
            preferences[genre] = GenrePreference(
                genre=genre,
                score=weighted_score,
                confidence=confidence,
                interaction_count=interaction_count,
                avg_rating=avg_rating,
                recent_trend=trend,
                first_seen=first_seen,
                last_seen=last_seen
            )
        
        return preferences
    
    def _calculate_time_weighted_score(self, interactions: List[Dict], current_time: datetime) -> float:
        """Calculate time-weighted preference score"""
        if not interactions:
            return 0.0
            
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for interaction in interactions:
            rating = interaction.get('rating', 0)
            if rating <= 0:
                continue
                
            timestamp = interaction.get('timestamp', current_time)
            days_old = (current_time - timestamp).days
            
            # Exponential decay weight
            weight = np.exp(-self.genre_decay_rate * days_old / 365)
            
            # Normalize rating to 0-1 scale
            normalized_rating = (rating - 1) / 4.0 if rating > 0 else 0
            
            weighted_sum += normalized_rating * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _analyze_preference_trend(self, interactions: List[Dict], window_days: int) -> str:
        """Analyze if preference is increasing, decreasing, or stable"""
        if len(interactions) < 4:
            return 'stable'
            
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=window_days)
        
        # Split into recent and older interactions
        recent_ratings = []
        older_ratings = []
        
        for interaction in interactions:
            rating = interaction.get('rating', 0)
            if rating <= 0:
                continue
                
            timestamp = interaction.get('timestamp', current_time)
            if timestamp >= cutoff_time:
                recent_ratings.append(rating)
            else:
                older_ratings.append(rating)
        
        if not recent_ratings or not older_ratings:
            return 'stable'
        
        recent_avg = np.mean(recent_ratings)
        older_avg = np.mean(older_ratings)
        
        diff = recent_avg - older_avg
        
        if diff > 0.5:
            return 'increasing'
        elif diff < -0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_rating_tendencies(self, interactions: List[Dict]) -> Dict[str, float]:
        """Analyze user's rating patterns and tendencies"""
        ratings = [i.get('rating', 0) for i in interactions if i.get('rating', 0) > 0]
        
        if not ratings:
            return {}
        
        ratings_array = np.array(ratings)
        
        return {
            'avg_rating': float(np.mean(ratings_array)),
            'rating_std': float(np.std(ratings_array)),
            'rating_range': float(np.max(ratings_array) - np.min(ratings_array)),
            'harsh_rater_score': self._calculate_harshness_score(ratings_array),
            'rating_consistency': self._calculate_rating_consistency(ratings_array),
            'positive_bias': float(np.sum(ratings_array >= 4) / len(ratings_array)),
            'critical_bias': float(np.sum(ratings_array <= 2) / len(ratings_array))
        }
    
    def _calculate_harshness_score(self, ratings: np.ndarray) -> float:
        """Calculate how harsh/lenient the user is compared to average"""
        if len(ratings) == 0:
            return 0.5  # Neutral
            
        # Assume global average is around 3.5
        global_avg = 3.5
        user_avg = np.mean(ratings)
        
        # Score from 0 (very harsh) to 1 (very lenient)
        return max(0.0, min(1.0, (user_avg - 1) / 4.0))
    
    def _calculate_rating_consistency(self, ratings: np.ndarray) -> float:
        """Calculate how consistent the user's ratings are"""
        if len(ratings) <= 1:
            return 1.0
            
        std = np.std(ratings)
        # Convert to 0-1 scale where 1 is very consistent
        return max(0.0, 1.0 - (std / 2.0))
    
    def _analyze_content_type_preferences(self, interactions: List[Dict]) -> Dict[str, float]:
        """Analyze preferences across different content types"""
        type_ratings = defaultdict(list)
        
        for interaction in interactions:
            content_type = interaction.get('content_type', 'unknown')
            rating = interaction.get('rating', 0)
            
            if rating > 0:
                type_ratings[content_type].append(rating)
        
        preferences = {}
        for content_type, ratings in type_ratings.items():
            if len(ratings) >= 2:  # Minimum interactions needed
                avg_rating = np.mean(ratings)
                interaction_count = len(ratings)
                
                # Weight by both rating and interaction frequency
                frequency_weight = min(1.0, interaction_count / 10)
                rating_weight = (avg_rating - 1) / 4.0  # Normalize to 0-1
                
                preferences[content_type] = rating_weight * frequency_weight
        
        return preferences
    
    def _analyze_temporal_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in user behavior"""
        if not interactions:
            return {}
        
        timestamps = []
        ratings = []
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            rating = interaction.get('rating', 0)
            
            if timestamp and rating > 0:
                timestamps.append(timestamp)
                ratings.append(rating)
        
        if not timestamps:
            return {}
        
        df = pd.DataFrame({'timestamp': timestamps, 'rating': ratings})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        patterns = {}
        
        # Hour of day patterns
        if len(df) >= 5:
            hourly_activity = df.groupby('hour').size()
            patterns['peak_hours'] = hourly_activity.nlargest(3).index.tolist()
            patterns['low_hours'] = hourly_activity.nsmallest(3).index.tolist()
        
        # Day of week patterns
        if len(df) >= 7:
            daily_activity = df.groupby('day_of_week').size()
            patterns['peak_days'] = daily_activity.nlargest(2).index.tolist()
            patterns['activity_distribution'] = daily_activity.to_dict()
        
        # Monthly patterns
        if len(df) >= 12:
            monthly_activity = df.groupby('month').size()
            patterns['peak_months'] = monthly_activity.nlargest(2).index.tolist()
        
        # Activity frequency
        date_counts = df.groupby(df['timestamp'].dt.date).size()
        patterns['avg_daily_interactions'] = float(date_counts.mean())
        patterns['max_daily_interactions'] = int(date_counts.max())
        patterns['active_days'] = len(date_counts)
        
        return patterns
    
    def _calculate_diversity_score(self, interactions: List[Dict]) -> float:
        """Calculate how diverse the user's content consumption is"""
        if not interactions:
            return 0.0
        
        # Collect all genres
        all_genres = set()
        for interaction in interactions:
            if interaction.get('genres'):
                genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                all_genres.update(g.strip() for g in genres if g.strip())
        
        # Collect all content types
        content_types = set(i.get('content_type', 'unknown') for i in interactions)
        
        # Calculate diversity metrics
        genre_diversity = len(all_genres) / max(1, len(interactions))
        type_diversity = len(content_types) / max(1, len(interactions))
        
        # Combined diversity score
        return min(1.0, (genre_diversity + type_diversity) / 2)
    
    def _calculate_exploration_tendency(self, interactions: List[Dict]) -> float:
        """Calculate user's tendency to explore new content"""
        if len(interactions) < 10:
            return 0.5  # Neutral for users with few interactions
        
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.get('timestamp', datetime.min))
        
        # Calculate how often user tries new genres over time
        seen_genres = set()
        new_genre_encounters = 0
        total_encounters = 0
        
        for interaction in sorted_interactions:
            if interaction.get('genres'):
                genres = interaction['genres'].split('|') if isinstance(interaction['genres'], str) else interaction['genres']
                interaction_genres = set(g.strip() for g in genres if g.strip())
                
                # Check for new genres
                new_genres = interaction_genres - seen_genres
                if new_genres:
                    new_genre_encounters += 1
                
                seen_genres.update(interaction_genres)
                total_encounters += 1
        
        # Calculate exploration tendency
        if total_encounters == 0:
            return 0.5
        
        exploration_rate = new_genre_encounters / total_encounters
        return min(1.0, exploration_rate * 2)  # Scale up to make more meaningful
    
    def get_preference_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user preferences for quick analysis"""
        profile = self.build_user_profile(user_id)
        
        # Top genres by score
        top_genres = sorted(
            profile.genre_preferences.items(),
            key=lambda x: x[1].score,
            reverse=True
        )[:5]
        
        # Top content types
        top_content_types = sorted(
            profile.content_type_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            'user_id': user_id,
            'top_genres': [{'genre': g, 'score': p.score, 'confidence': p.confidence} 
                          for g, p in top_genres],
            'top_content_types': [{'type': t, 'score': s} for t, s in top_content_types],
            'rating_profile': profile.rating_tendencies,
            'diversity_score': profile.diversity_score,
            'exploration_tendency': profile.exploration_tendency,
            'temporal_patterns': profile.temporal_patterns,
            'last_updated': profile.last_updated
        }
    
    def update_preferences_incremental(self, user_id: str, new_interactions: List[Dict]) -> PreferenceProfile:
        """Update user preferences with new interactions without full rebuild"""
        try:
            # Get existing profile
            existing_profile = self.preference_cache.get(user_id)
            if not existing_profile:
                return self.build_user_profile(user_id)
            
            # For now, do a full rebuild - incremental updates would be more complex
            # TODO: Implement true incremental updates for better performance
            return self.build_user_profile(user_id, force_refresh=True)
            
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {str(e)}")
            return existing_profile or PreferenceProfile(user_id=user_id)
    
    def compare_users(self, user_id1: str, user_id2: str) -> Dict[str, float]:
        """Compare preference similarity between two users"""
        try:
            profile1 = self.build_user_profile(user_id1)
            profile2 = self.build_user_profile(user_id2)
            
            # Genre preference similarity
            genre_similarity = self._calculate_genre_similarity(
                profile1.genre_preferences, 
                profile2.genre_preferences
            )
            
            # Rating tendency similarity
            rating_similarity = self._calculate_rating_similarity(
                profile1.rating_tendencies,
                profile2.rating_tendencies
            )
            
            # Content type similarity
            content_similarity = self._calculate_content_type_similarity(
                profile1.content_type_preferences,
                profile2.content_type_preferences
            )
            
            # Overall similarity
            overall_similarity = (genre_similarity + rating_similarity + content_similarity) / 3
            
            return {
                'genre_similarity': genre_similarity,
                'rating_similarity': rating_similarity,
                'content_type_similarity': content_similarity,
                'overall_similarity': overall_similarity
            }
            
        except Exception as e:
            logger.error(f"Error comparing users {user_id1} and {user_id2}: {str(e)}")
            return {
                'genre_similarity': 0.0,
                'rating_similarity': 0.0,
                'content_type_similarity': 0.0,
                'overall_similarity': 0.0
            }
    
    def _calculate_genre_similarity(self, prefs1: Dict[str, GenrePreference], 
                                  prefs2: Dict[str, GenrePreference]) -> float:
        """Calculate cosine similarity between genre preferences"""
        if not prefs1 or not prefs2:
            return 0.0
        
        # Get common genres
        common_genres = set(prefs1.keys()) & set(prefs2.keys())
        if not common_genres:
            return 0.0
        
        # Create vectors
        vector1 = np.array([prefs1[genre].score for genre in common_genres])
        vector2 = np.array([prefs2[genre].score for genre in common_genres])
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vector1, vector2) / (norm1 * norm2))
    
    def _calculate_rating_similarity(self, tendencies1: Dict[str, float], 
                                   tendencies2: Dict[str, float]) -> float:
        """Calculate similarity in rating tendencies"""
        if not tendencies1 or not tendencies2:
            return 0.0
        
        # Compare key metrics
        metrics = ['avg_rating', 'rating_std', 'harsh_rater_score', 'rating_consistency']
        similarities = []
        
        for metric in metrics:
            if metric in tendencies1 and metric in tendencies2:
                val1 = tendencies1[metric]
                val2 = tendencies2[metric]
                
                # Calculate normalized difference (higher = more similar)
                max_diff = 5.0 if metric == 'avg_rating' else 1.0
                diff = abs(val1 - val2) / max_diff
                similarity = max(0.0, 1.0 - diff)
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _calculate_content_type_similarity(self, prefs1: Dict[str, float], 
                                         prefs2: Dict[str, float]) -> float:
        """Calculate similarity in content type preferences"""
        if not prefs1 or not prefs2:
            return 0.0
        
        # Get common content types
        common_types = set(prefs1.keys()) & set(prefs2.keys())
        if not common_types:
            return 0.0
        
        # Create vectors
        vector1 = np.array([prefs1[ctype] for ctype in common_types])
        vector2 = np.array([prefs2[ctype] for ctype in common_types])
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vector1, vector2) / (norm1 * norm2))
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
