"""
Advanced Preference Tracking System

This module implements comprehensive preference tracking with:
- Multi-timescale preference modeling (short/medium/long-term/seasonal)
- Preference confidence scoring and uncertainty quantification
- Temporal preference evolution analysis
- Context-aware preference tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
from enum import Enum

logger = logging.getLogger(__name__)


class PreferenceTimescale(Enum):
    """Different timescales for preference tracking"""
    SHORT_TERM = "short_term"      # Last 7 days
    MEDIUM_TERM = "medium_term"    # Last 30 days  
    LONG_TERM = "long_term"        # Last 6 months
    SEASONAL = "seasonal"          # Yearly patterns


@dataclass
class PreferenceVector:
    """Multi-dimensional preference vector with metadata"""
    user_id: str
    timescale: PreferenceTimescale
    preferences: Dict[str, float]
    confidence_scores: Dict[str, float]
    timestamp: datetime
    interaction_count: int
    context: Optional[Dict[str, Any]] = None
    stability_score: float = 0.0
    diversity_score: float = 0.0


@dataclass
class PreferenceChange:
    """Represents a change in user preferences"""
    user_id: str
    preference_key: str
    old_value: float
    new_value: float
    change_magnitude: float
    change_type: str  # 'increase', 'decrease', 'new', 'removed'
    timestamp: datetime
    confidence: float
    context: Optional[Dict[str, Any]] = None


@dataclass
class PreferenceProfile:
    """Complete preference profile for a user"""
    user_id: str
    short_term: PreferenceVector
    medium_term: PreferenceVector
    long_term: PreferenceVector
    seasonal: Optional[PreferenceVector]
    overall_confidence: float
    stability_score: float
    volatility_score: float
    last_updated: datetime
    interaction_count: int
    preference_breadth: int  # Number of different preference categories


class AdvancedPreferenceTracker:
    """
    Advanced preference tracking system with multi-timescale analysis
    """
    
    def __init__(self,
                 short_term_days: int = 7,
                 medium_term_days: int = 30, 
                 long_term_days: int = 180,
                 min_interactions_confidence: int = 5,
                 confidence_decay_rate: float = 0.95,
                 stability_window_size: int = 10):
        """
        Initialize preference tracker
        
        Args:
            short_term_days: Days for short-term preferences
            medium_term_days: Days for medium-term preferences
            long_term_days: Days for long-term preferences
            min_interactions_confidence: Minimum interactions for reliable confidence
            confidence_decay_rate: Rate of confidence decay over time
            stability_window_size: Window size for stability calculation
        """
        self.short_term_days = short_term_days
        self.medium_term_days = medium_term_days
        self.long_term_days = long_term_days
        self.min_interactions_confidence = min_interactions_confidence
        self.confidence_decay_rate = confidence_decay_rate
        self.stability_window_size = stability_window_size
        
        # User preference storage
        self.user_preferences: Dict[str, PreferenceProfile] = {}
        self.preference_history: Dict[str, List[PreferenceVector]] = defaultdict(list)
        self.preference_changes: Dict[str, List[PreferenceChange]] = defaultdict(list)
        
        # Context tracking
        self.context_preferences: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        
        # Statistics
        self.global_preference_stats = {
            'most_common_preferences': defaultdict(int),
            'preference_stability_distribution': [],
            'average_preference_breadth': 0.0,
            'total_users_tracked': 0
        }
    
    def update_user_preferences(self,
                              user_id: str,
                              interactions: List[Dict[str, Any]],
                              context: Optional[Dict[str, Any]] = None) -> PreferenceProfile:
        """
        Update user preferences based on new interactions
        
        Args:
            user_id: User identifier
            interactions: List of user interactions
            context: Interaction context
            
        Returns:
            Updated preference profile
        """
        try:
            current_time = datetime.now()
            
            # Convert interactions to DataFrame for easier processing
            if not interactions:
                return self.user_preferences.get(user_id)
            
            df = pd.DataFrame(interactions)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = current_time
            
            df = df.sort_values('timestamp')
            
            # Calculate preferences for each timescale
            preference_vectors = {}
            
            for timescale in PreferenceTimescale:
                if timescale == PreferenceTimescale.SEASONAL:
                    # Skip seasonal for now if insufficient data
                    if len(df) < 50:  # Need substantial data for seasonal patterns
                        continue
                
                vector = self._calculate_preference_vector(
                    user_id, df, timescale, current_time, context
                )
                preference_vectors[timescale] = vector
            
            # Create or update preference profile
            profile = self._create_preference_profile(
                user_id, preference_vectors, df, current_time
            )
            
            # Store preference profile
            self.user_preferences[user_id] = profile
            
            # Update preference history
            for timescale, vector in preference_vectors.items():
                self.preference_history[f"{user_id}_{timescale.value}"].append(vector)
                
                # Maintain history size
                max_history = 100
                if len(self.preference_history[f"{user_id}_{timescale.value}"]) > max_history:
                    self.preference_history[f"{user_id}_{timescale.value}"] = \
                        self.preference_history[f"{user_id}_{timescale.value}"][-max_history:]
            
            # Detect and record preference changes
            self._detect_preference_changes(user_id, profile)
            
            # Update global statistics
            self._update_global_statistics(profile)
            
            logger.debug(f"Updated preferences for user {user_id} - "
                        f"confidence: {profile.overall_confidence:.3f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return self.user_preferences.get(user_id)
    
    def _calculate_preference_vector(self,
                                   user_id: str,
                                   interactions_df: pd.DataFrame,
                                   timescale: PreferenceTimescale,
                                   current_time: datetime,
                                   context: Optional[Dict[str, Any]]) -> PreferenceVector:
        """Calculate preference vector for specific timescale"""
        try:
            # Filter interactions by timescale
            if timescale == PreferenceTimescale.SHORT_TERM:
                cutoff_time = current_time - timedelta(days=self.short_term_days)
            elif timescale == PreferenceTimescale.MEDIUM_TERM:
                cutoff_time = current_time - timedelta(days=self.medium_term_days)
            elif timescale == PreferenceTimescale.LONG_TERM:
                cutoff_time = current_time - timedelta(days=self.long_term_days)
            else:  # SEASONAL
                cutoff_time = current_time - timedelta(days=365)
            
            filtered_df = interactions_df[interactions_df['timestamp'] >= cutoff_time]
            
            if len(filtered_df) == 0:
                return PreferenceVector(
                    user_id=user_id,
                    timescale=timescale,
                    preferences={},
                    confidence_scores={},
                    timestamp=current_time,
                    interaction_count=0
                )
            
            # Extract preferences from interactions
            preferences = {}
            confidence_scores = {}
            
            # Genre preferences
            if 'genres' in filtered_df.columns:
                genre_prefs = self._calculate_genre_preferences(filtered_df)
                preferences.update(genre_prefs)
                
                # Calculate confidence for genre preferences
                for genre, pref_score in genre_prefs.items():
                    genre_interactions = len(filtered_df[
                        filtered_df['genres'].str.contains(genre.replace('genre_', ''), na=False)
                    ])
                    confidence_scores[genre] = self._calculate_confidence(
                        genre_interactions, pref_score
                    )
            
            # Content type preferences
            if 'content_type' in filtered_df.columns:
                content_prefs = self._calculate_content_type_preferences(filtered_df)
                preferences.update(content_prefs)
                
                for content_type, pref_score in content_prefs.items():
                    type_interactions = len(filtered_df[
                        filtered_df['content_type'] == content_type.replace('content_', '')
                    ])
                    confidence_scores[content_type] = self._calculate_confidence(
                        type_interactions, pref_score
                    )
            
            # Rating-based preferences
            if 'rating' in filtered_df.columns:
                rating_prefs = self._calculate_rating_preferences(filtered_df)
                preferences.update(rating_prefs)
                
                for rating_pref, pref_score in rating_prefs.items():
                    confidence_scores[rating_pref] = self._calculate_confidence(
                        len(filtered_df), pref_score
                    )
            
            # Temporal preferences
            temporal_prefs = self._calculate_temporal_preferences(filtered_df)
            preferences.update(temporal_prefs)
            
            for temporal_pref, pref_score in temporal_prefs.items():
                confidence_scores[temporal_pref] = self._calculate_confidence(
                    len(filtered_df), pref_score
                )
            
            # Calculate stability and diversity scores
            stability_score = self._calculate_stability_score(user_id, timescale, preferences)
            diversity_score = self._calculate_diversity_score(preferences)
            
            return PreferenceVector(
                user_id=user_id,
                timescale=timescale,
                preferences=preferences,
                confidence_scores=confidence_scores,
                timestamp=current_time,
                interaction_count=len(filtered_df),
                context=context,
                stability_score=stability_score,
                diversity_score=diversity_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating preference vector: {e}")
            return PreferenceVector(
                user_id=user_id,
                timescale=timescale,
                preferences={},
                confidence_scores={},
                timestamp=current_time,
                interaction_count=0
            )
    
    def _calculate_genre_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate genre preferences from interactions"""
        try:
            genre_scores = defaultdict(list)
            
            for _, row in df.iterrows():
                if pd.isna(row.get('genres')) or pd.isna(row.get('rating')):
                    continue
                
                genres = row['genres']
                rating = float(row['rating'])
                
                # Handle different genre formats
                if isinstance(genres, str):
                    if '|' in genres:
                        genre_list = genres.split('|')
                    else:
                        genre_list = [genres]
                elif isinstance(genres, list):
                    genre_list = genres
                else:
                    continue
                
                # Normalize rating to preference score
                preference_score = (rating - 2.5) / 2.5  # Convert 1-5 to -1 to 1
                
                for genre in genre_list:
                    genre = genre.strip()
                    if genre:
                        genre_scores[f"genre_{genre.lower()}"].append(preference_score)
            
            # Calculate average preferences
            genre_preferences = {}
            for genre, scores in genre_scores.items():
                if len(scores) >= 2:  # Need at least 2 interactions
                    genre_preferences[genre] = float(np.mean(scores))
            
            return genre_preferences
            
        except Exception as e:
            logger.error(f"Error calculating genre preferences: {e}")
            return {}
    
    def _calculate_content_type_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate content type preferences"""
        try:
            content_scores = defaultdict(list)
            
            for _, row in df.iterrows():
                if pd.isna(row.get('content_type')) or pd.isna(row.get('rating')):
                    continue
                
                content_type = str(row['content_type']).lower()
                rating = float(row['rating'])
                preference_score = (rating - 2.5) / 2.5
                
                content_scores[f"content_{content_type}"].append(preference_score)
            
            # Calculate average preferences
            content_preferences = {}
            for content_type, scores in content_scores.items():
                if len(scores) >= 2:
                    content_preferences[content_type] = float(np.mean(scores))
            
            return content_preferences
            
        except Exception as e:
            logger.error(f"Error calculating content type preferences: {e}")
            return {}
    
    def _calculate_rating_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate rating-based preferences"""
        try:
            if 'rating' not in df.columns:
                return {}
            
            ratings = df['rating'].dropna()
            if len(ratings) < 3:
                return {}
            
            preferences = {}
            
            # Average rating preference
            avg_rating = float(ratings.mean())
            preferences['avg_rating'] = (avg_rating - 2.5) / 2.5
            
            # Rating variance (preference for consistency vs diversity)
            rating_std = float(ratings.std())
            preferences['rating_consistency'] = max(0.0, 1.0 - (rating_std / 2.0))
            
            # High rating preference (tendency to rate things highly)
            high_ratings = len(ratings[ratings >= 4]) / len(ratings)
            preferences['high_rating_tendency'] = high_ratings
            
            # Low rating preference (tendency to be critical)
            low_ratings = len(ratings[ratings <= 2]) / len(ratings)
            preferences['critical_tendency'] = low_ratings
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error calculating rating preferences: {e}")
            return {}
    
    def _calculate_temporal_preferences(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate temporal usage preferences"""
        try:
            if 'timestamp' not in df.columns:
                return {}
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            preferences = {}
            
            # Hour preferences
            if len(df) >= 10:
                hourly_activity = df['hour'].value_counts(normalize=True)
                
                # Morning preference (6-12)
                morning_activity = hourly_activity.loc[
                    hourly_activity.index.intersection(range(6, 12))
                ].sum()
                preferences['morning_preference'] = morning_activity
                
                # Evening preference (18-24)
                evening_activity = hourly_activity.loc[
                    hourly_activity.index.intersection(range(18, 24))
                ].sum()
                preferences['evening_preference'] = evening_activity
                
                # Night preference (0-6)
                night_activity = hourly_activity.loc[
                    hourly_activity.index.intersection(range(0, 6))
                ].sum()
                preferences['night_preference'] = night_activity
            
            # Day of week preferences
            if len(df) >= 14:  # At least 2 weeks of data
                daily_activity = df['day_of_week'].value_counts(normalize=True)
                
                # Weekend preference (Saturday=5, Sunday=6)
                weekend_activity = daily_activity.loc[
                    daily_activity.index.intersection([5, 6])
                ].sum()
                preferences['weekend_preference'] = weekend_activity
                
                # Weekday preference (Monday=0 to Friday=4)
                weekday_activity = daily_activity.loc[
                    daily_activity.index.intersection(range(0, 5))
                ].sum()
                preferences['weekday_preference'] = weekday_activity
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error calculating temporal preferences: {e}")
            return {}
    
    def _calculate_confidence(self, interaction_count: int, preference_strength: float) -> float:
        """Calculate confidence score for a preference"""
        try:
            # Base confidence from interaction count
            interaction_confidence = min(1.0, interaction_count / self.min_interactions_confidence)
            
            # Strength confidence from how strong the preference is
            strength_confidence = min(1.0, abs(preference_strength))
            
            # Combined confidence
            confidence = (interaction_confidence * 0.7) + (strength_confidence * 0.3)
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _calculate_stability_score(self, user_id: str, timescale: PreferenceTimescale,
                                 current_preferences: Dict[str, float]) -> float:
        """Calculate preference stability score"""
        try:
            history_key = f"{user_id}_{timescale.value}"
            history = self.preference_history.get(history_key, [])
            
            if len(history) < 2:
                return 0.5  # Neutral stability for insufficient history
            
            # Get recent preference vectors for comparison
            recent_vectors = history[-self.stability_window_size:]
            
            if len(recent_vectors) < 2:
                return 0.5
            
            # Calculate stability as average cosine similarity between consecutive vectors
            similarities = []
            
            for i in range(len(recent_vectors) - 1):
                prev_prefs = recent_vectors[i].preferences
                curr_prefs = recent_vectors[i + 1].preferences
                
                similarity = self._calculate_preference_similarity(prev_prefs, curr_prefs)
                similarities.append(similarity)
            
            # Include current preferences in comparison
            if recent_vectors:
                similarity = self._calculate_preference_similarity(
                    recent_vectors[-1].preferences, current_preferences
                )
                similarities.append(similarity)
            
            stability_score = float(np.mean(similarities)) if similarities else 0.5
            return max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 0.5
    
    def _calculate_preference_similarity(self, prefs1: Dict[str, float], 
                                       prefs2: Dict[str, float]) -> float:
        """Calculate similarity between two preference dictionaries"""
        try:
            if not prefs1 or not prefs2:
                return 0.0
            
            # Get common keys
            common_keys = set(prefs1.keys()) & set(prefs2.keys())
            
            if not common_keys:
                return 0.0
            
            # Calculate cosine similarity
            vec1 = np.array([prefs1[key] for key in common_keys])
            vec2 = np.array([prefs2[key] for key in common_keys])
            
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating preference similarity: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, preferences: Dict[str, float]) -> float:
        """Calculate preference diversity score"""
        try:
            if not preferences:
                return 0.0
            
            # Shannon entropy as diversity measure
            preference_values = list(preferences.values())
            
            # Normalize to positive values for entropy calculation
            min_val = min(preference_values) if preference_values else 0
            if min_val < 0:
                adjusted_values = [v - min_val + 0.001 for v in preference_values]
            else:
                adjusted_values = [v + 0.001 for v in preference_values]  # Add small constant
            
            # Calculate entropy
            total = sum(adjusted_values)
            probabilities = [v / total for v in adjusted_values]
            
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
            
            # Normalize entropy to [0, 1]
            max_entropy = math.log(len(preferences)) if len(preferences) > 1 else 1
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
            
            return float(diversity_score)
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.0
    
    def _create_preference_profile(self,
                                 user_id: str,
                                 preference_vectors: Dict[PreferenceTimescale, PreferenceVector],
                                 interactions_df: pd.DataFrame,
                                 current_time: datetime) -> PreferenceProfile:
        """Create complete preference profile from vectors"""
        try:
            # Get vectors for each timescale
            short_term = preference_vectors.get(PreferenceTimescale.SHORT_TERM)
            medium_term = preference_vectors.get(PreferenceTimescale.MEDIUM_TERM)
            long_term = preference_vectors.get(PreferenceTimescale.LONG_TERM)
            seasonal = preference_vectors.get(PreferenceTimescale.SEASONAL)
            
            # Calculate overall confidence as weighted average
            confidences = []
            weights = []
            
            if short_term and short_term.confidence_scores:
                avg_conf = np.mean(list(short_term.confidence_scores.values()))
                confidences.append(avg_conf)
                weights.append(0.3)  # Recent preferences weighted higher
            
            if medium_term and medium_term.confidence_scores:
                avg_conf = np.mean(list(medium_term.confidence_scores.values()))
                confidences.append(avg_conf)
                weights.append(0.4)
            
            if long_term and long_term.confidence_scores:
                avg_conf = np.mean(list(long_term.confidence_scores.values()))
                confidences.append(avg_conf)
                weights.append(0.3)
            
            if confidences:
                overall_confidence = float(np.average(confidences, weights=weights))
            else:
                overall_confidence = 0.0
            
            # Calculate stability score (use medium-term as representative)
            stability_score = medium_term.stability_score if medium_term else 0.0
            
            # Calculate volatility score (inverse of stability)
            volatility_score = 1.0 - stability_score
            
            # Calculate preference breadth (number of unique preference categories)
            all_preferences = set()
            for vector in preference_vectors.values():
                if vector:
                    all_preferences.update(vector.preferences.keys())
            
            preference_breadth = len(all_preferences)
            
            # Total interaction count
            interaction_count = len(interactions_df)
            
            profile = PreferenceProfile(
                user_id=user_id,
                short_term=short_term,
                medium_term=medium_term,
                long_term=long_term,
                seasonal=seasonal,
                overall_confidence=overall_confidence,
                stability_score=stability_score,
                volatility_score=volatility_score,
                last_updated=current_time,
                interaction_count=interaction_count,
                preference_breadth=preference_breadth
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating preference profile: {e}")
            # Return minimal profile
            return PreferenceProfile(
                user_id=user_id,
                short_term=None,
                medium_term=None,
                long_term=None,
                seasonal=None,
                overall_confidence=0.0,
                stability_score=0.0,
                volatility_score=1.0,
                last_updated=current_time,
                interaction_count=0,
                preference_breadth=0
            )
    
    def _detect_preference_changes(self, user_id: str, current_profile: PreferenceProfile):
        """Detect and record significant preference changes"""
        try:
            if user_id not in self.user_preferences:
                return  # No previous profile to compare
            
            previous_profile = self.user_preferences[user_id]
            changes = []
            
            # Compare medium-term preferences (most stable for change detection)
            if (current_profile.medium_term and previous_profile.medium_term and
                current_profile.medium_term.preferences and previous_profile.medium_term.preferences):
                
                current_prefs = current_profile.medium_term.preferences
                previous_prefs = previous_profile.medium_term.preferences
                
                # Check for changes in existing preferences
                for pref_key in set(current_prefs.keys()) | set(previous_prefs.keys()):
                    current_value = current_prefs.get(pref_key, 0.0)
                    previous_value = previous_prefs.get(pref_key, 0.0)
                    
                    change_magnitude = abs(current_value - previous_value)
                    
                    # Only record significant changes
                    if change_magnitude > 0.2:  # Threshold for significant change
                        if pref_key not in previous_prefs:
                            change_type = 'new'
                        elif pref_key not in current_prefs:
                            change_type = 'removed'
                        elif current_value > previous_value:
                            change_type = 'increase'
                        else:
                            change_type = 'decrease'
                        
                        # Calculate confidence in change detection
                        confidence = min(1.0, change_magnitude * 2)  # Scale to [0, 1]
                        
                        change = PreferenceChange(
                            user_id=user_id,
                            preference_key=pref_key,
                            old_value=previous_value,
                            new_value=current_value,
                            change_magnitude=change_magnitude,
                            change_type=change_type,
                            timestamp=datetime.now(),
                            confidence=confidence
                        )
                        
                        changes.append(change)
            
            # Store changes
            if changes:
                self.preference_changes[user_id].extend(changes)
                
                # Maintain change history size
                max_changes = 50
                if len(self.preference_changes[user_id]) > max_changes:
                    self.preference_changes[user_id] = self.preference_changes[user_id][-max_changes:]
                
                logger.debug(f"Detected {len(changes)} preference changes for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error detecting preference changes: {e}")
    
    def _update_global_statistics(self, profile: PreferenceProfile):
        """Update global preference statistics"""
        try:
            # Update most common preferences
            if profile.medium_term and profile.medium_term.preferences:
                for pref_key in profile.medium_term.preferences.keys():
                    self.global_preference_stats['most_common_preferences'][pref_key] += 1
            
            # Update stability distribution
            if len(self.global_preference_stats['preference_stability_distribution']) > 1000:
                self.global_preference_stats['preference_stability_distribution'] = \
                    self.global_preference_stats['preference_stability_distribution'][-1000:]
            
            self.global_preference_stats['preference_stability_distribution'].append(
                profile.stability_score
            )
            
            # Update average preference breadth
            current_breadth = self.global_preference_stats['average_preference_breadth']
            total_users = self.global_preference_stats['total_users_tracked']
            
            new_breadth = ((current_breadth * total_users) + profile.preference_breadth) / (total_users + 1)
            self.global_preference_stats['average_preference_breadth'] = new_breadth
            self.global_preference_stats['total_users_tracked'] = total_users + 1
            
        except Exception as e:
            logger.error(f"Error updating global statistics: {e}")
    
    def get_user_preference_profile(self, user_id: str) -> Optional[PreferenceProfile]:
        """Get complete preference profile for user"""
        return self.user_preferences.get(user_id)
    
    def get_user_preference_changes(self, user_id: str, 
                                  days_back: int = 30) -> List[PreferenceChange]:
        """Get recent preference changes for user"""
        try:
            if user_id not in self.preference_changes:
                return []
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_changes = [
                change for change in self.preference_changes[user_id]
                if change.timestamp >= cutoff_time
            ]
            
            return sorted(recent_changes, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting user preference changes: {e}")
            return []
    
    def get_preference_trends(self, user_id: str, 
                            timescale: PreferenceTimescale = PreferenceTimescale.MEDIUM_TERM,
                            days_back: int = 90) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get preference trends over time"""
        try:
            history_key = f"{user_id}_{timescale.value}"
            history = self.preference_history.get(history_key, [])
            
            if not history:
                return {}
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_history = [
                vector for vector in history
                if vector.timestamp >= cutoff_time
            ]
            
            # Group by preference key
            trends = defaultdict(list)
            
            for vector in recent_history:
                for pref_key, pref_value in vector.preferences.items():
                    trends[pref_key].append((vector.timestamp, pref_value))
            
            # Sort by timestamp
            for pref_key in trends:
                trends[pref_key].sort(key=lambda x: x[0])
            
            return dict(trends)
            
        except Exception as e:
            logger.error(f"Error getting preference trends: {e}")
            return {}
    
    def get_global_preference_statistics(self) -> Dict[str, Any]:
        """Get global preference statistics"""
        try:
            # Calculate additional statistics
            stability_stats = {}
            if self.global_preference_stats['preference_stability_distribution']:
                stability_values = self.global_preference_stats['preference_stability_distribution']
                stability_stats = {
                    'mean': float(np.mean(stability_values)),
                    'std': float(np.std(stability_values)),
                    'median': float(np.median(stability_values)),
                    'min': float(np.min(stability_values)),
                    'max': float(np.max(stability_values))
                }
            
            # Top preferences
            top_preferences = dict(
                sorted(
                    self.global_preference_stats['most_common_preferences'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            )
            
            return {
                'total_users_tracked': self.global_preference_stats['total_users_tracked'],
                'average_preference_breadth': self.global_preference_stats['average_preference_breadth'],
                'stability_statistics': stability_stats,
                'top_preferences': top_preferences,
                'users_with_preferences': len(self.user_preferences),
                'users_with_changes': len(self.preference_changes),
                'total_preference_changes': sum(
                    len(changes) for changes in self.preference_changes.values()
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting global statistics: {e}")
            return {}
    
    def predict_preference_evolution(self, user_id: str, 
                                   days_ahead: int = 30) -> Dict[str, float]:
        """Predict how user preferences might evolve"""
        try:
            profile = self.user_preferences.get(user_id)
            if not profile or not profile.medium_term:
                return {}
            
            # Get preference trends
            trends = self.get_preference_trends(user_id, PreferenceTimescale.MEDIUM_TERM, 90)
            
            predictions = {}
            
            for pref_key, trend_data in trends.items():
                if len(trend_data) < 3:
                    continue
                
                # Extract values and timestamps
                timestamps = [t for t, v in trend_data]
                values = [v for t, v in trend_data]
                
                # Simple linear trend extrapolation
                if len(values) >= 3:
                    # Use last 3 points for trend
                    recent_values = values[-3:]
                    x = np.arange(len(recent_values))
                    
                    # Linear regression
                    slope, intercept = np.polyfit(x, recent_values, 1)
                    
                    # Predict future value
                    future_x = len(recent_values) + (days_ahead / 30)  # Rough approximation
                    predicted_value = slope * future_x + intercept
                    
                    # Clip to reasonable bounds
                    predicted_value = max(-1.0, min(1.0, predicted_value))
                    predictions[pref_key] = float(predicted_value)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting preference evolution: {e}")
            return {}
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences in a dictionary format"""
        try:
            profile = self.user_preferences.get(user_id)
            if not profile:
                return {
                    'genre_preferences': {},
                    'content_type_preferences': {},
                    'feature_preferences': {},
                    'temporal_preferences': {},
                    'preference_breadth': 0.0,
                    'exploration_tendency': 0.5,
                    'stability_score': 0.0
                }
            
            return {
                'genre_preferences': dict(profile.genre_preferences),
                'content_type_preferences': dict(profile.content_type_preferences),
                'feature_preferences': dict(profile.feature_preferences),
                'temporal_preferences': dict(profile.temporal_preferences),
                'preference_breadth': profile.preference_breadth,
                'exploration_tendency': profile.exploration_tendency,
                'stability_score': profile.stability_score,
                'confidence_score': getattr(profile, 'confidence_score', 0.5)
            }
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            return {}
    
    def update_preferences(self, user_id: str, feedback_data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update preferences based on feedback data (wrapper for update_user_preferences)
        
        Args:
            user_id: User identifier
            feedback_data: Processed feedback data or list of interactions
            context: Optional context information
            
        Returns:
            True if update was successful
        """
        try:
            # Convert feedback_data to interactions format if needed
            if isinstance(feedback_data, dict):
                # Single feedback item
                interactions = [feedback_data]
            elif isinstance(feedback_data, list):
                # List of interactions
                interactions = feedback_data
            else:
                logger.warning(f"Unexpected feedback_data type: {type(feedback_data)}")
                return False
            
            # Update user preferences
            profile = self.update_user_preferences(user_id, interactions, context)
            return profile is not None
            
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False
