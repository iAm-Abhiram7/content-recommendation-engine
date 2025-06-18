"""
Knowledge-Based Recommender

Implements rule-based and knowledge-driven recommendations:
- Trending and popular content
- New releases and critically acclaimed items
- User-defined filters (genre, language, mood, etc.)
- Contextual recommendations (time, device, location)
- Business rule integration
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class KnowledgeBasedRecommender:
    """
    Knowledge-based and rule-based recommender system
    """
    
    def __init__(self, 
                 trending_window_days: int = 7,
                 new_release_days: int = 30,
                 min_ratings_for_trending: int = 50,
                 acclaim_threshold: float = 4.0):
        """
        Initialize knowledge-based recommender
        
        Args:
            trending_window_days: Window for calculating trending items
            new_release_days: Days to consider item as new release
            min_ratings_for_trending: Minimum ratings needed for trending status
            acclaim_threshold: Minimum rating for acclaimed status
        """
        self.trending_window_days = trending_window_days
        self.new_release_days = new_release_days
        self.min_ratings_for_trending = min_ratings_for_trending
        self.acclaim_threshold = acclaim_threshold
        
        # Data storage
        self.items_df = None
        self.ratings_df = None
        self.user_context = None
        
        # Computed lists
        self.trending_items = []
        self.new_releases = []
        self.acclaimed_items = []
        self.popular_items = []
        
        # Business rules
        self.rules = {}
        self.genre_mappings = {}
        self.mood_mappings = {}
        
        # Context handlers
        self.context_handlers = {
            'time_of_day': self._handle_time_context,
            'day_of_week': self._handle_day_context,
            'season': self._handle_season_context,
            'device': self._handle_device_context,
            'location': self._handle_location_context,
            'weather': self._handle_weather_context
        }
        
    def fit(self, 
            items_df: pd.DataFrame,
            ratings_df: pd.DataFrame,
            user_context: Optional[pd.DataFrame] = None,
            business_rules: Optional[Dict] = None):
        """
        Initialize knowledge base with data
        
        Args:
            items_df: DataFrame with item metadata
            ratings_df: DataFrame with user ratings and timestamps
            user_context: Optional user context data
            business_rules: Optional business rules dictionary
        """
        logger.info("Initializing knowledge-based recommender")
        
        self.items_df = items_df.copy()
        self.ratings_df = ratings_df.copy()
        self.user_context = user_context.copy() if user_context is not None else None
        
        if business_rules:
            self.rules.update(business_rules)
        
        # Compute knowledge lists
        self._compute_trending_items()
        self._compute_new_releases()
        self._compute_acclaimed_items()
        self._compute_popular_items()
        
        # Initialize genre and mood mappings
        self._initialize_mappings()
        
        logger.info("Knowledge-based recommender initialized")
    
    def _compute_trending_items(self):
        """Compute currently trending items"""
        if self.ratings_df is None:
            return
        
        # Get recent ratings within trending window
        cutoff_date = datetime.now() - timedelta(days=self.trending_window_days)
        
        if 'timestamp' in self.ratings_df.columns:
            recent_ratings = self.ratings_df[
                pd.to_datetime(self.ratings_df['timestamp']) >= cutoff_date
            ]
        else:
            # If no timestamp, use all ratings
            recent_ratings = self.ratings_df
        
        # Calculate trending score (rating count * average rating)
        trending_stats = recent_ratings.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).round(3)
        
        trending_stats.columns = ['rating_count', 'avg_rating']
        trending_stats = trending_stats.reset_index()
        
        # Filter by minimum ratings threshold
        trending_stats = trending_stats[
            trending_stats['rating_count'] >= self.min_ratings_for_trending
        ]
        
        # Calculate trending score
        trending_stats['trending_score'] = (
            trending_stats['rating_count'] * trending_stats['avg_rating']
        )
        
        # Sort by trending score
        trending_stats = trending_stats.sort_values(
            'trending_score', ascending=False
        )
        
        self.trending_items = trending_stats['item_id'].tolist()
        
        logger.info(f"Computed {len(self.trending_items)} trending items")
    
    def _compute_new_releases(self):
        """Compute new release items"""
        if self.items_df is None or 'release_date' not in self.items_df.columns:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.new_release_days)
        
        # Convert release_date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.items_df['release_date']):
            self.items_df['release_date'] = pd.to_datetime(
                self.items_df['release_date'], errors='coerce'
            )
        
        # Filter new releases
        new_releases = self.items_df[
            self.items_df['release_date'] >= cutoff_date
        ]
        
        # Sort by release date (newest first)
        new_releases = new_releases.sort_values('release_date', ascending=False)
        
        self.new_releases = new_releases['item_id'].tolist()
        
        logger.info(f"Found {len(self.new_releases)} new releases")
    
    def _compute_acclaimed_items(self):
        """Compute critically acclaimed items"""
        if self.ratings_df is None:
            return
        
        # Calculate average ratings
        item_ratings = self.ratings_df.groupby('item_id')['rating'].agg([
            'mean', 'count'
        ]).reset_index()
        
        # Filter by acclaim threshold and minimum ratings
        acclaimed = item_ratings[
            (item_ratings['mean'] >= self.acclaim_threshold) &
            (item_ratings['count'] >= self.min_ratings_for_trending)
        ]
        
        # Sort by average rating
        acclaimed = acclaimed.sort_values('mean', ascending=False)
        
        self.acclaimed_items = acclaimed['item_id'].tolist()
        
        logger.info(f"Found {len(self.acclaimed_items)} acclaimed items")
    
    def _compute_popular_items(self):
        """Compute overall popular items"""
        if self.ratings_df is None:
            return
        
        # Calculate popularity score (rating count weighted by average rating)
        popularity_stats = self.ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean']
        })
        
        popularity_stats.columns = ['rating_count', 'avg_rating']
        popularity_stats = popularity_stats.reset_index()
        
        # Calculate popularity score
        popularity_stats['popularity_score'] = (
            np.log1p(popularity_stats['rating_count']) * 
            popularity_stats['avg_rating']
        )
        
        # Sort by popularity
        popularity_stats = popularity_stats.sort_values(
            'popularity_score', ascending=False
        )
        
        self.popular_items = popularity_stats['item_id'].tolist()
        
        logger.info(f"Computed {len(self.popular_items)} popular items")
    
    def _initialize_mappings(self):
        """Initialize genre and mood mappings"""
        # Genre-based mood mappings
        self.mood_mappings = {
            'relaxing': ['drama', 'documentary', 'romance'],
            'exciting': ['action', 'thriller', 'adventure'],
            'fun': ['comedy', 'animation', 'family'],
            'thoughtful': ['drama', 'documentary', 'biography'],
            'romantic': ['romance', 'romantic comedy'],
            'scary': ['horror', 'thriller'],
            'uplifting': ['comedy', 'family', 'musical']
        }
        
        # Time-based genre mappings
        self.time_mappings = {
            'morning': ['news', 'documentary', 'educational'],
            'afternoon': ['comedy', 'family', 'animation'],
            'evening': ['drama', 'thriller', 'action'],
            'night': ['horror', 'thriller', 'adult'],
            'weekend': ['family', 'comedy', 'adventure'],
            'weekday': ['news', 'documentary', 'educational']
        }
    
    def recommend_trending(self, 
                          n_recommendations: int = 10,
                          filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recommend trending items
        
        Args:
            n_recommendations: Number of recommendations
            filters: Optional filters to apply
            
        Returns:
            List of trending recommendations
        """
        filtered_items = self._apply_filters(self.trending_items, filters)
        
        recommendations = []
        for i, item_id in enumerate(filtered_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': 1.0 - (i * 0.1),  # Decreasing score by rank
                'method': 'trending',
                'rank': i + 1,
                'reason': 'Currently trending'
            })
        
        return recommendations
    
    def recommend_new_releases(self, 
                              n_recommendations: int = 10,
                              filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recommend new releases
        
        Args:
            n_recommendations: Number of recommendations
            filters: Optional filters to apply
            
        Returns:
            List of new release recommendations
        """
        filtered_items = self._apply_filters(self.new_releases, filters)
        
        recommendations = []
        for i, item_id in enumerate(filtered_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': 1.0 - (i * 0.05),  # Decreasing score by rank
                'method': 'new_releases',
                'rank': i + 1,
                'reason': 'New release'
            })
        
        return recommendations
    
    def recommend_acclaimed(self, 
                           n_recommendations: int = 10,
                           filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recommend critically acclaimed items
        
        Args:
            n_recommendations: Number of recommendations
            filters: Optional filters to apply
            
        Returns:
            List of acclaimed recommendations
        """
        filtered_items = self._apply_filters(self.acclaimed_items, filters)
        
        recommendations = []
        for i, item_id in enumerate(filtered_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': 1.0 - (i * 0.02),  # Smaller decrease for quality
                'method': 'acclaimed',
                'rank': i + 1,
                'reason': 'Critically acclaimed'
            })
        
        return recommendations
    
    def recommend_by_context(self, 
                            user_id: str,
                            context: Dict[str, Any],
                            n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend based on contextual information
        
        Args:
            user_id: User identifier
            context: Context dictionary (time, location, device, etc.)
            n_recommendations: Number of recommendations
            
        Returns:
            Context-aware recommendations
        """
        context_scores = defaultdict(list)
        
        # Process each context dimension
        for context_type, context_value in context.items():
            if context_type in self.context_handlers:
                handler = self.context_handlers[context_type]
                scores = handler(context_value)
                
                for item_id, score in scores.items():
                    context_scores[item_id].append(score)
        
        # Aggregate context scores
        final_scores = {}
        for item_id, scores in context_scores.items():
            final_scores[item_id] = np.mean(scores)
        
        # Sort by aggregated score
        sorted_items = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        recommendations = []
        for i, (item_id, score) in enumerate(sorted_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': float(score),
                'method': 'contextual',
                'context': context,
                'reason': f'Matches your current context'
            })
        
        return recommendations
    
    def recommend_by_mood(self, 
                         mood: str,
                         n_recommendations: int = 10,
                         filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recommend based on user mood
        
        Args:
            mood: User's current mood
            n_recommendations: Number of recommendations
            filters: Optional additional filters
            
        Returns:
            Mood-based recommendations
        """
        if mood not in self.mood_mappings:
            logger.warning(f"Unknown mood: {mood}")
            return []
        
        # Get genres matching the mood
        mood_genres = self.mood_mappings[mood]
        
        # Filter items by mood genres
        if self.items_df is not None and 'genre' in self.items_df.columns:
            mood_items = self.items_df[
                self.items_df['genre'].str.lower().isin(
                    [g.lower() for g in mood_genres]
                )
            ]['item_id'].tolist()
        else:
            # Fallback to popular items
            mood_items = self.popular_items
        
        # Apply additional filters
        filtered_items = self._apply_filters(mood_items, filters)
        
        recommendations = []
        for i, item_id in enumerate(filtered_items[:n_recommendations]):
            recommendations.append({
                'item_id': item_id,
                'score': 1.0 - (i * 0.05),
                'method': 'mood_based',
                'mood': mood,
                'reason': f'Matches your {mood} mood'
            })
        
        return recommendations
    
    def apply_business_rules(self, 
                           recommendations: List[Dict[str, Any]],
                           user_id: str,
                           rules: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Apply business rules to filter/modify recommendations
        
        Args:
            recommendations: Input recommendations
            user_id: User identifier
            rules: Optional rules to apply
            
        Returns:
            Filtered recommendations
        """
        if rules is None:
            rules = self.rules
        
        filtered_recs = recommendations.copy()
        
        # Apply each rule
        for rule_name, rule_config in rules.items():
            filtered_recs = self._apply_single_rule(
                filtered_recs, user_id, rule_name, rule_config
            )
        
        return filtered_recs
    
    def _apply_filters(self, 
                      item_list: List[str], 
                      filters: Optional[Dict]) -> List[str]:
        """Apply filters to item list"""
        if not filters or self.items_df is None:
            return item_list
        
        filtered_df = self.items_df[self.items_df['item_id'].isin(item_list)]
        
        # Apply each filter
        for filter_key, filter_value in filters.items():
            if filter_key in filtered_df.columns:
                if isinstance(filter_value, list):
                    filtered_df = filtered_df[
                        filtered_df[filter_key].isin(filter_value)
                    ]
                else:
                    filtered_df = filtered_df[
                        filtered_df[filter_key] == filter_value
                    ]
        
        return filtered_df['item_id'].tolist()
    
    def _apply_single_rule(self, 
                          recommendations: List[Dict[str, Any]],
                          user_id: str,
                          rule_name: str,
                          rule_config: Dict) -> List[Dict[str, Any]]:
        """Apply a single business rule"""
        rule_type = rule_config.get('type', 'filter')
        
        if rule_type == 'filter':
            return self._apply_filter_rule(recommendations, rule_config)
        elif rule_type == 'boost':
            return self._apply_boost_rule(recommendations, rule_config)
        elif rule_type == 'limit':
            return self._apply_limit_rule(recommendations, rule_config)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return recommendations
    
    def _apply_filter_rule(self, 
                          recommendations: List[Dict[str, Any]],
                          rule_config: Dict) -> List[Dict[str, Any]]:
        """Apply filter rule (remove items matching criteria)"""
        filter_field = rule_config.get('field')
        filter_values = rule_config.get('values', [])
        
        if not filter_field or not filter_values:
            return recommendations
        
        # Get item metadata
        if self.items_df is None or filter_field not in self.items_df.columns:
            return recommendations
        
        # Filter out items matching criteria
        filtered_recs = []
        for rec in recommendations:
            item_data = self.items_df[
                self.items_df['item_id'] == rec['item_id']
            ]
            
            if not item_data.empty:
                item_value = item_data[filter_field].iloc[0]
                if item_value not in filter_values:
                    filtered_recs.append(rec)
            else:
                filtered_recs.append(rec)  # Keep if no metadata
        
        return filtered_recs
    
    def _apply_boost_rule(self, 
                         recommendations: List[Dict[str, Any]],
                         rule_config: Dict) -> List[Dict[str, Any]]:
        """Apply boost rule (increase scores for matching items)"""
        boost_field = rule_config.get('field')
        boost_values = rule_config.get('values', [])
        boost_factor = rule_config.get('boost_factor', 1.5)
        
        if not boost_field or not boost_values:
            return recommendations
        
        # Get item metadata
        if self.items_df is None or boost_field not in self.items_df.columns:
            return recommendations
        
        # Apply boost to matching items
        for rec in recommendations:
            item_data = self.items_df[
                self.items_df['item_id'] == rec['item_id']
            ]
            
            if not item_data.empty:
                item_value = item_data[boost_field].iloc[0]
                if item_value in boost_values:
                    rec['score'] *= boost_factor
                    rec['boosted'] = True
        
        # Re-sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def _apply_limit_rule(self, 
                         recommendations: List[Dict[str, Any]],
                         rule_config: Dict) -> List[Dict[str, Any]]:
        """Apply limit rule (limit items from specific category)"""
        limit_field = rule_config.get('field')
        limit_values = rule_config.get('values', [])
        max_count = rule_config.get('max_count', 1)
        
        if not limit_field or not limit_values:
            return recommendations
        
        # Get item metadata
        if self.items_df is None or limit_field not in self.items_df.columns:
            return recommendations
        
        # Count items per category and limit
        category_counts = defaultdict(int)
        filtered_recs = []
        
        for rec in recommendations:
            item_data = self.items_df[
                self.items_df['item_id'] == rec['item_id']
            ]
            
            if not item_data.empty:
                item_value = item_data[limit_field].iloc[0]
                
                if item_value in limit_values:
                    if category_counts[item_value] < max_count:
                        category_counts[item_value] += 1
                        filtered_recs.append(rec)
                    # Skip if limit reached
                else:
                    filtered_recs.append(rec)  # No limit for other categories
            else:
                filtered_recs.append(rec)  # Keep if no metadata
        
        return filtered_recs
    
    # Context handlers
    def _handle_time_context(self, time_value: str) -> Dict[str, float]:
        """Handle time of day context"""
        time_preferences = self.time_mappings.get(time_value, [])
        
        scores = {}
        if self.items_df is not None and 'genre' in self.items_df.columns:
            for _, item in self.items_df.iterrows():
                item_genre = item['genre'].lower()
                
                # Score based on time preference
                if any(pref in item_genre for pref in time_preferences):
                    scores[item['item_id']] = 0.8
                else:
                    scores[item['item_id']] = 0.5
        
        return scores
    
    def _handle_day_context(self, day_value: str) -> Dict[str, float]:
        """Handle day of week context"""
        if day_value in ['saturday', 'sunday']:
            return self._handle_time_context('weekend')
        else:
            return self._handle_time_context('weekday')
    
    def _handle_season_context(self, season_value: str) -> Dict[str, float]:
        """Handle seasonal context"""
        season_preferences = {
            'summer': ['action', 'adventure', 'comedy'],
            'winter': ['drama', 'horror', 'thriller'],
            'spring': ['romance', 'family', 'comedy'],
            'fall': ['thriller', 'drama', 'horror']
        }
        
        preferences = season_preferences.get(season_value, [])
        
        scores = {}
        if self.items_df is not None and 'genre' in self.items_df.columns:
            for _, item in self.items_df.iterrows():
                item_genre = item['genre'].lower()
                
                if any(pref in item_genre for pref in preferences):
                    scores[item['item_id']] = 0.7
                else:
                    scores[item['item_id']] = 0.5
        
        return scores
    
    def _handle_device_context(self, device_value: str) -> Dict[str, float]:
        """Handle device context"""
        device_preferences = {
            'mobile': ['short_form', 'music', 'podcast'],
            'tv': ['movie', 'series', 'documentary'],
            'tablet': ['book', 'magazine', 'comic'],
            'desktop': ['game', 'software', 'course']
        }
        
        preferences = device_preferences.get(device_value, [])
        
        scores = {}
        if self.items_df is not None and 'content_type' in self.items_df.columns:
            for _, item in self.items_df.iterrows():
                content_type = item['content_type'].lower()
                
                if content_type in preferences:
                    scores[item['item_id']] = 0.8
                else:
                    scores[item['item_id']] = 0.5
        
        return scores
    
    def _handle_location_context(self, location_value: str) -> Dict[str, float]:
        """Handle location context"""
        # Simple location-based preferences
        location_preferences = {
            'home': ['movie', 'series', 'book'],
            'work': ['podcast', 'audiobook', 'music'],
            'commute': ['podcast', 'audiobook', 'music'],
            'gym': ['music', 'podcast'],
            'travel': ['book', 'movie', 'music']
        }
        
        preferences = location_preferences.get(location_value, [])
        
        scores = {}
        if self.items_df is not None and 'content_type' in self.items_df.columns:
            for _, item in self.items_df.iterrows():
                content_type = item['content_type'].lower()
                
                if content_type in preferences:
                    scores[item['item_id']] = 0.7
                else:
                    scores[item['item_id']] = 0.4
        
        return scores
    
    def _handle_weather_context(self, weather_value: str) -> Dict[str, float]:
        """Handle weather context"""
        weather_preferences = {
            'sunny': ['adventure', 'comedy', 'action'],
            'rainy': ['drama', 'romance', 'documentary'],
            'cloudy': ['thriller', 'mystery', 'drama'],
            'stormy': ['horror', 'thriller', 'action']
        }
        
        preferences = weather_preferences.get(weather_value, [])
        
        scores = {}
        if self.items_df is not None and 'genre' in self.items_df.columns:
            for _, item in self.items_df.iterrows():
                item_genre = item['genre'].lower()
                
                if any(pref in item_genre for pref in preferences):
                    scores[item['item_id']] = 0.6
                else:
                    scores[item['item_id']] = 0.5
        
        return scores
    
    def explain_recommendation(self, item_id: str, method: str, **kwargs) -> Dict[str, Any]:
        """Generate explanation for knowledge-based recommendation"""
        explanation = {
            'method': method,
            'item_id': item_id,
            'reasoning': []
        }
        
        if method == 'trending':
            explanation['reasoning'].append("This item is currently trending")
        elif method == 'new_releases':
            explanation['reasoning'].append("This is a new release")
        elif method == 'acclaimed':
            explanation['reasoning'].append("This item is critically acclaimed")
        elif method == 'contextual':
            context = kwargs.get('context', {})
            explanation['reasoning'].append(f"Recommended based on your current context: {context}")
        elif method == 'mood_based':
            mood = kwargs.get('mood', '')
            explanation['reasoning'].append(f"This matches your {mood} mood")
        
        return explanation
