"""
Advanced Feature Extractor for Content Recommendation Engine
Extracts comprehensive features from user, content, and interaction data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from collections import Counter
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import settings, FeatureConstants
from ..utils.logging import get_data_processing_logger


class ContentFeatureExtractor:
    """Extract features from content metadata"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("ContentFeatureExtractor")
        self.genre_encoders = {}
        self.text_vectorizers = {}
        
    def extract_genre_features(self, df: pd.DataFrame, genre_column: str = 'genres') -> pd.DataFrame:
        """Extract and encode genre features"""
        df_features = df.copy()
        
        if genre_column not in df.columns:
            return df_features
        
        self.logger.logger.info(f"Extracting genre features from {genre_column}")
        
        try:
            # Parse genres (handle JSON format)
            genres_parsed = []
            all_genres = set()
            
            for idx, genre_value in df[genre_column].items():
                if pd.isna(genre_value):
                    genres_parsed.append([])
                else:
                    try:
                        if isinstance(genre_value, str):
                            if genre_value.startswith('['):
                                # JSON format
                                parsed_genres = json.loads(genre_value)
                            else:
                                # Pipe-separated or comma-separated
                                parsed_genres = [g.strip() for g in genre_value.replace('|', ',').split(',')]
                        else:
                            parsed_genres = []
                        
                        genres_parsed.append(parsed_genres)
                        all_genres.update(parsed_genres)
                    except:
                        genres_parsed.append([])
            
            # Create binary genre features
            for genre in sorted(all_genres):
                if genre and genre != '(no genres listed)':
                    df_features[f'genre_{genre.lower().replace(" ", "_")}'] = [
                        1 if genre in genres else 0 for genres in genres_parsed
                    ]
            
            # Additional genre-based features
            df_features['genre_count'] = [len(genres) for genres in genres_parsed]
            df_features['is_multi_genre'] = [1 if len(genres) > 1 else 0 for genres in genres_parsed]
            
            # Genre diversity score
            df_features['genre_diversity'] = df_features['genre_count'] / max(1, len(all_genres))
            
            # Popular genre indicators
            genre_popularity = Counter([genre for genres in genres_parsed for genre in genres])
            popular_genres = set([genre for genre, count in genre_popularity.most_common(10)])
            
            df_features['has_popular_genre'] = [
                1 if any(genre in popular_genres for genre in genres) else 0 
                for genres in genres_parsed
            ]
            
        except Exception as e:
            self.logger.log_error_with_context("extract_genre_features", e, {"column": genre_column})
        
        return df_features
    
    def extract_temporal_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from content metadata"""
        df_features = df.copy()
        
        # Publication/Release year features
        if 'publication_year' in df.columns:
            current_year = datetime.now().year
            df_features['content_age'] = current_year - df_features['publication_year']
            df_features['is_recent'] = (df_features['content_age'] <= 5).astype(int)
            df_features['is_classic'] = (df_features['content_age'] >= 20).astype(int)
            df_features['decade'] = (df_features['publication_year'] // 10) * 10
            
            # Era categorization
            df_features['era'] = pd.cut(
                df_features['publication_year'],
                bins=[0, 1950, 1980, 2000, 2010, current_year + 1],
                labels=['vintage', 'classic', 'modern', 'contemporary', 'recent'],
                include_lowest=True
            )
        
        # Duration-based features
        if 'duration' in df.columns:
            df_features['duration_category'] = pd.cut(
                df_features['duration'],
                bins=[0, 90, 120, 180, float('inf')],
                labels=['short', 'medium', 'long', 'very_long'],
                include_lowest=True
            )
            
            df_features['is_binge_worthy'] = (df_features['duration'] >= 120).astype(int)
            df_features['duration_normalized'] = df_features['duration'] / df_features['duration'].max()
        
        return df_features
    
    def extract_text_features(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """Extract features from text descriptions and titles"""
        df_features = df.copy()
        
        if text_columns is None:
            text_columns = ['title', 'description', 'plot_summary']
        
        available_text_columns = [col for col in text_columns if col in df.columns]
        
        for text_column in available_text_columns:
            self.logger.logger.info(f"Extracting text features from {text_column}")
            
            try:
                # Basic text statistics
                df_features[f'{text_column}_length'] = df[text_column].str.len().fillna(0)
                df_features[f'{text_column}_word_count'] = df[text_column].str.split().str.len().fillna(0)
                
                # Text complexity indicators
                df_features[f'{text_column}_avg_word_length'] = (
                    df_features[f'{text_column}_length'] / 
                    df_features[f'{text_column}_word_count'].replace(0, 1)
                )
                
                # Sentiment approximation (simple keyword-based)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
                
                df_features[f'{text_column}_positive_sentiment'] = df[text_column].str.lower().apply(
                    lambda x: sum(1 for word in positive_words if word in str(x)) if pd.notna(x) else 0
                )
                df_features[f'{text_column}_negative_sentiment'] = df[text_column].str.lower().apply(
                    lambda x: sum(1 for word in negative_words if word in str(x)) if pd.notna(x) else 0
                )
                
                # TF-IDF features for the most important terms
                if df[text_column].notna().sum() > 10:  # Only if enough text data
                    vectorizer = TfidfVectorizer(
                        max_features=50,
                        stop_words='english',
                        ngram_range=(1, 2),
                        min_df=2
                    )
                    
                    text_data = df[text_column].fillna('').astype(str)
                    tfidf_matrix = vectorizer.fit_transform(text_data)
                    
                    # Get top features
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[f'{text_column}_tfidf_{name}' for name in feature_names],
                        index=df.index
                    )
                    
                    df_features = pd.concat([df_features, tfidf_df], axis=1)
                    self.text_vectorizers[text_column] = vectorizer
                
            except Exception as e:
                self.logger.log_error_with_context(f"extract_text_features_{text_column}", e, {"column": text_column})
        
        return df_features
    
    def extract_creator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to content creators"""
        df_features = df.copy()
        
        creator_columns = ['director', 'author', 'artist', 'creator_info']
        available_creator_columns = [col for col in creator_columns if col in df.columns]
        
        for creator_column in available_creator_columns:
            try:
                # Creator popularity (based on frequency)
                creator_counts = df[creator_column].value_counts()
                df_features[f'{creator_column}_popularity'] = df[creator_column].map(creator_counts).fillna(0)
                
                # Is prolific creator (more than 5 works)
                df_features[f'is_prolific_{creator_column}'] = (
                    df_features[f'{creator_column}_popularity'] > 5
                ).astype(int)
                
                # Creator category (new, established, legendary)
                df_features[f'{creator_column}_category'] = pd.cut(
                    df_features[f'{creator_column}_popularity'],
                    bins=[0, 1, 5, 15, float('inf')],
                    labels=['new', 'emerging', 'established', 'legendary'],
                    include_lowest=True
                )
                
            except Exception as e:
                self.logger.log_error_with_context(f"extract_creator_features_{creator_column}", e, {"column": creator_column})
        
        return df_features
    
    def extract_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract quality and popularity related features"""
        df_features = df.copy()
        
        # Rating-based quality features
        rating_columns = ['user_score', 'critical_score', 'quality_score']
        available_rating_columns = [col for col in rating_columns if col in df.columns]
        
        for rating_column in available_rating_columns:
            try:
                # Quality tiers
                df_features[f'{rating_column}_tier'] = pd.cut(
                    df[rating_column],
                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                    labels=['poor', 'average', 'good', 'excellent'],
                    include_lowest=True
                )
                
                # Above/below average indicators
                mean_score = df[rating_column].mean()
                df_features[f'{rating_column}_above_average'] = (
                    df[rating_column] > mean_score
                ).astype(int)
                
            except Exception as e:
                self.logger.log_error_with_context(f"extract_quality_features_{rating_column}", e, {"column": rating_column})
        
        # Popularity features
        if 'popularity_score' in df.columns:
            # Popularity percentiles
            df_features['popularity_percentile'] = df['popularity_score'].rank(pct=True)
            df_features['is_mainstream'] = (df_features['popularity_percentile'] > 0.7).astype(int)
            df_features['is_niche'] = (df_features['popularity_percentile'] < 0.3).astype(int)
        
        return df_features


class UserFeatureExtractor:
    """Extract features from user profiles and behavior"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("UserFeatureExtractor")
        
    def extract_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode demographic features"""
        df_features = df.copy()
        
        # Age-based features
        if 'age' in df.columns:
            # Age groups
            df_features['age_group'] = pd.cut(
                df['age'],
                bins=[0, 18, 25, 35, 50, 65, 100],
                labels=['teen', 'young_adult', 'adult', 'middle_aged', 'senior', 'elderly'],
                include_lowest=True
            )
            
            # Generation categorization
            current_year = datetime.now().year
            birth_year = current_year - df['age']
            df_features['generation'] = pd.cut(
                birth_year,
                bins=[1900, 1946, 1964, 1981, 1997, 2012, current_year],
                labels=['silent', 'boomer', 'gen_x', 'millennial', 'gen_z', 'gen_alpha'],
                include_lowest=True
            )
        
        # Gender features
        if 'gender' in df.columns:
            # Standardize gender values
            gender_mapping = {
                'M': 'male', 'F': 'female', 'Male': 'male', 'Female': 'female',
                'male': 'male', 'female': 'female', 'Other': 'other', 'other': 'other'
            }
            df_features['gender_standardized'] = df['gender'].map(gender_mapping).fillna('unknown')
        
        # Location features
        if 'location' in df.columns:
            # Extract location components (simple parsing)
            df_features['location_provided'] = df['location'].notna().astype(int)
            
            # Country detection (simple)
            common_countries = ['usa', 'uk', 'canada', 'germany', 'france', 'australia']
            for country in common_countries:
                df_features[f'location_is_{country}'] = df['location'].str.lower().str.contains(
                    country, na=False
                ).astype(int)
        
        return df_features
    
    def extract_activity_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract user activity and engagement features"""
        
        # Group by user to calculate activity metrics
        user_activity = interactions_df.groupby('user_id').agg({
            'content_id': 'count',  # Total interactions
            'rating': ['mean', 'std', 'count'],  # Rating statistics
            'timestamp': ['min', 'max']  # Activity period
        }).round(4)
        
        # Flatten column names
        user_activity.columns = ['_'.join(col).strip() for col in user_activity.columns.values]
        user_activity = user_activity.reset_index()
        
        # Calculate additional activity features
        user_activity['activity_span_days'] = (
            user_activity['timestamp_max'] - user_activity['timestamp_min']
        ).dt.days
        
        user_activity['avg_interactions_per_day'] = (
            user_activity['content_id_count'] / user_activity['activity_span_days'].replace(0, 1)
        )
        
        # Engagement level categorization
        user_activity['engagement_level'] = pd.cut(
            user_activity['content_id_count'],
            bins=[0, 10, 50, 200, float('inf')],
            labels=['low', 'medium', 'high', 'very_high'],
            include_lowest=True
        )
        
        # Rating behavior patterns
        user_activity['is_generous_rater'] = (user_activity['rating_mean'] > 0.7).astype(int)
        user_activity['is_harsh_rater'] = (user_activity['rating_mean'] < 0.3).astype(int)
        user_activity['rating_consistency'] = 1 / (1 + user_activity['rating_std'].fillna(0))
        
        return user_activity
    
    def extract_preference_features(self, interactions_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
        """Extract user preference patterns"""
        
        # Merge interactions with content features
        interaction_content = interactions_df.merge(
            content_df[['content_id', 'genres', 'publication_year', 'duration']], 
            on='content_id', 
            how='left'
        )
        
        user_preferences = []
        
        for user_id in interactions_df['user_id'].unique():
            user_data = interaction_content[interaction_content['user_id'] == user_id]
            
            user_pref = {'user_id': user_id}
            
            # Genre preferences
            user_genres = []
            for genres_str in user_data['genres'].dropna():
                try:
                    if isinstance(genres_str, str) and genres_str.startswith('['):
                        genres = json.loads(genres_str)
                        user_genres.extend(genres)
                except:
                    continue
            
            if user_genres:
                genre_counter = Counter(user_genres)
                top_genres = [genre for genre, count in genre_counter.most_common(3)]
                user_pref['top_genres'] = json.dumps(top_genres)
                user_pref['genre_diversity'] = len(set(user_genres)) / len(user_genres)
            else:
                user_pref['top_genres'] = json.dumps([])
                user_pref['genre_diversity'] = 0.0
            
            # Temporal preferences
            if 'publication_year' in user_data.columns:
                year_preferences = user_data['publication_year'].dropna()
                if not year_preferences.empty:
                    user_pref['prefers_recent'] = (year_preferences > 2010).mean()
                    user_pref['prefers_classic'] = (year_preferences < 1990).mean()
                    user_pref['year_range'] = year_preferences.max() - year_preferences.min()
            
            # Duration preferences
            if 'duration' in user_data.columns:
                duration_preferences = user_data['duration'].dropna()
                if not duration_preferences.empty:
                    user_pref['avg_preferred_duration'] = duration_preferences.mean()
                    user_pref['prefers_long_content'] = (duration_preferences > 150).mean()
            
            # Rating patterns by content characteristics
            high_rated = user_data[user_data['rating'] > 0.7]
            if not high_rated.empty:
                user_pref['high_rated_count'] = len(high_rated)
                user_pref['selectivity'] = len(high_rated) / len(user_data)
            else:
                user_pref['high_rated_count'] = 0
                user_pref['selectivity'] = 0.0
            
            user_preferences.append(user_pref)
        
        return pd.DataFrame(user_preferences)
    
    def extract_temporal_behavior_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal behavior patterns"""
        
        # Ensure timestamp is datetime
        interactions_df = interactions_df.copy()
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        # Extract temporal components
        interactions_df['hour'] = interactions_df['timestamp'].dt.hour
        interactions_df['day_of_week'] = interactions_df['timestamp'].dt.dayofweek
        interactions_df['month'] = interactions_df['timestamp'].dt.month
        
        # Aggregate temporal behavior by user
        temporal_behavior = interactions_df.groupby('user_id').agg({
            'hour': lambda x: x.mode().iloc[0] if not x.mode().empty else 12,  # Most common hour
            'day_of_week': lambda x: x.mode().iloc[0] if not x.mode().empty else 0,  # Most common day
            'month': lambda x: x.mode().iloc[0] if not x.mode().empty else 6  # Most common month
        }).round(0)
        
        # Additional temporal features
        user_temporal_patterns = []
        
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            
            temporal_pattern = {'user_id': user_id}
            
            # Time of day patterns
            hour_dist = user_data['hour'].value_counts(normalize=True)
            temporal_pattern['is_morning_user'] = hour_dist.get(range(6, 12), pd.Series([0])).sum()
            temporal_pattern['is_evening_user'] = hour_dist.get(range(18, 24), pd.Series([0])).sum()
            temporal_pattern['is_night_owl'] = hour_dist.get(range(22, 24), pd.Series([0])).sum() + hour_dist.get(range(0, 6), pd.Series([0])).sum()
            
            # Weekend vs weekday patterns
            weekend_interactions = user_data[user_data['day_of_week'].isin([5, 6])]
            temporal_pattern['weekend_activity_ratio'] = len(weekend_interactions) / len(user_data)
            
            # Seasonal patterns
            season_mapping = {12: 'winter', 1: 'winter', 2: 'winter',
                            3: 'spring', 4: 'spring', 5: 'spring',
                            6: 'summer', 7: 'summer', 8: 'summer',
                            9: 'fall', 10: 'fall', 11: 'fall'}
            user_data['season'] = user_data['month'].map(season_mapping)
            season_dist = user_data['season'].value_counts(normalize=True)
            for season in ['winter', 'spring', 'summer', 'fall']:
                temporal_pattern[f'{season}_activity'] = season_dist.get(season, 0)
            
            # Activity consistency
            daily_counts = user_data.groupby(user_data['timestamp'].dt.date).size()
            temporal_pattern['activity_consistency'] = 1 / (1 + daily_counts.std()) if len(daily_counts) > 1 else 1.0
            
            user_temporal_patterns.append(temporal_pattern)
        
        temporal_df = pd.DataFrame(user_temporal_patterns)
        
        # Merge with basic temporal aggregations
        temporal_behavior = temporal_behavior.reset_index()
        final_temporal = temporal_behavior.merge(temporal_df, on='user_id', how='outer')
        
        return final_temporal


class InteractionFeatureExtractor:
    """Extract features from user-content interactions"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("InteractionFeatureExtractor")
        
    def extract_interaction_context_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract contextual features from interactions"""
        df_features = interactions_df.copy()
        
        # Session-based features
        if 'session_id' in df_features.columns:
            session_stats = df_features.groupby('session_id').agg({
                'content_id': 'count',
                'rating': 'mean',
                'interaction_duration': 'sum'
            }).add_suffix('_session')
            
            df_features = df_features.merge(
                session_stats, left_on='session_id', right_index=True, how='left'
            )
        
        # Device context features
        if 'device_info' in df_features.columns:
            # Extract device type (mobile, desktop, tablet, tv)
            df_features['is_mobile'] = df_features['device_info'].str.contains('mobile', case=False, na=False).astype(int)
            df_features['is_desktop'] = df_features['device_info'].str.contains('desktop', case=False, na=False).astype(int)
            df_features['is_tablet'] = df_features['device_info'].str.contains('tablet', case=False, na=False).astype(int)
            df_features['is_tv'] = df_features['device_info'].str.contains('tv', case=False, na=False).astype(int)
        
        # Completion rate features
        if 'completion_status' in df_features.columns:
            df_features['completed'] = (df_features['completion_status'] >= 0.8).astype(int)
            df_features['partially_consumed'] = ((df_features['completion_status'] >= 0.3) & 
                                               (df_features['completion_status'] < 0.8)).astype(int)
            df_features['abandoned'] = (df_features['completion_status'] < 0.3).astype(int)
        
        # Interaction sequence features
        df_features = df_features.sort_values(['user_id', 'timestamp'])
        df_features['interaction_sequence'] = df_features.groupby('user_id').cumcount() + 1
        
        # Time between interactions
        df_features['time_since_last_interaction'] = df_features.groupby('user_id')['timestamp'].diff()
        df_features['time_since_last_interaction_hours'] = (
            df_features['time_since_last_interaction'].dt.total_seconds() / 3600
        )
        
        # Rating velocity (rating change over time)
        df_features['prev_rating'] = df_features.groupby('user_id')['rating'].shift(1)
        df_features['rating_change'] = df_features['rating'] - df_features['prev_rating']
        
        return df_features
    
    def extract_user_item_history_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on user and item interaction history"""
        df_features = interactions_df.copy()
        
        # User interaction statistics
        user_stats = df_features.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'content_id': 'nunique',
            'interaction_duration': 'mean'
        }).round(4)
        
        user_stats.columns = ['user_' + '_'.join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        
        # Item interaction statistics
        item_stats = df_features.groupby('content_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique',
            'interaction_duration': 'mean'
        }).round(4)
        
        item_stats.columns = ['item_' + '_'.join(col).strip() for col in item_stats.columns.values]
        item_stats = item_stats.reset_index()
        
        # Merge statistics back to interactions
        df_features = df_features.merge(user_stats, on='user_id', how='left')
        df_features = df_features.merge(item_stats, on='content_id', how='left')
        
        # User-item affinity features
        df_features['user_rating_vs_avg'] = df_features['rating'] - df_features['user_rating_mean']
        df_features['item_rating_vs_avg'] = df_features['rating'] - df_features['item_rating_mean']
        
        # Popularity and user activity features
        df_features['item_popularity_percentile'] = df_features['item_rating_count'].rank(pct=True)
        df_features['user_activity_percentile'] = df_features['user_rating_count'].rank(pct=True)
        
        # Cold start indicators
        df_features['is_new_user'] = (df_features['user_rating_count'] <= 5).astype(int)
        df_features['is_new_item'] = (df_features['item_rating_count'] <= 5).astype(int)
        
        return df_features
    
    def extract_similarity_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract user and item similarity features"""
        
        # Create user-item matrix for similarity calculations
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='rating', 
            fill_value=0
        )
        
        # Calculate user similarities (top-K for efficiency)
        n_similar_users = min(50, len(user_item_matrix))
        user_similarities = {}
        
        for user in user_item_matrix.index:
            user_vector = user_item_matrix.loc[user].values
            similarities = []
            
            for other_user in user_item_matrix.index:
                if user != other_user:
                    other_vector = user_item_matrix.loc[other_user].values
                    # Only calculate similarity if users have common items
                    common_items = (user_vector != 0) & (other_vector != 0)
                    if common_items.sum() > 0:
                        similarity = 1 - cosine(user_vector[common_items], other_vector[common_items])
                        similarities.append((other_user, similarity))
            
            # Keep top similar users
            similarities.sort(key=lambda x: x[1], reverse=True)
            user_similarities[user] = similarities[:n_similar_users]
        
        # Calculate item similarities
        item_similarities = {}
        for item in user_item_matrix.columns:
            item_vector = user_item_matrix[item].values
            similarities = []
            
            for other_item in user_item_matrix.columns:
                if item != other_item:
                    other_vector = user_item_matrix[other_item].values
                    common_users = (item_vector != 0) & (other_vector != 0)
                    if common_users.sum() > 0:
                        similarity = 1 - cosine(item_vector[common_users], other_vector[common_users])
                        similarities.append((other_item, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            item_similarities[item] = similarities[:n_similar_users]
        
        # Add similarity features to interactions
        similarity_features = []
        
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            content_id = interaction['content_id']
            
            features = {
                'user_id': user_id,
                'content_id': content_id
            }
            
            # User similarity features
            if user_id in user_similarities:
                similar_users = user_similarities[user_id]
                if similar_users:
                    features['max_user_similarity'] = max([sim for _, sim in similar_users])
                    features['avg_user_similarity'] = np.mean([sim for _, sim in similar_users])
                    features['num_similar_users'] = len(similar_users)
                else:
                    features['max_user_similarity'] = 0
                    features['avg_user_similarity'] = 0
                    features['num_similar_users'] = 0
            
            # Item similarity features
            if content_id in item_similarities:
                similar_items = item_similarities[content_id]
                if similar_items:
                    features['max_item_similarity'] = max([sim for _, sim in similar_items])
                    features['avg_item_similarity'] = np.mean([sim for _, sim in similar_items])
                    features['num_similar_items'] = len(similar_items)
                else:
                    features['max_item_similarity'] = 0
                    features['avg_item_similarity'] = 0
                    features['num_similar_items'] = 0
            
            similarity_features.append(features)
        
        similarity_df = pd.DataFrame(similarity_features)
        
        # Merge with original interactions
        enhanced_interactions = interactions_df.merge(
            similarity_df, on=['user_id', 'content_id'], how='left'
        )
        
        return enhanced_interactions


class ComprehensiveFeatureExtractor:
    """Main feature extractor orchestrating all feature extraction processes"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("ComprehensiveFeatureExtractor")
        self.content_extractor = ContentFeatureExtractor()
        self.user_extractor = UserFeatureExtractor()
        self.interaction_extractor = InteractionFeatureExtractor()
        
        self.feature_store = {}
        self.extraction_report = {}
    
    def extract_all_features(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Extract all features from provided datasets"""
        self.logger.log_processing_start("extract_all_features", sum(len(df) for df in datasets.values()))
        
        feature_sets = {}
        
        try:
            # Extract content features
            if 'content' in datasets or 'movies' in datasets or 'books' in datasets:
                content_df = datasets.get('content', datasets.get('movies', datasets.get('books')))
                self.logger.logger.info("Extracting content features...")
                
                content_features = self.content_extractor.extract_genre_features(content_df)
                content_features = self.content_extractor.extract_temporal_content_features(content_features)
                content_features = self.content_extractor.extract_text_features(content_features)
                content_features = self.content_extractor.extract_creator_features(content_features)
                content_features = self.content_extractor.extract_quality_features(content_features)
                
                feature_sets['content_features'] = content_features
            
            # Extract user features
            if 'users' in datasets:
                self.logger.logger.info("Extracting user demographic features...")
                user_features = self.user_extractor.extract_demographic_features(datasets['users'])
                feature_sets['user_demographic_features'] = user_features
            
            # Extract interaction-based features
            if 'interactions' in datasets or 'ratings' in datasets:
                interactions_df = datasets.get('interactions', datasets.get('ratings'))
                self.logger.logger.info("Extracting interaction features...")
                
                # Basic interaction features
                interaction_features = self.interaction_extractor.extract_interaction_context_features(interactions_df)
                interaction_features = self.interaction_extractor.extract_user_item_history_features(interaction_features)
                
                # User activity and preference features
                user_activity = self.user_extractor.extract_activity_features(interactions_df)
                user_temporal = self.user_extractor.extract_temporal_behavior_features(interactions_df)
                
                feature_sets['interaction_features'] = interaction_features
                feature_sets['user_activity_features'] = user_activity
                feature_sets['user_temporal_features'] = user_temporal
                
                # User preferences (requires content data)
                if 'content_features' in feature_sets:
                    user_preferences = self.user_extractor.extract_preference_features(
                        interactions_df, feature_sets['content_features']
                    )
                    feature_sets['user_preference_features'] = user_preferences
                
                # Similarity features (computationally intensive - optional)
                try:
                    similarity_features = self.interaction_extractor.extract_similarity_features(interactions_df)
                    feature_sets['similarity_features'] = similarity_features
                except Exception as e:
                    self.logger.logger.warning(f"Similarity feature extraction failed: {e}")
            
            # Store in feature store
            self.feature_store.update(feature_sets)
            
            # Generate extraction report
            self.extraction_report = self._generate_extraction_report(feature_sets)
            
            self.logger.log_processing_complete("extract_all_features", time.time(), len(feature_sets), 0)
            self.logger.log_data_quality_metrics("feature_extraction", self.extraction_report)
            
            return feature_sets
            
        except Exception as e:
            self.logger.log_error_with_context(
                "extract_all_features",
                e,
                {"datasets": list(datasets.keys())}
            )
            return {}
    
    def create_unified_feature_matrix(self, feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a unified feature matrix for machine learning"""
        self.logger.logger.info("Creating unified feature matrix...")
        
        # Start with interaction features as base
        if 'interaction_features' in feature_sets:
            unified_matrix = feature_sets['interaction_features'].copy()
            
            # Add user features
            if 'user_activity_features' in feature_sets:
                unified_matrix = unified_matrix.merge(
                    feature_sets['user_activity_features'], 
                    on='user_id', 
                    how='left'
                )
            
            if 'user_temporal_features' in feature_sets:
                unified_matrix = unified_matrix.merge(
                    feature_sets['user_temporal_features'], 
                    on='user_id', 
                    how='left'
                )
            
            if 'user_preference_features' in feature_sets:
                unified_matrix = unified_matrix.merge(
                    feature_sets['user_preference_features'], 
                    on='user_id', 
                    how='left'
                )
            
            # Add content features
            if 'content_features' in feature_sets:
                content_features_subset = feature_sets['content_features'].select_dtypes(include=[np.number])
                content_features_subset['content_id'] = feature_sets['content_features']['content_id']
                
                unified_matrix = unified_matrix.merge(
                    content_features_subset, 
                    on='content_id', 
                    how='left'
                )
            
            # Fill missing values
            numeric_columns = unified_matrix.select_dtypes(include=[np.number]).columns
            unified_matrix[numeric_columns] = unified_matrix[numeric_columns].fillna(0)
            
            categorical_columns = unified_matrix.select_dtypes(include=['object', 'category']).columns
            unified_matrix[categorical_columns] = unified_matrix[categorical_columns].fillna('unknown')
            
            self.logger.logger.info(f"Created unified feature matrix with shape: {unified_matrix.shape}")
            return unified_matrix
            
        else:
            self.logger.logger.error("No interaction features found for unified matrix creation")
            return pd.DataFrame()
    
    def _generate_extraction_report(self, feature_sets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive feature extraction report"""
        
        report = {
            'extraction_timestamp': datetime.now().isoformat(),
            'feature_sets_created': len(feature_sets),
            'total_features_extracted': 0,
            'feature_set_summaries': {},
            'feature_categories': {
                'content_features': 0,
                'user_features': 0,
                'interaction_features': 0,
                'temporal_features': 0,
                'similarity_features': 0
            }
        }
        
        for feature_set_name, feature_df in feature_sets.items():
            # Basic statistics
            summary = {
                'shape': feature_df.shape,
                'numeric_features': len(feature_df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(feature_df.select_dtypes(include=['object', 'category']).columns),
                'missing_values_percentage': (feature_df.isnull().sum().sum() / (feature_df.shape[0] * feature_df.shape[1])) * 100
            }
            
            # Feature type breakdown
            if 'content' in feature_set_name:
                report['feature_categories']['content_features'] += summary['numeric_features'] + summary['categorical_features']
            elif 'user' in feature_set_name:
                report['feature_categories']['user_features'] += summary['numeric_features'] + summary['categorical_features']
            elif 'interaction' in feature_set_name:
                report['feature_categories']['interaction_features'] += summary['numeric_features'] + summary['categorical_features']
            elif 'temporal' in feature_set_name:
                report['feature_categories']['temporal_features'] += summary['numeric_features'] + summary['categorical_features']
            elif 'similarity' in feature_set_name:
                report['feature_categories']['similarity_features'] += summary['numeric_features'] + summary['categorical_features']
            
            report['feature_set_summaries'][feature_set_name] = summary
            report['total_features_extracted'] += summary['numeric_features'] + summary['categorical_features']
        
        return report
    
    def get_feature_importance_analysis(self, feature_matrix: pd.DataFrame, target_column: str = 'rating') -> Dict[str, Any]:
        """Analyze feature importance using various methods"""
        if target_column not in feature_matrix.columns:
            return {"error": f"Target column '{target_column}' not found"}
        
        # Prepare features and target
        X = feature_matrix.select_dtypes(include=[np.number]).drop(columns=[target_column])
        y = feature_matrix[target_column]
        
        importance_analysis = {}
        
        try:
            # Correlation analysis
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            importance_analysis['correlation_importance'] = correlations.head(20).to_dict()
            
            # Mutual information
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X.fillna(0), y)
            mi_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            importance_analysis['mutual_info_importance'] = mi_importance.head(20).to_dict()
            
            # Random Forest feature importance
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X.fillna(0), y)
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            importance_analysis['random_forest_importance'] = rf_importance.head(20).to_dict()
            
        except Exception as e:
            importance_analysis['error'] = str(e)
        
        return importance_analysis
