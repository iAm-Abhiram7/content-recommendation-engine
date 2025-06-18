"""
Quality Scorer for Content Recommendation Engine
Assesses content quality, popularity, and freshness metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .gemini_client import GeminiClient
from ..utils.config import settings
from ..utils.logging import get_data_processing_logger


class QualityScorer:
    """Comprehensive quality scoring for content items"""
    
    def __init__(self, gemini_client: GeminiClient = None):
        self.gemini_client = gemini_client or GeminiClient()
        self.logger = get_data_processing_logger("QualityScorer")
        
        # Scoring weights
        self.quality_weights = {
            'popularity': 0.25,
            'critical_acclaim': 0.30,
            'user_ratings': 0.25,
            'freshness': 0.10,
            'content_quality': 0.10
        }
        
        # Scalers for normalization
        self.scalers = {}
        
    def calculate_comprehensive_quality_scores(self, content_df: pd.DataFrame,
                                             interactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate comprehensive quality scores for all content"""
        
        self.logger.log_processing_start("calculate_quality_scores", len(content_df))
        
        quality_df = content_df.copy()
        
        # Calculate individual quality components
        quality_df = self._calculate_popularity_scores(quality_df, interactions_df)
        quality_df = self._calculate_critical_acclaim_scores(quality_df)
        quality_df = self._calculate_user_rating_scores(quality_df, interactions_df)
        quality_df = self._calculate_freshness_scores(quality_df)
        quality_df = self._calculate_content_quality_scores(quality_df)
        
        # Calculate composite quality score
        quality_df = self._calculate_composite_quality_score(quality_df)
        
        # Add quality tiers and rankings
        quality_df = self._add_quality_tiers(quality_df)
        
        self.logger.log_processing_complete("calculate_quality_scores", time.time(), len(content_df), 0)
        
        return quality_df
    
    def calculate_quality_score(self, content_df: pd.DataFrame,
                              interactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """Alias for calculate_comprehensive_quality_scores for backward compatibility"""
        return self.calculate_comprehensive_quality_scores(content_df, interactions_df)
    
    def calculate_single_quality_score(self, content_item: Dict[str, Any]) -> float:
        """Calculate quality score for a single content item"""
        # Convert dict to DataFrame
        df = pd.DataFrame([content_item])
        
        # Add required columns if missing
        if 'content_id' not in df.columns:
            df['content_id'] = 'temp_id'
        
        # Calculate a simple quality score based on available data
        score = 0.5  # Base score
        
        # Title quality
        if 'title' in content_item and content_item['title']:
            title_len = len(content_item['title'])
            if 10 <= title_len <= 100:
                score += 0.1
        
        # Description quality
        if 'description' in content_item and content_item['description']:
            desc_len = len(content_item['description'])
            if 50 <= desc_len <= 1000:
                score += 0.2
        
        # Genre presence
        if 'genres' in content_item and content_item['genres']:
            if isinstance(content_item['genres'], list) and len(content_item['genres']) > 0:
                score += 0.1
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _calculate_popularity_scores(self, df: pd.DataFrame, 
                                   interactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate popularity-based quality scores"""
        
        df = df.copy()
        
        # Initialize popularity score
        df['raw_popularity_score'] = 0.0
        
        if interactions_df is not None:
            # Calculate interaction-based popularity
            interaction_counts = interactions_df['content_id'].value_counts()
            
            # Map interaction counts to content
            df['interaction_count'] = df['content_id'].map(interaction_counts).fillna(0)
            
            # Calculate recency-weighted popularity
            recent_cutoff = datetime.now() - timedelta(days=365)
            if 'timestamp' in interactions_df.columns:
                recent_interactions = interactions_df[
                    pd.to_datetime(interactions_df['timestamp']) > recent_cutoff
                ]
                recent_counts = recent_interactions['content_id'].value_counts()
                df['recent_interaction_count'] = df['content_id'].map(recent_counts).fillna(0)
            else:
                df['recent_interaction_count'] = df['interaction_count']
            
            # Calculate popularity metrics
            df['popularity_rank'] = df['interaction_count'].rank(pct=True)
            df['recent_popularity_rank'] = df['recent_interaction_count'].rank(pct=True)
            
            # Weighted popularity score
            df['raw_popularity_score'] = (
                0.6 * df['popularity_rank'] + 
                0.4 * df['recent_popularity_rank']
            )
        
        # Use existing popularity score if available
        if 'popularity_score' in df.columns:
            existing_popularity = df['popularity_score'].fillna(0)
            # Normalize existing scores
            if existing_popularity.max() > 1:
                existing_popularity = existing_popularity / existing_popularity.max()
            
            # Combine with interaction-based popularity
            df['raw_popularity_score'] = (
                0.5 * df['raw_popularity_score'] + 
                0.5 * existing_popularity
            )
        
        # Handle content without interactions
        df['raw_popularity_score'] = df['raw_popularity_score'].fillna(0.1)  # Small baseline
        
        # Normalize to 0-1 scale
        scaler = MinMaxScaler()
        df['popularity_quality_score'] = scaler.fit_transform(
            df[['raw_popularity_score']]
        ).flatten()
        
        self.scalers['popularity'] = scaler
        
        return df
    
    def _calculate_critical_acclaim_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate critical acclaim quality scores"""
        
        df = df.copy()
        df['critical_acclaim_score'] = 0.5  # Default neutral score
        
        # Use existing critical scores if available
        critical_columns = ['critical_score', 'metacritic_score', 'rotten_tomatoes_score']
        available_critical_columns = [col for col in critical_columns if col in df.columns]
        
        if available_critical_columns:
            # Combine multiple critical scores
            critical_scores = []
            
            for col in available_critical_columns:
                # Normalize scores to 0-1 scale
                score_series = df[col].fillna(0)
                
                # Handle different score scales
                if col == 'rotten_tomatoes_score':
                    # Assuming 0-100 scale
                    normalized_score = score_series / 100
                elif col == 'metacritic_score':
                    # Assuming 0-100 scale
                    normalized_score = score_series / 100
                else:
                    # Assume already normalized or use as-is
                    max_score = score_series.max()
                    if max_score > 1:
                        normalized_score = score_series / max_score
                    else:
                        normalized_score = score_series
                
                critical_scores.append(normalized_score)
            
            # Average all available critical scores
            df['critical_acclaim_score'] = pd.concat(critical_scores, axis=1).mean(axis=1)
        
        # Use AI to assess content quality if no critical scores available
        elif self.gemini_client:
            df = self._ai_assess_critical_quality(df)
        
        # Ensure scores are in valid range
        df['critical_acclaim_score'] = df['critical_acclaim_score'].clip(0, 1)
        
        return df
    
    def _calculate_user_rating_scores(self, df: pd.DataFrame, 
                                    interactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate user rating quality scores"""
        
        df = df.copy()
        df['user_rating_quality_score'] = 0.5  # Default neutral score
        
        if interactions_df is not None and 'rating' in interactions_df.columns:
            # Calculate user rating statistics per content item
            rating_stats = interactions_df.groupby('content_id')['rating'].agg([
                'mean', 'count', 'std'
            ]).round(4)
            
            # Merge with content dataframe
            df = df.merge(rating_stats, left_on='content_id', right_index=True, how='left')
            
            # Calculate confidence-weighted rating score
            df['rating_count'] = df['count'].fillna(0)
            df['rating_mean'] = df['mean'].fillna(0.5)
            df['rating_std'] = df['std'].fillna(0.2)
            
            # Confidence factor (more ratings = higher confidence)
            df['rating_confidence'] = np.log(df['rating_count'] + 1) / np.log(100)  # Log scale
            df['rating_confidence'] = df['rating_confidence'].clip(0, 1)
            
            # Consensus factor (lower std = higher consensus)
            df['rating_consensus'] = 1 / (1 + df['rating_std'])
            df['rating_consensus'] = df['rating_consensus'].clip(0, 1)
            
            # Combined user rating quality score
            df['user_rating_quality_score'] = (
                0.6 * df['rating_mean'] + 
                0.2 * df['rating_confidence'] + 
                0.2 * df['rating_consensus']
            )
            
            # Clean up temporary columns
            df = df.drop(columns=['mean', 'count', 'std'], errors='ignore')
        
        return df
    
    def _calculate_freshness_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate content freshness scores"""
        
        df = df.copy()
        df['freshness_score'] = 0.5  # Default neutral score
        
        if 'publication_year' in df.columns:
            current_year = datetime.now().year
            content_age = current_year - df['publication_year'].fillna(current_year)
            
            # Freshness decay function
            # Recent content (0-2 years) gets highest score
            # Gradual decay for older content
            df['freshness_score'] = np.exp(-content_age / 10)  # Exponential decay
            df['freshness_score'] = df['freshness_score'].clip(0, 1)
            
            # Boost for classic content (20+ years old with high quality)
            is_classic = content_age >= 20
            has_high_rating = df.get('critical_acclaim_score', 0.5) > 0.7
            
            classic_boost = is_classic & has_high_rating
            df.loc[classic_boost, 'freshness_score'] = np.minimum(
                df.loc[classic_boost, 'freshness_score'] + 0.2, 1.0
            )
        
        # Consider trending patterns if interaction data available
        if 'recent_interaction_count' in df.columns and 'interaction_count' in df.columns:
            # Calculate trending factor
            df['trending_factor'] = (
                df['recent_interaction_count'] / 
                (df['interaction_count'] + 1)
            ).clip(0, 2)  # Allow boost up to 2x
            
            # Apply trending boost to freshness
            df['freshness_score'] = (
                df['freshness_score'] * df['trending_factor']
            ).clip(0, 1)
        
        return df
    
    def _calculate_content_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate intrinsic content quality scores"""
        
        df = df.copy()
        df['content_quality_score'] = 0.5  # Default neutral score
        
        # Use AI-generated quality indicators if available
        ai_quality_columns = ['complexity_score', 'sentiment_score', 'quality_score']
        available_ai_columns = [col for col in ai_quality_columns if col in df.columns]
        
        if available_ai_columns:
            ai_scores = []
            
            for col in available_ai_columns:
                score_series = df[col].fillna(0.5)
                
                # Normalize if needed
                if score_series.max() > 1:
                    score_series = score_series / score_series.max()
                
                ai_scores.append(score_series)
            
            # Average AI quality scores
            df['ai_quality_average'] = pd.concat(ai_scores, axis=1).mean(axis=1)
            df['content_quality_score'] = df['ai_quality_average']
        
        # Use text-based quality indicators
        text_quality_features = []
        
        # Description length and quality
        if 'description' in df.columns:
            df['description_length'] = df['description'].str.len().fillna(0)
            df['description_quality'] = np.log(df['description_length'] + 1) / np.log(1000)
            df['description_quality'] = df['description_quality'].clip(0, 1)
            text_quality_features.append('description_quality')
        
        # Title quality
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len().fillna(0)
            df['title_quality'] = (df['title_length'] / 100).clip(0, 1)
            text_quality_features.append('title_quality')
        
        # Genre diversity as quality indicator
        if 'genres' in df.columns:
            df['genre_count'] = df['genres'].apply(self._count_genres)
            df['genre_diversity'] = (df['genre_count'] / 5).clip(0, 1)  # Max 5 genres
            text_quality_features.append('genre_diversity')
        
        # Combine text-based quality features
        if text_quality_features:
            text_quality_score = df[text_quality_features].mean(axis=1)
            
            # Combine with AI quality if available
            if 'ai_quality_average' in df.columns:
                df['content_quality_score'] = (
                    0.7 * df['ai_quality_average'] + 
                    0.3 * text_quality_score
                )
            else:
                df['content_quality_score'] = text_quality_score
        
        return df
    
    def _calculate_composite_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted composite quality score"""
        
        df = df.copy()
        
        # Ensure all component scores exist
        component_scores = {
            'popularity_quality_score': self.quality_weights['popularity'],
            'critical_acclaim_score': self.quality_weights['critical_acclaim'],
            'user_rating_quality_score': self.quality_weights['user_ratings'],
            'freshness_score': self.quality_weights['freshness'],
            'content_quality_score': self.quality_weights['content_quality']
        }
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_column, weight in component_scores.items():
            if score_column in df.columns:
                weighted_sum += df[score_column].fillna(0.5) * weight
                total_weight += weight
        
        # Normalize by actual total weight
        df['composite_quality_score'] = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Ensure scores are in valid range
        df['composite_quality_score'] = df['composite_quality_score'].clip(0, 1)
        
        return df
    
    def _add_quality_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality tiers and rankings"""
        
        df = df.copy()
        
        # Quality percentiles
        df['quality_percentile'] = df['composite_quality_score'].rank(pct=True)
        
        # Quality tiers
        df['quality_tier'] = pd.cut(
            df['quality_percentile'],
            bins=[0, 0.25, 0.50, 0.75, 0.90, 1.0],
            labels=['poor', 'below_average', 'average', 'good', 'excellent'],
            include_lowest=True
        )
        
        # Overall quality ranking
        df['quality_rank'] = df['composite_quality_score'].rank(ascending=False)
        
        # Quality confidence score
        quality_components = [
            'popularity_quality_score', 'critical_acclaim_score', 
            'user_rating_quality_score', 'freshness_score', 'content_quality_score'
        ]
        available_components = [col for col in quality_components if col in df.columns]
        
        if len(available_components) > 1:
            # Calculate standard deviation of quality components as confidence measure
            quality_matrix = df[available_components].fillna(0.5)
            df['quality_confidence'] = 1 / (1 + quality_matrix.std(axis=1))
        else:
            df['quality_confidence'] = 0.5  # Default moderate confidence
        
        return df
    
    def _ai_assess_critical_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use AI to assess content quality for items without critical scores"""
        
        # Sample a subset for AI analysis (to manage API costs)
        sample_size = min(100, len(df))
        sample_indices = df.sample(n=sample_size).index if len(df) > sample_size else df.index
        
        ai_scores = {}
        
        for idx in sample_indices:
            row = df.loc[idx]
            
            try:
                # Prepare content for AI analysis
                content_text = self._prepare_content_for_ai_analysis(row)
                
                if content_text:
                    # Use Gemini to analyze content quality
                    themes = self.gemini_client.extract_themes(content_text)
                    sentiment = self.gemini_client.analyze_sentiment(content_text)
                    
                    # Calculate quality score from AI analysis
                    complexity_score = 0.6 if themes.get('complexity_level') == 'complex' else 0.4
                    sentiment_quality = sentiment.get('sentiment_score', 0.5)
                    
                    ai_quality_score = (complexity_score + sentiment_quality) / 2
                    ai_scores[idx] = ai_quality_score
                
            except Exception as e:
                self.logger.logger.warning(f"AI quality assessment failed for content {idx}: {e}")
                ai_scores[idx] = 0.5  # Default score
        
        # Apply AI scores to dataframe
        for idx, score in ai_scores.items():
            df.loc[idx, 'critical_acclaim_score'] = score
        
        return df
    
    def _prepare_content_for_ai_analysis(self, content_row: pd.Series) -> str:
        """Prepare content text for AI quality analysis"""
        
        text_parts = []
        
        if 'title' in content_row and pd.notna(content_row['title']):
            text_parts.append(f"Title: {content_row['title']}")
        
        if 'description' in content_row and pd.notna(content_row['description']):
            description = content_row['description'][:500]  # Limit length
            text_parts.append(f"Description: {description}")
        
        if 'genres' in content_row and pd.notna(content_row['genres']):
            text_parts.append(f"Genres: {content_row['genres']}")
        
        return ". ".join(text_parts) if text_parts else ""
    
    def _count_genres(self, genres_str) -> int:
        """Count number of genres in genres string"""
        if pd.isna(genres_str):
            return 0
        
        try:
            if isinstance(genres_str, str):
                if genres_str.startswith('['):
                    genres = json.loads(genres_str)
                    return len(genres)
                else:
                    return len([g.strip() for g in genres_str.split(',') if g.strip()])
            return 0
        except:
            return 0
    
    def analyze_quality_distribution(self, quality_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the distribution of quality scores"""
        
        analysis = {
            'total_content_items': len(quality_df),
            'quality_statistics': {},
            'tier_distribution': {},
            'component_correlations': {},
            'quality_insights': []
        }
        
        # Overall quality statistics
        if 'composite_quality_score' in quality_df.columns:
            quality_scores = quality_df['composite_quality_score']
            analysis['quality_statistics'] = {
                'mean': quality_scores.mean(),
                'median': quality_scores.median(),
                'std': quality_scores.std(),
                'min': quality_scores.min(),
                'max': quality_scores.max(),
                'skewness': quality_scores.skew()
            }
        
        # Quality tier distribution
        if 'quality_tier' in quality_df.columns:
            tier_counts = quality_df['quality_tier'].value_counts()
            tier_percentages = tier_counts / len(quality_df) * 100
            
            for tier, count in tier_counts.items():
                analysis['tier_distribution'][tier] = {
                    'count': count,
                    'percentage': tier_percentages[tier]
                }
        
        # Component correlations
        quality_components = [
            'popularity_quality_score', 'critical_acclaim_score', 
            'user_rating_quality_score', 'freshness_score', 'content_quality_score'
        ]
        available_components = [col for col in quality_components if col in quality_df.columns]
        
        if len(available_components) > 1:
            correlation_matrix = quality_df[available_components].corr()
            analysis['component_correlations'] = correlation_matrix.to_dict()
        
        # Quality insights
        analysis['quality_insights'] = self._generate_quality_insights(quality_df, analysis)
        
        return analysis
    
    def _generate_quality_insights(self, quality_df: pd.DataFrame, 
                                 analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about content quality distribution"""
        
        insights = []
        
        # Quality distribution insights
        if 'quality_statistics' in analysis:
            stats = analysis['quality_statistics']
            
            if stats['mean'] > 0.7:
                insights.append("Content library has generally high quality scores")
            elif stats['mean'] < 0.4:
                insights.append("Content library has below-average quality scores")
            
            if stats['std'] > 0.3:
                insights.append("High variance in quality scores - diverse content quality")
            elif stats['std'] < 0.1:
                insights.append("Low variance in quality scores - consistent content quality")
        
        # Tier distribution insights
        if 'tier_distribution' in analysis:
            excellent_pct = analysis['tier_distribution'].get('excellent', {}).get('percentage', 0)
            poor_pct = analysis['tier_distribution'].get('poor', {}).get('percentage', 0)
            
            if excellent_pct > 15:
                insights.append(f"{excellent_pct:.1f}% of content is excellent quality")
            
            if poor_pct > 20:
                insights.append(f"{poor_pct:.1f}% of content needs quality improvement")
        
        # Component correlation insights
        if 'component_correlations' in analysis:
            correlations = analysis['component_correlations']
            
            # Check for high correlations
            for comp1 in correlations:
                for comp2 in correlations[comp1]:
                    if comp1 != comp2 and abs(correlations[comp1][comp2]) > 0.8:
                        insights.append(f"High correlation between {comp1} and {comp2}")
        
        return insights

    def get_quality_scores(self, item_id: str) -> Dict[str, Any]:
        """Get quality scores for a specific item"""
        try:
            # This is a simplified implementation - in a real system, 
            # you would fetch from your quality scores database/cache
            return {
                "content_id": item_id,
                "overall_score": 0.75,
                "popularity_score": 0.80,
                "acclaim_score": 0.70,
                "rating_score": 0.85,
                "freshness_score": 0.60,
                "content_quality_score": 0.75,
                "last_updated": datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error getting quality scores for {item_id}: {e}")
            return {
                "content_id": item_id,
                "overall_score": 0.0,
                "popularity_score": 0.0,
                "acclaim_score": 0.0,
                "rating_score": 0.0,
                "freshness_score": 0.0,
                "content_quality_score": 0.0,
                "last_updated": datetime.now()
            }
