"""
Recommendation Scoring and Evaluation Module

This module provides comprehensive scoring and evaluation metrics for 
recommendation systems, including accuracy, diversity, novelty, and fairness metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import logging
from sklearn.metrics import (
    ndcg_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, roc_auc_score
)
from scipy.spatial.distance import cosine, jaccard
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RecommendationScorer:
    """Comprehensive scoring system for recommendation quality."""
    
    def __init__(self):
        """Initialize the scorer."""
        self.item_popularity = {}
        self.user_profiles = {}
        self.global_stats = {}
        
    def update_item_popularity(self, interactions: List[Dict[str, Any]]):
        """Update item popularity statistics."""
        try:
            item_counts = Counter()
            for interaction in interactions:
                item_id = interaction.get('item_id')
                if item_id:
                    item_counts[item_id] += 1
            
            total_interactions = sum(item_counts.values())
            self.item_popularity = {
                item_id: count / total_interactions 
                for item_id, count in item_counts.items()
            }
            
            logger.info(f"Updated popularity for {len(self.item_popularity)} items")
            
        except Exception as e:
            logger.error(f"Error updating item popularity: {e}")
    
    def compute_accuracy_metrics(self, 
                                recommendations: List[Dict[str, Any]], 
                                ground_truth: List[Dict[str, Any]],
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Compute accuracy metrics for recommendations.
        
        Args:
            recommendations: List of recommendations with 'item_id' and 'score'
            ground_truth: List of actual interactions with 'item_id' and 'rating'
            k_values: List of k values for top-k metrics
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            if not recommendations or not ground_truth:
                return {}
            
            # Convert to sets for easier computation
            rec_items = {rec['item_id'] for rec in recommendations}
            true_items = {gt['item_id'] for gt in ground_truth if gt.get('rating', 0) > 3.5}
            
            metrics = {}
            
            # Overall metrics
            intersection = rec_items & true_items
            metrics['precision'] = len(intersection) / len(rec_items) if rec_items else 0.0
            metrics['recall'] = len(intersection) / len(true_items) if true_items else 0.0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
            
            # Top-k metrics
            for k in k_values:
                top_k_recs = set([rec['item_id'] for rec in recommendations[:k]])
                top_k_intersection = top_k_recs & true_items
                
                metrics[f'precision@{k}'] = len(top_k_intersection) / min(k, len(rec_items))
                metrics[f'recall@{k}'] = len(top_k_intersection) / len(true_items) if true_items else 0.0
                metrics[f'hit_rate@{k}'] = 1.0 if top_k_intersection else 0.0
            
            # NDCG computation
            try:
                # Create relevance scores
                true_relevance = {item['item_id']: item.get('rating', 0) for item in ground_truth}
                
                for k in k_values:
                    y_true = []
                    y_score = []
                    
                    for rec in recommendations[:k]:
                        item_id = rec['item_id']
                        y_score.append(rec.get('score', 0))
                        y_true.append(true_relevance.get(item_id, 0))
                    
                    if y_true and max(y_true) > 0:
                        ndcg = ndcg_score([y_true], [y_score], k=k)
                        metrics[f'ndcg@{k}'] = ndcg
                    else:
                        metrics[f'ndcg@{k}'] = 0.0
                        
            except Exception as e:
                logger.warning(f"Error computing NDCG: {e}")
                for k in k_values:
                    metrics[f'ndcg@{k}'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing accuracy metrics: {e}")
            return {}
    
    def compute_diversity_metrics(self, 
                                 recommendations: List[Dict[str, Any]], 
                                 item_features: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute diversity metrics for recommendations.
        
        Args:
            recommendations: List of recommendations
            item_features: Dictionary of item features
            
        Returns:
            Dictionary of diversity metrics
        """
        try:
            if not recommendations:
                return {'intra_list_diversity': 0.0, 'genre_diversity': 0.0}
            
            metrics = {}
            
            # Intra-list diversity (average pairwise distance)
            similarities = []
            for i, rec1 in enumerate(recommendations):
                for j, rec2 in enumerate(recommendations[i+1:], i+1):
                    item1_id = rec1['item_id']
                    item2_id = rec2['item_id']
                    
                    if item1_id in item_features and item2_id in item_features:
                        sim = self._compute_item_similarity(
                            item_features[item1_id], 
                            item_features[item2_id]
                        )
                        similarities.append(1 - sim)  # Convert similarity to distance
            
            metrics['intra_list_diversity'] = np.mean(similarities) if similarities else 0.0
            
            # Genre diversity (entropy of genre distribution)
            genres = []
            for rec in recommendations:
                item_id = rec['item_id']
                if item_id in item_features:
                    item_genres = item_features[item_id].get('genres', [])
                    if isinstance(item_genres, str):
                        item_genres = item_genres.split('|')
                    genres.extend(item_genres)
            
            if genres:
                genre_counts = Counter(genres)
                genre_probs = np.array(list(genre_counts.values())) / len(genres)
                metrics['genre_diversity'] = entropy(genre_probs)
            else:
                metrics['genre_diversity'] = 0.0
            
            # Coverage (percentage of unique items in recommendations)
            unique_items = len(set(rec['item_id'] for rec in recommendations))
            metrics['coverage'] = unique_items / len(recommendations)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing diversity metrics: {e}")
            return {}
    
    def compute_novelty_metrics(self, 
                               recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute novelty metrics for recommendations.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dictionary of novelty metrics
        """
        try:
            if not recommendations or not self.item_popularity:
                return {'novelty': 0.0, 'long_tail_coverage': 0.0}
            
            metrics = {}
            
            # Average novelty (negative log popularity)
            novelties = []
            for rec in recommendations:
                item_id = rec['item_id']
                popularity = self.item_popularity.get(item_id, 1e-6)
                novelty = -np.log2(popularity + 1e-8)
                novelties.append(novelty)
            
            metrics['novelty'] = np.mean(novelties) if novelties else 0.0
            
            # Long-tail coverage (percentage of long-tail items)
            # Define long-tail as bottom 80% of items by popularity
            if self.item_popularity:
                sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
                long_tail_threshold = int(0.2 * len(sorted_items))
                long_tail_items = set(item for item, _ in sorted_items[long_tail_threshold:])
                
                rec_items = set(rec['item_id'] for rec in recommendations)
                long_tail_recs = rec_items & long_tail_items
                
                metrics['long_tail_coverage'] = len(long_tail_recs) / len(rec_items) if rec_items else 0.0
            else:
                metrics['long_tail_coverage'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing novelty metrics: {e}")
            return {}
    
    def compute_serendipity_metrics(self, 
                                   recommendations: List[Dict[str, Any]], 
                                   user_profile: Dict[str, Any],
                                   item_features: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute serendipity metrics for recommendations.
        
        Args:
            recommendations: List of recommendations
            user_profile: User's historical preferences
            item_features: Dictionary of item features
            
        Returns:
            Dictionary of serendipity metrics
        """
        try:
            if not recommendations or not user_profile:
                return {'serendipity': 0.0}
            
            # Get user's historical genres/features
            user_genres = set()
            if 'preferred_genres' in user_profile:
                user_genres.update(user_profile['preferred_genres'])
            
            # Calculate unexpectedness for each recommendation
            unexpectedness_scores = []
            for rec in recommendations:
                item_id = rec['item_id']
                if item_id in item_features:
                    item_genres = item_features[item_id].get('genres', [])
                    if isinstance(item_genres, str):
                        item_genres = item_genres.split('|')
                    
                    item_genre_set = set(item_genres)
                    
                    # Unexpectedness = 1 - overlap with user preferences
                    if user_genres and item_genre_set:
                        overlap = len(user_genres & item_genre_set) / len(user_genres | item_genre_set)
                        unexpectedness = 1 - overlap
                    else:
                        unexpectedness = 0.5  # Neutral if no genre info
                    
                    # Weight by recommendation score (serendipity = unexpectedness * relevance)
                    relevance = rec.get('score', 0.5)
                    serendipity = unexpectedness * relevance
                    unexpectedness_scores.append(serendipity)
            
            metrics = {
                'serendipity': np.mean(unexpectedness_scores) if unexpectedness_scores else 0.0,
                'unexpectedness': np.mean([score / (rec.get('score', 1) + 1e-8) 
                                         for score, rec in zip(unexpectedness_scores, recommendations)]) 
                                 if unexpectedness_scores else 0.0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing serendipity metrics: {e}")
            return {}
    
    def compute_fairness_metrics(self, 
                                recommendations: List[Dict[str, Any]], 
                                item_features: Dict[str, Dict[str, Any]],
                                protected_attributes: List[str] = ['genre', 'year', 'country']) -> Dict[str, float]:
        """
        Compute fairness metrics for recommendations.
        
        Args:
            recommendations: List of recommendations
            item_features: Dictionary of item features
            protected_attributes: List of attributes to check fairness for
            
        Returns:
            Dictionary of fairness metrics
        """
        try:
            if not recommendations or not item_features:
                return {}
            
            metrics = {}
            
            for attr in protected_attributes:
                attr_values = []
                for rec in recommendations:
                    item_id = rec['item_id']
                    if item_id in item_features:
                        value = item_features[item_id].get(attr)
                        if value:
                            if isinstance(value, list):
                                attr_values.extend(value)
                            else:
                                attr_values.append(value)
                
                if attr_values:
                    # Calculate entropy (higher entropy = more fair distribution)
                    value_counts = Counter(attr_values)
                    total = sum(value_counts.values())
                    probs = np.array(list(value_counts.values())) / total
                    
                    fairness_entropy = entropy(probs)
                    # Normalize by maximum possible entropy
                    max_entropy = np.log(len(value_counts))
                    normalized_fairness = fairness_entropy / max_entropy if max_entropy > 0 else 0.0
                    
                    metrics[f'fairness_{attr}'] = normalized_fairness
                    
                    # Gini coefficient for inequality measurement
                    sorted_probs = np.sort(probs)
                    n = len(sorted_probs)
                    index = np.arange(1, n + 1)
                    gini = 2 * np.sum(index * sorted_probs) / (n * np.sum(sorted_probs)) - (n + 1) / n
                    metrics[f'gini_{attr}'] = gini
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing fairness metrics: {e}")
            return {}
    
    def compute_temporal_metrics(self, 
                                recommendations: List[Dict[str, Any]],
                                item_features: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute temporal diversity and freshness metrics.
        
        Args:
            recommendations: List of recommendations
            item_features: Dictionary of item features
            
        Returns:
            Dictionary of temporal metrics
        """
        try:
            if not recommendations or not item_features:
                return {}
            
            metrics = {}
            years = []
            
            for rec in recommendations:
                item_id = rec['item_id']
                if item_id in item_features:
                    year = item_features[item_id].get('year')
                    if year:
                        years.append(int(year))
            
            if years:
                # Temporal diversity (standard deviation of years)
                metrics['temporal_diversity'] = np.std(years)
                
                # Freshness (average recency, assuming current year is reference)
                current_year = 2024  # Could be made dynamic
                recency_scores = [(current_year - year) for year in years]
                metrics['freshness'] = 1 / (1 + np.mean(recency_scores))  # Inverse of average age
                
                # Temporal coverage (span of years)
                metrics['temporal_coverage'] = max(years) - min(years) if len(years) > 1 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing temporal metrics: {e}")
            return {}
    
    def _compute_item_similarity(self, item1_features: Dict[str, Any], 
                                item2_features: Dict[str, Any]) -> float:
        """Compute similarity between two items based on their features."""
        try:
            # Genre similarity
            genres1 = item1_features.get('genres', [])
            genres2 = item2_features.get('genres', [])
            
            if isinstance(genres1, str):
                genres1 = genres1.split('|')
            if isinstance(genres2, str):
                genres2 = genres2.split('|')
            
            genres1_set = set(genres1)
            genres2_set = set(genres2)
            
            if genres1_set and genres2_set:
                genre_similarity = len(genres1_set & genres2_set) / len(genres1_set | genres2_set)
            else:
                genre_similarity = 0.0
            
            # Year similarity
            year1 = item1_features.get('year')
            year2 = item2_features.get('year')
            
            if year1 and year2:
                year_diff = abs(int(year1) - int(year2))
                year_similarity = 1 / (1 + year_diff / 10)  # Normalize by decade
            else:
                year_similarity = 0.0
            
            # Combined similarity
            similarity = 0.7 * genre_similarity + 0.3 * year_similarity
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing item similarity: {e}")
            return 0.0
    
    def compute_comprehensive_score(self, 
                                   recommendations: List[Dict[str, Any]], 
                                   ground_truth: List[Dict[str, Any]],
                                   user_profile: Dict[str, Any],
                                   item_features: Dict[str, Dict[str, Any]],
                                   weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compute comprehensive recommendation quality score.
        
        Args:
            recommendations: List of recommendations
            ground_truth: List of actual interactions
            user_profile: User's historical preferences
            item_features: Dictionary of item features
            weights: Weights for different metric categories
            
        Returns:
            Dictionary with all metrics and overall score
        """
        try:
            if weights is None:
                weights = {
                    'accuracy': 0.4,
                    'diversity': 0.2,
                    'novelty': 0.15,
                    'serendipity': 0.15,
                    'fairness': 0.1
                }
            
            # Compute all metric categories
            accuracy_metrics = self.compute_accuracy_metrics(recommendations, ground_truth)
            diversity_metrics = self.compute_diversity_metrics(recommendations, item_features)
            novelty_metrics = self.compute_novelty_metrics(recommendations)
            serendipity_metrics = self.compute_serendipity_metrics(recommendations, user_profile, item_features)
            fairness_metrics = self.compute_fairness_metrics(recommendations, item_features)
            temporal_metrics = self.compute_temporal_metrics(recommendations, item_features)
            
            # Aggregate scores for each category
            accuracy_score = accuracy_metrics.get('ndcg@10', 0.0)
            diversity_score = diversity_metrics.get('intra_list_diversity', 0.0)
            novelty_score = min(novelty_metrics.get('novelty', 0.0) / 10, 1.0)  # Normalize
            serendipity_score = serendipity_metrics.get('serendipity', 0.0)
            fairness_score = np.mean([v for k, v in fairness_metrics.items() if 'fairness_' in k]) if fairness_metrics else 0.0
            
            # Calculate weighted overall score
            overall_score = (
                weights.get('accuracy', 0) * accuracy_score +
                weights.get('diversity', 0) * diversity_score +
                weights.get('novelty', 0) * novelty_score +
                weights.get('serendipity', 0) * serendipity_score +
                weights.get('fairness', 0) * fairness_score
            )
            
            return {
                'overall_score': overall_score,
                'category_scores': {
                    'accuracy': accuracy_score,
                    'diversity': diversity_score,
                    'novelty': novelty_score,
                    'serendipity': serendipity_score,
                    'fairness': fairness_score
                },
                'detailed_metrics': {
                    'accuracy': accuracy_metrics,
                    'diversity': diversity_metrics,
                    'novelty': novelty_metrics,
                    'serendipity': serendipity_metrics,
                    'fairness': fairness_metrics,
                    'temporal': temporal_metrics
                },
                'weights_used': weights
            }
            
        except Exception as e:
            logger.error(f"Error computing comprehensive score: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
