"""
Comprehensive Recommendation Quality Metrics
Implements all standard recommendation system evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import math

logger = logging.getLogger(__name__)

class AccuracyMetrics:
    """Calculate accuracy metrics for recommendation systems"""
    
    @staticmethod
    def ndcg_at_k(recommendations: List[int], relevance_scores: Dict[int, float], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        Target: > 0.35 for interview requirements
        """
        if not recommendations or k <= 0:
            return 0.0
        
        # Get top-k recommendations
        top_k = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(top_k):
            relevance = relevance_scores.get(item_id, 0.0)
            dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
        """Calculate Precision at K"""
        if not recommendations or k <= 0:
            return 0.0
        
        top_k = set(recommendations[:k])
        relevant_in_top_k = len(top_k.intersection(relevant_items))
        
        return relevant_in_top_k / min(k, len(recommendations))
    
    @staticmethod
    def recall_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
        """Calculate Recall at K"""
        if not recommendations or not relevant_items or k <= 0:
            return 0.0
        
        top_k = set(recommendations[:k])
        relevant_in_top_k = len(top_k.intersection(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def f1_score_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
        """Calculate F1 Score at K"""
        precision = AccuracyMetrics.precision_at_k(recommendations, relevant_items, k)
        recall = AccuracyMetrics.recall_at_k(recommendations, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(recommendations: List[int], relevant_items: Set[int]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        for i, item_id in enumerate(recommendations):
            if item_id in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def hit_rate_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
        """Calculate Hit Rate at K (binary: 1 if any relevant item in top-k, 0 otherwise)"""
        if not recommendations or not relevant_items or k <= 0:
            return 0.0
        
        top_k = set(recommendations[:k])
        return 1.0 if len(top_k.intersection(relevant_items)) > 0 else 0.0
    
    @staticmethod
    def mae(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        if len(predicted_ratings) != len(actual_ratings) or not predicted_ratings:
            return float('inf')
        
        return mean_absolute_error(actual_ratings, predicted_ratings)
    
    @staticmethod
    def rmse(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """Calculate Root Mean Square Error"""
        if len(predicted_ratings) != len(actual_ratings) or not predicted_ratings:
            return float('inf')
        
        return math.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    
    @staticmethod
    def coverage(recommendations_by_user: Dict[str, List[int]], total_items: int) -> float:
        """Calculate catalog coverage - percentage of items that appear in recommendations"""
        all_recommended_items = set()
        for recommendations in recommendations_by_user.values():
            all_recommended_items.update(recommendations)
        
        return len(all_recommended_items) / total_items if total_items > 0 else 0.0

class DiversityMetrics:
    """Calculate diversity and novelty metrics"""
    
    @staticmethod
    def intra_list_diversity(recommendations: List[int], item_features: Dict[int, List[str]], 
                           similarity_func=None) -> float:
        """
        Calculate intra-list diversity (average dissimilarity between pairs)
        Target: > 0.7 for interview requirements
        """
        if len(recommendations) < 2:
            return 0.0
        
        if similarity_func is None:
            # Default Jaccard similarity for categorical features
            similarity_func = DiversityMetrics._jaccard_similarity
        
        total_dissimilarity = 0.0
        pair_count = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item1_features = item_features.get(recommendations[i], [])
                item2_features = item_features.get(recommendations[j], [])
                
                similarity = similarity_func(item1_features, item2_features)
                dissimilarity = 1.0 - similarity
                
                total_dissimilarity += dissimilarity
                pair_count += 1
        
        return total_dissimilarity / pair_count if pair_count > 0 else 0.0
    
    @staticmethod
    def _jaccard_similarity(features1: List[str], features2: List[str]) -> float:
        """Calculate Jaccard similarity between two feature sets"""
        set1, set2 = set(features1), set(features2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def novelty_score(recommendations: List[int], item_popularity: Dict[int, float]) -> float:
        """Calculate novelty as inverse of popularity"""
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item_id in recommendations:
            popularity = item_popularity.get(item_id, 0.0)
            # Higher novelty for less popular items
            novelty = 1.0 - popularity if popularity > 0 else 1.0
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    @staticmethod
    def serendipity_score(recommendations: List[int], user_profile: Dict[str, Any], 
                         item_features: Dict[int, List[str]]) -> float:
        """Calculate serendipity - unexpected but relevant recommendations"""
        if not recommendations:
            return 0.0
        
        user_genres = set(user_profile.get('preferred_genres', []))
        serendipity_scores = []
        
        for item_id in recommendations:
            item_genres = set(item_features.get(item_id, []))
            
            # Serendipity: item is relevant but from different genre
            genre_overlap = len(user_genres.intersection(item_genres))
            genre_novelty = len(item_genres - user_genres)
            
            # High serendipity: some relevance but mostly new genres
            if genre_overlap > 0 and genre_novelty > 0:
                serendipity = genre_novelty / (genre_overlap + genre_novelty)
            else:
                serendipity = 0.0
            
            serendipity_scores.append(serendipity)
        
        return np.mean(serendipity_scores)
    
    @staticmethod
    def personalization_score(recommendations_by_user: Dict[str, List[int]]) -> float:
        """Calculate personalization - how different are recommendations across users"""
        users = list(recommendations_by_user.keys())
        if len(users) < 2:
            return 0.0
        
        total_dissimilarity = 0.0
        pair_count = 0
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_recs = set(recommendations_by_user[users[i]])
                user2_recs = set(recommendations_by_user[users[j]])
                
                # Jaccard dissimilarity
                intersection = len(user1_recs.intersection(user2_recs))
                union = len(user1_recs.union(user2_recs))
                
                dissimilarity = 1.0 - (intersection / union) if union > 0 else 1.0
                total_dissimilarity += dissimilarity
                pair_count += 1
        
        return total_dissimilarity / pair_count if pair_count > 0 else 0.0

class FairnessMetrics:
    """Calculate fairness metrics across user demographics"""
    
    @staticmethod
    def demographic_parity(recommendations_by_group: Dict[str, List[List[int]]], 
                          item_categories: Dict[int, str]) -> Dict[str, float]:
        """Calculate demographic parity across user groups"""
        category_distribution = {}
        
        for group, group_recommendations in recommendations_by_group.items():
            category_counts = defaultdict(int)
            total_recommendations = 0
            
            for user_recommendations in group_recommendations:
                for item_id in user_recommendations:
                    category = item_categories.get(item_id, 'unknown')
                    category_counts[category] += 1
                    total_recommendations += 1
            
            # Calculate distribution for this group
            group_distribution = {}
            for category, count in category_counts.items():
                group_distribution[category] = count / total_recommendations if total_recommendations > 0 else 0.0
            
            category_distribution[group] = group_distribution
        
        return category_distribution
    
    @staticmethod
    def calculate_bias_score(metric_by_group: Dict[str, float]) -> float:
        """Calculate bias as coefficient of variation across groups"""
        values = list(metric_by_group.values())
        if not values:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        return std_val / mean_val if mean_val > 0 else 0.0

class BusinessMetrics:
    """Calculate business-oriented metrics"""
    
    @staticmethod
    def click_through_rate(clicks: int, impressions: int) -> float:
        """Calculate Click-Through Rate"""
        return clicks / impressions if impressions > 0 else 0.0
    
    @staticmethod
    def conversion_rate(conversions: int, clicks: int) -> float:
        """Calculate Conversion Rate"""
        return conversions / clicks if clicks > 0 else 0.0
    
    @staticmethod
    def average_session_duration(session_durations: List[float]) -> float:
        """Calculate average session duration"""
        return np.mean(session_durations) if session_durations else 0.0
    
    @staticmethod
    def user_retention_rate(active_users_period1: Set[str], 
                           active_users_period2: Set[str]) -> float:
        """Calculate user retention rate between two periods"""
        if not active_users_period1:
            return 0.0
        
        retained_users = len(active_users_period1.intersection(active_users_period2))
        return retained_users / len(active_users_period1)
    
    @staticmethod
    def revenue_per_user(total_revenue: float, num_users: int) -> float:
        """Calculate Revenue Per User"""
        return total_revenue / num_users if num_users > 0 else 0.0

class MetricsEvaluator:
    """Main class for comprehensive metrics evaluation"""
    
    def __init__(self):
        self.accuracy_metrics = AccuracyMetrics()
        self.diversity_metrics = DiversityMetrics()
        self.fairness_metrics = FairnessMetrics()
        self.business_metrics = BusinessMetrics()
    
    def evaluate_comprehensive(self, 
                             recommendations_by_user: Dict[str, List[int]],
                             relevance_data: Dict[str, Dict[int, float]],
                             item_features: Dict[int, List[str]],
                             item_popularity: Dict[int, float],
                             user_demographics: Optional[Dict[str, str]] = None,
                             business_data: Optional[Dict[str, Any]] = None,
                             k: int = 10) -> Dict[str, Any]:
        """
        Comprehensive evaluation of recommendation system
        
        Returns all metrics required for production evaluation
        """
        results = {
            'accuracy': {},
            'diversity': {},
            'fairness': {},
            'business': {},
            'summary': {}
        }
        
        # Accuracy metrics
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []
        hit_rates = []
        
        for user_id, recommendations in recommendations_by_user.items():
            user_relevance = relevance_data.get(user_id, {})
            relevant_items = set(item_id for item_id, score in user_relevance.items() if score > 0.5)
            
            ndcg = self.accuracy_metrics.ndcg_at_k(recommendations, user_relevance, k)
            precision = self.accuracy_metrics.precision_at_k(recommendations, relevant_items, k)
            recall = self.accuracy_metrics.recall_at_k(recommendations, relevant_items, k)
            f1 = self.accuracy_metrics.f1_score_at_k(recommendations, relevant_items, k)
            mrr = self.accuracy_metrics.mean_reciprocal_rank(recommendations, relevant_items)
            hit_rate = self.accuracy_metrics.hit_rate_at_k(recommendations, relevant_items, k)
            
            ndcg_scores.append(ndcg)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            mrr_scores.append(mrr)
            hit_rates.append(hit_rate)
        
        results['accuracy'] = {
            'ndcg_at_10': np.mean(ndcg_scores),
            'precision_at_10': np.mean(precision_scores),
            'recall_at_10': np.mean(recall_scores),
            'f1_at_10': np.mean(f1_scores),
            'mrr': np.mean(mrr_scores),
            'hit_rate_at_10': np.mean(hit_rates),
            'coverage': self.accuracy_metrics.coverage(recommendations_by_user, len(item_features))
        }
        
        # Diversity metrics
        diversity_scores = []
        novelty_scores = []
        
        for recommendations in recommendations_by_user.values():
            diversity = self.diversity_metrics.intra_list_diversity(recommendations, item_features)
            novelty = self.diversity_metrics.novelty_score(recommendations, item_popularity)
            
            diversity_scores.append(diversity)
            novelty_scores.append(novelty)
        
        results['diversity'] = {
            'intra_list_diversity': np.mean(diversity_scores),
            'novelty': np.mean(novelty_scores),
            'personalization': self.diversity_metrics.personalization_score(recommendations_by_user)
        }
        
        # Fairness metrics (if demographic data available)
        if user_demographics:
            # Group recommendations by demographics
            recommendations_by_group = defaultdict(list)
            for user_id, recommendations in recommendations_by_user.items():
                group = user_demographics.get(user_id, 'unknown')
                recommendations_by_group[group].append(recommendations)
            
            # Calculate NDCG by group for fairness assessment
            ndcg_by_group = {}
            for group, group_recs in recommendations_by_group.items():
                group_ndcg_scores = []
                for recommendations in group_recs:
                    user_relevance = relevance_data.get(user_id, {})  # This should be matched properly
                    ndcg = self.accuracy_metrics.ndcg_at_k(recommendations, user_relevance, k)
                    group_ndcg_scores.append(ndcg)
                ndcg_by_group[group] = np.mean(group_ndcg_scores) if group_ndcg_scores else 0.0
            
            bias_score = self.fairness_metrics.calculate_bias_score(ndcg_by_group)
            
            results['fairness'] = {
                'ndcg_by_group': ndcg_by_group,
                'bias_score': bias_score
            }
        
        # Business metrics (if business data available)
        if business_data:
            results['business'] = {
                'ctr': self.business_metrics.click_through_rate(
                    business_data.get('clicks', 0),
                    business_data.get('impressions', 1)
                ),
                'conversion_rate': self.business_metrics.conversion_rate(
                    business_data.get('conversions', 0),
                    business_data.get('clicks', 1)
                ),
                'avg_session_duration': self.business_metrics.average_session_duration(
                    business_data.get('session_durations', [])
                )
            }
        
        # Summary metrics for quick assessment
        results['summary'] = {
            'ndcg_at_10_meets_target': results['accuracy']['ndcg_at_10'] > 0.35,
            'diversity_meets_target': results['diversity']['intra_list_diversity'] > 0.7,
            'overall_score': (
                results['accuracy']['ndcg_at_10'] * 0.4 +
                results['diversity']['intra_list_diversity'] * 0.3 +
                results['accuracy']['precision_at_10'] * 0.2 +
                results['diversity']['novelty'] * 0.1
            ),
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = evaluation_results['summary']
        report.append("SUMMARY:")
        report.append(f"  Overall Score: {summary['overall_score']:.3f}")
        report.append(f"  NDCG@10 Target Met: {'✓' if summary['ndcg_at_10_meets_target'] else '✗'}")
        report.append(f"  Diversity Target Met: {'✓' if summary['diversity_meets_target'] else '✗'}")
        report.append("")
        
        # Accuracy metrics
        accuracy = evaluation_results['accuracy']
        report.append("ACCURACY METRICS:")
        report.append(f"  NDCG@10: {accuracy['ndcg_at_10']:.3f} (target: >0.35)")
        report.append(f"  Precision@10: {accuracy['precision_at_10']:.3f}")
        report.append(f"  Recall@10: {accuracy['recall_at_10']:.3f}")
        report.append(f"  F1@10: {accuracy['f1_at_10']:.3f}")
        report.append(f"  MRR: {accuracy['mrr']:.3f}")
        report.append(f"  Hit Rate@10: {accuracy['hit_rate_at_10']:.3f}")
        report.append(f"  Coverage: {accuracy['coverage']:.3f}")
        report.append("")
        
        # Diversity metrics
        diversity = evaluation_results['diversity']
        report.append("DIVERSITY METRICS:")
        report.append(f"  Intra-list Diversity: {diversity['intra_list_diversity']:.3f} (target: >0.7)")
        report.append(f"  Novelty: {diversity['novelty']:.3f}")
        report.append(f"  Personalization: {diversity['personalization']:.3f}")
        report.append("")
        
        # Fairness metrics
        if 'fairness' in evaluation_results:
            fairness = evaluation_results['fairness']
            report.append("FAIRNESS METRICS:")
            report.append(f"  Bias Score: {fairness['bias_score']:.3f} (lower is better)")
            report.append("  NDCG by Group:")
            for group, ndcg in fairness['ndcg_by_group'].items():
                report.append(f"    {group}: {ndcg:.3f}")
            report.append("")
        
        # Business metrics
        if 'business' in evaluation_results:
            business = evaluation_results['business']
            report.append("BUSINESS METRICS:")
            report.append(f"  Click-Through Rate: {business.get('ctr', 0):.3f}")
            report.append(f"  Conversion Rate: {business.get('conversion_rate', 0):.3f}")
            report.append(f"  Avg Session Duration: {business.get('avg_session_duration', 0):.2f}s")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
