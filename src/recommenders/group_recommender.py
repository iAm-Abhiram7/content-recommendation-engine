"""
Group Recommender

Implements group recommendation strategies:
- Multiple consensus strategies (average, least misery, most pleasure)
- Group preference aggregation
- Fairness and satisfaction optimization
- Group explanation generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from collections import defaultdict, Counter
from scipy.stats import entropy
import joblib

logger = logging.getLogger(__name__)


class GroupRecommender:
    """
    Group recommendation system for multiple users
    """
    
    def __init__(self,
                 base_recommender=None,
                 aggregation_strategy: str = 'average',
                 fairness_weight: float = 0.3,
                 min_satisfaction_threshold: float = 0.5):
        """
        Initialize group recommender
        
        Args:
            base_recommender: Base recommender to use for individual recommendations
            aggregation_strategy: Strategy for aggregating preferences ('average', 'least_misery', 'most_pleasure', 'fairness')
            fairness_weight: Weight for fairness consideration (0-1)
            min_satisfaction_threshold: Minimum satisfaction threshold for users
        """
        self.base_recommender = base_recommender
        self.aggregation_strategy = aggregation_strategy
        self.fairness_weight = fairness_weight
        self.min_satisfaction_threshold = min_satisfaction_threshold
        
        # Group profiles and preferences
        self.group_profiles = {}
        self.user_similarities = {}
        
        # Aggregation strategies
        self.aggregation_functions = {
            'average': self._average_aggregation,
            'least_misery': self._least_misery_aggregation,
            'most_pleasure': self._most_pleasure_aggregation,
            'fairness': self._fairness_aggregation,
            'weighted_average': self._weighted_average_aggregation,
            'approval_voting': self._approval_voting_aggregation,
            'borda_count': self._borda_count_aggregation
        }
        
        # Performance tracking
        self.group_satisfaction_history = defaultdict(list)
        
    def recommend_for_group(self,
                           user_ids: List[str],
                           n_recommendations: int = 10,
                           user_histories: Optional[Dict[str, List[str]]] = None,
                           user_weights: Optional[Dict[str, float]] = None,
                           filters: Optional[Dict] = None,
                           context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a group of users
        
        Args:
            user_ids: List of user identifiers in the group
            n_recommendations: Number of recommendations to return
            user_histories: Optional dict of user interaction histories
            user_weights: Optional weights for each user in group decisions
            filters: Optional filters to apply
            context: Optional contextual information
            
        Returns:
            List of group recommendations with satisfaction scores
        """
        logger.info(f"Generating group recommendations for {len(user_ids)} users")
        
        if not user_ids:
            return []
        
        # Handle single user case
        if len(user_ids) == 1:
            return self._single_user_recommendations(
                user_ids[0], n_recommendations, user_histories, filters, context
            )
        
        # Get individual recommendations for each user
        individual_recommendations = {}
        
        for user_id in user_ids:
            try:
                user_history = user_histories.get(user_id, []) if user_histories else []
                
                if self.base_recommender:
                    recs = self.base_recommender.recommend(
                        user_id=user_id,
                        n_recommendations=n_recommendations * 3,  # Get more for aggregation
                        user_history=user_history,
                        filters=filters,
                        context=context,
                        exclude_seen=True
                    )
                else:
                    recs = []  # Fallback needed
                
                individual_recommendations[user_id] = recs
                
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                individual_recommendations[user_id] = []
        
        # Calculate user similarities for weighted aggregation
        self._calculate_user_similarities(user_ids, individual_recommendations)
        
        # Aggregate recommendations using selected strategy
        aggregation_func = self.aggregation_functions.get(
            self.aggregation_strategy, 
            self._average_aggregation
        )
        
        group_recommendations = aggregation_func(
            individual_recommendations, 
            user_weights or {}
        )
        
        # Apply fairness optimization
        if self.fairness_weight > 0:
            group_recommendations = self._optimize_fairness(
                group_recommendations, 
                individual_recommendations,
                user_ids
            )
        
        # Sort and limit to requested number
        group_recommendations.sort(key=lambda x: x['group_score'], reverse=True)
        final_recommendations = group_recommendations[:n_recommendations]
        
        # Generate group explanations
        for rec in final_recommendations:
            rec['group_explanation'] = self._generate_group_explanation(
                rec, individual_recommendations, user_ids
            )
        
        # Calculate and store satisfaction metrics
        satisfaction_metrics = self._calculate_group_satisfaction(
            final_recommendations, individual_recommendations, user_ids
        )
        
        # Add satisfaction info to recommendations
        for i, rec in enumerate(final_recommendations):
            rec['satisfaction_metrics'] = satisfaction_metrics
            rec['recommendation_rank'] = i + 1
        
        return final_recommendations
    
    def _single_user_recommendations(self,
                                   user_id: str,
                                   n_recommendations: int,
                                   user_histories: Optional[Dict],
                                   filters: Optional[Dict],
                                   context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Handle single user case"""
        if not self.base_recommender:
            return []
        
        user_history = user_histories.get(user_id, []) if user_histories else []
        
        recs = self.base_recommender.recommend(
            user_id=user_id,
            n_recommendations=n_recommendations,
            user_history=user_history,
            filters=filters,
            context=context
        )
        
        # Add group-specific fields
        for rec in recs:
            rec['group_score'] = rec['score']
            rec['user_scores'] = {user_id: rec['score']}
            rec['group_explanation'] = {
                'strategy': 'single_user',
                'reasoning': f"Personalized recommendation for {user_id}"
            }
        
        return recs
    
    def _calculate_user_similarities(self,
                                   user_ids: List[str],
                                   individual_recommendations: Dict[str, List[Dict]]):
        """Calculate pairwise user similarities based on recommendation overlap"""
        similarities = {}
        
        for i, user1 in enumerate(user_ids):
            for j, user2 in enumerate(user_ids[i+1:], i+1):
                # Get top items for each user
                items1 = set([rec['item_id'] for rec in individual_recommendations[user1][:20]])
                items2 = set([rec['item_id'] for rec in individual_recommendations[user2][:20]])
                
                # Calculate Jaccard similarity
                if items1 or items2:
                    intersection = len(items1.intersection(items2))
                    union = len(items1.union(items2))
                    similarity = intersection / union if union > 0 else 0
                else:
                    similarity = 0
                
                similarities[(user1, user2)] = similarity
                similarities[(user2, user1)] = similarity
        
        self.user_similarities = similarities
    
    def _average_aggregation(self,
                           individual_recs: Dict[str, List[Dict]],
                           user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Average aggregation strategy"""
        item_scores = defaultdict(list)
        item_details = {}
        
        # Collect scores for each item from all users
        for user_id, recs in individual_recs.items():
            weight = user_weights.get(user_id, 1.0)
            
            for rec in recs:
                item_id = rec['item_id']
                weighted_score = rec['score'] * weight
                item_scores[item_id].append((user_id, weighted_score, rec['score']))
                
                if item_id not in item_details:
                    item_details[item_id] = rec.copy()
        
        # Calculate average scores
        group_recommendations = []
        
        for item_id, scores in item_scores.items():
            if len(scores) >= len(individual_recs) * 0.3:  # At least 30% agreement
                user_scores = {user_id: orig_score for user_id, _, orig_score in scores}
                avg_score = np.mean([weighted_score for _, weighted_score, _ in scores])
                
                group_rec = item_details[item_id].copy()
                group_rec['group_score'] = avg_score
                group_rec['user_scores'] = user_scores
                group_rec['agreement_count'] = len(scores)
                group_rec['aggregation_method'] = 'average'
                
                group_recommendations.append(group_rec)
        
        return group_recommendations
    
    def _least_misery_aggregation(self,
                                individual_recs: Dict[str, List[Dict]],
                                user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Least misery aggregation strategy (minimize maximum dissatisfaction)"""
        item_scores = defaultdict(dict)
        item_details = {}
        
        # Collect scores for each item from all users
        for user_id, recs in individual_recs.items():
            for rec in recs:
                item_id = rec['item_id']
                item_scores[item_id][user_id] = rec['score']
                
                if item_id not in item_details:
                    item_details[item_id] = rec.copy()
        
        # Calculate least misery scores
        group_recommendations = []
        
        for item_id, user_score_dict in item_scores.items():
            # Only consider items that appear for multiple users
            if len(user_score_dict) >= 2:
                # Least misery = minimum score among users
                min_score = min(user_score_dict.values())
                
                group_rec = item_details[item_id].copy()
                group_rec['group_score'] = min_score
                group_rec['user_scores'] = user_score_dict
                group_rec['agreement_count'] = len(user_score_dict)
                group_rec['aggregation_method'] = 'least_misery'
                
                group_recommendations.append(group_rec)
        
        return group_recommendations
    
    def _most_pleasure_aggregation(self,
                                 individual_recs: Dict[str, List[Dict]],
                                 user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Most pleasure aggregation strategy (maximize maximum satisfaction)"""
        item_scores = defaultdict(dict)
        item_details = {}
        
        # Collect scores for each item from all users
        for user_id, recs in individual_recs.items():
            for rec in recs:
                item_id = rec['item_id']
                item_scores[item_id][user_id] = rec['score']
                
                if item_id not in item_details:
                    item_details[item_id] = rec.copy()
        
        # Calculate most pleasure scores
        group_recommendations = []
        
        for item_id, user_score_dict in item_scores.items():
            # Most pleasure = maximum score among users
            max_score = max(user_score_dict.values())
            
            group_rec = item_details[item_id].copy()
            group_rec['group_score'] = max_score
            group_rec['user_scores'] = user_score_dict
            group_rec['agreement_count'] = len(user_score_dict)
            group_rec['aggregation_method'] = 'most_pleasure'
            
            group_recommendations.append(group_rec)
        
        return group_recommendations
    
    def _fairness_aggregation(self,
                            individual_recs: Dict[str, List[Dict]],
                            user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Fairness-aware aggregation strategy"""
        # Start with average aggregation
        group_recs = self._average_aggregation(individual_recs, user_weights)
        
        # Apply fairness adjustments
        user_ids = list(individual_recs.keys())
        
        for rec in group_recs:
            user_scores = rec['user_scores']
            
            # Calculate fairness metrics
            scores = list(user_scores.values())
            
            if len(scores) > 1:
                # Standard deviation as unfairness measure
                unfairness = np.std(scores)
                
                # Adjust group score to penalize unfairness
                fairness_penalty = unfairness * self.fairness_weight
                rec['group_score'] = rec['group_score'] * (1 - fairness_penalty)
                rec['fairness_score'] = 1 - unfairness
                rec['aggregation_method'] = 'fairness'
        
        return group_recs
    
    def _weighted_average_aggregation(self,
                                    individual_recs: Dict[str, List[Dict]],
                                    user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Weighted average based on user similarities and explicit weights"""
        # Similar to average but uses user similarities as weights
        return self._average_aggregation(individual_recs, user_weights)
    
    def _approval_voting_aggregation(self,
                                   individual_recs: Dict[str, List[Dict]],
                                   user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Approval voting aggregation (count of users who approve)"""
        item_votes = defaultdict(list)
        item_details = {}
        approval_threshold = 0.5  # Items with score > 0.5 are "approved"
        
        # Count approvals for each item
        for user_id, recs in individual_recs.items():
            weight = user_weights.get(user_id, 1.0)
            
            for rec in recs:
                item_id = rec['item_id']
                
                if rec['score'] > approval_threshold:
                    item_votes[item_id].append((user_id, weight, rec['score']))
                
                if item_id not in item_details:
                    item_details[item_id] = rec.copy()
        
        # Calculate approval scores
        group_recommendations = []
        
        for item_id, votes in item_votes.items():
            if len(votes) >= 2:  # At least 2 approvals
                # Weighted approval count
                approval_score = sum(weight for _, weight, _ in votes)
                user_scores = {user_id: score for user_id, _, score in votes}
                
                group_rec = item_details[item_id].copy()
                group_rec['group_score'] = approval_score
                group_rec['user_scores'] = user_scores
                group_rec['approval_count'] = len(votes)
                group_rec['aggregation_method'] = 'approval_voting'
                
                group_recommendations.append(group_rec)
        
        return group_recommendations
    
    def _borda_count_aggregation(self,
                               individual_recs: Dict[str, List[Dict]],
                               user_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Borda count aggregation (rank-based voting)"""
        item_ranks = defaultdict(list)
        item_details = {}
        
        # Collect ranks for each item from all users
        for user_id, recs in individual_recs.items():
            weight = user_weights.get(user_id, 1.0)
            
            for rank, rec in enumerate(recs):
                item_id = rec['item_id']
                # Higher rank = better position (inverse of list index)
                borda_score = (len(recs) - rank) * weight
                item_ranks[item_id].append((user_id, borda_score, rec['score']))
                
                if item_id not in item_details:
                    item_details[item_id] = rec.copy()
        
        # Calculate Borda scores
        group_recommendations = []
        
        for item_id, ranks in item_ranks.items():
            if len(ranks) >= 2:  # At least 2 users ranked this item
                total_borda_score = sum(borda_score for _, borda_score, _ in ranks)
                user_scores = {user_id: orig_score for user_id, _, orig_score in ranks}
                
                group_rec = item_details[item_id].copy()
                group_rec['group_score'] = total_borda_score
                group_rec['user_scores'] = user_scores
                group_rec['rank_count'] = len(ranks)
                group_rec['aggregation_method'] = 'borda_count'
                
                group_recommendations.append(group_rec)
        
        return group_recommendations
    
    def _optimize_fairness(self,
                         group_recommendations: List[Dict[str, Any]],
                         individual_recommendations: Dict[str, List[Dict]],
                         user_ids: List[str]) -> List[Dict[str, Any]]:
        """Apply fairness optimization to ensure balanced satisfaction"""
        # Track satisfaction for each user across top recommendations
        top_n = min(20, len(group_recommendations))
        top_recs = sorted(group_recommendations, key=lambda x: x['group_score'], reverse=True)[:top_n]
        
        user_satisfaction = {user_id: 0 for user_id in user_ids}
        
        # Calculate cumulative satisfaction for each user
        for rec in top_recs:
            for user_id in user_ids:
                if user_id in rec.get('user_scores', {}):
                    user_satisfaction[user_id] += rec['user_scores'][user_id]
        
        # Identify under-satisfied users
        mean_satisfaction = np.mean(list(user_satisfaction.values()))
        under_satisfied = [
            user_id for user_id, satisfaction in user_satisfaction.items()
            if satisfaction < mean_satisfaction * 0.8
        ]
        
        # Boost recommendations that benefit under-satisfied users
        if under_satisfied:
            for rec in group_recommendations:
                boost_factor = 1.0
                
                for user_id in under_satisfied:
                    if user_id in rec.get('user_scores', {}):
                        # Boost based on how much this helps the under-satisfied user
                        user_score = rec['user_scores'][user_id]
                        boost_factor += user_score * self.fairness_weight
                
                rec['group_score'] *= boost_factor
                rec['fairness_boosted'] = boost_factor > 1.0
        
        return group_recommendations
    
    def _generate_group_explanation(self,
                                  recommendation: Dict[str, Any],
                                  individual_recommendations: Dict[str, List[Dict]],
                                  user_ids: List[str]) -> Dict[str, Any]:
        """Generate explanation for group recommendation"""
        explanation = {
            'aggregation_strategy': self.aggregation_strategy,
            'item_id': recommendation['item_id'],
            'group_score': recommendation['group_score'],
            'user_agreement': {},
            'reasoning': []
        }
        
        # Analyze user agreement
        user_scores = recommendation.get('user_scores', {})
        
        for user_id in user_ids:
            if user_id in user_scores:
                score = user_scores[user_id]
                # Find rank of this item in user's individual recommendations
                user_recs = individual_recommendations.get(user_id, [])
                item_rank = None
                
                for i, rec in enumerate(user_recs):
                    if rec['item_id'] == recommendation['item_id']:
                        item_rank = i + 1
                        break
                
                explanation['user_agreement'][user_id] = {
                    'score': score,
                    'rank': item_rank,
                    'satisfied': score >= self.min_satisfaction_threshold
                }
        
        # Generate reasoning text
        satisfied_users = [
            user_id for user_id, data in explanation['user_agreement'].items()
            if data['satisfied']
        ]
        
        total_users = len(user_ids)
        satisfied_count = len(satisfied_users)
        
        if satisfied_count == total_users:
            explanation['reasoning'].append("All group members would enjoy this recommendation")
        elif satisfied_count >= total_users * 0.7:
            explanation['reasoning'].append(f"Most group members ({satisfied_count}/{total_users}) would enjoy this")
        else:
            explanation['reasoning'].append(f"Some group members ({satisfied_count}/{total_users}) would enjoy this")
        
        # Add strategy-specific reasoning
        if self.aggregation_strategy == 'least_misery':
            min_score = min(user_scores.values()) if user_scores else 0
            explanation['reasoning'].append(f"Ensures minimum satisfaction of {min_score:.2f} for all members")
        elif self.aggregation_strategy == 'most_pleasure':
            max_score = max(user_scores.values()) if user_scores else 0
            explanation['reasoning'].append(f"Maximizes enjoyment with peak satisfaction of {max_score:.2f}")
        elif self.aggregation_strategy == 'fairness':
            fairness_score = recommendation.get('fairness_score', 0)
            explanation['reasoning'].append(f"Balances preferences fairly (fairness score: {fairness_score:.2f})")
        
        return explanation
    
    def _calculate_group_satisfaction(self,
                                    recommendations: List[Dict[str, Any]],
                                    individual_recommendations: Dict[str, List[Dict]],
                                    user_ids: List[str]) -> Dict[str, Any]:
        """Calculate group satisfaction metrics"""
        user_satisfactions = {}
        
        for user_id in user_ids:
            user_satisfaction = 0
            user_rec_items = set([rec['item_id'] for rec in individual_recommendations.get(user_id, [])])
            
            for rec in recommendations:
                if rec['item_id'] in user_rec_items:
                    # This item was in user's individual recommendations
                    user_satisfaction += rec.get('user_scores', {}).get(user_id, 0)
            
            user_satisfactions[user_id] = user_satisfaction
        
        satisfaction_values = list(user_satisfactions.values())
        
        metrics = {
            'individual_satisfaction': user_satisfactions,
            'mean_satisfaction': np.mean(satisfaction_values),
            'min_satisfaction': np.min(satisfaction_values),
            'max_satisfaction': np.max(satisfaction_values),
            'satisfaction_std': np.std(satisfaction_values),
            'satisfied_users_count': sum(1 for s in satisfaction_values if s >= self.min_satisfaction_threshold),
            'satisfaction_fairness': 1 - (np.std(satisfaction_values) / (np.mean(satisfaction_values) + 1e-8))
        }
        
        return metrics
    
    def analyze_group_dynamics(self,
                             user_ids: List[str],
                             user_histories: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze group dynamics and preferences"""
        analysis = {
            'group_size': len(user_ids),
            'user_similarities': {},
            'preference_diversity': 0,
            'consensus_difficulty': 0,
            'recommended_strategy': self.aggregation_strategy
        }
        
        # Calculate pairwise similarities
        if hasattr(self, 'user_similarities'):
            similarities = []
            for i, user1 in enumerate(user_ids):
                for user2 in user_ids[i+1:]:
                    sim = self.user_similarities.get((user1, user2), 0)
                    similarities.append(sim)
                    analysis['user_similarities'][f"{user1}_{user2}"] = sim
            
            if similarities:
                analysis['mean_similarity'] = np.mean(similarities)
                analysis['preference_diversity'] = 1 - np.mean(similarities)
        
        # Recommend strategy based on group characteristics
        if analysis.get('mean_similarity', 0) > 0.7:
            analysis['recommended_strategy'] = 'average'
            analysis['consensus_difficulty'] = 'low'
        elif analysis.get('preference_diversity', 0) > 0.8:
            analysis['recommended_strategy'] = 'fairness'
            analysis['consensus_difficulty'] = 'high'
        else:
            analysis['recommended_strategy'] = 'least_misery'
            analysis['consensus_difficulty'] = 'medium'
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save group recommender configuration"""
        model_data = {
            'aggregation_strategy': self.aggregation_strategy,
            'fairness_weight': self.fairness_weight,
            'min_satisfaction_threshold': self.min_satisfaction_threshold,
            'group_satisfaction_history': dict(self.group_satisfaction_history),
            'user_similarities': self.user_similarities
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Group recommender saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load group recommender configuration"""
        model_data = joblib.load(filepath)
        
        self.aggregation_strategy = model_data['aggregation_strategy']
        self.fairness_weight = model_data['fairness_weight']
        self.min_satisfaction_threshold = model_data['min_satisfaction_threshold']
        self.group_satisfaction_history = defaultdict(list, model_data.get('group_satisfaction_history', {}))
        self.user_similarities = model_data.get('user_similarities', {})
        
        logger.info(f"Group recommender loaded from {filepath}")
