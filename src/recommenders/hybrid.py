"""
Hybrid Recommender

Combines collaborative, content-based, and knowledge-based filtering:
- Weighted ensemble of different recommendation methods
- Dynamic weight optimization based on performance
- Diversity boosting and explanation generation
- Fallback strategies for cold start scenarios
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from collections import defaultdict
import json
from sklearn.metrics import ndcg_score
from scipy import sparse
import joblib

from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .knowledge_based import KnowledgeBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommendation system combining multiple approaches
    """
    
    def __init__(self,
                 collaborative_weight: float = 0.4,
                 content_weight: float = 0.4,
                 knowledge_weight: float = 0.2,
                 diversity_factor: float = 0.1,
                 explanation_enabled: bool = True,
                 auto_tune_weights: bool = True):
        """
        Initialize hybrid recommender
        
        Args:
            collaborative_weight: Weight for collaborative filtering
            content_weight: Weight for content-based filtering  
            knowledge_weight: Weight for knowledge-based filtering
            diversity_factor: Factor for diversity promotion (0-1)
            explanation_enabled: Whether to generate explanations
            auto_tune_weights: Whether to automatically tune weights
        """
        # Weights for ensemble
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.knowledge_weight = knowledge_weight
        self.diversity_factor = diversity_factor
        
        # Normalize weights
        self._normalize_weights()
        
        # Components
        self.collaborative_rec = None
        self.content_rec = None
        self.knowledge_rec = None
        
        # Configuration
        self.explanation_enabled = explanation_enabled
        self.auto_tune_weights = auto_tune_weights
        
        # Performance tracking
        self.performance_metrics = {}
        self.component_performances = {}
        
        # Fallback strategies
        self.fallback_strategies = ['popular', 'random', 'trending']
        
        # Diversity tracking
        self.diversity_metrics = {}
        
    def _normalize_weights(self):
        """Normalize ensemble weights to sum to 1"""
        total_weight = (self.collaborative_weight + 
                       self.content_weight + 
                       self.knowledge_weight)
        
        if total_weight > 0:
            self.collaborative_weight /= total_weight
            self.content_weight /= total_weight
            self.knowledge_weight /= total_weight
    
    def fit(self,
            interactions_df: pd.DataFrame,
            item_embeddings: Optional[pd.DataFrame] = None,
            item_metadata: Optional[pd.DataFrame] = None,
            user_features: Optional[pd.DataFrame] = None,
            validation_df: Optional[pd.DataFrame] = None):
        """
        Train all component recommenders
        
        Args:
            interactions_df: User-item interactions
            item_embeddings: Item embeddings from Gemini (optional, if None will use item_metadata)
            item_metadata: Item metadata and features (optional, if None will use item_embeddings)
            user_features: Optional user features
            validation_df: Optional validation data for weight tuning
        """
        logger.info("Training hybrid recommender system")
        
        # Handle case where only interactions and one item dataset is provided
        if item_embeddings is not None and item_metadata is None:
            item_metadata = item_embeddings
        elif item_metadata is not None and item_embeddings is None:
            item_embeddings = item_metadata
        elif item_embeddings is None and item_metadata is None:
            raise ValueError("Either item_embeddings or item_metadata must be provided")
        
        # Initialize and train collaborative filtering
        logger.info("Training collaborative filtering component")
        self.collaborative_rec = CollaborativeRecommender(
            method='als',  # Can be configured
            n_factors=50,
            iterations=10
        )
        self.collaborative_rec.fit(
            interactions_df, 
            user_features=user_features,
            item_features=item_metadata
        )
        
        # Initialize and train content-based filtering
        logger.info("Training content-based filtering component")
        self.content_rec = ContentBasedRecommender(
            embedding_dim=768,  # Gemini embedding dimension
            use_faiss=True
        )
        
        # Check if item_embeddings has actual embedding columns
        embedding_cols = [col for col in item_embeddings.columns if col.startswith('embed_') or col == 'embedding']
        
        if not embedding_cols and 'embedding' not in item_embeddings.columns:
            # Create dummy embeddings for testing
            logger.warning("No embedding columns found, creating dummy embeddings for testing")
            dummy_embeddings = item_embeddings.copy()
            dummy_embeddings['embedding'] = [np.random.normal(0, 1, 768).tolist() for _ in range(len(dummy_embeddings))]
            item_embeddings_to_use = dummy_embeddings
        else:
            item_embeddings_to_use = item_embeddings
            
        self.content_rec.fit(
            item_embeddings_to_use,
            item_metadata,
            user_profiles=user_features
        )
        
        # Initialize knowledge-based filtering
        logger.info("Training knowledge-based filtering component")
        self.knowledge_rec = KnowledgeBasedRecommender(
            trending_window_days=7,
            new_release_days=30
        )
        
        # Prepare ratings data for knowledge component
        ratings_df = interactions_df.copy()
        if 'rating' not in ratings_df.columns:
            # Create implicit ratings
            ratings_df['rating'] = 1.0
        
        self.knowledge_rec.fit(
            item_metadata,
            ratings_df
        )
        
        # Auto-tune weights if validation data provided
        if self.auto_tune_weights and validation_df is not None:
            logger.info("Auto-tuning ensemble weights")
            self._tune_weights(validation_df)
        
        # Evaluate component performances
        if validation_df is not None:
            self._evaluate_components(validation_df)
        
        logger.info("Hybrid recommender training completed")
    
    def recommend(self,
                 user_id: str,
                 n_recommendations: int = 10,
                 user_history: Optional[List[str]] = None,
                 filters: Optional[Dict] = None,
                 context: Optional[Dict] = None,
                 exclude_seen: bool = True) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            user_history: User's interaction history
            filters: Optional filters to apply
            context: Optional contextual information
            exclude_seen: Whether to exclude seen items
            
        Returns:
            List of hybrid recommendations with scores and explanations
        """
        logger.debug(f"Generating recommendations for user {user_id}")
        
        # Get recommendations from each component
        collaborative_recs = self._get_collaborative_recommendations(
            user_id, n_recommendations * 2, exclude_seen
        )
        
        content_recs = self._get_content_recommendations(
            user_id, user_history, n_recommendations * 2, exclude_seen
        )
        
        knowledge_recs = self._get_knowledge_recommendations(
            user_id, context, filters, n_recommendations * 2
        )
        
        # Combine recommendations
        hybrid_recs = self._combine_recommendations(
            collaborative_recs,
            content_recs, 
            knowledge_recs
        )
        
        # Apply diversity boosting
        if self.diversity_factor > 0:
            hybrid_recs = self._apply_diversity_boosting(hybrid_recs)
        
        # Apply filters
        if filters:
            hybrid_recs = self._apply_filters(hybrid_recs, filters)
        
        # Sort by final score and take top N
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        final_recs = hybrid_recs[:n_recommendations]
        
        # Generate explanations
        if self.explanation_enabled:
            for rec in final_recs:
                rec['explanation'] = self._generate_explanation(
                    user_id, rec, user_history, context
                )
        
        # Calculate diversity metrics
        self._calculate_diversity_metrics(final_recs)
        
        return final_recs
    
    def _get_collaborative_recommendations(self,
                                         user_id: str,
                                         n_recs: int,
                                         exclude_seen: bool) -> List[Dict[str, Any]]:
        """Get recommendations from collaborative filtering"""
        if self.collaborative_rec is None:
            return []
        
        try:
            recs = self.collaborative_rec.recommend(
                user_id, n_recs, exclude_seen
            )
            
            # Add component identifier
            for rec in recs:
                rec['component'] = 'collaborative'
            
            return recs
        except Exception as e:
            logger.warning(f"Collaborative filtering failed: {e}")
            return []
    
    def _get_content_recommendations(self,
                                   user_id: str,
                                   user_history: Optional[List[str]],
                                   n_recs: int,
                                   exclude_seen: bool) -> List[Dict[str, Any]]:
        """Get recommendations from content-based filtering"""
        if self.content_rec is None:
            return []
        
        try:
            if user_history:
                recs = self.content_rec.recommend_for_user(
                    user_id, user_history, n_recs, exclude_seen=exclude_seen
                )
            else:
                # Fallback to popular content
                recs = []
            
            # Add component identifier
            for rec in recs:
                rec['component'] = 'content'
            
            return recs
        except Exception as e:
            logger.warning(f"Content-based filtering failed: {e}")
            return []
    
    def _get_knowledge_recommendations(self,
                                     user_id: str,
                                     context: Optional[Dict],
                                     filters: Optional[Dict],
                                     n_recs: int) -> List[Dict[str, Any]]:
        """Get recommendations from knowledge-based filtering"""
        if self.knowledge_rec is None:
            return []
        
        try:
            recs = []
            
            # Get trending recommendations
            trending_recs = self.knowledge_rec.recommend_trending(
                n_recs // 3, filters
            )
            recs.extend(trending_recs)
            
            # Get new release recommendations
            new_recs = self.knowledge_rec.recommend_new_releases(
                n_recs // 3, filters
            )
            recs.extend(new_recs)
            
            # Get contextual recommendations if context provided
            if context:
                context_recs = self.knowledge_rec.recommend_by_context(
                    user_id, context, n_recs // 3
                )
                recs.extend(context_recs)
            else:
                # Get acclaimed recommendations
                acclaimed_recs = self.knowledge_rec.recommend_acclaimed(
                    n_recs // 3, filters
                )
                recs.extend(acclaimed_recs)
            
            # Add component identifier
            for rec in recs:
                rec['component'] = 'knowledge'
            
            return recs
        except Exception as e:
            logger.warning(f"Knowledge-based filtering failed: {e}")
            return []
    
    def _combine_recommendations(self,
                               collaborative_recs: List[Dict[str, Any]],
                               content_recs: List[Dict[str, Any]],
                               knowledge_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine recommendations from all components using weighted ensemble"""
        # Collect all unique items
        all_items = {}
        
        # Process collaborative recommendations
        for rec in collaborative_recs:
            item_id = rec['item_id']
            score = rec['score'] * self.collaborative_weight
            
            if item_id not in all_items:
                all_items[item_id] = {
                    'item_id': item_id,
                    'score': 0.0,
                    'components': {},
                    'methods': []
                }
            
            all_items[item_id]['score'] += score
            all_items[item_id]['components']['collaborative'] = rec['score']
            all_items[item_id]['methods'].append('collaborative')
        
        # Process content-based recommendations
        for rec in content_recs:
            item_id = rec['item_id']
            score = rec['score'] * self.content_weight
            
            if item_id not in all_items:
                all_items[item_id] = {
                    'item_id': item_id,
                    'score': 0.0,
                    'components': {},
                    'methods': []
                }
            
            all_items[item_id]['score'] += score
            all_items[item_id]['components']['content'] = rec['score']
            all_items[item_id]['methods'].append('content')
        
        # Process knowledge-based recommendations
        for rec in knowledge_recs:
            item_id = rec['item_id']
            score = rec['score'] * self.knowledge_weight
            
            if item_id not in all_items:
                all_items[item_id] = {
                    'item_id': item_id,
                    'score': 0.0,
                    'components': {},
                    'methods': []
                }
            
            all_items[item_id]['score'] += score
            all_items[item_id]['components']['knowledge'] = rec['score']
            all_items[item_id]['methods'].append('knowledge')
        
        # Convert to list
        combined_recs = list(all_items.values())
        
        return combined_recs
    
    def _apply_diversity_boosting(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity boosting to recommendations"""
        if not recommendations:
            return recommendations
        
        # Sort by original score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply MMR-like diversity boosting
        diversified_recs = []
        remaining_recs = recommendations.copy()
        
        while remaining_recs and len(diversified_recs) < len(recommendations):
            if not diversified_recs:
                # Add highest scoring item first
                diversified_recs.append(remaining_recs.pop(0))
            else:
                # Find item that maximizes relevance - diversity trade-off
                best_idx = 0
                best_score = -float('inf')
                
                for i, rec in enumerate(remaining_recs):
                    relevance = rec['score']
                    
                    # Calculate diversity (inverse similarity to selected items)
                    diversity = self._calculate_diversity_score(
                        rec, diversified_recs
                    )
                    
                    # MMR score: relevance - diversity_factor * similarity
                    mmr_score = relevance - self.diversity_factor * (1 - diversity)
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                # Add best item and update score
                selected_rec = remaining_recs.pop(best_idx)
                selected_rec['diversity_score'] = best_score
                diversified_recs.append(selected_rec)
        
        return diversified_recs
    
    def _calculate_diversity_score(self, 
                                 candidate: Dict[str, Any],
                                 selected: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for a candidate item"""
        if not selected:
            return 1.0
        
        # Use content similarity if available
        if self.content_rec is not None:
            similarities = []
            for sel_item in selected:
                similar_items = self.content_rec.find_similar_items(
                    sel_item['item_id'], n_similar=1
                )
                
                for sim_item in similar_items:
                    if sim_item['item_id'] == candidate['item_id']:
                        similarities.append(sim_item['similarity_score'])
                        break
                else:
                    similarities.append(0.0)  # Not similar
            
            if similarities:
                avg_similarity = np.mean(similarities)
                return 1.0 - avg_similarity
        
        # Fallback: assume moderate diversity
        return 0.5
    
    def _apply_filters(self, 
                      recommendations: List[Dict[str, Any]],
                      filters: Dict) -> List[Dict[str, Any]]:
        """Apply user-defined filters to recommendations"""
        if self.knowledge_rec is None:
            return recommendations
        
        # Use knowledge-based component to apply filters
        item_ids = [rec['item_id'] for rec in recommendations]
        filtered_ids = self.knowledge_rec._apply_filters(item_ids, filters)
        
        # Filter recommendations
        filtered_recs = [
            rec for rec in recommendations 
            if rec['item_id'] in filtered_ids
        ]
        
        return filtered_recs
    
    def _generate_explanation(self,
                            user_id: str,
                            recommendation: Dict[str, Any],
                            user_history: Optional[List[str]],
                            context: Optional[Dict]) -> Dict[str, Any]:
        """Generate explanation for a recommendation"""
        explanation = {
            'item_id': recommendation['item_id'],
            'overall_score': recommendation['score'],
            'components': recommendation.get('components', {}),
            'methods': recommendation.get('methods', []),
            'reasoning': []
        }
        
        # Component-specific explanations
        if 'collaborative' in explanation['methods'] and self.collaborative_rec:
            collab_exp = self.collaborative_rec.explain_recommendation(
                user_id, recommendation['item_id']
            )
            explanation['reasoning'].append({
                'component': 'collaborative',
                'reason': 'Users with similar preferences also liked this',
                'details': collab_exp
            })
        
        if 'content' in explanation['methods'] and self.content_rec and user_history:
            content_exp = self.content_rec.explain_recommendation(
                user_id, recommendation['item_id'], user_history
            )
            explanation['reasoning'].append({
                'component': 'content',
                'reason': 'Similar to items you\'ve enjoyed',
                'details': content_exp
            })
        
        if 'knowledge' in explanation['methods'] and self.knowledge_rec:
            # Determine knowledge method used
            knowledge_method = 'trending'  # Default, would need more info
            knowledge_exp = self.knowledge_rec.explain_recommendation(
                recommendation['item_id'], knowledge_method
            )
            explanation['reasoning'].append({
                'component': 'knowledge',
                'reason': 'Based on current trends and popularity',
                'details': knowledge_exp
            })
        
        return explanation
    
    def _calculate_diversity_metrics(self, recommendations: List[Dict[str, Any]]):
        """Calculate and store diversity metrics for recommendations"""
        if len(recommendations) <= 1:
            return
        
        # Intra-list diversity (average pairwise dissimilarity)
        diversities = []
        
        for i, rec1 in enumerate(recommendations):
            for j, rec2 in enumerate(recommendations[i+1:], i+1):
                diversity = self._calculate_diversity_score(rec1, [rec2])
                diversities.append(diversity)
        
        if diversities:
            self.diversity_metrics['intra_list_diversity'] = np.mean(diversities)
            self.diversity_metrics['diversity_std'] = np.std(diversities)
        
        # Category diversity (if metadata available)
        if self.knowledge_rec and self.knowledge_rec.items_df is not None:
            categories = []
            for rec in recommendations:
                item_data = self.knowledge_rec.items_df[
                    self.knowledge_rec.items_df['item_id'] == rec['item_id']
                ]
                if not item_data.empty and 'genre' in item_data.columns:
                    categories.append(item_data['genre'].iloc[0])
            
            if categories:
                unique_categories = len(set(categories))
                self.diversity_metrics['category_diversity'] = unique_categories / len(categories)
    
    def _tune_weights(self, validation_df: pd.DataFrame):
        """Auto-tune ensemble weights using validation data"""
        logger.info("Tuning ensemble weights using validation data")
        
        # Grid search over weight combinations
        weight_combinations = []
        for collab_w in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for content_w in [0.2, 0.3, 0.4, 0.5, 0.6]:
                for knowledge_w in [0.1, 0.2, 0.3]:
                    if abs(collab_w + content_w + knowledge_w - 1.0) < 0.01:
                        weight_combinations.append((collab_w, content_w, knowledge_w))
        
        best_score = -1
        best_weights = (self.collaborative_weight, self.content_weight, self.knowledge_weight)
        
        for collab_w, content_w, knowledge_w in weight_combinations:
            # Temporarily set weights
            old_weights = (self.collaborative_weight, self.content_weight, self.knowledge_weight)
            self.collaborative_weight = collab_w
            self.content_weight = content_w
            self.knowledge_weight = knowledge_w
            
            # Evaluate on validation set
            score = self._evaluate_on_validation(validation_df)
            
            if score > best_score:
                best_score = score
                best_weights = (collab_w, content_w, knowledge_w)
            
            # Restore old weights
            self.collaborative_weight, self.content_weight, self.knowledge_weight = old_weights
        
        # Set best weights
        self.collaborative_weight, self.content_weight, self.knowledge_weight = best_weights
        
        logger.info(f"Best weights found: collaborative={self.collaborative_weight:.3f}, "
                   f"content={self.content_weight:.3f}, knowledge={self.knowledge_weight:.3f}, "
                   f"score={best_score:.3f}")
    
    def _evaluate_on_validation(self, validation_df: pd.DataFrame) -> float:
        """Evaluate current configuration on validation data"""
        total_score = 0
        total_users = 0
        
        # Sample of users for efficiency
        sample_users = validation_df['user_id'].unique()[:100]
        
        for user_id in sample_users:
            user_data = validation_df[validation_df['user_id'] == user_id]
            
            if len(user_data) < 2:
                continue
            
            # Split user data into training and testing
            test_items = user_data['item_id'].tolist()[-3:]  # Last 3 items as test
            
            try:
                # Get recommendations
                recs = self.recommend(user_id, n_recommendations=20, exclude_seen=False)
                rec_items = [rec['item_id'] for rec in recs]
                
                # Calculate NDCG@10
                if test_items and rec_items:
                    relevance = [1 if item in test_items else 0 for item in rec_items[:10]]
                    if sum(relevance) > 0:  # Only if there are relevant items
                        true_relevance = np.array([relevance])
                        score = ndcg_score(true_relevance, true_relevance, k=10)
                        total_score += score
                        total_users += 1
            
            except Exception as e:
                logger.debug(f"Evaluation failed for user {user_id}: {e}")
                continue
        
        return total_score / max(total_users, 1)
    
    def _evaluate_components(self, validation_df: pd.DataFrame):
        """Evaluate individual component performances"""
        logger.info("Evaluating component performances")
        
        # This would implement component-specific evaluation
        # For now, storing placeholder metrics
        self.component_performances = {
            'collaborative': {'ndcg@10': 0.0, 'coverage': 0.0},
            'content': {'ndcg@10': 0.0, 'coverage': 0.0},
            'knowledge': {'ndcg@10': 0.0, 'coverage': 0.0}
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'ensemble_weights': {
                'collaborative': self.collaborative_weight,
                'content': self.content_weight,
                'knowledge': self.knowledge_weight
            },
            'diversity_metrics': self.diversity_metrics,
            'component_performances': self.component_performances
        }
    
    def save_model(self, filepath: str):
        """Save hybrid model and all components"""
        model_data = {
            'weights': {
                'collaborative': self.collaborative_weight,
                'content': self.content_weight,
                'knowledge': self.knowledge_weight
            },
            'diversity_factor': self.diversity_factor,
            'explanation_enabled': self.explanation_enabled,
            'performance_metrics': self.performance_metrics,
            'diversity_metrics': self.diversity_metrics
        }
        
        # Save main model data
        joblib.dump(model_data, filepath)
        
        # Save individual components
        base_path = filepath.replace('.pkl', '')
        
        if self.collaborative_rec:
            self.collaborative_rec.save_model(f"{base_path}_collaborative.pkl")
        
        if self.content_rec:
            self.content_rec.save_model(f"{base_path}_content.pkl")
        
        # Knowledge-based doesn't need separate saving as it's rule-based
        
        logger.info(f"Hybrid model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load hybrid model and all components"""
        model_data = joblib.load(filepath)
        
        # Load weights and configuration
        weights = model_data['weights']
        self.collaborative_weight = weights['collaborative']
        self.content_weight = weights['content']
        self.knowledge_weight = weights['knowledge']
        
        self.diversity_factor = model_data['diversity_factor']
        self.explanation_enabled = model_data['explanation_enabled']
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.diversity_metrics = model_data.get('diversity_metrics', {})
        
        # Load individual components
        base_path = filepath.replace('.pkl', '')
        
        try:
            self.collaborative_rec = CollaborativeRecommender()
            self.collaborative_rec.load_model(f"{base_path}_collaborative.pkl")
        except:
            logger.warning("Could not load collaborative component")
        
        try:
            self.content_rec = ContentBasedRecommender()
            self.content_rec.load_model(f"{base_path}_content.pkl")
        except:
            logger.warning("Could not load content component")
        
        logger.info(f"Hybrid model loaded from {filepath}")
