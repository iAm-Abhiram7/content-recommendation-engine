"""
Collaborative Filtering Recommender

Implements both user-based and item-based collaborative filtering using:
- Matrix factorization (SVD, ALS)
- User-user and item-item similarity
- Cold-start handling with content-based fallback
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import logging
from cachetools import TTLCache
import joblib
from pathlib import Path

# Optional imports with fallbacks
try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    IMPLICIT_AVAILABLE = True
except ImportError:
    AlternatingLeastSquares = None
    BayesianPersonalizedRanking = None
    IMPLICIT_AVAILABLE = False
    logging.warning("implicit library not available - matrix factorization features will be limited")

try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SVD = None
    Dataset = None
    Reader = None
    train_test_split = None
    SURPRISE_AVAILABLE = False
    logging.warning("surprise library not available - SVD features will be limited")

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    """
    Collaborative filtering recommender with multiple algorithms
    """
    
    def __init__(self, 
                 method: str = 'als',
                 n_factors: int = 50,
                 regularization: float = 0.1,
                 iterations: int = 10,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600):
        """
        Initialize collaborative recommender
        
        Args:
            method: Algorithm to use ('als', 'bpr', 'svd', 'user_based', 'item_based')
            n_factors: Number of latent factors for matrix factorization
            regularization: Regularization parameter
            iterations: Number of training iterations
            cache_size: Size of prediction cache
            cache_ttl: Cache TTL in seconds
        """
        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        
        # Initialize model based on method
        self.model = self._initialize_model()
        
        # Caching for performance
        self.prediction_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        
        # Data matrices
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        self.user_similarity = None
        self.item_similarity = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        # Performance metrics
        self.training_metrics = {}
        
    def _initialize_model(self):
        """Initialize the specific collaborative filtering model"""
        if self.method == 'als':
            if not IMPLICIT_AVAILABLE or AlternatingLeastSquares is None:
                logger.warning("ALS not available, falling back to user-based collaborative filtering")
                self.method = 'user_based'
                return None
            return AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=self.regularization,
                iterations=self.iterations,
                use_gpu=False
            )
        elif self.method == 'bpr':
            if not IMPLICIT_AVAILABLE or BayesianPersonalizedRanking is None:
                logger.warning("BPR not available, falling back to user-based collaborative filtering")
                self.method = 'user_based'
                return None
            return BayesianPersonalizedRanking(
                factors=self.n_factors,
                regularization=self.regularization,
                iterations=self.iterations
            )
        elif self.method == 'svd':
            if not SURPRISE_AVAILABLE or SVD is None:
                logger.warning("SVD not available, falling back to user-based collaborative filtering")
                self.method = 'user_based'
                return None
            return SVD(
                n_factors=self.n_factors,
                reg_all=self.regularization,
                n_epochs=self.iterations
            )
        else:
            return None
    
    def fit(self, interactions_df: pd.DataFrame, 
            user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None):
        """
        Train the collaborative filtering model
        
        Args:
            interactions_df: DataFrame with columns [user_id, item_id, rating, timestamp]
            user_features: Optional user feature matrix
            item_features: Optional item feature matrix
        """
        logger.info(f"Training collaborative filter with method: {self.method}")
        
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix(interactions_df)
        
        # Store features
        self.user_features = user_features
        self.item_features = item_features
        
        # Train based on method
        if self.method in ['als', 'bpr']:
            self._train_implicit_model(interactions_df)
        elif self.method == 'svd':
            self._train_surprise_model(interactions_df)
        elif self.method in ['user_based', 'item_based']:
            self._compute_similarity_matrices()
        
        # Clear cache after training
        self.prediction_cache.clear()
        
        logger.info("Collaborative filtering training completed")
    
    def _create_user_item_matrix(self, interactions_df: pd.DataFrame) -> csr_matrix:
        """Create sparse user-item interaction matrix"""
        # Map user and item IDs to indices
        user_indices = interactions_df['user_id'].map(self.user_to_idx)
        item_indices = interactions_df['item_id'].map(self.item_to_idx)
        
        # For implicit feedback, use confidence scores or binary interactions
        if 'rating' in interactions_df.columns:
            ratings = interactions_df['rating'].values
        else:
            ratings = np.ones(len(interactions_df))
        
        # Create sparse matrix
        matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )
        
        return matrix
    
    def _train_implicit_model(self, interactions_df: pd.DataFrame):
        """Train implicit feedback models (ALS, BPR)"""
        # Convert to implicit format (item x user)
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        # Train model
        self.model.fit(item_user_matrix)
        
        # Calculate training metrics
        self.training_metrics['sparsity'] = 1 - (self.user_item_matrix.nnz / 
                                                (self.user_item_matrix.shape[0] * 
                                                 self.user_item_matrix.shape[1]))
    
    def _train_surprise_model(self, interactions_df: pd.DataFrame):
        """Train Surprise library models (SVD)"""
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            interactions_df[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        # Train test split
        trainset, testset = train_test_split(data, test_size=0.2)
        
        # Train model
        self.model.fit(trainset)
        
        # Calculate metrics on test set
        predictions = self.model.test(testset)
        rmse = np.sqrt(np.mean([(pred.est - pred.r_ui) ** 2 for pred in predictions]))
        self.training_metrics['rmse'] = rmse
    
    def _compute_similarity_matrices(self):
        """Compute user-user and item-item similarity matrices"""
        # User-user similarity
        user_matrix = self.user_item_matrix.toarray()
        self.user_similarity = cosine_similarity(user_matrix)
        
        # Item-item similarity
        item_matrix = self.user_item_matrix.T.toarray()
        self.item_similarity = cosine_similarity(item_matrix)
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Predicted rating/score
        """
        cache_key = f"{user_id}_{item_id}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Handle cold start
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            score = self._handle_cold_start(user_id, item_id)
        else:
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            if self.method in ['als', 'bpr']:
                # For implicit models, get score from user factors
                score = self.model.user_factors[user_idx].dot(
                    self.model.item_factors[item_idx]
                )
            elif self.method == 'svd':
                score = self.model.predict(user_id, item_id).est
            elif self.method == 'user_based':
                score = self._user_based_predict(user_idx, item_idx)
            elif self.method == 'item_based':
                score = self._item_based_predict(user_idx, item_idx)
            else:
                score = 0.0
        
        self.prediction_cache[cache_key] = score
        return score
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 exclude_seen: bool = True, 
                 filter_items: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude already interacted items
            filter_items: Optional list of items to consider
            
        Returns:
            List of recommendation dictionaries with scores
        """
        if user_id not in self.user_to_idx:
            return self._cold_start_recommendations(user_id, n_recommendations)
        
        user_idx = self.user_to_idx[user_id]
        
        # Get scores for all items
        if self.method in ['als', 'bpr']:
            # Use model's recommend method for efficiency
            item_scores = self.model.recommend(
                user_idx, 
                self.user_item_matrix[user_idx],
                N=n_recommendations * 2,  # Get extra to account for filtering
                filter_already_liked_items=exclude_seen
            )
            
            recommendations = []
            for item_idx, score in item_scores:
                item_id = self.idx_to_item[item_idx]
                if filter_items is None or item_id in filter_items:
                    recommendations.append({
                        'item_id': item_id,
                        'score': float(score),
                        'method': self.method
                    })
                    if len(recommendations) >= n_recommendations:
                        break
        
        else:
            # For other methods, compute scores manually
            item_scores = []
            for item_id, item_idx in self.item_to_idx.items():
                if filter_items is not None and item_id not in filter_items:
                    continue
                    
                if exclude_seen and self.user_item_matrix[user_idx, item_idx] > 0:
                    continue
                
                score = self.predict(user_id, item_id)
                item_scores.append((item_id, score))
            
            # Sort by score and take top N
            item_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [
                {
                    'item_id': item_id,
                    'score': score,
                    'method': self.method
                }
                for item_id, score in item_scores[:n_recommendations]
            ]
        
        return recommendations
    
    def _user_based_predict(self, user_idx: int, item_idx: int) -> float:
        """User-based collaborative filtering prediction"""
        user_similarities = self.user_similarity[user_idx]
        item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
        
        # Find users who rated this item
        rated_users = np.where(item_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return 0.0
        
        # Calculate weighted average
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        if np.sum(np.abs(similarities)) == 0:
            return np.mean(ratings)
        
        return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
    
    def _item_based_predict(self, user_idx: int, item_idx: int) -> float:
        """Item-based collaborative filtering prediction"""
        item_similarities = self.item_similarity[item_idx]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Find items rated by this user
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Calculate weighted average
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        if np.sum(np.abs(similarities)) == 0:
            return np.mean(ratings)
        
        return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
    
    def _handle_cold_start(self, user_id: str, item_id: str) -> float:
        """Handle cold start users/items"""
        # Return global average or item/user average
        if hasattr(self, 'global_mean'):
            return self.global_mean
        return 3.0  # Default middle rating
    
    def _cold_start_recommendations(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """Generate recommendations for cold start users"""
        # Return popular items or random sample
        if hasattr(self, 'popular_items'):
            return [
                {
                    'item_id': item_id,
                    'score': 1.0,
                    'method': 'popularity_fallback'
                }
                for item_id in self.popular_items[:n_recommendations]
            ]
        
        # Return random sample
        random_items = np.random.choice(
            list(self.item_to_idx.keys()), 
            size=min(n_recommendations, len(self.item_to_idx)),
            replace=False
        )
        
        return [
            {
                'item_id': item_id,
                'score': 0.5,
                'method': 'random_fallback'
            }
            for item_id in random_items
        ]
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'method': self.method,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Collaborative filtering model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = model_data['idx_to_item']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity = model_data.get('user_similarity')
        self.item_similarity = model_data.get('item_similarity')
        self.method = model_data['method']
        self.training_metrics = model_data.get('training_metrics', {})
        
        logger.info(f"Collaborative filtering model loaded from {filepath}")
    
    def get_user_embeddings(self, user_ids: Optional[List[str]] = None) -> np.ndarray:
        """Get user embedding vectors"""
        if self.method in ['als', 'bpr'] and hasattr(self.model, 'user_factors'):
            if user_ids is None:
                return self.model.user_factors
            else:
                indices = [self.user_to_idx.get(uid, 0) for uid in user_ids]
                return self.model.user_factors[indices]
        return None
    
    def get_item_embeddings(self, item_ids: Optional[List[str]] = None) -> np.ndarray:
        """Get item embedding vectors"""
        if self.method in ['als', 'bpr'] and hasattr(self.model, 'item_factors'):
            if item_ids is None:
                return self.model.item_factors
            else:
                indices = [self.item_to_idx.get(iid, 0) for iid in item_ids]
                return self.model.item_factors[indices]
        return None
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict[str, Any]:
        """Generate explanation for a recommendation"""
        explanation = {
            'method': self.method,
            'user_id': user_id,
            'item_id': item_id,
            'score': self.predict(user_id, item_id)
        }
        
        if self.method == 'user_based' and user_id in self.user_to_idx:
            # Find similar users who liked this item
            user_idx = self.user_to_idx[user_id]
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
                rated_users = np.where(item_ratings > 0)[0]
                
                if len(rated_users) > 0:
                    similarities = self.user_similarity[user_idx][rated_users]
                    top_similar = np.argsort(similarities)[-5:]  # Top 5 similar users
                    
                    explanation['similar_users'] = [
                        {
                            'user_id': self.idx_to_user[rated_users[idx]],
                            'similarity': float(similarities[idx]),
                            'rating': float(item_ratings[rated_users[idx]])
                        }
                        for idx in top_similar if similarities[idx] > 0
                    ]
        
        return explanation
