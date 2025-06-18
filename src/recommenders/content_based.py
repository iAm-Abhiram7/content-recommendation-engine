"""
Content-Based Filtering Recommender

Uses Gemini-generated embeddings and content features to recommend similar items:
- Embedding-based similarity using cosine similarity and nearest neighbors
- Content feature matching (genre, tags, metadata)
- Cross-domain recommendations
- Personalized content matching based on user profiles
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
import logging
from cachetools import TTLCache
import joblib
from pathlib import Path
import json

# Optional import for FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available - will use sklearn for similarity search")

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based filtering using embeddings and metadata
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 similarity_metric: str = 'cosine',
                 n_neighbors: int = 50,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 use_faiss: bool = True):
        """
        Initialize content-based recommender
        
        Args:
            embedding_dim: Dimension of content embeddings
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'dot')
            n_neighbors: Number of neighbors for KNN
            cache_size: Size of similarity cache
            cache_ttl: Cache TTL in seconds
            use_faiss: Whether to use FAISS for efficient similarity search
        """
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.use_faiss = use_faiss
        
        # Caching for performance
        self.similarity_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        
        # Data storage
        self.item_embeddings = None
        self.item_metadata = None
        self.user_profiles = None
        self.content_features = None
        
        # Models and indices
        self.nn_model = None
        self.faiss_index = None
        self.scaler = StandardScaler()
        
        # Mappings
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.feature_columns = []
        
        # Cross-domain mappings
        self.domain_mappings = {}
        
    def fit(self, 
            item_embeddings: pd.DataFrame,
            item_metadata: pd.DataFrame,
            user_profiles: Optional[pd.DataFrame] = None,
            content_features: Optional[pd.DataFrame] = None):
        """
        Train the content-based model
        
        Args:
            item_embeddings: DataFrame with item_id and embedding columns
            item_metadata: DataFrame with item metadata (genre, tags, etc.)
            user_profiles: Optional user profile features
            content_features: Optional additional content features
        """
        logger.info("Training content-based recommender")
        
        # Store data
        self.item_embeddings = item_embeddings.copy()
        self.item_metadata = item_metadata.copy()
        self.user_profiles = user_profiles.copy() if user_profiles is not None else None
        self.content_features = content_features.copy() if content_features is not None else None
        
        # Create item mappings
        unique_items = item_embeddings['item_id'].unique()
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Prepare embedding matrix
        self._prepare_embeddings()
        
        # Build similarity index
        self._build_similarity_index()
        
        # Process metadata features
        self._process_metadata_features()
        
        # Clear cache
        self.similarity_cache.clear()
        
        logger.info("Content-based recommender training completed")
    
    def _prepare_embeddings(self):
        """Prepare and normalize embedding matrix"""
        # Extract embedding columns (assuming they're named 'emb_0', 'emb_1', etc.)
        embedding_cols = [col for col in self.item_embeddings.columns if col.startswith('emb_')]
        
        if not embedding_cols:
            # If embeddings are stored as arrays in a single column
            if 'embedding' in self.item_embeddings.columns:
                embeddings = np.vstack(self.item_embeddings['embedding'].values)
            else:
                raise ValueError("No embedding columns found in item_embeddings")
        else:
            embeddings = self.item_embeddings[embedding_cols].values
        
        # Normalize embeddings for cosine similarity
        if self.similarity_metric == 'cosine':
            embeddings = normalize(embeddings, norm='l2')
        
        self.embedding_matrix = embeddings.astype(np.float32)
        
    def _build_similarity_index(self):
        """Build similarity search index"""
        if self.use_faiss and FAISS_AVAILABLE and faiss and self.embedding_matrix.shape[0] > 100:
            # Use FAISS for large datasets
            if self.similarity_metric == 'cosine':
                # For cosine similarity with normalized vectors, use inner product
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            
            self.faiss_index.add(self.embedding_matrix)
            logger.info(f"Built FAISS index with {self.embedding_matrix.shape[0]} items")
        else:
            # Use sklearn for smaller datasets or when FAISS is not available
            if not FAISS_AVAILABLE and self.use_faiss:
                logger.warning("FAISS not available, falling back to sklearn")
            metric = 'cosine' if self.similarity_metric == 'cosine' else 'euclidean'
            self.nn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=metric,
                algorithm='auto'
            )
            self.nn_model.fit(self.embedding_matrix)
            logger.info(f"Built sklearn NearestNeighbors with {self.embedding_matrix.shape[0]} items")
    
    def _process_metadata_features(self):
        """Process and encode metadata features"""
        if self.item_metadata is None:
            return
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in self.item_metadata.columns:
            if col == 'item_id':
                continue
            
            if self.item_metadata[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.feature_columns = categorical_features + numerical_features
        
        # One-hot encode categorical features
        if categorical_features:
            categorical_encoded = pd.get_dummies(
                self.item_metadata[categorical_features], 
                prefix=categorical_features
            )
        else:
            categorical_encoded = pd.DataFrame()
        
        # Scale numerical features
        if numerical_features:
            numerical_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.item_metadata[numerical_features]),
                columns=numerical_features,
                index=self.item_metadata.index
            )
        else:
            numerical_scaled = pd.DataFrame()
        
        # Combine features
        if not categorical_encoded.empty and not numerical_scaled.empty:
            self.metadata_features = pd.concat([categorical_encoded, numerical_scaled], axis=1)
        elif not categorical_encoded.empty:
            self.metadata_features = categorical_encoded
        elif not numerical_scaled.empty:
            self.metadata_features = numerical_scaled
        else:
            self.metadata_features = pd.DataFrame()
        
        # Add item_id for alignment
        self.metadata_features['item_id'] = self.item_metadata['item_id'].values
    
    def find_similar_items(self, 
                          item_id: str, 
                          n_similar: int = 10,
                          use_metadata: bool = False,
                          metadata_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find similar items to a given item
        
        Args:
            item_id: Item to find similarities for
            n_similar: Number of similar items to return
            use_metadata: Whether to include metadata features
            metadata_weight: Weight for metadata features vs embeddings
            
        Returns:
            List of similar items with similarity scores
        """
        cache_key = f"similar_{item_id}_{n_similar}_{use_metadata}_{metadata_weight}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if item_id not in self.item_to_idx:
            logger.warning(f"Item {item_id} not found in content index")
            return []
        
        item_idx = self.item_to_idx[item_id]
        
        # Get embedding-based similarities
        if self.use_faiss and self.faiss_index is not None:
            similarities, indices = self.faiss_index.search(
                self.embedding_matrix[item_idx:item_idx+1], 
                n_similar + 1  # +1 to exclude self
            )
            similarities = similarities[0]
            indices = indices[0]
        else:
            similarities, indices = self.nn_model.kneighbors(
                self.embedding_matrix[item_idx:item_idx+1],
                n_neighbors=n_similar + 1
            )
            similarities = similarities[0]
            indices = indices[0]
        
        # Convert to similarity scores (distance -> similarity)
        if self.similarity_metric == 'cosine' and not self.use_faiss:
            similarities = 1 - similarities  # Convert distance to similarity
        elif self.similarity_metric == 'euclidean':
            similarities = 1 / (1 + similarities)  # Convert distance to similarity
        
        similar_items = []
        for i, (sim_score, sim_idx) in enumerate(zip(similarities, indices)):
            if sim_idx == item_idx:  # Skip self
                continue
            
            sim_item_id = self.idx_to_item[sim_idx]
            
            # Combine with metadata similarity if requested
            if use_metadata and hasattr(self, 'metadata_features'):
                metadata_sim = self._compute_metadata_similarity(item_id, sim_item_id)
                combined_score = ((1 - metadata_weight) * sim_score + 
                                metadata_weight * metadata_sim)
            else:
                combined_score = sim_score
            
            similar_items.append({
                'item_id': sim_item_id,
                'similarity_score': float(combined_score),
                'embedding_similarity': float(sim_score),
                'metadata_similarity': self._compute_metadata_similarity(item_id, sim_item_id) if use_metadata else None
            })
            
            if len(similar_items) >= n_similar:
                break
        
        # Sort by combined score
        similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.similarity_cache[cache_key] = similar_items
        return similar_items
    
    def recommend_for_user(self, 
                          user_id: str,
                          user_history: List[str],
                          n_recommendations: int = 10,
                          diversity_factor: float = 0.1,
                          exclude_seen: bool = True) -> List[Dict[str, Any]]:
        """
        Generate content-based recommendations for a user
        
        Args:
            user_id: User identifier
            user_history: List of items user has interacted with
            n_recommendations: Number of recommendations to return
            diversity_factor: Factor to promote diversity (0-1)
            exclude_seen: Whether to exclude items in user history
            
        Returns:
            List of recommendations with scores
        """
        if not user_history:
            return self._cold_start_recommendations(user_id, n_recommendations)
        
        # Get user profile embedding
        user_embedding = self._compute_user_profile_embedding(user_history)
        
        if user_embedding is None:
            return self._cold_start_recommendations(user_id, n_recommendations)
        
        # Find items similar to user profile
        if self.use_faiss and self.faiss_index is not None:
            similarities, indices = self.faiss_index.search(
                user_embedding.reshape(1, -1).astype(np.float32),
                n_recommendations * 3  # Get more for filtering
            )
            similarities = similarities[0]
            indices = indices[0]
        else:
            similarities, indices = self.nn_model.kneighbors(
                user_embedding.reshape(1, -1),
                n_neighbors=n_recommendations * 3
            )
            similarities = similarities[0]
            indices = indices[0]
        
        # Convert to similarity scores
        if self.similarity_metric == 'cosine' and not self.use_faiss:
            similarities = 1 - similarities
        elif self.similarity_metric == 'euclidean':
            similarities = 1 / (1 + similarities)
        
        # Filter and rank recommendations
        recommendations = []
        seen_items = set(user_history) if exclude_seen else set()
        
        for sim_score, item_idx in zip(similarities, indices):
            item_id = self.idx_to_item[item_idx]
            
            if item_id in seen_items:
                continue
            
            # Apply diversity penalty
            diversity_penalty = self._compute_diversity_penalty(
                item_id, [r['item_id'] for r in recommendations], diversity_factor
            )
            
            final_score = sim_score * (1 - diversity_penalty)
            
            recommendations.append({
                'item_id': item_id,
                'score': float(final_score),
                'similarity_score': float(sim_score),
                'diversity_penalty': float(diversity_penalty),
                'method': 'content_based'
            })
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def _compute_user_profile_embedding(self, user_history: List[str]) -> Optional[np.ndarray]:
        """Compute user profile embedding from interaction history"""
        item_embeddings = []
        
        for item_id in user_history:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                item_embeddings.append(self.embedding_matrix[item_idx])
        
        if not item_embeddings:
            return None
        
        # Simple average for now - could be weighted by rating/recency
        user_embedding = np.mean(item_embeddings, axis=0)
        
        # Normalize if using cosine similarity
        if self.similarity_metric == 'cosine':
            user_embedding = normalize(user_embedding.reshape(1, -1), norm='l2')[0]
        
        return user_embedding
    
    def _compute_metadata_similarity(self, item1_id: str, item2_id: str) -> float:
        """Compute similarity based on metadata features"""
        if not hasattr(self, 'metadata_features') or self.metadata_features.empty:
            return 0.0
        
        # Find items in metadata
        item1_meta = self.metadata_features[self.metadata_features['item_id'] == item1_id]
        item2_meta = self.metadata_features[self.metadata_features['item_id'] == item2_id]
        
        if item1_meta.empty or item2_meta.empty:
            return 0.0
        
        # Remove item_id column and compute cosine similarity
        feature_cols = [col for col in self.metadata_features.columns if col != 'item_id']
        
        if not feature_cols:
            return 0.0
        
        vec1 = item1_meta[feature_cols].values[0]
        vec2 = item2_meta[feature_cols].values[0]
        
        # Handle missing values
        vec1 = np.nan_to_num(vec1)
        vec2 = np.nan_to_num(vec2)
        
        # Compute cosine similarity
        similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]
        
        return float(similarity)
    
    def _compute_diversity_penalty(self, 
                                  candidate_item: str, 
                                  selected_items: List[str], 
                                  diversity_factor: float) -> float:
        """Compute diversity penalty for a candidate item"""
        if not selected_items or diversity_factor == 0:
            return 0.0
        
        similarities = []
        for selected_item in selected_items:
            similar_items = self.find_similar_items(selected_item, n_similar=1)
            for sim_item in similar_items:
                if sim_item['item_id'] == candidate_item:
                    similarities.append(sim_item['similarity_score'])
                    break
        
        if not similarities:
            return 0.0
        
        # Average similarity to already selected items
        avg_similarity = np.mean(similarities)
        
        # Convert to penalty (higher similarity = higher penalty)
        penalty = avg_similarity * diversity_factor
        
        return min(penalty, 1.0)  # Cap at 1.0
    
    def cross_domain_recommend(self,
                              source_items: List[str],
                              target_domain: str,
                              n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Generate cross-domain recommendations
        
        Args:
            source_items: Items from source domain
            target_domain: Target domain for recommendations
            n_recommendations: Number of recommendations
            
        Returns:
            Cross-domain recommendations
        """
        # Get average embedding of source items
        source_embeddings = []
        for item_id in source_items:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                source_embeddings.append(self.embedding_matrix[item_idx])
        
        if not source_embeddings:
            return []
        
        source_profile = np.mean(source_embeddings, axis=0)
        
        # Filter items by target domain
        target_items = self._get_items_by_domain(target_domain)
        
        if not target_items:
            return []
        
        # Find similar items in target domain
        similarities = []
        for item_id in target_items:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                item_embedding = self.embedding_matrix[item_idx]
                
                similarity = cosine_similarity(
                    source_profile.reshape(1, -1),
                    item_embedding.reshape(1, -1)
                )[0, 0]
                
                similarities.append((item_id, similarity))
        
        # Sort and return top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, similarity in similarities[:n_recommendations]:
            recommendations.append({
                'item_id': item_id,
                'score': float(similarity),
                'method': 'cross_domain',
                'source_domain': self._get_item_domain(source_items[0]),
                'target_domain': target_domain
            })
        
        return recommendations
    
    def _get_items_by_domain(self, domain: str) -> List[str]:
        """Get items belonging to a specific domain"""
        if self.item_metadata is None or 'domain' not in self.item_metadata.columns:
            # Fallback: infer domain from item_id pattern
            return [item_id for item_id in self.item_to_idx.keys() 
                   if domain.lower() in item_id.lower()]
        
        domain_items = self.item_metadata[
            self.item_metadata['domain'] == domain
        ]['item_id'].tolist()
        
        return domain_items
    
    def _get_item_domain(self, item_id: str) -> str:
        """Get domain of an item"""
        if self.item_metadata is None or 'domain' not in self.item_metadata.columns:
            # Infer from item_id pattern
            if 'movie' in item_id.lower():
                return 'movies'
            elif 'book' in item_id.lower():
                return 'books'
            elif 'music' in item_id.lower():
                return 'music'
            else:
                return 'unknown'
        
        item_data = self.item_metadata[self.item_metadata['item_id'] == item_id]
        if not item_data.empty:
            return item_data['domain'].iloc[0]
        
        return 'unknown'
    
    def _cold_start_recommendations(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """Generate recommendations for users with no history"""
        # Return popular or high-quality items
        if hasattr(self, 'popular_items'):
            recommendations = []
            for item_id in self.popular_items[:n_recommendations]:
                recommendations.append({
                    'item_id': item_id,
                    'score': 0.5,
                    'method': 'content_popular'
                })
            return recommendations
        
        # Random sample from available items
        random_items = np.random.choice(
            list(self.item_to_idx.keys()),
            size=min(n_recommendations, len(self.item_to_idx)),
            replace=False
        )
        
        return [
            {
                'item_id': item_id,
                'score': 0.3,
                'method': 'content_random'
            }
            for item_id in random_items
        ]
    
    def explain_recommendation(self, user_id: str, item_id: str, user_history: List[str]) -> Dict[str, Any]:
        """Generate explanation for content-based recommendation"""
        explanation = {
            'method': 'content_based',
            'user_id': user_id,
            'item_id': item_id,
            'reasoning': []
        }
        
        # Find most similar items in user history
        similar_to_history = []
        for hist_item in user_history[-5:]:  # Check last 5 items
            similar_items = self.find_similar_items(hist_item, n_similar=1)
            for sim_item in similar_items:
                if sim_item['item_id'] == item_id:
                    similar_to_history.append({
                        'history_item': hist_item,
                        'similarity': sim_item['similarity_score']
                    })
        
        if similar_to_history:
            best_match = max(similar_to_history, key=lambda x: x['similarity'])
            explanation['reasoning'].append(
                f"Similar to {best_match['history_item']} "
                f"(similarity: {best_match['similarity']:.3f})"
            )
        
        # Add metadata-based reasoning if available
        if hasattr(self, 'metadata_features') and not self.metadata_features.empty:
            item_meta = self.metadata_features[
                self.metadata_features['item_id'] == item_id
            ]
            if not item_meta.empty:
                # Extract top features (this would need domain-specific logic)
                explanation['metadata'] = 'Based on content features and metadata'
        
        return explanation
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'embedding_matrix': self.embedding_matrix,
            'similarity_metric': self.similarity_metric,
            'metadata_features': getattr(self, 'metadata_features', None),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        # Save FAISS index separately if it exists
        if self.faiss_index is not None and FAISS_AVAILABLE and faiss:
            faiss_path = filepath.replace('.pkl', '_faiss.index')
            faiss.write_index(self.faiss_index, faiss_path)
            model_data['faiss_index_path'] = faiss_path
        
        joblib.dump(model_data, filepath)
        logger.info(f"Content-based model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = model_data['idx_to_item']
        self.embedding_matrix = model_data['embedding_matrix']
        self.similarity_metric = model_data['similarity_metric']
        self.metadata_features = model_data.get('metadata_features')
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        # Load FAISS index if it exists
        if 'faiss_index_path' in model_data and FAISS_AVAILABLE and faiss:
            try:
                self.faiss_index = faiss.read_index(model_data['faiss_index_path'])
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self.faiss_index = None
        
        # Rebuild sklearn model if FAISS not available
        if self.faiss_index is None:
            self._build_similarity_index()
        
        logger.info(f"Content-based model loaded from {filepath}")
