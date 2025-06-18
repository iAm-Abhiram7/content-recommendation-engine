"""
Online Learning Algorithm Implementation

This module implements adaptive learning mechanisms for real-time model updates:
- Incremental Matrix Factorization with online SGD
- Online Content-Based Learning with streaming updates
- Ensemble Weight Adaptation with multi-armed bandit
- Dynamic model adaptation for new users and items
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from concurrent.futures import ThreadPoolExecutor
import joblib
import threading
from collections import defaultdict, deque
import random
import math
from enum import Enum

logger = logging.getLogger(__name__)


class LearningAlgorithm(Enum):
    """Types of online learning algorithms"""
    SGD = "sgd"
    INCREMENTAL = "incremental"
    ENSEMBLE = "ensemble"
    BANDIT = "bandit"
    MATRIX_FACTORIZATION = "matrix_factorization"
    CONTENT_BASED = "content_based"


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning algorithms"""
    learning_rate: float = 0.01
    regularization: float = 0.01
    forgetting_factor: float = 0.95
    n_factors: int = 50
    batch_size: int = 100
    update_threshold: int = 10
    cold_start_threshold: int = 5
    bandit_exploration_rate: float = 0.1
    weight_decay: float = 0.001
    max_iterations: int = 100


@dataclass
class UserProfile:
    """Online user profile for incremental learning"""
    user_id: str
    factors: np.ndarray
    bias: float
    interaction_count: int
    last_updated: datetime
    confidence: float = 1.0
    preferences: Dict[str, float] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass
class ItemProfile:
    """Online item profile for incremental learning"""
    item_id: str
    factors: np.ndarray
    bias: float
    features: Dict[str, Any]
    interaction_count: int
    last_updated: datetime
    popularity_score: float = 0.0


@dataclass
class EnsembleWeights:
    """Dynamic ensemble weights for hybrid recommendation"""
    collaborative_weight: float = 0.4
    content_weight: float = 0.4
    knowledge_weight: float = 0.2
    last_updated: datetime = None
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.performance_history is None:
            self.performance_history = []


class MultiArmedBandit:
    """Multi-armed bandit for ensemble weight optimization"""
    
    def __init__(self, n_arms: int = 3, exploration_rate: float = 0.1):
        self.n_arms = n_arms
        self.exploration_rate = exploration_rate
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_count = 0
    
    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy"""
        if random.random() < self.exploration_rate or self.total_count < self.n_arms:
            return random.randint(0, self.n_arms - 1)
        
        # Exploit: choose arm with highest average reward
        avg_rewards = np.divide(self.arm_rewards, self.arm_counts, 
                               out=np.zeros_like(self.arm_rewards), 
                               where=self.arm_counts!=0)
        return np.argmax(avg_rewards)
    
    def update_reward(self, arm: int, reward: float):
        """Update reward for selected arm"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_count += 1
    
    def get_arm_probabilities(self) -> np.ndarray:
        """Get probability distribution over arms"""
        if self.total_count == 0:
            return np.ones(self.n_arms) / self.n_arms
        
        avg_rewards = np.divide(self.arm_rewards, self.arm_counts,
                               out=np.zeros_like(self.arm_rewards),
                               where=self.arm_counts!=0)
        
        # Softmax transformation
        exp_rewards = np.exp(avg_rewards - np.max(avg_rewards))
        return exp_rewards / np.sum(exp_rewards)


class IncrementalMatrixFactorization:
    """Incremental matrix factorization for collaborative filtering"""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.user_profiles: Dict[str, UserProfile] = {}
        self.item_profiles: Dict[str, ItemProfile] = {}
        self.global_bias = 0.0
        self.user_index = {}
        self.item_index = {}
        self.next_user_idx = 0
        self.next_item_idx = 0
        self.scaler = StandardScaler()
        
    def add_user(self, user_id: str) -> UserProfile:
        """Add new user with cold-start initialization"""
        if user_id not in self.user_profiles:
            factors = np.random.normal(0, 0.1, self.config.n_factors)
            profile = UserProfile(
                user_id=user_id,
                factors=factors,
                bias=0.0,
                interaction_count=0,
                last_updated=datetime.now()
            )
            self.user_profiles[user_id] = profile
            self.user_index[user_id] = self.next_user_idx
            self.next_user_idx += 1
            
        return self.user_profiles[user_id]
    
    def add_item(self, item_id: str, features: Dict[str, Any] = None) -> ItemProfile:
        """Add new item with cold-start initialization"""
        if item_id not in self.item_profiles:
            factors = np.random.normal(0, 0.1, self.config.n_factors)
            profile = ItemProfile(
                item_id=item_id,
                factors=factors,
                bias=0.0,
                features=features or {},
                interaction_count=0,
                last_updated=datetime.now()
            )
            self.item_profiles[item_id] = profile
            self.item_index[item_id] = self.next_item_idx
            self.next_item_idx += 1
            
        return self.item_profiles[item_id]
    
    def update_single_interaction(self, user_id: str, item_id: str, rating: float, 
                                 timestamp: datetime = None):
        """Update model with a single user-item interaction"""
        try:
            # Ensure user and item exist
            user_profile = self.add_user(user_id)
            item_profile = self.add_item(item_id)
            
            # Calculate prediction error
            prediction = self.predict_rating(user_id, item_id)
            error = rating - prediction
            
            # Apply forgetting factor based on time
            if timestamp:
                time_decay = self._calculate_time_decay(user_profile.last_updated, timestamp)
                effective_lr = self.config.learning_rate * time_decay
            else:
                effective_lr = self.config.learning_rate
            
            # Update biases
            user_profile.bias += effective_lr * (error - self.config.regularization * user_profile.bias)
            item_profile.bias += effective_lr * (error - self.config.regularization * item_profile.bias)
            
            # Update factors using SGD
            user_factors_old = user_profile.factors.copy()
            user_profile.factors += effective_lr * (
                error * item_profile.factors - self.config.regularization * user_profile.factors
            )
            item_profile.factors += effective_lr * (
                error * user_factors_old - self.config.regularization * item_profile.factors
            )
            
            # Update metadata
            user_profile.interaction_count += 1
            item_profile.interaction_count += 1
            user_profile.last_updated = timestamp or datetime.now()
            item_profile.last_updated = timestamp or datetime.now()
            
            # Update confidence based on interaction count
            user_profile.confidence = min(1.0, user_profile.interaction_count / self.config.cold_start_threshold)
            
            logger.debug(f"Updated profiles for {user_id} -> {item_id}, error: {error:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating single interaction: {e}")
    
    def batch_update(self, interactions_df: pd.DataFrame):
        """Update model with batch of interactions"""
        try:
            for _, row in interactions_df.iterrows():
                self.update_single_interaction(
                    user_id=str(row['user_id']),
                    item_id=str(row['item_id']),
                    rating=float(row['rating']),
                    timestamp=pd.to_datetime(row.get('timestamp', datetime.now()))
                )
                
            logger.info(f"Batch updated {len(interactions_df)} interactions")
            
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for user-item pair"""
        try:
            if user_id not in self.user_profiles or item_id not in self.item_profiles:
                return self.global_bias
            
            user_profile = self.user_profiles[user_id]
            item_profile = self.item_profiles[item_id]
            
            # Matrix factorization prediction
            prediction = (self.global_bias + 
                         user_profile.bias + 
                         item_profile.bias + 
                         np.dot(user_profile.factors, item_profile.factors))
            
            return float(np.clip(prediction, 0.0, 5.0))
            
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return self.global_bias
    
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 10,
                                exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Get top N recommendations for user"""
        try:
            if user_id not in self.user_profiles:
                return []
            
            recommendations = []
            user_profile = self.user_profiles[user_id]
            
            # Get items user hasn't interacted with
            if exclude_seen:
                # This would need interaction history tracking
                available_items = list(self.item_profiles.keys())
            else:
                available_items = list(self.item_profiles.keys())
            
            # Calculate predictions for all items
            for item_id in available_items:
                prediction = self.predict_rating(user_id, item_id)
                recommendations.append((item_id, prediction))
            
            # Sort by prediction score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _calculate_time_decay(self, last_update: datetime, current_time: datetime) -> float:
        """Calculate time-based decay factor"""
        try:
            time_diff = (current_time - last_update).total_seconds() / 3600  # hours
            decay = self.config.forgetting_factor ** time_diff
            return max(0.1, decay)  # Minimum decay factor
        except:
            return 1.0


class OnlineContentBasedLearner:
    """Online content-based learning with streaming updates"""
    
    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.user_preference_profiles: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.item_content_profiles: Dict[str, Dict[str, float]] = {}
        self.feature_weights: Dict[str, float] = defaultdict(float)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_similarity_cache = {}
        self.update_counter = 0
        
    def update_user_preferences(self, user_id: str, item_id: str, rating: float,
                               item_features: Dict[str, Any]):
        """Update user preference profile with new interaction"""
        try:
            # Extract content features
            content_vector = self._extract_content_features(item_features)
            
            # Update user preferences using weighted average
            user_prefs = self.user_preference_profiles[user_id]
            
            for feature, value in content_vector.items():
                if feature in user_prefs:
                    # Exponential moving average update
                    alpha = self.config.learning_rate
                    user_prefs[feature] = ((1 - alpha) * user_prefs[feature] + 
                                         alpha * rating * value)
                else:
                    user_prefs[feature] = rating * value
            
            self.user_preference_profiles[user_id] = user_prefs
            
            # Update item content profile
            self.item_content_profiles[item_id] = content_vector
            
            # Update global feature weights
            self._update_feature_weights(content_vector, rating)
            
            self.update_counter += 1
            
            logger.debug(f"Updated content preferences for {user_id} -> {item_id}")
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def _extract_content_features(self, item_features: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize content features"""
        try:
            content_vector = {}
            
            # Handle different feature types
            for feature_name, feature_value in item_features.items():
                if isinstance(feature_value, (list, tuple)):
                    # Multi-valued features (e.g., genres)
                    for value in feature_value:
                        key = f"{feature_name}_{value}"
                        content_vector[key] = 1.0
                elif isinstance(feature_value, str):
                    # String features
                    if len(feature_value) > 20:  # Long text - use TF-IDF
                        # Simplified TF-IDF for online learning
                        words = feature_value.lower().split()
                        for word in words[:10]:  # Limit to top 10 words
                            key = f"{feature_name}_word_{word}"
                            content_vector[key] = 1.0 / len(words)
                    else:
                        # Short categorical string
                        key = f"{feature_name}_{feature_value}"
                        content_vector[key] = 1.0
                elif isinstance(feature_value, (int, float)):
                    # Numerical features - normalize
                    key = f"{feature_name}_numeric"
                    content_vector[key] = float(feature_value)
            
            return content_vector
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return {}
    
    def _update_feature_weights(self, content_vector: Dict[str, float], rating: float):
        """Update global feature importance weights"""
        try:
            rating_normalized = (rating - 2.5) / 2.5  # Normalize to [-1, 1]
            
            for feature, value in content_vector.items():
                # Update feature weight using gradient descent
                current_weight = self.feature_weights[feature]
                gradient = rating_normalized * value
                
                self.feature_weights[feature] = (
                    current_weight + self.config.learning_rate * gradient
                )
                
        except Exception as e:
            logger.error(f"Error updating feature weights: {e}")
    
    def predict_user_item_affinity(self, user_id: str, item_id: str,
                                  item_features: Dict[str, Any]) -> float:
        """Predict user affinity for item based on content"""
        try:
            if user_id not in self.user_preference_profiles:
                return 2.5  # Neutral rating
            
            user_prefs = self.user_preference_profiles[user_id]
            item_vector = self._extract_content_features(item_features)
            
            # Calculate content-based affinity
            affinity_score = 0.0
            feature_count = 0
            
            for feature, item_value in item_vector.items():
                if feature in user_prefs:
                    feature_weight = self.feature_weights.get(feature, 1.0)
                    affinity_score += user_prefs[feature] * item_value * feature_weight
                    feature_count += 1
            
            if feature_count > 0:
                affinity_score /= feature_count
            
            # Normalize to rating scale
            predicted_rating = 2.5 + affinity_score  # Center around 2.5
            return float(np.clip(predicted_rating, 0.0, 5.0))
            
        except Exception as e:
            logger.error(f"Error predicting affinity: {e}")
            return 2.5
    
    def get_similar_items(self, target_item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get items similar to target item based on content"""
        try:
            if target_item_id not in self.item_content_profiles:
                return []
            
            target_vector = self.item_content_profiles[target_item_id]
            similarities = []
            
            for item_id, item_vector in self.item_content_profiles.items():
                if item_id != target_item_id:
                    similarity = self._calculate_content_similarity(target_vector, item_vector)
                    similarities.append((item_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_similar]
            
        except Exception as e:
            logger.error(f"Error getting similar items: {e}")
            return []
    
    def _calculate_content_similarity(self, vector1: Dict[str, float], 
                                    vector2: Dict[str, float]) -> float:
        """Calculate cosine similarity between content vectors"""
        try:
            # Get common features
            common_features = set(vector1.keys()) & set(vector2.keys())
            
            if not common_features:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = sum(vector1[f] * vector2[f] for f in common_features)
            norm1 = math.sqrt(sum(v**2 for v in vector1.values()))
            norm2 = math.sqrt(sum(v**2 for v in vector2.values()))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


class OnlineLearner:
    """
    Main online learning engine coordinating different learning approaches
    """
    
    def __init__(self, config: OnlineLearningConfig = None):
        self.config = config or OnlineLearningConfig()
        
        # Initialize learning components
        self.matrix_factorization = IncrementalMatrixFactorization(self.config)
        self.content_learner = OnlineContentBasedLearner(self.config)
        
        # Ensemble management
        self.ensemble_weights = EnsembleWeights()
        self.bandit = MultiArmedBandit(n_arms=3, exploration_rate=self.config.bandit_exploration_rate)
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=1000)
        self.update_queue = deque(maxlen=self.config.batch_size * 2)
        
        # Threading for asynchronous updates
        self.update_lock = threading.Lock()
        self.background_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.is_running = False
        self.start_background_processing()
    
    def start_background_processing(self):
        """Start background processing for queued updates"""
        if not self.is_running:
            self.is_running = True
            self.background_thread = threading.Thread(target=self._background_processor)
            self.background_thread.daemon = True
            self.background_thread.start()
            logger.info("Started online learning background processing")
    
    def stop_background_processing(self):
        """Stop background processing"""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Stopped online learning background processing")
    
    def _background_processor(self):
        """Background thread for processing update queue"""
        while self.is_running:
            try:
                if self.update_queue:
                    with self.update_lock:
                        # Process batch of updates
                        batch_size = min(self.config.batch_size, len(self.update_queue))
                        batch = []
                        for _ in range(batch_size):
                            if self.update_queue:
                                batch.append(self.update_queue.popleft())
                    
                    if batch:
                        self._process_update_batch(batch)
                
                # Sleep briefly to prevent high CPU usage
                asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                asyncio.sleep(1)
    
    def _process_update_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of updates"""
        try:
            for update in batch:
                update_type = update.get('type')
                
                if update_type == 'interaction':
                    self._process_interaction_update(update)
                elif update_type == 'weights':
                    self._process_weight_update(update)
                
        except Exception as e:
            logger.error(f"Error processing update batch: {e}")
    
    def _process_interaction_update(self, update: Dict[str, Any]):
        """Process single interaction update"""
        try:
            user_id = update['user_id']
            item_id = update['item_id']
            rating = update['rating']
            timestamp = update.get('timestamp', datetime.now())
            item_features = update.get('item_features', {})
            
            # Update matrix factorization
            self.matrix_factorization.update_single_interaction(
                user_id, item_id, rating, timestamp
            )
            
            # Update content-based learning
            if item_features:
                self.content_learner.update_user_preferences(
                    user_id, item_id, rating, item_features
                )
                
        except Exception as e:
            logger.error(f"Error processing interaction update: {e}")
    
    def _process_weight_update(self, update: Dict[str, Any]):
        """Process ensemble weight update"""
        try:
            performance_scores = update['performance_scores']
            
            # Update bandit with performance feedback
            for i, score in enumerate(performance_scores):
                self.bandit.update_reward(i, score)
            
            # Update ensemble weights based on bandit probabilities
            probabilities = self.bandit.get_arm_probabilities()
            
            self.ensemble_weights.collaborative_weight = probabilities[0]
            self.ensemble_weights.content_weight = probabilities[1]
            self.ensemble_weights.knowledge_weight = probabilities[2]
            self.ensemble_weights.last_updated = datetime.now()
            
            logger.debug(f"Updated ensemble weights: {probabilities}")
            
        except Exception as e:
            logger.error(f"Error processing weight update: {e}")
    
    async def update_with_feedback(self, user_id: str, item_id: str, rating: float,
                                 item_features: Dict[str, Any] = None,
                                 timestamp: datetime = None) -> bool:
        """
        Update models with new user feedback
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: User rating (0-5)
            item_features: Item content features
            timestamp: Interaction timestamp
            
        Returns:
            Success status
        """
        try:
            update_data = {
                'type': 'interaction',
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'item_features': item_features or {},
                'timestamp': timestamp or datetime.now()
            }
            
            with self.update_lock:
                self.update_queue.append(update_data)
            
            logger.debug(f"Queued feedback update: {user_id} -> {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating with feedback: {e}")
            return False
    
    def predict_rating(self, user_id: str, item_id: str,
                      item_features: Dict[str, Any] = None) -> float:
        """
        Predict rating using ensemble of models
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            item_features: Item content features
            
        Returns:
            Predicted rating
        """
        try:
            predictions = []
            weights = []
            
            # Collaborative filtering prediction
            cf_prediction = self.matrix_factorization.predict_rating(user_id, item_id)
            predictions.append(cf_prediction)
            weights.append(self.ensemble_weights.collaborative_weight)
            
            # Content-based prediction
            if item_features:
                cb_prediction = self.content_learner.predict_user_item_affinity(
                    user_id, item_id, item_features
                )
                predictions.append(cb_prediction)
                weights.append(self.ensemble_weights.content_weight)
            
            # Knowledge-based prediction (simplified popularity-based)
            kb_prediction = self._get_popularity_prediction(item_id)
            predictions.append(kb_prediction)
            weights.append(self.ensemble_weights.knowledge_weight)
            
            # Weighted ensemble prediction
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_prediction = sum(p * w for p, w in zip(predictions, weights)) / total_weight
            else:
                weighted_prediction = np.mean(predictions)
            
            return float(np.clip(weighted_prediction, 0.0, 5.0))
            
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return 2.5  # Neutral prediction
    
    def _get_popularity_prediction(self, item_id: str) -> float:
        """Get popularity-based prediction"""
        try:
            if item_id in self.matrix_factorization.item_profiles:
                item_profile = self.matrix_factorization.item_profiles[item_id]
                # Simple popularity score based on interaction count
                popularity = min(5.0, 2.5 + item_profile.interaction_count / 100)
                return popularity
            return 2.5
        except:
            return 2.5
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                           available_items: List[str] = None) -> List[Tuple[str, float]]:
        """
        Get personalized recommendations for user
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            available_items: List of available items to consider
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        try:
            if available_items is None:
                available_items = list(self.matrix_factorization.item_profiles.keys())
            
            recommendations = []
            
            for item_id in available_items:
                # Get item features if available
                item_features = {}
                if item_id in self.content_learner.item_content_profiles:
                    item_features = self.content_learner.item_content_profiles[item_id]
                
                # Predict rating
                predicted_rating = self.predict_rating(user_id, item_id, item_features)
                recommendations.append((item_id, predicted_rating))
            
            # Sort by predicted rating
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def update_ensemble_weights(self, performance_scores: List[float]):
        """
        Update ensemble weights based on performance feedback
        
        Args:
            performance_scores: Performance scores for [collaborative, content, knowledge]
        """
        try:
            update_data = {
                'type': 'weights',
                'performance_scores': performance_scores
            }
            
            with self.update_lock:
                self.update_queue.append(update_data)
                
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the online learning models"""
        try:
            return {
                'matrix_factorization': {
                    'num_users': len(self.matrix_factorization.user_profiles),
                    'num_items': len(self.matrix_factorization.item_profiles),
                    'total_interactions': sum(
                        profile.interaction_count 
                        for profile in self.matrix_factorization.user_profiles.values()
                    )
                },
                'content_learning': {
                    'num_user_profiles': len(self.content_learner.user_preference_profiles),
                    'num_item_profiles': len(self.content_learner.item_content_profiles),
                    'num_features': len(self.content_learner.feature_weights),
                    'update_count': self.content_learner.update_counter
                },
                'ensemble': {
                    'weights': {
                        'collaborative': self.ensemble_weights.collaborative_weight,
                        'content': self.ensemble_weights.content_weight,
                        'knowledge': self.ensemble_weights.knowledge_weight
                    },
                    'bandit_stats': {
                        'arm_counts': self.bandit.arm_counts.tolist(),
                        'arm_rewards': self.bandit.arm_rewards.tolist(),
                        'total_count': int(self.bandit.total_count)
                    }
                },
                'queue_size': len(self.update_queue),
                'is_running': self.is_running
            }
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save online learning model to file"""
        try:
            model_data = {
                'config': self.config,
                'matrix_factorization': {
                    'user_profiles': self.matrix_factorization.user_profiles,
                    'item_profiles': self.matrix_factorization.item_profiles,
                    'global_bias': self.matrix_factorization.global_bias,
                    'user_index': self.matrix_factorization.user_index,
                    'item_index': self.matrix_factorization.item_index
                },
                'content_learner': {
                    'user_preference_profiles': dict(self.content_learner.user_preference_profiles),
                    'item_content_profiles': self.content_learner.item_content_profiles,
                    'feature_weights': dict(self.content_learner.feature_weights)
                },
                'ensemble_weights': self.ensemble_weights,
                'bandit': {
                    'arm_counts': self.bandit.arm_counts,
                    'arm_rewards': self.bandit.arm_rewards,
                    'total_count': self.bandit.total_count
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved online learning model to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load online learning model from file"""
        try:
            model_data = joblib.load(filepath)
            
            # Restore configuration
            self.config = model_data['config']
            
            # Restore matrix factorization
            mf_data = model_data['matrix_factorization']
            self.matrix_factorization.user_profiles = mf_data['user_profiles']
            self.matrix_factorization.item_profiles = mf_data['item_profiles']
            self.matrix_factorization.global_bias = mf_data['global_bias']
            self.matrix_factorization.user_index = mf_data['user_index']
            self.matrix_factorization.item_index = mf_data['item_index']
            
            # Restore content learner
            cl_data = model_data['content_learner']
            self.content_learner.user_preference_profiles = defaultdict(dict, cl_data['user_preference_profiles'])
            self.content_learner.item_content_profiles = cl_data['item_content_profiles']
            self.content_learner.feature_weights = defaultdict(float, cl_data['feature_weights'])
            
            # Restore ensemble weights
            self.ensemble_weights = model_data['ensemble_weights']
            
            # Restore bandit
            bandit_data = model_data['bandit']
            self.bandit.arm_counts = bandit_data['arm_counts']
            self.bandit.arm_rewards = bandit_data['arm_rewards']
            self.bandit.total_count = bandit_data['total_count']
            
            logger.info(f"Loaded online learning model from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_background_processing()
        except:
            pass
    
    async def update_model(self, data: Dict[str, Any]) -> bool:
        """
        Update the model with new data
        
        Args:
            data: Data to update the model with
            
        Returns:
            True if update was successful
        """
        try:
            # Add to update queue for background processing
            self.update_queue.append(data)
            return True
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the online learner."""
        return {
            "status": "healthy",
            "model_initialized": hasattr(self, 'collaborative_learner') and self.collaborative_learner is not None,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "last_update": self.last_updated.isoformat() if hasattr(self, 'last_updated') and self.last_updated else None,
            "total_updates": len(self.update_queue) if hasattr(self, 'update_queue') else 0,
            "ensemble_weights": self.ensemble_weights.tolist() if hasattr(self, 'ensemble_weights') else None,
            "background_processing": self.processing_thread.is_alive() if hasattr(self, 'processing_thread') and self.processing_thread else False
        }
