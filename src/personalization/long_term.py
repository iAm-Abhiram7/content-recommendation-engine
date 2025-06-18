"""
Long-Term Preference Model

Captures and models long-term user preferences:
- Historical preference patterns and stability
- User lifecycle stages and evolution
- Preference persistence across time
- Long-term trend analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)


class LongTermPreferenceModel:
    """
    Models long-term user preferences and historical patterns
    """
    
    def __init__(self,
                 min_history_days: int = 30,
                 preference_decay_rate: float = 0.01,
                 stability_window_days: int = 90,
                 n_preference_clusters: int = 5):
        """
        Initialize long-term preference model
        
        Args:
            min_history_days: Minimum days of history required
            preference_decay_rate: Decay rate for old preferences
            stability_window_days: Window for stability analysis
            n_preference_clusters: Number of preference clusters
        """
        self.min_history_days = min_history_days
        self.preference_decay_rate = preference_decay_rate
        self.stability_window_days = stability_window_days
        self.n_preference_clusters = n_preference_clusters
        
        # User data storage
        self.user_long_term_preferences = {}
        self.user_preference_history = defaultdict(list)
        self.user_lifecycle_stages = {}
        
        # Preference clusters
        self.preference_clusters = None
        self.cluster_model = None
        
        # Stability metrics
        self.preference_stability_scores = {}
        
    def update_user_preferences(self,
                              user_id: str,
                              interactions_df: pd.DataFrame,
                              content_metadata: Optional[pd.DataFrame] = None):
        """
        Update user's long-term preferences based on interaction history
        
        Args:
            user_id: User identifier
            interactions_df: User's interaction history
            content_metadata: Optional content metadata
        """
        if len(interactions_df) == 0:
            return
        
        # Calculate preference vector from interactions
        preference_vector = self._calculate_preference_vector(
            interactions_df, content_metadata
        )
        
        # Update preference history
        self.user_preference_history[user_id].append({
            'timestamp': datetime.now(),
            'preferences': preference_vector,
            'interaction_count': len(interactions_df)
        })
        
        # Maintain history size
        if len(self.user_preference_history[user_id]) > 100:
            self.user_preference_history[user_id] = self.user_preference_history[user_id][-100:]
        
        # Calculate current long-term preferences
        self.user_long_term_preferences[user_id] = self._calculate_long_term_preferences(user_id)
        
        # Update lifecycle stage
        self._update_lifecycle_stage(user_id, interactions_df)
        
        # Calculate stability score
        self.preference_stability_scores[user_id] = self._calculate_stability_score(user_id)
    
    def _calculate_preference_vector(self,
                                   interactions_df: pd.DataFrame,
                                   content_metadata: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate preference vector from interactions"""
        preferences = defaultdict(float)
        
        # Weight interactions by type and recency
        current_time = datetime.now()
        
        for _, interaction in interactions_df.iterrows():
            # Time decay weight
            if 'timestamp' in interaction and pd.notna(interaction['timestamp']):
                interaction_time = pd.to_datetime(interaction['timestamp'])
                days_ago = (current_time - interaction_time).days
                time_weight = np.exp(-self.preference_decay_rate * days_ago)
            else:
                time_weight = 1.0
            
            # Interaction type weight
            interaction_type = interaction.get('interaction_type', 'view')
            type_weights = {
                'view': 1.0,
                'like': 3.0,
                'share': 4.0,
                'rating': 5.0,
                'purchase': 6.0
            }
            interaction_weight = type_weights.get(interaction_type, 1.0)
            
            # Rating weight
            rating_weight = 1.0
            if 'rating' in interaction and pd.notna(interaction['rating']):
                rating_weight = interaction['rating'] / 5.0
            
            # Extract content features if metadata available
            if content_metadata is not None:
                content_id = interaction.get('content_id') or interaction.get('item_id')
                content_info = content_metadata[content_metadata['item_id'] == content_id]
                
                if not content_info.empty:
                    content_row = content_info.iloc[0]
                    
                    # Genre preferences
                    if 'genre' in content_row and pd.notna(content_row['genre']):
                        genre = content_row['genre']
                        weight = time_weight * interaction_weight * rating_weight
                        preferences[f'genre_{genre}'] += weight
                    
                    # Domain preferences  
                    if 'domain' in content_row and pd.notna(content_row['domain']):
                        domain = content_row['domain']
                        weight = time_weight * interaction_weight * rating_weight
                        preferences[f'domain_{domain}'] += weight
                    
                    # Other metadata features
                    for feature in ['language', 'format', 'year']:
                        if feature in content_row and pd.notna(content_row[feature]):
                            value = content_row[feature]
                            weight = time_weight * interaction_weight * rating_weight * 0.5
                            preferences[f'{feature}_{value}'] += weight
            
            # Item-level preferences
            item_id = interaction.get('content_id') or interaction.get('item_id')
            if item_id:
                weight = time_weight * interaction_weight * rating_weight * 0.1
                preferences[f'item_{item_id}'] += weight
        
        # Normalize preferences
        if preferences:
            max_pref = max(preferences.values())
            if max_pref > 0:
                preferences = {k: v / max_pref for k, v in preferences.items()}
        
        return dict(preferences)
    
    def _calculate_long_term_preferences(self, user_id: str) -> Dict[str, float]:
        """Calculate aggregated long-term preferences"""
        preference_history = self.user_preference_history.get(user_id, [])
        
        if not preference_history:
            return {}
        
        # Aggregate preferences with time decay
        current_time = datetime.now()
        aggregated_prefs = defaultdict(float)
        total_weight = 0
        
        for pref_entry in preference_history:
            timestamp = pref_entry['timestamp']
            preferences = pref_entry['preferences']
            
            # Time decay from entry timestamp
            days_ago = (current_time - timestamp).days
            entry_weight = np.exp(-self.preference_decay_rate * days_ago)
            
            for pref_key, pref_value in preferences.items():
                aggregated_prefs[pref_key] += pref_value * entry_weight
            
            total_weight += entry_weight
        
        # Normalize by total weight
        if total_weight > 0:
            aggregated_prefs = {
                k: v / total_weight 
                for k, v in aggregated_prefs.items()
            }
        
        return dict(aggregated_prefs)
    
    def _update_lifecycle_stage(self, user_id: str, interactions_df: pd.DataFrame):
        """Update user's lifecycle stage"""
        # Calculate user metrics
        total_interactions = len(interactions_df)
        
        # Time span of interactions
        if 'timestamp' in interactions_df.columns:
            timestamps = pd.to_datetime(interactions_df['timestamp'])
            time_span_days = (timestamps.max() - timestamps.min()).days
            interaction_frequency = total_interactions / max(time_span_days, 1)
        else:
            time_span_days = 1
            interaction_frequency = total_interactions
        
        # Determine lifecycle stage
        if total_interactions < 10:
            stage = 'new_user'
        elif total_interactions < 50:
            stage = 'developing_user'
        elif interaction_frequency > 1.0:  # More than 1 interaction per day
            stage = 'active_user'
        elif interaction_frequency > 0.1:  # More than 1 interaction per 10 days
            stage = 'regular_user'
        else:
            stage = 'inactive_user'
        
        self.user_lifecycle_stages[user_id] = {
            'stage': stage,
            'total_interactions': total_interactions,
            'time_span_days': time_span_days,
            'interaction_frequency': interaction_frequency,
            'last_updated': datetime.now()
        }
    
    def _calculate_stability_score(self, user_id: str) -> float:
        """Calculate preference stability score for user"""
        preference_history = self.user_preference_history.get(user_id, [])
        
        if len(preference_history) < 3:
            return 0.0  # Not enough history
        
        # Get recent preference vectors
        recent_prefs = preference_history[-10:]  # Last 10 entries
        
        # Calculate pairwise similarities between consecutive preference vectors
        similarities = []
        
        for i in range(len(recent_prefs) - 1):
            prefs1 = recent_prefs[i]['preferences']
            prefs2 = recent_prefs[i + 1]['preferences']
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(prefs1, prefs2)
            similarities.append(similarity)
        
        # Stability = average similarity between consecutive preferences
        return np.mean(similarities) if similarities else 0.0
    
    def _cosine_similarity(self, prefs1: Dict[str, float], prefs2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two preference vectors"""
        # Get all keys
        all_keys = set(prefs1.keys()) | set(prefs2.keys())
        
        if not all_keys:
            return 1.0
        
        # Create vectors
        vec1 = np.array([prefs1.get(key, 0) for key in all_keys])
        vec2 = np.array([prefs2.get(key, 0) for key in all_keys])
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def cluster_user_preferences(self, user_ids: Optional[List[str]] = None):
        """Cluster users based on their long-term preferences"""
        if user_ids is None:
            user_ids = list(self.user_long_term_preferences.keys())
        
        if len(user_ids) < self.n_preference_clusters:
            logger.warning("Not enough users for clustering")
            return
        
        # Prepare preference matrix
        all_preference_keys = set()
        for user_id in user_ids:
            prefs = self.user_long_term_preferences.get(user_id, {})
            all_preference_keys.update(prefs.keys())
        
        preference_keys = list(all_preference_keys)
        
        # Create preference matrix
        preference_matrix = []
        valid_users = []
        
        for user_id in user_ids:
            prefs = self.user_long_term_preferences.get(user_id, {})
            if prefs:  # Only include users with preferences
                pref_vector = [prefs.get(key, 0) for key in preference_keys]
                preference_matrix.append(pref_vector)
                valid_users.append(user_id)
        
        if len(preference_matrix) < self.n_preference_clusters:
            return
        
        preference_matrix = np.array(preference_matrix)
        
        # Apply PCA for dimensionality reduction if needed
        if preference_matrix.shape[1] > 50:
            pca = PCA(n_components=50)
            preference_matrix = pca.fit_transform(preference_matrix)
        
        # Perform clustering
        self.cluster_model = KMeans(n_clusters=self.n_preference_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(preference_matrix)
        
        # Store cluster assignments
        self.preference_clusters = {}
        for user_id, cluster_label in zip(valid_users, cluster_labels):
            self.preference_clusters[user_id] = int(cluster_label)
        
        logger.info(f"Clustered {len(valid_users)} users into {self.n_preference_clusters} preference groups")
    
    def get_user_long_term_preferences(self, user_id: str) -> Dict[str, float]:
        """Get user's long-term preferences"""
        return self.user_long_term_preferences.get(user_id, {})
    
    def get_user_lifecycle_stage(self, user_id: str) -> Dict[str, Any]:
        """Get user's lifecycle stage information"""
        return self.user_lifecycle_stages.get(user_id, {})
    
    def get_user_preference_cluster(self, user_id: str) -> Optional[int]:
        """Get user's preference cluster"""
        return self.preference_clusters.get(user_id) if self.preference_clusters else None
    
    def get_similar_users(self, user_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get users with similar long-term preferences"""
        user_prefs = self.user_long_term_preferences.get(user_id, {})
        
        if not user_prefs:
            return []
        
        similarities = []
        
        for other_user_id, other_prefs in self.user_long_term_preferences.items():
            if other_user_id != user_id and other_prefs:
                similarity = self._cosine_similarity(user_prefs, other_prefs)
                similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def predict_preference_evolution(self, user_id: str) -> Dict[str, Any]:
        """Predict how user preferences might evolve"""
        preference_history = self.user_preference_history.get(user_id, [])
        
        if len(preference_history) < 5:
            return {"prediction": "insufficient_data"}
        
        # Analyze preference trends
        recent_prefs = preference_history[-5:]
        older_prefs = preference_history[-10:-5] if len(preference_history) >= 10 else []
        
        if not older_prefs:
            return {"prediction": "stable", "confidence": 0.5}
        
        # Calculate preference changes
        recent_avg = self._average_preferences([p['preferences'] for p in recent_prefs])
        older_avg = self._average_preferences([p['preferences'] for p in older_prefs])
        
        # Find growing and declining preferences
        growing_prefs = {}
        declining_prefs = {}
        
        all_keys = set(recent_avg.keys()) | set(older_avg.keys())
        
        for key in all_keys:
            recent_val = recent_avg.get(key, 0)
            older_val = older_avg.get(key, 0)
            
            change = recent_val - older_val
            
            if change > 0.1:
                growing_prefs[key] = change
            elif change < -0.1:
                declining_prefs[key] = abs(change)
        
        # Calculate overall stability
        stability = self.preference_stability_scores.get(user_id, 0.5)
        
        prediction = {
            "prediction": "evolving" if stability < 0.7 else "stable",
            "confidence": stability,
            "growing_preferences": growing_prefs,
            "declining_preferences": declining_prefs,
            "stability_score": stability
        }
        
        return prediction
    
    def _average_preferences(self, preference_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average preferences across multiple preference dictionaries"""
        if not preference_list:
            return {}
        
        aggregated = defaultdict(float)
        
        for prefs in preference_list:
            for key, value in prefs.items():
                aggregated[key] += value
        
        # Average
        for key in aggregated:
            aggregated[key] /= len(preference_list)
        
        return dict(aggregated)
    
    def save_model(self, filepath: str):
        """Save long-term preference model"""
        model_data = {
            'min_history_days': self.min_history_days,
            'preference_decay_rate': self.preference_decay_rate,
            'stability_window_days': self.stability_window_days,
            'n_preference_clusters': self.n_preference_clusters,
            'user_long_term_preferences': self.user_long_term_preferences,
            'user_preference_history': dict(self.user_preference_history),
            'user_lifecycle_stages': self.user_lifecycle_stages,
            'preference_clusters': self.preference_clusters,
            'preference_stability_scores': self.preference_stability_scores
        }
        
        joblib.dump(model_data, filepath)
        
        # Save cluster model separately if it exists
        if self.cluster_model is not None:
            cluster_path = filepath.replace('.pkl', '_cluster_model.pkl')
            joblib.dump(self.cluster_model, cluster_path)
        
        logger.info(f"Long-term preference model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load long-term preference model"""
        model_data = joblib.load(filepath)
        
        self.min_history_days = model_data['min_history_days']
        self.preference_decay_rate = model_data['preference_decay_rate']
        self.stability_window_days = model_data['stability_window_days']
        self.n_preference_clusters = model_data['n_preference_clusters']
        
        self.user_long_term_preferences = model_data['user_long_term_preferences']
        self.user_preference_history = defaultdict(list, model_data['user_preference_history'])
        self.user_lifecycle_stages = model_data['user_lifecycle_stages']
        self.preference_clusters = model_data.get('preference_clusters')
        self.preference_stability_scores = model_data['preference_stability_scores']
        
        # Load cluster model if it exists
        cluster_path = filepath.replace('.pkl', '_cluster_model.pkl')
        try:
            self.cluster_model = joblib.load(cluster_path)
        except FileNotFoundError:
            self.cluster_model = None
        
        logger.info(f"Long-term preference model loaded from {filepath}")
