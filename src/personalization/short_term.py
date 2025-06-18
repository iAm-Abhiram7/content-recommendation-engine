"""
Short-Term Preference Model

Captures and models short-term user preferences:
- Recent interaction patterns and trends
- Session-based preferences
- Temporary mood and context shifts
- Recency-weighted recommendation scoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from scipy.stats import entropy
import joblib

logger = logging.getLogger(__name__)


class ShortTermPreferenceModel:
    """
    Models short-term user preferences and recent behavior patterns
    """
    
    def __init__(self,
                 time_window_hours: int = 24,
                 session_timeout_minutes: int = 30,
                 recency_decay_rate: float = 0.1,
                 min_interactions_for_pattern: int = 3):
        """
        Initialize short-term preference model
        
        Args:
            time_window_hours: Time window for short-term preferences (hours)
            session_timeout_minutes: Session timeout in minutes
            recency_decay_rate: Decay rate for recency weighting
            min_interactions_for_pattern: Minimum interactions to detect patterns
        """
        self.time_window_hours = time_window_hours
        self.session_timeout_minutes = session_timeout_minutes
        self.recency_decay_rate = recency_decay_rate
        self.min_interactions_for_pattern = min_interactions_for_pattern
        
        # User data storage
        self.user_sessions = defaultdict(list)  # Recent sessions per user
        self.user_recent_interactions = defaultdict(deque)  # Recent interactions
        self.user_preference_vectors = {}  # Current short-term preference vectors
        
        # Pattern detection
        self.session_patterns = defaultdict(list)  # Detected session patterns
        self.temporal_patterns = defaultdict(dict)  # Time-based patterns
        
        # Context tracking
        self.context_preferences = defaultdict(dict)  # Context-specific preferences
        
        # Performance metrics
        self.prediction_accuracy = {}
        
    def update_user_interaction(self,
                              user_id: str,
                              item_id: str,
                              interaction_type: str = 'view',
                              rating: Optional[float] = None,
                              timestamp: Optional[datetime] = None,
                              context: Optional[Dict] = None):
        """
        Update user's short-term preferences with new interaction
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            interaction_type: Type of interaction ('view', 'click', 'rating', 'purchase')
            rating: Optional rating score
            timestamp: Interaction timestamp
            context: Optional context information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        interaction = {
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': timestamp,
            'context': context or {}
        }
        
        # Add to recent interactions
        self.user_recent_interactions[user_id].append(interaction)
        
        # Maintain sliding window
        self._maintain_sliding_window(user_id)
        
        # Update or create session
        self._update_user_session(user_id, interaction)
        
        # Update preference vector
        self._update_preference_vector(user_id)
        
        # Update context preferences
        if context:
            self._update_context_preferences(user_id, context, interaction)
    
    def _maintain_sliding_window(self, user_id: str):
        """Maintain sliding window of recent interactions"""
        cutoff_time = datetime.now() - timedelta(hours=self.time_window_hours)
        
        interactions = self.user_recent_interactions[user_id]
        
        # Remove old interactions
        while interactions and interactions[0]['timestamp'] < cutoff_time:
            interactions.popleft()
    
    def _update_user_session(self, user_id: str, interaction: Dict):
        """Update user's current session or create new one"""
        current_time = interaction['timestamp']
        
        # Check if this continues the current session
        if self.user_sessions[user_id]:
            last_session = self.user_sessions[user_id][-1]
            last_interaction_time = last_session['interactions'][-1]['timestamp']
            
            time_diff = (current_time - last_interaction_time).total_seconds() / 60
            
            if time_diff <= self.session_timeout_minutes:
                # Continue current session
                last_session['interactions'].append(interaction)
                last_session['end_time'] = current_time
                last_session['duration'] = (current_time - last_session['start_time']).total_seconds() / 60
            else:
                # Start new session
                self._start_new_session(user_id, interaction)
        else:
            # First session for user
            self._start_new_session(user_id, interaction)
        
        # Maintain session history (keep last 10 sessions)
        if len(self.user_sessions[user_id]) > 10:
            self.user_sessions[user_id] = self.user_sessions[user_id][-10:]
    
    def _start_new_session(self, user_id: str, interaction: Dict):
        """Start a new session for the user"""
        session = {
            'session_id': f"{user_id}_{interaction['timestamp'].isoformat()}",
            'start_time': interaction['timestamp'],
            'end_time': interaction['timestamp'],
            'duration': 0,
            'interactions': [interaction],
            'context': interaction.get('context', {})
        }
        
        self.user_sessions[user_id].append(session)
    
    def _update_preference_vector(self, user_id: str):
        """Update user's short-term preference vector"""
        recent_interactions = list(self.user_recent_interactions[user_id])
        
        if not recent_interactions:
            return
        
        # Calculate recency weights
        current_time = datetime.now()
        preference_scores = defaultdict(float)
        
        for interaction in recent_interactions:
            # Time decay weight
            time_diff_hours = (current_time - interaction['timestamp']).total_seconds() / 3600
            recency_weight = np.exp(-self.recency_decay_rate * time_diff_hours)
            
            # Interaction type weight
            type_weights = {
                'view': 1.0,
                'click': 2.0,
                'like': 3.0,
                'share': 3.0,
                'rating': 4.0,
                'purchase': 5.0
            }
            interaction_weight = type_weights.get(interaction['interaction_type'], 1.0)
            
            # Rating weight (if available)
            rating_weight = 1.0
            if interaction['rating'] is not None:
                rating_weight = interaction['rating'] / 5.0  # Normalize to 0-1
            
            # Combined weight
            total_weight = recency_weight * interaction_weight * rating_weight
            
            preference_scores[interaction['item_id']] += total_weight
        
        # Normalize scores
        if preference_scores:
            max_score = max(preference_scores.values())
            if max_score > 0:
                self.user_preference_vectors[user_id] = {
                    item_id: score / max_score 
                    for item_id, score in preference_scores.items()
                }
    
    def _update_context_preferences(self, user_id: str, context: Dict, interaction: Dict):
        """Update context-specific preferences"""
        for context_key, context_value in context.items():
            if context_key not in self.context_preferences[user_id]:
                self.context_preferences[user_id][context_key] = defaultdict(float)
            
            # Weight by interaction type and rating
            weight = 1.0
            if interaction['rating'] is not None:
                weight = interaction['rating'] / 5.0
            
            self.context_preferences[user_id][context_key][context_value] += weight
    
    def get_user_short_term_preferences(self, user_id: str) -> Dict[str, float]:
        """Get user's current short-term preferences"""
        return self.user_preference_vectors.get(user_id, {})
    
    def get_user_recent_sessions(self, user_id: str, n_sessions: int = 5) -> List[Dict]:
        """Get user's recent sessions"""
        sessions = self.user_sessions.get(user_id, [])
        return sessions[-n_sessions:] if sessions else []
    
    def detect_session_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Detect patterns in user's session behavior"""
        sessions = self.user_sessions.get(user_id, [])
        
        if len(sessions) < self.min_interactions_for_pattern:
            return []
        
        patterns = []
        
        # Pattern 1: Preferred session duration
        durations = [s['duration'] for s in sessions if s['duration'] > 0]
        if durations:
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            patterns.append({
                'pattern_type': 'session_duration',
                'avg_duration_minutes': avg_duration,
                'std_duration_minutes': std_duration,
                'preferred_duration_range': (max(0, avg_duration - std_duration), 
                                           avg_duration + std_duration)
            })
        
        # Pattern 2: Common session contexts
        context_counts = defaultdict(int)
        for session in sessions:
            context = session.get('context', {})
            for key, value in context.items():
                context_counts[f"{key}:{value}"] += 1
        
        if context_counts:
            # Find most common contexts
            sorted_contexts = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)
            common_contexts = [(ctx, count) for ctx, count in sorted_contexts[:3] 
                             if count >= self.min_interactions_for_pattern]
            
            if common_contexts:
                patterns.append({
                    'pattern_type': 'common_contexts',
                    'contexts': common_contexts
                })
        
        # Pattern 3: Session interaction patterns
        interaction_sequences = []
        for session in sessions:
            sequence = [i['interaction_type'] for i in session['interactions']]
            if len(sequence) >= 2:
                interaction_sequences.append(sequence)
        
        if interaction_sequences:
            # Find common interaction sequences
            sequence_counts = defaultdict(int)
            for sequence in interaction_sequences:
                for i in range(len(sequence) - 1):
                    bigram = (sequence[i], sequence[i + 1])
                    sequence_counts[bigram] += 1
            
            common_sequences = [(seq, count) for seq, count in sequence_counts.items() 
                              if count >= self.min_interactions_for_pattern]
            
            if common_sequences:
                patterns.append({
                    'pattern_type': 'interaction_sequences',
                    'common_sequences': common_sequences
                })
        
        # Store detected patterns
        self.session_patterns[user_id] = patterns
        
        return patterns
    
    def predict_next_interaction_type(self, user_id: str, current_session: List[str]) -> Dict[str, float]:
        """Predict next interaction type based on current session"""
        if not current_session:
            return {}
        
        # Get user's historical patterns
        patterns = self.session_patterns.get(user_id, [])
        
        # Find interaction sequence patterns
        sequence_patterns = None
        for pattern in patterns:
            if pattern['pattern_type'] == 'interaction_sequences':
                sequence_patterns = pattern['common_sequences']
                break
        
        if not sequence_patterns:
            return {}
        
        # Predict based on last interaction in current session
        last_interaction = current_session[-1]
        
        predictions = defaultdict(float)
        total_count = 0
        
        for (first_type, second_type), count in sequence_patterns:
            if first_type == last_interaction:
                predictions[second_type] += count
                total_count += count
        
        # Normalize to probabilities
        if total_count > 0:
            predictions = {
                interaction_type: count / total_count 
                for interaction_type, count in predictions.items()
            }
        
        return dict(predictions)
    
    def get_contextual_preferences(self, user_id: str, context: Dict[str, str]) -> Dict[str, float]:
        """Get user preferences filtered by context"""
        user_context_prefs = self.context_preferences.get(user_id, {})
        
        if not user_context_prefs or not context:
            return self.get_user_short_term_preferences(user_id)
        
        # Calculate context similarity scores
        context_scores = {}
        
        for context_key, context_value in context.items():
            if context_key in user_context_prefs:
                context_prefs = user_context_prefs[context_key]
                
                # Get preference for this context value
                if context_value in context_prefs:
                    context_scores[context_key] = context_prefs[context_value]
                else:
                    # Use average preference for this context type
                    context_scores[context_key] = np.mean(list(context_prefs.values()))
        
        # Weight base preferences by context scores
        base_preferences = self.get_user_short_term_preferences(user_id)
        
        if not context_scores:
            return base_preferences
        
        # Apply context weighting
        context_weight = np.mean(list(context_scores.values()))
        weighted_preferences = {
            item_id: score * (1 + context_weight)
            for item_id, score in base_preferences.items()
        }
        
        return weighted_preferences
    
    def calculate_preference_volatility(self, user_id: str) -> float:
        """Calculate how volatile/stable user's short-term preferences are"""
        sessions = self.user_sessions.get(user_id, [])
        
        if len(sessions) < 3:
            return 0.0  # Not enough data
        
        # Calculate preference entropy for each session
        session_entropies = []
        
        for session in sessions[-5:]:  # Last 5 sessions
            # Get item distribution in session
            items = [i['item_id'] for i in session['interactions']]
            
            if len(set(items)) <= 1:
                session_entropies.append(0.0)  # Single item = no entropy
                continue
            
            # Calculate entropy
            item_counts = defaultdict(int)
            for item in items:
                item_counts[item] += 1
            
            total_interactions = len(items)
            probabilities = [count / total_interactions for count in item_counts.values()]
            
            session_entropy = entropy(probabilities)
            session_entropies.append(session_entropy)
        
        # Volatility = standard deviation of session entropies
        if len(session_entropies) > 1:
            return np.std(session_entropies)
        else:
            return 0.0
    
    def get_trending_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get trending preferences based on recent behavior"""
        recent_interactions = list(self.user_recent_interactions[user_id])
        
        if len(recent_interactions) < self.min_interactions_for_pattern:
            return {}
        
        # Split into time periods
        current_time = datetime.now()
        recent_cutoff = current_time - timedelta(hours=self.time_window_hours // 2)
        
        recent_items = []
        older_items = []
        
        for interaction in recent_interactions:
            if interaction['timestamp'] >= recent_cutoff:
                recent_items.append(interaction['item_id'])
            else:
                older_items.append(interaction['item_id'])
        
        # Calculate item frequency in each period
        recent_counts = defaultdict(int)
        older_counts = defaultdict(int)
        
        for item in recent_items:
            recent_counts[item] += 1
        
        for item in older_items:
            older_counts[item] += 1
        
        # Find trending items (higher frequency in recent period)
        trending_items = {}
        
        for item in recent_counts:
            recent_freq = recent_counts[item] / max(len(recent_items), 1)
            older_freq = older_counts.get(item, 0) / max(len(older_items), 1)
            
            # Trending score = recent frequency - older frequency
            trending_score = recent_freq - older_freq
            
            if trending_score > 0:
                trending_items[item] = trending_score
        
        return {
            'trending_items': trending_items,
            'recent_period_hours': self.time_window_hours // 2,
            'total_recent_interactions': len(recent_items),
            'total_older_interactions': len(older_items)
        }
    
    def save_model(self, filepath: str):
        """Save short-term preference model"""
        # Convert deque objects to lists for serialization
        serializable_interactions = {}
        for user_id, interactions in self.user_recent_interactions.items():
            serializable_interactions[user_id] = list(interactions)
        
        model_data = {
            'time_window_hours': self.time_window_hours,
            'session_timeout_minutes': self.session_timeout_minutes,
            'recency_decay_rate': self.recency_decay_rate,
            'min_interactions_for_pattern': self.min_interactions_for_pattern,
            'user_sessions': dict(self.user_sessions),
            'user_recent_interactions': serializable_interactions,
            'user_preference_vectors': self.user_preference_vectors,
            'session_patterns': dict(self.session_patterns),
            'temporal_patterns': dict(self.temporal_patterns),
            'context_preferences': dict(self.context_preferences)
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Short-term preference model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load short-term preference model"""
        model_data = joblib.load(filepath)
        
        self.time_window_hours = model_data['time_window_hours']
        self.session_timeout_minutes = model_data['session_timeout_minutes']
        self.recency_decay_rate = model_data['recency_decay_rate']
        self.min_interactions_for_pattern = model_data['min_interactions_for_pattern']
        
        self.user_sessions = defaultdict(list, model_data['user_sessions'])
        
        # Convert lists back to deques
        interactions_data = model_data['user_recent_interactions']
        self.user_recent_interactions = defaultdict(deque)
        for user_id, interactions in interactions_data.items():
            self.user_recent_interactions[user_id] = deque(interactions)
        
        self.user_preference_vectors = model_data['user_preference_vectors']
        self.session_patterns = defaultdict(list, model_data['session_patterns'])
        self.temporal_patterns = defaultdict(dict, model_data['temporal_patterns'])
        self.context_preferences = defaultdict(dict, model_data['context_preferences'])
        
        logger.info(f"Short-term preference model loaded from {filepath}")
