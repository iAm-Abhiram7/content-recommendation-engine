"""
Sequential Pattern Mining

Implements sequential pattern mining for next-item prediction:
- Markov chain models for sequence prediction
- Session-based patterns and transitions
- Sequential rule mining
- Next-item recommendation based on sequences
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
import joblib
from itertools import combinations

logger = logging.getLogger(__name__)


class SequentialPatternMiner:
    """
    Sequential pattern mining for next-item prediction
    """
    
    def __init__(self,
                 max_sequence_length: int = 10,
                 min_support: float = 0.01,
                 min_confidence: float = 0.1,
                 session_timeout_minutes: int = 30):
        """
        Initialize sequential pattern miner
        
        Args:
            max_sequence_length: Maximum length of sequences to consider
            min_support: Minimum support threshold for patterns
            min_confidence: Minimum confidence threshold for rules
            session_timeout_minutes: Session timeout in minutes
        """
        self.max_sequence_length = max_sequence_length
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.session_timeout_minutes = session_timeout_minutes
        
        # Pattern storage
        self.frequent_sequences = {}
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.sequential_rules = []
        
        # User sessions
        self.user_sessions = defaultdict(list)
        
        # Performance metrics
        self.pattern_stats = {}
        
    def fit(self, interactions_df: pd.DataFrame):
        """
        Train sequential pattern mining model
        
        Args:
            interactions_df: DataFrame with user interactions including timestamps
        """
        logger.info("Training sequential pattern mining model")
        
        # Extract user sessions
        self._extract_user_sessions(interactions_df)
        
        # Mine frequent sequences
        self._mine_frequent_sequences()
        
        # Build transition matrix
        self._build_transition_matrix()
        
        # Generate sequential rules
        self._generate_sequential_rules()
        
        # Calculate statistics
        self._calculate_pattern_statistics()
        
        logger.info("Sequential pattern mining completed")
    
    def _extract_user_sessions(self, interactions_df: pd.DataFrame):
        """Extract user sessions from interaction data"""
        # Sort by user and timestamp
        df = interactions_df.sort_values(['user_id', 'timestamp'])
        
        current_sessions = {}
        
        for _, interaction in df.iterrows():
            user_id = interaction['user_id']
            item_id = interaction.get('content_id') or interaction.get('item_id')
            timestamp = pd.to_datetime(interaction['timestamp'])
            
            # Check if this continues current session or starts new one
            if user_id in current_sessions:
                last_timestamp = current_sessions[user_id]['last_timestamp']
                time_diff = (timestamp - last_timestamp).total_seconds() / 60
                
                if time_diff <= self.session_timeout_minutes:
                    # Continue current session
                    current_sessions[user_id]['items'].append(item_id)
                    current_sessions[user_id]['last_timestamp'] = timestamp
                else:
                    # End current session and start new one
                    if len(current_sessions[user_id]['items']) >= 2:
                        self.user_sessions[user_id].append(
                            current_sessions[user_id]['items']
                        )
                    
                    current_sessions[user_id] = {
                        'items': [item_id],
                        'last_timestamp': timestamp
                    }
            else:
                # Start first session for user
                current_sessions[user_id] = {
                    'items': [item_id],
                    'last_timestamp': timestamp
                }
        
        # Add remaining sessions
        for user_id, session_data in current_sessions.items():
            if len(session_data['items']) >= 2:
                self.user_sessions[user_id].append(session_data['items'])
        
        logger.info(f"Extracted {sum(len(sessions) for sessions in self.user_sessions.values())} sessions")
    
    def _mine_frequent_sequences(self):
        """Mine frequent sequential patterns"""
        # Count all sequences of different lengths
        sequence_counts = defaultdict(int)
        total_sequences = 0
        
        # Count sequences from all user sessions
        for user_sessions in self.user_sessions.values():
            for session in user_sessions:
                # Generate all subsequences
                for length in range(2, min(len(session) + 1, self.max_sequence_length + 1)):
                    for start in range(len(session) - length + 1):
                        sequence = tuple(session[start:start + length])
                        sequence_counts[sequence] += 1
                        total_sequences += 1
        
        # Filter by minimum support
        min_count = self.min_support * total_sequences
        
        self.frequent_sequences = {
            sequence: count for sequence, count in sequence_counts.items()
            if count >= min_count
        }
        
        logger.info(f"Found {len(self.frequent_sequences)} frequent sequences")
    
    def _build_transition_matrix(self):
        """Build item-to-item transition matrix"""
        transition_counts = defaultdict(lambda: defaultdict(int))
        item_counts = defaultdict(int)
        
        # Count transitions from all sessions
        for user_sessions in self.user_sessions.values():
            for session in user_sessions:
                for i in range(len(session) - 1):
                    current_item = session[i]
                    next_item = session[i + 1]
                    
                    transition_counts[current_item][next_item] += 1
                    item_counts[current_item] += 1
        
        # Convert counts to probabilities
        for current_item, next_items in transition_counts.items():
            total_transitions = item_counts[current_item]
            
            for next_item, count in next_items.items():
                probability = count / total_transitions
                self.transition_matrix[current_item][next_item] = probability
        
        logger.info(f"Built transition matrix with {len(self.transition_matrix)} items")
    
    def _generate_sequential_rules(self):
        """Generate sequential association rules"""
        self.sequential_rules = []
        
        # Generate rules from frequent sequences
        for sequence, support_count in self.frequent_sequences.items():
            if len(sequence) < 2:
                continue
            
            # Try different splits of the sequence into antecedent and consequent
            for split_point in range(1, len(sequence)):
                antecedent = sequence[:split_point]
                consequent = sequence[split_point:]
                
                # Calculate confidence
                antecedent_count = sum(
                    1 for seq in self.frequent_sequences.keys()
                    if len(seq) >= len(antecedent) and seq[:len(antecedent)] == antecedent
                )
                
                if antecedent_count > 0:
                    confidence = support_count / antecedent_count
                    
                    if confidence >= self.min_confidence:
                        rule = {
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support_count,
                            'confidence': confidence,
                            'length': len(sequence)
                        }
                        self.sequential_rules.append(rule)
        
        # Sort rules by confidence and support
        self.sequential_rules.sort(
            key=lambda x: (x['confidence'], x['support']), 
            reverse=True
        )
        
        logger.info(f"Generated {len(self.sequential_rules)} sequential rules")
    
    def predict_next_items(self, 
                          user_id: str,
                          current_sequence: List[str],
                          n_predictions: int = 5,
                          method: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Predict next items based on current sequence
        
        Args:
            user_id: User identifier
            current_sequence: Current sequence of items
            n_predictions: Number of predictions to return
            method: Prediction method ('markov', 'rules', 'hybrid')
            
        Returns:
            List of next item predictions with probabilities
        """
        if not current_sequence:
            return []
        
        predictions = {}
        
        if method in ['markov', 'hybrid']:
            # Markov chain prediction
            markov_predictions = self._predict_markov(current_sequence)
            for item, prob in markov_predictions.items():
                predictions[item] = predictions.get(item, 0) + prob * 0.6
        
        if method in ['rules', 'hybrid']:
            # Rule-based prediction
            rule_predictions = self._predict_rules(current_sequence)
            for item, prob in rule_predictions.items():
                predictions[item] = predictions.get(item, 0) + prob * 0.4
        
        # Convert to list and sort
        prediction_list = [
            {
                'item_id': item_id,
                'probability': prob,
                'confidence': min(prob * 2, 1.0),  # Simple confidence estimate
                'method': method
            }
            for item_id, prob in predictions.items()
        ]
        
        prediction_list.sort(key=lambda x: x['probability'], reverse=True)
        
        return prediction_list[:n_predictions]
    
    def _predict_markov(self, current_sequence: List[str]) -> Dict[str, float]:
        """Predict next items using Markov chain"""
        predictions = defaultdict(float)
        
        # Use last item for first-order Markov chain
        if current_sequence:
            last_item = current_sequence[-1]
            
            if last_item in self.transition_matrix:
                for next_item, prob in self.transition_matrix[last_item].items():
                    predictions[next_item] += prob
        
        # Use last two items for second-order if available
        if len(current_sequence) >= 2:
            last_two = tuple(current_sequence[-2:])
            
            # Find sequences that start with last_two
            for sequence, count in self.frequent_sequences.items():
                if (len(sequence) >= 3 and 
                    sequence[:2] == last_two):
                    next_item = sequence[2]
                    # Weight by sequence frequency
                    weight = count / sum(self.frequent_sequences.values())
                    predictions[next_item] += weight * 0.5
        
        return dict(predictions)
    
    def _predict_rules(self, current_sequence: List[str]) -> Dict[str, float]:
        """Predict next items using sequential rules"""
        predictions = defaultdict(float)
        
        # Find matching rules
        for rule in self.sequential_rules:
            antecedent = rule['antecedent']
            consequent = rule['consequent']
            confidence = rule['confidence']
            
            # Check if current sequence ends with the antecedent
            if (len(current_sequence) >= len(antecedent) and
                tuple(current_sequence[-len(antecedent):]) == antecedent):
                
                # Add consequent items as predictions
                for item in consequent:
                    predictions[item] += confidence / len(consequent)
        
        return dict(predictions)
    
    def get_session_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Get session patterns for a specific user"""
        user_sessions = self.user_sessions.get(user_id, [])
        
        if not user_sessions:
            return []
        
        patterns = []
        
        # Analyze session lengths
        session_lengths = [len(session) for session in user_sessions]
        
        patterns.append({
            'pattern_type': 'session_length',
            'avg_length': np.mean(session_lengths),
            'std_length': np.std(session_lengths),
            'min_length': np.min(session_lengths),
            'max_length': np.max(session_lengths)
        })
        
        # Find common items in sessions
        all_items = []
        for session in user_sessions:
            all_items.extend(session)
        
        item_counts = Counter(all_items)
        common_items = item_counts.most_common(5)
        
        patterns.append({
            'pattern_type': 'common_items',
            'items': common_items
        })
        
        # Find common session starting items
        start_items = [session[0] for session in user_sessions if session]
        start_item_counts = Counter(start_items)
        
        patterns.append({
            'pattern_type': 'session_starters',
            'items': start_item_counts.most_common(3)
        })
        
        return patterns
    
    def _calculate_pattern_statistics(self):
        """Calculate pattern mining statistics"""
        total_sessions = sum(len(sessions) for sessions in self.user_sessions.values())
        total_users = len(self.user_sessions)
        
        # Calculate average session length
        all_session_lengths = []
        for user_sessions in self.user_sessions.values():
            all_session_lengths.extend([len(session) for session in user_sessions])
        
        self.pattern_stats = {
            'total_users': total_users,
            'total_sessions': total_sessions,
            'avg_sessions_per_user': total_sessions / max(total_users, 1),
            'avg_session_length': np.mean(all_session_lengths) if all_session_lengths else 0,
            'total_frequent_sequences': len(self.frequent_sequences),
            'total_sequential_rules': len(self.sequential_rules),
            'unique_items': len(set(
                item for user_sessions in self.user_sessions.values()
                for session in user_sessions
                for item in session
            ))
        }
    
    def evaluate_predictions(self, 
                           test_sessions: List[List[str]],
                           method: str = 'hybrid') -> Dict[str, float]:
        """
        Evaluate prediction accuracy on test sessions
        
        Args:
            test_sessions: List of test sessions
            method: Prediction method to evaluate
            
        Returns:
            Evaluation metrics
        """
        correct_predictions = 0
        total_predictions = 0
        
        for session in test_sessions:
            if len(session) < 2:
                continue
            
            # Use all but last item as input, predict last item
            input_sequence = session[:-1]
            true_next_item = session[-1]
            
            predictions = self.predict_next_items(
                user_id='test_user',
                current_sequence=input_sequence,
                n_predictions=5,
                method=method
            )
            
            # Check if true item is in top predictions
            predicted_items = [p['item_id'] for p in predictions]
            
            if true_next_item in predicted_items:
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def get_user_sequence_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users based on their sequences"""
        sessions1 = self.user_sessions.get(user1, [])
        sessions2 = self.user_sessions.get(user2, [])
        
        if not sessions1 or not sessions2:
            return 0.0
        
        # Get all bigrams from both users
        bigrams1 = set()
        bigrams2 = set()
        
        for session in sessions1:
            for i in range(len(session) - 1):
                bigrams1.add((session[i], session[i + 1]))
        
        for session in sessions2:
            for i in range(len(session) - 1):
                bigrams2.add((session[i], session[i + 1]))
        
        # Calculate Jaccard similarity
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / max(union, 1)
    
    def save_model(self, filepath: str):
        """Save sequential pattern mining model"""
        model_data = {
            'max_sequence_length': self.max_sequence_length,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'session_timeout_minutes': self.session_timeout_minutes,
            'frequent_sequences': dict(self.frequent_sequences),
            'transition_matrix': {
                k: dict(v) for k, v in self.transition_matrix.items()
            },
            'sequential_rules': self.sequential_rules,
            'user_sessions': dict(self.user_sessions),
            'pattern_stats': self.pattern_stats
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Sequential pattern mining model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load sequential pattern mining model"""
        model_data = joblib.load(filepath)
        
        self.max_sequence_length = model_data['max_sequence_length']
        self.min_support = model_data['min_support']
        self.min_confidence = model_data['min_confidence']
        self.session_timeout_minutes = model_data['session_timeout_minutes']
        
        self.frequent_sequences = model_data['frequent_sequences']
        self.transition_matrix = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in model_data['transition_matrix'].items()}
        )
        self.sequential_rules = model_data['sequential_rules']
        self.user_sessions = defaultdict(list, model_data['user_sessions'])
        self.pattern_stats = model_data['pattern_stats']
        
        logger.info(f"Sequential pattern mining model loaded from {filepath}")
