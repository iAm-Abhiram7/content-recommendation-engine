"""
Preference Drift Detection Module

This module implements algorithms to detect changes in user preferences over time,
enabling the recommendation system to adapt to evolving user tastes and interests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PreferenceDriftDetector:
    """Detects preference drift in user behavior patterns."""
    
    def __init__(self, 
                 time_window_days: int = 30,
                 drift_threshold: float = 0.3,
                 min_interactions: int = 5):
        """
        Initialize drift detector.
        
        Args:
            time_window_days: Size of time windows for comparison
            drift_threshold: Threshold for detecting significant drift
            min_interactions: Minimum interactions required for drift detection
        """
        self.time_window_days = time_window_days
        self.drift_threshold = drift_threshold
        self.min_interactions = min_interactions
        self.user_profiles = {}
        self.drift_history = {}
        
    def update_user_profile(self, user_id: str, interactions: List[Dict[str, Any]]):
        """Update user profile with new interactions."""
        try:
            if not interactions:
                return
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(interactions)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Store in user profiles
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = []
            
            self.user_profiles[user_id].extend(interactions)
            
            # Keep only recent interactions (last 6 months)
            cutoff_date = datetime.now() - timedelta(days=180)
            self.user_profiles[user_id] = [
                interaction for interaction in self.user_profiles[user_id]
                if pd.to_datetime(interaction.get('timestamp', datetime.now())) >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error updating user profile for {user_id}: {e}")
    
    def detect_preference_drift(self, user_id: str) -> Dict[str, Any]:
        """
        Detect preference drift for a user.
        
        Returns:
            Dict containing drift detection results
        """
        try:
            if user_id not in self.user_profiles:
                return {"drift_detected": False, "confidence": 0.0, "reason": "No profile found"}
            
            interactions = self.user_profiles[user_id]
            if len(interactions) < self.min_interactions * 2:
                return {"drift_detected": False, "confidence": 0.0, "reason": "Insufficient data"}
            
            # Analyze different types of drift
            temporal_drift = self._detect_temporal_drift(interactions)
            genre_drift = self._detect_genre_drift(interactions)
            rating_drift = self._detect_rating_drift(interactions)
            behavioral_drift = self._detect_behavioral_drift(interactions)
            
            # Combine drift signals
            drift_signals = [temporal_drift, genre_drift, rating_drift, behavioral_drift]
            avg_confidence = np.mean([signal["confidence"] for signal in drift_signals])
            max_confidence = max([signal["confidence"] for signal in drift_signals])
            
            drift_detected = max_confidence > self.drift_threshold
            
            result = {
                "drift_detected": drift_detected,
                "confidence": max_confidence,
                "average_confidence": avg_confidence,
                "signals": {
                    "temporal": temporal_drift,
                    "genre": genre_drift,
                    "rating": rating_drift,
                    "behavioral": behavioral_drift
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in drift history
            if user_id not in self.drift_history:
                self.drift_history[user_id] = []
            self.drift_history[user_id].append(result)
            
            # Keep only recent drift history
            self.drift_history[user_id] = self.drift_history[user_id][-20:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting drift for user {user_id}: {e}")
            return {"drift_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _detect_temporal_drift(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect drift based on temporal patterns."""
        try:
            df = pd.DataFrame(interactions)
            if 'timestamp' not in df.columns:
                return {"confidence": 0.0, "type": "temporal", "details": "No timestamps"}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Split into time windows
            now = datetime.now()
            window1_start = now - timedelta(days=self.time_window_days * 2)
            window2_start = now - timedelta(days=self.time_window_days)
            
            window1 = df[(df['timestamp'] >= window1_start) & (df['timestamp'] < window2_start)]
            window2 = df[df['timestamp'] >= window2_start]
            
            if len(window1) < self.min_interactions or len(window2) < self.min_interactions:
                return {"confidence": 0.0, "type": "temporal", "details": "Insufficient data in windows"}
            
            # Compare activity patterns
            window1_hourly = window1['timestamp'].dt.hour.value_counts(normalize=True)
            window2_hourly = window2['timestamp'].dt.hour.value_counts(normalize=True)
            
            # Calculate distribution similarity
            hours = list(range(24))
            dist1 = [window1_hourly.get(h, 0) for h in hours]
            dist2 = [window2_hourly.get(h, 0) for h in hours]
            
            # Use KL divergence to measure difference
            kl_div = stats.entropy(dist1 + [1e-8], dist2 + [1e-8])
            confidence = min(kl_div / 2.0, 1.0)  # Normalize to [0,1]
            
            return {
                "confidence": confidence,
                "type": "temporal",
                "details": {
                    "kl_divergence": kl_div,
                    "window1_peak_hour": window1_hourly.idxmax() if not window1_hourly.empty else None,
                    "window2_peak_hour": window2_hourly.idxmax() if not window2_hourly.empty else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in temporal drift detection: {e}")
            return {"confidence": 0.0, "type": "temporal", "error": str(e)}
    
    def _detect_genre_drift(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect drift in genre preferences."""
        try:
            df = pd.DataFrame(interactions)
            if 'genres' not in df.columns and 'genre' not in df.columns:
                return {"confidence": 0.0, "type": "genre", "details": "No genre information"}
            
            genre_col = 'genres' if 'genres' in df.columns else 'genre'
            df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.now()))
            df = df.sort_values('timestamp')
            
            # Split into time windows
            mid_point = len(df) // 2
            window1 = df.iloc[:mid_point]
            window2 = df.iloc[mid_point:]
            
            if len(window1) < 3 or len(window2) < 3:
                return {"confidence": 0.0, "type": "genre", "details": "Insufficient data"}
            
            # Extract genre preferences
            def extract_genres(genre_data):
                genres = []
                for item in genre_data:
                    if isinstance(item, str):
                        if '|' in item:
                            genres.extend(item.split('|'))
                        else:
                            genres.append(item)
                    elif isinstance(item, list):
                        genres.extend(item)
                return genres
            
            genres1 = extract_genres(window1[genre_col].dropna())
            genres2 = extract_genres(window2[genre_col].dropna())
            
            # Calculate genre distributions
            from collections import Counter
            dist1 = Counter(genres1)
            dist2 = Counter(genres2)
            
            # Normalize distributions
            total1 = sum(dist1.values())
            total2 = sum(dist2.values())
            if total1 == 0 or total2 == 0:
                return {"confidence": 0.0, "type": "genre", "details": "Empty distributions"}
            
            dist1_norm = {k: v/total1 for k, v in dist1.items()}
            dist2_norm = {k: v/total2 for k, v in dist2.items()}
            
            # Calculate Jensen-Shannon divergence
            all_genres = set(dist1_norm.keys()) | set(dist2_norm.keys())
            p = np.array([dist1_norm.get(g, 0) for g in all_genres])
            q = np.array([dist2_norm.get(g, 0) for g in all_genres])
            
            m = 0.5 * (p + q)
            js_div = 0.5 * stats.entropy(p + 1e-8, m + 1e-8) + 0.5 * stats.entropy(q + 1e-8, m + 1e-8)
            confidence = min(js_div, 1.0)
            
            return {
                "confidence": confidence,
                "type": "genre",
                "details": {
                    "js_divergence": js_div,
                    "top_genres_window1": dict(Counter(dist1_norm).most_common(3)),
                    "top_genres_window2": dict(Counter(dist2_norm).most_common(3))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in genre drift detection: {e}")
            return {"confidence": 0.0, "type": "genre", "error": str(e)}
    
    def _detect_rating_drift(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect drift in rating patterns."""
        try:
            df = pd.DataFrame(interactions)
            if 'rating' not in df.columns:
                return {"confidence": 0.0, "type": "rating", "details": "No rating information"}
            
            df = df.dropna(subset=['rating'])
            if len(df) < self.min_interactions * 2:
                return {"confidence": 0.0, "type": "rating", "details": "Insufficient ratings"}
            
            # Split into time windows
            df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.now()))
            df = df.sort_values('timestamp')
            
            mid_point = len(df) // 2
            ratings1 = df.iloc[:mid_point]['rating'].values
            ratings2 = df.iloc[mid_point:]['rating'].values
            
            # Statistical tests for distribution change
            ks_stat, ks_pvalue = stats.ks_2samp(ratings1, ratings2)
            
            # Mean and variance change
            mean1, mean2 = np.mean(ratings1), np.mean(ratings2)
            var1, var2 = np.var(ratings1), np.var(ratings2)
            
            mean_change = abs(mean2 - mean1) / (max(abs(mean1), abs(mean2)) + 1e-8)
            var_change = abs(var2 - var1) / (max(var1, var2) + 1e-8)
            
            # Combined confidence
            confidence = max(ks_stat, mean_change, var_change)
            
            return {
                "confidence": min(confidence, 1.0),
                "type": "rating",
                "details": {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "mean_change": mean_change,
                    "variance_change": var_change,
                    "mean1": mean1,
                    "mean2": mean2
                }
            }
            
        except Exception as e:
            logger.error(f"Error in rating drift detection: {e}")
            return {"confidence": 0.0, "type": "rating", "error": str(e)}
    
    def _detect_behavioral_drift(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect drift in behavioral patterns."""
        try:
            df = pd.DataFrame(interactions)
            df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.now()))
            df = df.sort_values('timestamp')
            
            if len(df) < 10:
                return {"confidence": 0.0, "type": "behavioral", "details": "Insufficient data"}
            
            # Calculate session patterns
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Split into windows
            mid_point = len(df) // 2
            window1 = df.iloc[:mid_point]
            window2 = df.iloc[mid_point:]
            
            # Compare session lengths (time between interactions)
            session1_lengths = window1['time_diff'].values[1:]  # Skip first NaN
            session2_lengths = window2['time_diff'].values[1:]
            
            if len(session1_lengths) < 3 or len(session2_lengths) < 3:
                return {"confidence": 0.0, "type": "behavioral", "details": "Insufficient session data"}
            
            # Statistical test for session length change
            try:
                ks_stat, _ = stats.ks_2samp(session1_lengths, session2_lengths)
            except:
                ks_stat = 0.0
            
            # Compare interaction frequency patterns
            freq1 = len(window1) / ((window1['timestamp'].max() - window1['timestamp'].min()).days + 1)
            freq2 = len(window2) / ((window2['timestamp'].max() - window2['timestamp'].min()).days + 1)
            
            freq_change = abs(freq2 - freq1) / (max(freq1, freq2) + 1e-8)
            
            confidence = max(ks_stat, freq_change)
            
            return {
                "confidence": min(confidence, 1.0),
                "type": "behavioral",
                "details": {
                    "session_ks_stat": ks_stat,
                    "frequency_change": freq_change,
                    "freq1": freq1,
                    "freq2": freq2,
                    "avg_session1": np.mean(session1_lengths),
                    "avg_session2": np.mean(session2_lengths)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in behavioral drift detection: {e}")
            return {"confidence": 0.0, "type": "behavioral", "error": str(e)}
    
    def get_drift_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get drift detection history for a user."""
        if user_id not in self.drift_history:
            return []
        
        return self.drift_history[user_id][-limit:]
    
    def get_drift_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze drift trends for a user."""
        try:
            if user_id not in self.drift_history:
                return {"trend": "no_data", "confidence_trend": []}
            
            history = self.drift_history[user_id]
            if len(history) < 3:
                return {"trend": "insufficient_data", "confidence_trend": []}
            
            # Extract confidence scores over time
            confidences = [h["confidence"] for h in history]
            timestamps = [h["timestamp"] for h in history]
            
            # Calculate trend
            if len(confidences) >= 3:
                x = np.arange(len(confidences))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, confidences)
                
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
                slope = 0.0
                r_value = 0.0
            
            return {
                "trend": trend,
                "slope": slope,
                "correlation": r_value,
                "confidence_trend": list(zip(timestamps, confidences)),
                "recent_avg": np.mean(confidences[-5:]) if len(confidences) >= 5 else np.mean(confidences)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing drift trends for {user_id}: {e}")
            return {"trend": "error", "error": str(e)}
    
    def reset_user_drift(self, user_id: str):
        """Reset drift detection for a user (useful after model retraining)."""
        if user_id in self.drift_history:
            self.drift_history[user_id] = []
        logger.info(f"Reset drift detection for user {user_id}")
    
    def get_system_drift_stats(self) -> Dict[str, Any]:
        """Get system-wide drift statistics."""
        try:
            if not self.drift_history:
                return {"total_users": 0, "avg_drift": 0.0}
            
            total_users = len(self.drift_history)
            recent_drifts = []
            
            for user_id, history in self.drift_history.items():
                if history:
                    recent_drifts.append(history[-1]["confidence"])
            
            avg_drift = np.mean(recent_drifts) if recent_drifts else 0.0
            high_drift_users = sum(1 for d in recent_drifts if d > self.drift_threshold)
            
            return {
                "total_users": total_users,
                "users_with_recent_drift": len(recent_drifts),
                "avg_drift_confidence": avg_drift,
                "high_drift_users": high_drift_users,
                "high_drift_percentage": high_drift_users / len(recent_drifts) * 100 if recent_drifts else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting system drift stats: {e}")
            return {"error": str(e)}
