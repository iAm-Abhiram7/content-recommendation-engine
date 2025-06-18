"""
Advanced Drift Detection and Adaptation Engine

This module implements sophisticated drift detection mechanisms:
- Statistical drift detection (ADWIN, Page-Hinkley, Kolmogorov-Smirnov)
- Behavioral pattern analysis for preference changes
- Temporal preference evolution tracking
- Multi-timescale preference modeling with confidence scoring
- Adaptive response strategies for different drift types
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import asyncio
import threading
import math
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftDetectionMethod(Enum):
    """Methods for drift detection"""
    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"
    KS_TEST = "ks_test"
    ENSEMBLE = "ensemble"


class DriftType(Enum):
    """Types of drift that can be detected"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    SEASONAL = "seasonal"
    CONCEPT = "concept"
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    drift_detected: bool
    drift_type: DriftType
    confidence: float
    severity: float
    timestamp: datetime
    details: Dict[str, Any]
    recommendation: str
    affected_preferences: List[str] = None


@dataclass
class PreferenceSnapshot:
    """Snapshot of user preferences at a point in time"""
    user_id: str
    timestamp: datetime
    preferences: Dict[str, float]
    confidence: float
    interaction_count: int
    context: Optional[Dict[str, Any]] = None


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection algorithms"""
    adwin_delta: float = 0.002
    page_hinkley_threshold: float = 50.0
    page_hinkley_alpha: float = 0.9999
    min_window_size: int = 30
    confidence_threshold: float = 0.7
    severity_threshold: float = 0.5
    adaptation_lag_hours: int = 24
    max_snapshots: int = 100
    seasonal_period_days: int = 7


class ADWINDetector:
    """Adaptive Windowing drift detector implementation"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        self.window.append(value)
        self.total += value
        self.width += 1
        
        # Update variance (Welford's online algorithm)
        if self.width == 1:
            self.variance = 0.0
        else:
            old_mean = (self.total - value) / (self.width - 1)
            new_mean = self.total / self.width
            self.variance = ((self.width - 2) * self.variance + 
                           (value - old_mean) * (value - new_mean)) / (self.width - 1)
        
        # Check for drift
        return self._detect_change()
    
    def _detect_change(self) -> bool:
        """Detect if change occurred using ADWIN algorithm"""
        if self.width < 2:
            return False
            
        # Simplified ADWIN implementation
        # Check if there's significant difference between sub-windows
        mid_point = self.width // 2
        if mid_point < 5:  # Need minimum window size
            return False
            
        window_list = list(self.window)
        left_window = window_list[:mid_point]
        right_window = window_list[mid_point:]
        
        left_mean = np.mean(left_window)
        right_mean = np.mean(right_window)
        
        # Calculate cut threshold based on variance
        if self.variance > 0:
            cut_threshold = np.sqrt(2 * self.variance * np.log(2 / self.delta) / mid_point)
            return abs(left_mean - right_mean) > cut_threshold
        
        return False
    
    def reset(self):
        """Reset the detector"""
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0


class PageHinkleyDetector:
    """Page-Hinkley test for gradual drift detection"""
    
    def __init__(self, threshold: float = 50.0, alpha: float = 0.9999):
        self.threshold = threshold
        self.alpha = alpha
        self.cumsum = 0.0
        self.mean = 0.0
        self.n = 0
        
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        self.n += 1
        
        # Update running mean
        old_mean = self.mean
        self.mean = old_mean + (value - old_mean) / self.n
        
        # Update cumulative sum
        self.cumsum = self.alpha * self.cumsum + (value - old_mean - self.alpha * (self.mean - old_mean))
        
        # Check threshold
        return abs(self.cumsum) > self.threshold
    
    def reset(self):
        """Reset the detector"""
        self.cumsum = 0.0
        self.mean = 0.0
        self.n = 0


class AdaptiveDriftDetector:
    """
    Advanced drift detection system with multiple algorithms and adaptation strategies
    """
    
    def __init__(self, config: DriftDetectionConfig = None):
        self.config = config or DriftDetectionConfig()
        
        # Drift detectors for different signals
        self.detectors = {
            'rating_adwin': ADWINDetector(self.config.adwin_delta),
            'rating_page_hinkley': PageHinkleyDetector(
                self.config.page_hinkley_threshold, 
                self.config.page_hinkley_alpha
            ),
            'engagement_adwin': ADWINDetector(self.config.adwin_delta),
            'preference_adwin': ADWINDetector(self.config.adwin_delta)
        }
        
        # User preference snapshots for trend analysis
        self.user_snapshots: Dict[str, List[PreferenceSnapshot]] = defaultdict(list)
        
        # Drift history and statistics
        self.drift_history: Dict[str, List[DriftDetectionResult]] = defaultdict(list)
        self.drift_statistics = {
            'total_drifts_detected': 0,
            'drift_types_count': defaultdict(int),
            'users_with_drift': set(),
            'avg_confidence': 0.0
        }
        
        # Behavioral pattern tracking
        self.behavior_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Threading for background analysis
        self.analysis_thread = None
        self.stop_analysis = threading.Event()
        self.is_running = False
        
        self.start_background_analysis()
    
    def start_background_analysis(self):
        """Start background drift analysis"""
        if not self.is_running:
            self.is_running = True
            self.stop_analysis.clear()
            self.analysis_thread = threading.Thread(target=self._background_analyzer)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            logger.info("Started background drift analysis")
    
    def stop_background_analysis(self):
        """Stop background drift analysis"""
        self.is_running = False
        self.stop_analysis.set()
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        logger.info("Stopped background drift analysis")
    
    def _background_analyzer(self):
        """Background thread for continuous drift analysis"""
        while not self.stop_analysis.wait(timeout=3600):  # Analyze every hour
            try:
                self._analyze_seasonal_patterns()
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in background analysis: {e}")
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, float],
                                    interaction_count: int = 1,
                                    context: Optional[Dict[str, Any]] = None) -> Optional[DriftDetectionResult]:
        """
        Update user preferences and detect drift
        
        Args:
            user_id: User identifier
            preferences: Current user preferences
            interaction_count: Number of interactions in this update
            context: Additional context information
            
        Returns:
            Drift detection result if drift detected
        """
        try:
            current_time = datetime.now()
            
            # Create preference snapshot
            snapshot = PreferenceSnapshot(
                user_id=user_id,
                timestamp=current_time,
                preferences=preferences.copy(),
                confidence=min(1.0, interaction_count / 10),  # Confidence based on interactions
                interaction_count=interaction_count,
                context=context
            )
            
            # Add to user snapshots
            self.user_snapshots[user_id].append(snapshot)
            
            # Maintain snapshot limit
            if len(self.user_snapshots[user_id]) > self.config.max_snapshots:
                self.user_snapshots[user_id] = self.user_snapshots[user_id][-self.config.max_snapshots:]
            
            # Check for drift if we have enough history
            if len(self.user_snapshots[user_id]) >= self.config.min_window_size:
                drift_result = await self._detect_preference_drift(user_id)
                
                if drift_result and drift_result.drift_detected:
                    # Store drift result
                    self.drift_history[user_id].append(drift_result)
                    
                    # Update statistics
                    self._update_drift_statistics(drift_result)
                    
                    logger.info(f"Drift detected for user {user_id}: {drift_result.drift_type.value} "
                              f"(confidence: {drift_result.confidence:.3f})")
                    
                    return drift_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating user preferences for drift detection: {e}")
            return None
    
    async def _detect_preference_drift(self, user_id: str) -> Optional[DriftDetectionResult]:
        """Detect preference drift for a specific user"""
        try:
            snapshots = self.user_snapshots[user_id]
            if len(snapshots) < self.config.min_window_size:
                return None
            
            # Analyze different types of drift
            drift_results = []
            
            # 1. Statistical drift detection
            statistical_drift = await self._detect_statistical_drift(snapshots)
            if statistical_drift:
                drift_results.append(statistical_drift)
            
            # 2. Behavioral pattern drift
            behavioral_drift = await self._detect_behavioral_drift(user_id, snapshots)
            if behavioral_drift:
                drift_results.append(behavioral_drift)
            
            # 3. Temporal preference evolution
            temporal_drift = await self._detect_temporal_drift(snapshots)
            if temporal_drift:
                drift_results.append(temporal_drift)
            
            # 4. Preference confidence drift
            confidence_drift = await self._detect_confidence_drift(snapshots)
            if confidence_drift:
                drift_results.append(confidence_drift)
            
            # Select most significant drift
            if drift_results:
                # Sort by confidence and severity
                drift_results.sort(key=lambda x: x.confidence * x.severity, reverse=True)
                return drift_results[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting preference drift: {e}")
            return None
    
    async def _detect_statistical_drift(self, snapshots: List[PreferenceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift using statistical methods"""
        try:
            if len(snapshots) < self.config.min_window_size:
                return None
            
            # Extract preference vectors
            recent_prefs = [s.preferences for s in snapshots[-15:]]  # Last 15 snapshots
            older_prefs = [s.preferences for s in snapshots[-30:-15]]  # Previous 15 snapshots
            
            if len(older_prefs) < 10 or len(recent_prefs) < 10:
                return None
            
            # Calculate preference stability using cosine similarity
            similarities = []
            
            # Get all preference keys
            all_keys = set()
            for prefs in recent_prefs + older_prefs:
                all_keys.update(prefs.keys())
            
            # Convert to vectors
            older_vectors = []
            recent_vectors = []
            
            for prefs in older_prefs:
                vector = [prefs.get(key, 0.0) for key in all_keys]
                older_vectors.append(vector)
            
            for prefs in recent_prefs:
                vector = [prefs.get(key, 0.0) for key in all_keys]
                recent_vectors.append(vector)
            
            # Calculate average vectors
            older_avg = np.mean(older_vectors, axis=0)
            recent_avg = np.mean(recent_vectors, axis=0)
            
            # Calculate similarity between periods
            similarity = cosine_similarity([older_avg], [recent_avg])[0, 0]
            drift_magnitude = 1.0 - similarity
            
            # Use ADWIN for rating-based drift detection
            rating_values = [np.mean(list(s.preferences.values())) for s in snapshots[-20:]]
            adwin_drift = False
            
            for rating in rating_values:
                if self.detectors['rating_adwin'].add_element(rating):
                    adwin_drift = True
                    break
            
            # Kolmogorov-Smirnov test for distribution changes
            older_ratings = [np.mean(list(s.preferences.values())) for s in snapshots[-30:-15]]
            recent_ratings = [np.mean(list(s.preferences.values())) for s in snapshots[-15:]]
            
            if len(older_ratings) >= 5 and len(recent_ratings) >= 5:
                ks_stat, ks_pvalue = stats.ks_2samp(older_ratings, recent_ratings)
                ks_drift = ks_pvalue < 0.05  # Significant change
            else:
                ks_drift = False
                ks_stat = 0.0
            
            # Determine if drift detected
            drift_detected = (drift_magnitude > 0.3 or adwin_drift or ks_drift)
            
            if drift_detected:
                # Determine drift type based on patterns
                if drift_magnitude > 0.6:
                    drift_type = DriftType.SUDDEN
                elif adwin_drift:
                    drift_type = DriftType.GRADUAL
                else:
                    drift_type = DriftType.CONCEPT
                
                confidence = min(1.0, drift_magnitude + (0.3 if adwin_drift else 0) + (0.3 if ks_drift else 0))
                
                return DriftDetectionResult(
                    drift_detected=True,
                    drift_type=drift_type,
                    confidence=confidence,
                    severity=drift_magnitude,
                    timestamp=datetime.now(),
                    details={
                        'cosine_similarity': similarity,
                        'drift_magnitude': drift_magnitude,
                        'adwin_drift': adwin_drift,
                        'ks_statistic': ks_stat,
                        'ks_drift': ks_drift
                    },
                    recommendation="Statistical drift detected - recommend model retraining"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in statistical drift detection: {e}")
            return None
    
    async def _detect_behavioral_drift(self, user_id: str, 
                                     snapshots: List[PreferenceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift in behavioral patterns"""
        try:
            # Analyze interaction patterns
            recent_interactions = [s.interaction_count for s in snapshots[-10:]]
            older_interactions = [s.interaction_count for s in snapshots[-20:-10]]
            
            if len(recent_interactions) < 5 or len(older_interactions) < 5:
                return None
            
            # Test for significant change in interaction frequency
            t_stat, t_pvalue = stats.ttest_ind(recent_interactions, older_interactions)
            interaction_drift = t_pvalue < 0.05
            
            # Analyze confidence patterns
            recent_confidence = [s.confidence for s in snapshots[-10:]]
            older_confidence = [s.confidence for s in snapshots[-20:-10]]
            
            conf_change = abs(np.mean(recent_confidence) - np.mean(older_confidence))
            confidence_drift = conf_change > 0.3
            
            # Analyze preference diversity
            recent_diversity = [len(s.preferences) for s in snapshots[-10:]]
            older_diversity = [len(s.preferences) for s in snapshots[-20:-10]]
            
            diversity_change = abs(np.mean(recent_diversity) - np.mean(older_diversity))
            diversity_drift = diversity_change > 2.0  # Significant change in preference breadth
            
            # Combine behavioral signals
            behavioral_drift = interaction_drift or confidence_drift or diversity_drift
            
            if behavioral_drift:
                confidence = (
                    (0.4 if interaction_drift else 0) +
                    (0.4 if confidence_drift else 0) +
                    (0.2 if diversity_drift else 0)
                )
                
                severity = max(
                    abs(t_stat) / 10.0 if interaction_drift else 0,
                    conf_change,
                    diversity_change / 5.0
                )
                
                return DriftDetectionResult(
                    drift_detected=True,
                    drift_type=DriftType.BEHAVIORAL,
                    confidence=min(1.0, confidence),
                    severity=min(1.0, severity),
                    timestamp=datetime.now(),
                    details={
                        'interaction_drift': interaction_drift,
                        'confidence_drift': confidence_drift,
                        'diversity_drift': diversity_drift,
                        't_statistic': t_stat,
                        'confidence_change': conf_change,
                        'diversity_change': diversity_change
                    },
                    recommendation="Behavioral drift detected - adjust recommendation strategy"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in behavioral drift detection: {e}")
            return None
    
    async def _detect_temporal_drift(self, snapshots: List[PreferenceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect temporal preference evolution patterns"""
        try:
            if len(snapshots) < 20:
                return None
            
            # Analyze preference trends over time
            timestamps = [s.timestamp for s in snapshots]
            
            # Calculate preference changes over time for each category
            preference_trends = defaultdict(list)
            
            all_keys = set()
            for s in snapshots:
                all_keys.update(s.preferences.keys())
            
            for key in all_keys:
                values = []
                for s in snapshots:
                    values.append(s.preferences.get(key, 0.0))
                
                if len(values) >= 10:
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    if np.std(values) > 0:  # Only if there's variance
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                        if abs(slope) > 0.01 and p_value < 0.1:  # Significant trend
                            preference_trends[key] = {
                                'slope': slope,
                                'r_value': r_value,
                                'p_value': p_value
                            }
            
            # Check for seasonal patterns (simplified)
            if len(snapshots) >= 28:  # At least 4 weeks of data
                seasonal_drift = self._detect_seasonal_patterns(snapshots)
            else:
                seasonal_drift = False
            
            # Determine if significant temporal drift
            significant_trends = len([t for t in preference_trends.values() 
                                   if abs(t['slope']) > 0.05 and t['p_value'] < 0.05])
            
            temporal_drift = significant_trends >= 2 or seasonal_drift
            
            if temporal_drift:
                confidence = min(1.0, significant_trends / len(all_keys) + (0.3 if seasonal_drift else 0))
                severity = np.mean([abs(t['slope']) for t in preference_trends.values()])
                
                drift_type = DriftType.SEASONAL if seasonal_drift else DriftType.GRADUAL
                
                return DriftDetectionResult(
                    drift_detected=True,
                    drift_type=drift_type,
                    confidence=confidence,
                    severity=min(1.0, severity),
                    timestamp=datetime.now(),
                    details={
                        'significant_trends': significant_trends,
                        'total_preferences': len(all_keys),
                        'seasonal_detected': seasonal_drift,
                        'preference_trends': dict(preference_trends)
                    },
                    recommendation="Temporal drift detected - implement time-aware recommendations",
                    affected_preferences=list(preference_trends.keys())
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in temporal drift detection: {e}")
            return None
    
    def _detect_seasonal_patterns(self, snapshots: List[PreferenceSnapshot]) -> bool:
        """Detect seasonal preference patterns"""
        try:
            # Simple seasonal pattern detection based on day of week
            weekday_prefs = defaultdict(list)
            
            for snapshot in snapshots:
                day_of_week = snapshot.timestamp.weekday()
                avg_pref = np.mean(list(snapshot.preferences.values()))
                weekday_prefs[day_of_week].append(avg_pref)
            
            # Check if there's significant variance across days
            daily_means = []
            for day in range(7):
                if day in weekday_prefs and len(weekday_prefs[day]) >= 3:
                    daily_means.append(np.mean(weekday_prefs[day]))
            
            if len(daily_means) >= 5:
                # Use coefficient of variation to detect seasonal patterns
                cv = np.std(daily_means) / np.mean(daily_means) if np.mean(daily_means) > 0 else 0
                return cv > 0.1  # 10% variation indicates seasonal pattern
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {e}")
            return False
    
    async def _detect_confidence_drift(self, snapshots: List[PreferenceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift in preference confidence levels"""
        try:
            confidence_values = [s.confidence for s in snapshots[-20:]]
            
            if len(confidence_values) < self.config.min_window_size:
                return None
            
            # Use Page-Hinkley test for confidence drift
            ph_drift = False
            for conf in confidence_values:
                if self.detectors['preference_adwin'].add_element(conf):
                    ph_drift = True
                    break
            
            # Check for significant confidence trend
            x = np.arange(len(confidence_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, confidence_values)
            
            significant_trend = abs(slope) > 0.01 and p_value < 0.05
            confidence_drift = ph_drift or significant_trend
            
            if confidence_drift:
                severity = abs(slope) * 10  # Scale slope to severity
                
                return DriftDetectionResult(
                    drift_detected=True,
                    drift_type=DriftType.PREFERENCE,
                    confidence=min(1.0, abs(r_value) + (0.3 if ph_drift else 0)),
                    severity=min(1.0, severity),
                    timestamp=datetime.now(),
                    details={
                        'confidence_trend_slope': slope,
                        'r_value': r_value,
                        'p_value': p_value,
                        'page_hinkley_drift': ph_drift
                    },
                    recommendation="Confidence drift detected - review data quality and user engagement"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in confidence drift detection: {e}")
            return None
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal patterns across all users (background task)"""
        try:
            # This would be a more comprehensive seasonal analysis
            # For now, just log that we're analyzing
            total_users = len(self.user_snapshots)
            if total_users > 0:
                logger.debug(f"Analyzing seasonal patterns for {total_users} users")
                
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old drift detection data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=90)  # Keep 90 days of data
            
            # Clean up snapshots
            for user_id in list(self.user_snapshots.keys()):
                snapshots = self.user_snapshots[user_id]
                self.user_snapshots[user_id] = [
                    s for s in snapshots if s.timestamp >= cutoff_time
                ]
                
                # Remove users with no recent data
                if not self.user_snapshots[user_id]:
                    del self.user_snapshots[user_id]
            
            # Clean up drift history
            for user_id in list(self.drift_history.keys()):
                history = self.drift_history[user_id]
                self.drift_history[user_id] = [
                    d for d in history if d.timestamp >= cutoff_time
                ]
                
                if not self.drift_history[user_id]:
                    del self.drift_history[user_id]
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def _update_drift_statistics(self, drift_result: DriftDetectionResult):
        """Update drift detection statistics"""
        try:
            self.drift_statistics['total_drifts_detected'] += 1
            self.drift_statistics['drift_types_count'][drift_result.drift_type.value] += 1
            self.drift_statistics['users_with_drift'].add(
                len([s for s in self.user_snapshots.keys() if s in self.drift_history])
            )
            
            # Update average confidence
            all_confidences = []
            for history in self.drift_history.values():
                all_confidences.extend([d.confidence for d in history])
            
            if all_confidences:
                self.drift_statistics['avg_confidence'] = np.mean(all_confidences)
                
        except Exception as e:
            logger.error(f"Error updating drift statistics: {e}")
    
    def get_user_drift_history(self, user_id: str, days_back: int = 30) -> List[DriftDetectionResult]:
        """Get drift detection history for a user"""
        try:
            if user_id not in self.drift_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_drifts = [
                d for d in self.drift_history[user_id]
                if d.timestamp >= cutoff_time
            ]
            
            return sorted(recent_drifts, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting user drift history: {e}")
            return []
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics"""
        try:
            return {
                'total_drifts_detected': self.drift_statistics['total_drifts_detected'],
                'drift_types': dict(self.drift_statistics['drift_types_count']),
                'users_monitored': len(self.user_snapshots),
                'users_with_drift': len(self.drift_statistics['users_with_drift']),
                'avg_drift_confidence': self.drift_statistics['avg_confidence'],
                'active_detectors': len(self.detectors),
                'recent_snapshots': sum(len(snapshots) for snapshots in self.user_snapshots.values()),
                'background_analysis_running': self.is_running
            }
            
        except Exception as e:
            logger.error(f"Error getting drift statistics: {e}")
            return {}
    
    def reset_user_drift_detection(self, user_id: str):
        """Reset drift detection for a specific user"""
        try:
            if user_id in self.user_snapshots:
                del self.user_snapshots[user_id]
            
            if user_id in self.drift_history:
                del self.drift_history[user_id]
            
            if user_id in self.behavior_patterns:
                del self.behavior_patterns[user_id]
            
            logger.info(f"Reset drift detection for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error resetting drift detection: {e}")
    
    def monitor_all_users(self) -> Dict[str, Any]:
        """Monitor drift for all users"""
        try:
            results = {
                'users_monitored': 0,
                'drift_detected': 0,
                'monitoring_results': {}
            }
            
            for user_id in self.user_snapshots.keys():
                drift_result = self.detect_preference_drift(user_id)
                results['monitoring_results'][user_id] = drift_result
                results['users_monitored'] += 1
                if drift_result.get('drift_detected', False):
                    results['drift_detected'] += 1
            
            return results
        except Exception as e:
            logger.error(f"Error monitoring all users: {e}")
            return {'error': str(e)}
    
    def analyze_user_drift(self, user_id: str, time_range: str = "7d") -> Dict[str, Any]:
        """Analyze drift for a specific user"""
        try:
            # Parse time range
            if time_range.endswith('d'):
                days = int(time_range[:-1])
            elif time_range.endswith('h'):
                days = int(time_range[:-1]) / 24
            else:
                days = 7  # default
            
            # Get drift history for the time period
            cutoff_time = datetime.now() - timedelta(days=days)
            user_drift_history = [
                drift for drift in self.drift_history.get(user_id, [])
                if drift.timestamp >= cutoff_time
            ]
            
            return {
                'user_id': user_id,
                'time_range': time_range,
                'drift_events': len(user_drift_history),
                'latest_drift': user_drift_history[-1].__dict__ if user_drift_history else None,
                'drift_pattern': self._analyze_drift_pattern(user_drift_history),
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing user drift: {e}")
            return {'error': str(e)}
    
    def _analyze_drift_pattern(self, drift_history: List) -> Dict[str, Any]:
        """Analyze pattern in drift history"""
        if not drift_history:
            return {'pattern': 'no_drift', 'frequency': 0}
        
        if len(drift_history) == 1:
            return {'pattern': 'single_drift', 'frequency': 1}
        
        # Calculate average time between drifts
        time_deltas = []
        for i in range(1, len(drift_history)):
            delta = drift_history[i].timestamp - drift_history[i-1].timestamp
            time_deltas.append(delta.total_seconds() / 3600)  # hours
        
        avg_hours = sum(time_deltas) / len(time_deltas)
        
        if avg_hours < 24:
            pattern = 'rapid_drift'
        elif avg_hours < 168:  # 1 week
            pattern = 'frequent_drift'
        else:
            pattern = 'gradual_drift'
        
        return {
            'pattern': pattern,
            'frequency': len(drift_history),
            'avg_hours_between_drifts': avg_hours
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_background_analysis()
        except:
            pass


# Alias for backward compatibility
DriftDetector = AdaptiveDriftDetector


# Alias for backward compatibility
DriftDetector = AdaptiveDriftDetector
