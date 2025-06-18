"""
Evolution Modeler for Content Recommendation Engine

This module models how user preferences evolve over time, detecting trends,
seasonal patterns, and long-term shifts in user behavior and interests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import scipy.stats as stats
from collections import defaultdict, deque
import pickle
import json

from ..utils.logging import setup_logger


class TrendType(Enum):
    """Types of preference trends"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    VOLATILE = "volatile"
    EMERGING = "emerging"
    DECLINING = "declining"


class SeasonalPattern(Enum):
    """Seasonal pattern types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class PreferenceTrend:
    """Preference trend information"""
    feature: str
    trend_type: TrendType
    strength: float  # 0-1
    confidence: float  # 0-1
    start_time: datetime
    last_updated: datetime
    slope: float
    r_squared: float
    p_value: float
    forecast: List[float]


@dataclass
class SeasonalComponent:
    """Seasonal component information"""
    pattern_type: SeasonalPattern
    amplitude: float
    phase: float
    period: float
    confidence: float
    last_occurrence: datetime


@dataclass
class EvolutionState:
    """Current evolution state"""
    stability_score: float  # 0-1, higher = more stable
    exploration_rate: float  # 0-1, higher = more exploratory
    preference_velocity: float  # Rate of change
    dominant_trends: List[str]
    emerging_interests: List[str]
    declining_interests: List[str]
    adaptation_speed: float  # How quickly user adapts to new content


class PreferenceEvolutionModeler:
    """
    Models how user preferences evolve over time using statistical
    and machine learning techniques
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any] = None):
        self.user_id = user_id
        self.config = config or {}
        
        # Configuration
        self.min_observations = self.config.get('min_observations', 10)
        self.trend_window = self.config.get('trend_window', timedelta(days=30))
        self.seasonal_window = self.config.get('seasonal_window', timedelta(days=90))
        self.significance_level = self.config.get('significance_level', 0.05)
        self.forecast_horizon = self.config.get('forecast_horizon', 7)
        
        # State
        self.preference_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.trends: Dict[str, PreferenceTrend] = {}
        self.seasonal_components: Dict[str, List[SeasonalComponent]] = defaultdict(list)
        self.evolution_state = EvolutionState(
            stability_score=0.5,
            exploration_rate=0.5,
            preference_velocity=0.0,
            dominant_trends=[],
            emerging_interests=[],
            declining_interests=[],
            adaptation_speed=0.5
        )
        
        # Models
        self.trend_models: Dict[str, Any] = {}
        self.seasonal_models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        
        # Caching
        self.last_analysis_time = None
        self.analysis_cache: Dict[str, Any] = {}
        
        self.logger = setup_logger(__name__)
    
    def update_preference(self, feature: str, value: float, 
                         timestamp: datetime = None):
        """Update preference value for a feature"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.preference_history[feature].append((timestamp, value))
        
        # Keep only recent data (memory efficiency)
        cutoff_time = timestamp - timedelta(days=365)  # Keep 1 year
        self.preference_history[feature] = [
            (ts, val) for ts, val in self.preference_history[feature]
            if ts > cutoff_time
        ]
        
        # Sort by timestamp
        self.preference_history[feature].sort(key=lambda x: x[0])
        
        # Trigger analysis if enough new data
        if len(self.preference_history[feature]) % 10 == 0:
            self._analyze_evolution()
    
    def analyze_trends(self, features: List[str] = None) -> Dict[str, PreferenceTrend]:
        """Analyze preference trends for specified features"""
        if features is None:
            features = list(self.preference_history.keys())
        
        trends = {}
        current_time = datetime.now()
        
        for feature in features:
            if len(self.preference_history[feature]) < self.min_observations:
                continue
            
            try:
                trend = self._analyze_feature_trend(feature, current_time)
                if trend:
                    trends[feature] = trend
                    self.trends[feature] = trend
                    
            except Exception as e:
                self.logger.error(f"Error analyzing trend for {feature}: {e}")
        
        return trends
    
    def detect_seasonal_patterns(self, features: List[str] = None) -> Dict[str, List[SeasonalComponent]]:
        """Detect seasonal patterns in preferences"""
        if features is None:
            features = list(self.preference_history.keys())
        
        seasonal_patterns = {}
        
        for feature in features:
            if len(self.preference_history[feature]) < 50:  # Need more data for seasonality
                continue
            
            try:
                patterns = self._detect_feature_seasonality(feature)
                if patterns:
                    seasonal_patterns[feature] = patterns
                    self.seasonal_components[feature] = patterns
                    
            except Exception as e:
                self.logger.error(f"Error detecting seasonality for {feature}: {e}")
        
        return seasonal_patterns
    
    def predict_future_preferences(self, feature: str, 
                                 horizon: int = None) -> List[Tuple[datetime, float, float]]:
        """Predict future preference values with confidence intervals"""
        if horizon is None:
            horizon = self.forecast_horizon
        
        if feature not in self.preference_history:
            return []
        
        try:
            return self._forecast_preference(feature, horizon)
        except Exception as e:
            self.logger.error(f"Error forecasting {feature}: {e}")
            return []
    
    def get_evolution_state(self) -> EvolutionState:
        """Get current evolution state"""
        self._update_evolution_state()
        return self.evolution_state
    
    def detect_preference_shifts(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """Detect significant shifts in preferences"""
        shifts = []
        
        for feature, history in self.preference_history.items():
            if len(history) < window_size * 2:
                continue
            
            try:
                shift_points = self._detect_change_points(feature, window_size)
                for shift_point in shift_points:
                    shifts.append({
                        'feature': feature,
                        'timestamp': shift_point['timestamp'],
                        'magnitude': shift_point['magnitude'],
                        'confidence': shift_point['confidence'],
                        'direction': shift_point['direction']
                    })
            except Exception as e:
                self.logger.error(f"Error detecting shifts for {feature}: {e}")
        
        return sorted(shifts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_preference_clusters(self, n_clusters: int = None) -> Dict[str, Any]:
        """Cluster preferences to identify patterns"""
        if len(self.preference_history) < 3:
            return {}
        
        try:
            # Prepare data matrix
            features = list(self.preference_history.keys())
            data_matrix = []
            timestamps = []
            
            # Get common time points
            all_timestamps = set()
            for history in self.preference_history.values():
                all_timestamps.update([ts for ts, _ in history])
            
            common_timestamps = sorted(list(all_timestamps))[-100:]  # Last 100 points
            
            for ts in common_timestamps:
                row = []
                valid_row = True
                
                for feature in features:
                    # Find closest value in time
                    closest_value = self._get_value_at_time(feature, ts)
                    if closest_value is not None:
                        row.append(closest_value)
                    else:
                        valid_row = False
                        break
                
                if valid_row:
                    data_matrix.append(row)
                    timestamps.append(ts)
            
            if len(data_matrix) < 5:
                return {}
            
            # Perform clustering
            data_array = np.array(data_matrix)
            
            # Use DBSCAN for automatic cluster detection
            clusterer = DBSCAN(eps=0.3, min_samples=3)
            cluster_labels = clusterer.fit_predict(data_array)
            
            # Analyze clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    clusters[label].append({
                        'timestamp': timestamps[i],
                        'preferences': {features[j]: data_matrix[i][j] 
                                     for j in range(len(features))}
                    })
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id, points in clusters.items():
                if len(points) < 2:
                    continue
                
                # Calculate centroid
                centroid = {}
                for feature in features:
                    values = [p['preferences'][feature] for p in points]
                    centroid[feature] = np.mean(values)
                
                # Calculate time span
                times = [p['timestamp'] for p in points]
                time_span = max(times) - min(times)
                
                cluster_stats[cluster_id] = {
                    'centroid': centroid,
                    'size': len(points),
                    'time_span': time_span.total_seconds() / 3600,  # hours
                    'recency': (datetime.now() - max(times)).total_seconds() / 3600,
                    'stability': np.std([p['preferences'][features[0]] for p in points])
                }
            
            return cluster_stats
            
        except Exception as e:
            self.logger.error(f"Error clustering preferences: {e}")
            return {}
    
    def _analyze_feature_trend(self, feature: str, current_time: datetime) -> Optional[PreferenceTrend]:
        """Analyze trend for a specific feature"""
        history = self.preference_history[feature]
        
        # Filter to trend window
        cutoff_time = current_time - self.trend_window
        recent_history = [(ts, val) for ts, val in history if ts > cutoff_time]
        
        if len(recent_history) < self.min_observations:
            return None
        
        # Prepare data for regression
        timestamps = [ts.timestamp() for ts, _ in recent_history]
        values = [val for _, val in recent_history]
        
        # Normalize timestamps to start from 0
        min_ts = min(timestamps)
        X = np.array([(ts - min_ts) / 3600 for ts in timestamps]).reshape(-1, 1)  # Hours
        y = np.array(values)
        
        # Fit robust regression
        model = HuberRegressor(epsilon=1.35)  # Robust to outliers
        model.fit(X, y)
        
        # Get predictions and calculate R²
        y_pred = model.predict(X)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Calculate p-value for slope significance
        # Simple t-test for slope
        slope = model.coef_[0]
        n = len(y)
        
        if n > 2:
            residuals = y - y_pred
            mse = np.sum(residuals ** 2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X.flatten())) ** 2))
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        # Determine trend type
        if p_value > self.significance_level:
            trend_type = TrendType.STABLE
        elif slope > 0:
            if r_squared > 0.7:
                trend_type = TrendType.INCREASING
            else:
                trend_type = TrendType.VOLATILE
        else:
            if r_squared > 0.7:
                trend_type = TrendType.DECREASING
            else:
                trend_type = TrendType.VOLATILE
        
        # Generate forecast
        forecast_x = np.arange(X[-1, 0] + 1, X[-1, 0] + self.forecast_horizon + 1).reshape(-1, 1)
        forecast = model.predict(forecast_x).tolist()
        
        return PreferenceTrend(
            feature=feature,
            trend_type=trend_type,
            strength=min(abs(slope), 1.0),
            confidence=max(0, r_squared),
            start_time=recent_history[0][0],
            last_updated=current_time,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            forecast=forecast
        )
    
    def _detect_feature_seasonality(self, feature: str) -> List[SeasonalComponent]:
        """Detect seasonal patterns for a feature"""
        history = self.preference_history[feature]
        
        if len(history) < 50:
            return []
        
        # Convert to time series
        timestamps = [ts.timestamp() for ts, _ in history]
        values = [val for _, val in history]
        
        # Test different seasonal periods
        periods_to_test = [
            (24 * 3600, SeasonalPattern.DAILY),      # 24 hours
            (7 * 24 * 3600, SeasonalPattern.WEEKLY), # 7 days
            (30 * 24 * 3600, SeasonalPattern.MONTHLY), # 30 days
            (90 * 24 * 3600, SeasonalPattern.QUARTERLY)  # 90 days
        ]
        
        seasonal_components = []
        
        for period_seconds, pattern_type in periods_to_test:
            try:
                # Check if we have enough data for this period
                time_span = timestamps[-1] - timestamps[0]
                if time_span < period_seconds * 2:
                    continue
                
                # Create sinusoidal features
                X = np.column_stack([
                    np.sin(2 * np.pi * np.array(timestamps) / period_seconds),
                    np.cos(2 * np.pi * np.array(timestamps) / period_seconds),
                    np.ones(len(timestamps))  # Intercept
                ])
                
                # Fit regression
                model = LinearRegression()
                model.fit(X, values)
                
                # Calculate amplitude and phase
                sin_coef = model.coef_[0]
                cos_coef = model.coef_[1]
                amplitude = np.sqrt(sin_coef ** 2 + cos_coef ** 2)
                phase = np.arctan2(sin_coef, cos_coef)
                
                # Calculate R²
                y_pred = model.predict(X)
                r_squared = 1 - np.sum((values - y_pred) ** 2) / np.sum((values - np.mean(values)) ** 2)
                
                # Only keep significant seasonal components
                if r_squared > 0.1 and amplitude > 0.01:
                    seasonal_components.append(SeasonalComponent(
                        pattern_type=pattern_type,
                        amplitude=amplitude,
                        phase=phase,
                        period=period_seconds,
                        confidence=r_squared,
                        last_occurrence=datetime.fromtimestamp(timestamps[-1])
                    ))
                    
            except Exception as e:
                self.logger.debug(f"Error testing {pattern_type.value} seasonality: {e}")
        
        return seasonal_components
    
    def _forecast_preference(self, feature: str, horizon: int) -> List[Tuple[datetime, float, float]]:
        """Forecast future preference values"""
        history = self.preference_history[feature]
        
        if len(history) < self.min_observations:
            return []
        
        # Get trend component
        trend = self.trends.get(feature)
        
        # Get seasonal components
        seasonal = self.seasonal_components.get(feature, [])
        
        # Base forecast on trend
        if trend and trend.forecast:
            base_forecast = trend.forecast[:horizon]
        else:
            # Use last value if no trend
            base_forecast = [history[-1][1]] * horizon
        
        # Add seasonal components
        last_timestamp = history[-1][0]
        forecasts = []
        
        for i in range(horizon):
            future_time = last_timestamp + timedelta(hours=i + 1)
            future_timestamp = future_time.timestamp()
            
            # Start with trend/base value
            forecast_value = base_forecast[i] if i < len(base_forecast) else base_forecast[-1]
            
            # Add seasonal components
            for component in seasonal:
                seasonal_value = component.amplitude * np.sin(
                    2 * np.pi * future_timestamp / component.period + component.phase
                )
                forecast_value += seasonal_value * component.confidence
            
            # Calculate confidence interval (simple approach)
            # In practice, this would be based on model residuals
            recent_values = [val for _, val in history[-20:]]
            std_dev = np.std(recent_values) if len(recent_values) > 1 else 0.1
            confidence_interval = 1.96 * std_dev  # 95% CI
            
            forecasts.append((future_time, forecast_value, confidence_interval))
        
        return forecasts
    
    def _detect_change_points(self, feature: str, window_size: int) -> List[Dict[str, Any]]:
        """Detect change points in preference time series"""
        history = self.preference_history[feature]
        
        if len(history) < window_size * 2:
            return []
        
        values = [val for _, val in history]
        timestamps = [ts for ts, _ in history]
        
        change_points = []
        
        # Sliding window approach
        for i in range(window_size, len(values) - window_size):
            before_window = values[i - window_size:i]
            after_window = values[i:i + window_size]
            
            # Statistical test for difference in means
            t_stat, p_value = stats.ttest_ind(before_window, after_window)
            
            if p_value < self.significance_level:
                # Calculate magnitude of change
                before_mean = np.mean(before_window)
                after_mean = np.mean(after_window)
                magnitude = abs(after_mean - before_mean)
                direction = "increase" if after_mean > before_mean else "decrease"
                
                change_points.append({
                    'timestamp': timestamps[i],
                    'magnitude': magnitude,
                    'confidence': 1 - p_value,
                    'direction': direction,
                    'before_mean': before_mean,
                    'after_mean': after_mean
                })
        
        return change_points
    
    def _get_value_at_time(self, feature: str, target_time: datetime, 
                          tolerance: timedelta = timedelta(hours=1)) -> Optional[float]:
        """Get preference value closest to target time"""
        history = self.preference_history[feature]
        
        closest_value = None
        min_diff = float('inf')
        
        for timestamp, value in history:
            diff = abs((timestamp - target_time).total_seconds())
            if diff < min_diff and diff <= tolerance.total_seconds():
                min_diff = diff
                closest_value = value
        
        return closest_value
    
    def _analyze_evolution(self):
        """Analyze overall preference evolution"""
        try:
            self.analyze_trends()
            self.detect_seasonal_patterns()
            self._update_evolution_state()
            self.last_analysis_time = datetime.now()
        except Exception as e:
            self.logger.error(f"Error in evolution analysis: {e}")
    
    def _update_evolution_state(self):
        """Update the current evolution state"""
        current_time = datetime.now()
        
        # Calculate stability score
        recent_changes = []
        for feature, history in self.preference_history.items():
            if len(history) > 10:
                recent_values = [val for _, val in history[-10:]]
                stability = 1 - min(np.std(recent_values), 1.0)  # Lower std = higher stability
                recent_changes.append(stability)
        
        if recent_changes:
            self.evolution_state.stability_score = np.mean(recent_changes)
        
        # Calculate exploration rate based on preference diversity
        if len(self.preference_history) > 1:
            current_preferences = []
            for feature, history in self.preference_history.items():
                if history:
                    current_preferences.append(history[-1][1])
            
            if current_preferences:
                # Higher std deviation = more exploration
                self.evolution_state.exploration_rate = min(np.std(current_preferences), 1.0)
        
        # Calculate preference velocity (rate of change)
        velocities = []
        for feature, trend in self.trends.items():
            velocities.append(abs(trend.slope))
        
        if velocities:
            self.evolution_state.preference_velocity = np.mean(velocities)
        
        # Identify dominant trends
        strong_trends = [
            feature for feature, trend in self.trends.items()
            if trend.confidence > 0.7 and trend.strength > 0.5
        ]
        self.evolution_state.dominant_trends = strong_trends[:5]  # Top 5
        
        # Identify emerging/declining interests
        emerging = []
        declining = []
        
        for feature, trend in self.trends.items():
            if trend.trend_type == TrendType.INCREASING and trend.confidence > 0.6:
                emerging.append(feature)
            elif trend.trend_type == TrendType.DECREASING and trend.confidence > 0.6:
                declining.append(feature)
        
        self.evolution_state.emerging_interests = emerging[:5]
        self.evolution_state.declining_interests = declining[:5]
        
        # Calculate adaptation speed
        # Based on how quickly user responds to new content
        if hasattr(self, 'adaptation_history'):
            recent_adaptations = self.adaptation_history[-10:]
            if recent_adaptations:
                self.evolution_state.adaptation_speed = np.mean(recent_adaptations)
    
    def save_state(self, filepath: str):
        """Save evolution model state"""
        state = {
            'user_id': self.user_id,
            'preference_history': {
                k: [(ts.isoformat(), val) for ts, val in v]
                for k, v in self.preference_history.items()
            },
            'trends': {k: asdict(v) for k, v in self.trends.items()},
            'seasonal_components': {
                k: [asdict(comp) for comp in v]
                for k, v in self.seasonal_components.items()
            },
            'evolution_state': asdict(self.evolution_state),
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load evolution model state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.user_id = state['user_id']
        
        # Restore preference history
        self.preference_history = {}
        for feature, history in state['preference_history'].items():
            self.preference_history[feature] = [
                (datetime.fromisoformat(ts), val) for ts, val in history
            ]
        
        # Restore trends
        self.trends = {}
        for feature, trend_data in state['trends'].items():
            trend_data['start_time'] = datetime.fromisoformat(trend_data['start_time'])
            trend_data['last_updated'] = datetime.fromisoformat(trend_data['last_updated'])
            trend_data['trend_type'] = TrendType(trend_data['trend_type'])
            self.trends[feature] = PreferenceTrend(**trend_data)
        
        # Restore seasonal components
        self.seasonal_components = {}
        for feature, components in state['seasonal_components'].items():
            self.seasonal_components[feature] = []
            for comp_data in components:
                comp_data['pattern_type'] = SeasonalPattern(comp_data['pattern_type'])
                comp_data['last_occurrence'] = datetime.fromisoformat(comp_data['last_occurrence'])
                self.seasonal_components[feature].append(SeasonalComponent(**comp_data))
        
        # Restore evolution state
        self.evolution_state = EvolutionState(**state['evolution_state'])
        
        if state['last_analysis_time']:
            self.last_analysis_time = datetime.fromisoformat(state['last_analysis_time'])
