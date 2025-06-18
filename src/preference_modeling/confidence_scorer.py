"""
Confidence Scorer for Content Recommendation Engine

This module provides confidence scoring for recommendations and predictions,
accounting for data quality, model uncertainty, and temporal factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
from collections import defaultdict, deque
import json

from ..utils.logging import setup_logger


class ConfidenceLevel(Enum):
    """Confidence levels"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # > 0.9


class UncertaintySource(Enum):
    """Sources of uncertainty"""
    DATA_SPARSITY = "data_sparsity"
    MODEL_UNCERTAINTY = "model_uncertainty"
    TEMPORAL_DRIFT = "temporal_drift"
    COLD_START = "cold_start"
    FEATURE_QUALITY = "feature_quality"
    PREDICTION_COMPLEXITY = "prediction_complexity"


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown"""
    overall_score: float  # 0-1
    level: ConfidenceLevel
    data_quality_score: float
    model_uncertainty_score: float
    temporal_reliability_score: float
    prediction_complexity_score: float
    uncertainty_sources: List[UncertaintySource]
    confidence_interval: Tuple[float, float]
    explanation: str


@dataclass
class DataQualityMetrics:
    """Data quality assessment"""
    completeness: float  # 0-1
    freshness: float     # 0-1
    consistency: float   # 0-1
    volume: float        # 0-1
    diversity: float     # 0-1
    reliability: float   # 0-1


@dataclass
class ModelUncertaintyMetrics:
    """Model uncertainty assessment"""
    prediction_variance: float
    model_disagreement: float
    calibration_error: float
    feature_importance_stability: float
    cross_validation_std: float


class ConfidenceScorer:
    """
    Provides comprehensive confidence scoring for recommendations
    and predictions, considering multiple uncertainty sources
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.min_data_points = self.config.get('min_data_points', 10)
        self.freshness_decay_hours = self.config.get('freshness_decay_hours', 24)
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.7)
        self.calibration_bins = self.config.get('calibration_bins', 10)
        
        # State tracking
        self.prediction_history: List[Dict[str, Any]] = []
        self.model_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.feature_stability_scores: Dict[str, float] = {}
        self.calibration_data: List[Tuple[float, float]] = []  # (confidence, accuracy)
        
        # Models for uncertainty estimation
        self.uncertainty_models: Dict[str, Any] = {}
        
        self.logger = setup_logger(__name__)
    
    def score_prediction_confidence(self, 
                                   prediction: float,
                                   features: Dict[str, Any],
                                   model_name: str,
                                   context: Dict[str, Any] = None) -> ConfidenceScore:
        """Score confidence for a single prediction"""
        context = context or {}
        
        try:
            # Assess data quality
            data_quality = self._assess_data_quality(features, context)
            
            # Assess model uncertainty
            model_uncertainty = self._assess_model_uncertainty(
                prediction, features, model_name, context
            )
            
            # Assess temporal reliability
            temporal_reliability = self._assess_temporal_reliability(context)
            
            # Assess prediction complexity
            complexity_score = self._assess_prediction_complexity(features, context)
            
            # Combine scores
            overall_score = self._combine_confidence_scores(
                data_quality.reliability,
                1 - model_uncertainty.prediction_variance,
                temporal_reliability,
                complexity_score
            )
            
            # Determine confidence level
            level = self._get_confidence_level(overall_score)
            
            # Identify uncertainty sources
            uncertainty_sources = self._identify_uncertainty_sources(
                data_quality, model_uncertainty, temporal_reliability, complexity_score
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                prediction, model_uncertainty.prediction_variance
            )
            
            # Generate explanation
            explanation = self._generate_confidence_explanation(
                overall_score, uncertainty_sources, data_quality
            )
            
            score = ConfidenceScore(
                overall_score=overall_score,
                level=level,
                data_quality_score=data_quality.reliability,
                model_uncertainty_score=1 - model_uncertainty.prediction_variance,
                temporal_reliability_score=temporal_reliability,
                prediction_complexity_score=complexity_score,
                uncertainty_sources=uncertainty_sources,
                confidence_interval=confidence_interval,
                explanation=explanation
            )
            
            # Store for calibration
            self._update_calibration_data(overall_score, features, context)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error scoring confidence: {e}")
            return self._default_confidence_score()
    
    def score_recommendation_set_confidence(self,
                                          recommendations: List[Dict[str, Any]],
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score confidence for a set of recommendations"""
        if not recommendations:
            return {'overall_confidence': 0.0, 'individual_scores': []}
        
        individual_scores = []
        
        for rec in recommendations:
            prediction = rec.get('score', 0.0)
            features = rec.get('features', {})
            model_name = rec.get('model', 'unknown')
            
            score = self.score_prediction_confidence(
                prediction, features, model_name, context
            )
            individual_scores.append(score)
        
        # Calculate set-level metrics
        overall_confidence = np.mean([s.overall_score for s in individual_scores])
        
        # Calculate diversity of confidence scores
        confidence_std = np.std([s.overall_score for s in individual_scores])
        
        # Calculate agreement between top recommendations
        top_scores = [s.overall_score for s in individual_scores[:5]]
        top_agreement = 1 - np.std(top_scores) if len(top_scores) > 1 else 1.0
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_std': confidence_std,
            'top_agreement': top_agreement,
            'individual_scores': individual_scores,
            'high_confidence_count': sum(1 for s in individual_scores 
                                       if s.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]),
            'low_confidence_count': sum(1 for s in individual_scores 
                                      if s.level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW])
        }
    
    def update_model_performance(self, model_name: str, actual: float, 
                               predicted: float, features: Dict[str, Any],
                               confidence_score: float = None):
        """Update model performance tracking"""
        error = abs(actual - predicted)
        self.model_performance_history[model_name].append(error)
        
        # Keep only recent history
        if len(self.model_performance_history[model_name]) > 1000:
            self.model_performance_history[model_name] = \
                self.model_performance_history[model_name][-1000:]
        
        # Update calibration data if confidence score provided
        if confidence_score is not None:
            # Binary accuracy based on error threshold
            accuracy = 1.0 if error < 0.1 else 0.0
            self.calibration_data.append((confidence_score, accuracy))
            
            # Keep only recent calibration data
            if len(self.calibration_data) > 1000:
                self.calibration_data = self.calibration_data[-1000:]
    
    def get_model_calibration(self, model_name: str = None) -> Dict[str, Any]:
        """Get model calibration metrics"""
        if not self.calibration_data:
            return {'calibration_error': 0.0, 'reliability_diagram': []}
        
        confidences, accuracies = zip(*self.calibration_data)
        
        # Calculate Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        reliability_diagram = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Select predictions in this bin
            in_bin = np.logical_and(
                np.array(confidences) > bin_lower,
                np.array(confidences) <= bin_upper
            )
            
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(accuracies)[in_bin].mean()
                avg_confidence_in_bin = np.array(confidences)[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                reliability_diagram.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'count': in_bin.sum()
                })
        
        return {
            'calibration_error': ece,
            'reliability_diagram': reliability_diagram,
            'total_predictions': len(self.calibration_data)
        }
    
    def _assess_data_quality(self, features: Dict[str, Any], 
                           context: Dict[str, Any]) -> DataQualityMetrics:
        """Assess quality of input data"""
        # Completeness: proportion of non-missing features
        total_features = len(features)
        non_missing = sum(1 for v in features.values() if v is not None)
        completeness = non_missing / total_features if total_features > 0 else 0
        
        # Freshness: how recent is the data
        data_timestamp = context.get('timestamp', datetime.now())
        if isinstance(data_timestamp, str):
            data_timestamp = datetime.fromisoformat(data_timestamp)
        
        age_hours = (datetime.now() - data_timestamp).total_seconds() / 3600
        freshness = max(0, 1 - age_hours / self.freshness_decay_hours)
        
        # Consistency: check for outliers in numerical features
        numerical_features = [v for v in features.values() 
                            if isinstance(v, (int, float)) and v is not None]
        
        if numerical_features:
            z_scores = np.abs(stats.zscore(numerical_features))
            outlier_proportion = np.mean(z_scores > 3)  # More than 3 std devs
            consistency = 1 - outlier_proportion
        else:
            consistency = 1.0
        
        # Volume: amount of historical data available
        data_points = context.get('data_points', 0)
        volume = min(1.0, data_points / self.min_data_points)
        
        # Diversity: variety in feature values
        if numerical_features:
            diversity = min(1.0, np.std(numerical_features))
        else:
            diversity = 0.5
        
        # Overall reliability
        reliability = np.mean([completeness, freshness, consistency, volume, diversity])
        
        return DataQualityMetrics(
            completeness=completeness,
            freshness=freshness,
            consistency=consistency,
            volume=volume,
            diversity=diversity,
            reliability=reliability
        )
    
    def _assess_model_uncertainty(self, prediction: float, features: Dict[str, Any],
                                model_name: str, context: Dict[str, Any]) -> ModelUncertaintyMetrics:
        """Assess model uncertainty"""
        # Prediction variance based on feature similarity to training data
        feature_values = [v for v in features.values() 
                         if isinstance(v, (int, float)) and v is not None]
        
        if feature_values:
            # Simple heuristic: higher variance for extreme values
            normalized_features = np.abs(stats.zscore(feature_values))
            prediction_variance = min(1.0, np.mean(normalized_features) / 3)
        else:
            prediction_variance = 0.5
        
        # Model disagreement (would be calculated with ensemble in practice)
        model_disagreement = context.get('model_disagreement', 0.0)
        
        # Calibration error from historical performance
        calibration_data = self.get_model_calibration(model_name)
        calibration_error = calibration_data.get('calibration_error', 0.0)
        
        # Feature importance stability
        feature_stability = np.mean(list(self.feature_stability_scores.values())) \
                          if self.feature_stability_scores else 0.5
        
        # Cross-validation standard deviation
        if model_name in self.model_performance_history:
            recent_errors = self.model_performance_history[model_name][-50:]
            cv_std = np.std(recent_errors) if len(recent_errors) > 1 else 0.5
        else:
            cv_std = 0.5
        
        return ModelUncertaintyMetrics(
            prediction_variance=prediction_variance,
            model_disagreement=model_disagreement,
            calibration_error=calibration_error,
            feature_importance_stability=feature_stability,
            cross_validation_std=cv_std
        )
    
    def _assess_temporal_reliability(self, context: Dict[str, Any]) -> float:
        """Assess temporal reliability of prediction"""
        # Time since last model update
        last_update = context.get('last_model_update')
        if last_update:
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)
            
            hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
            update_freshness = max(0, 1 - hours_since_update / (7 * 24))  # Decay over week
        else:
            update_freshness = 0.5
        
        # Concept drift indicator
        drift_score = context.get('concept_drift_score', 0.0)
        drift_reliability = 1 - min(1.0, drift_score)
        
        # Seasonal stability
        seasonal_alignment = context.get('seasonal_alignment', 1.0)
        
        return np.mean([update_freshness, drift_reliability, seasonal_alignment])
    
    def _assess_prediction_complexity(self, features: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """Assess complexity of the prediction task"""
        # Feature interaction complexity
        numerical_features = [v for v in features.values() 
                            if isinstance(v, (int, float)) and v is not None]
        
        if len(numerical_features) > 1:
            # Calculate correlation matrix to assess feature interactions
            correlation_matrix = np.corrcoef(numerical_features)
            interaction_complexity = np.mean(np.abs(correlation_matrix))
        else:
            interaction_complexity = 0.0
        
        # Prediction target complexity (e.g., rating vs. binary preference)
        target_type = context.get('target_type', 'continuous')
        type_complexity = 0.3 if target_type == 'binary' else 0.7
        
        # Domain complexity
        domain = context.get('domain', 'general')
        domain_complexity = {
            'movies': 0.6,
            'music': 0.7,
            'books': 0.5,
            'news': 0.8,
            'general': 0.5
        }.get(domain, 0.5)
        
        # Overall complexity (higher complexity = lower confidence)
        complexity = np.mean([interaction_complexity, type_complexity, domain_complexity])
        
        # Return simplicity score (inverse of complexity)
        return 1 - complexity
    
    def _combine_confidence_scores(self, data_quality: float, model_certainty: float,
                                 temporal_reliability: float, simplicity: float) -> float:
        """Combine individual confidence components"""
        # Weighted combination
        weights = {
            'data_quality': 0.3,
            'model_certainty': 0.3,
            'temporal_reliability': 0.2,
            'simplicity': 0.2
        }
        
        combined_score = (
            weights['data_quality'] * data_quality +
            weights['model_certainty'] * model_certainty +
            weights['temporal_reliability'] * temporal_reliability +
            weights['simplicity'] * simplicity
        )
        
        return max(0.0, min(1.0, combined_score))
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical score to confidence level"""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _identify_uncertainty_sources(self, data_quality: DataQualityMetrics,
                                    model_uncertainty: ModelUncertaintyMetrics,
                                    temporal_reliability: float,
                                    complexity_score: float) -> List[UncertaintySource]:
        """Identify primary sources of uncertainty"""
        sources = []
        
        if data_quality.volume < 0.5:
            sources.append(UncertaintySource.DATA_SPARSITY)
        
        if data_quality.completeness < 0.7:
            sources.append(UncertaintySource.FEATURE_QUALITY)
        
        if model_uncertainty.prediction_variance > 0.7:
            sources.append(UncertaintySource.MODEL_UNCERTAINTY)
        
        if temporal_reliability < 0.5:
            sources.append(UncertaintySource.TEMPORAL_DRIFT)
        
        if data_quality.volume < 0.3:
            sources.append(UncertaintySource.COLD_START)
        
        if complexity_score < 0.5:
            sources.append(UncertaintySource.PREDICTION_COMPLEXITY)
        
        return sources
    
    def _calculate_confidence_interval(self, prediction: float, 
                                     variance: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simple approach using variance
        std_dev = np.sqrt(variance) if variance > 0 else 0.1
        margin = 1.96 * std_dev  # 95% confidence interval
        
        lower_bound = prediction - margin
        upper_bound = prediction + margin
        
        return (lower_bound, upper_bound)
    
    def _generate_confidence_explanation(self, score: float, 
                                       uncertainty_sources: List[UncertaintySource],
                                       data_quality: DataQualityMetrics) -> str:
        """Generate human-readable explanation of confidence score"""
        level = self._get_confidence_level(score)
        
        explanations = {
            ConfidenceLevel.VERY_HIGH: "Very high confidence due to abundant, fresh data and stable model performance.",
            ConfidenceLevel.HIGH: "High confidence with good data quality and reliable model predictions.",
            ConfidenceLevel.MEDIUM: "Medium confidence - prediction is reasonably reliable but some uncertainty remains.",
            ConfidenceLevel.LOW: "Low confidence due to limited data or model uncertainty.",
            ConfidenceLevel.VERY_LOW: "Very low confidence - prediction should be used with caution."
        }
        
        base_explanation = explanations[level]
        
        # Add specific uncertainty sources
        if uncertainty_sources:
            uncertainty_descriptions = {
                UncertaintySource.DATA_SPARSITY: "limited historical data",
                UncertaintySource.MODEL_UNCERTAINTY: "model prediction uncertainty",
                UncertaintySource.TEMPORAL_DRIFT: "temporal changes in patterns",
                UncertaintySource.COLD_START: "new user/item with minimal data",
                UncertaintySource.FEATURE_QUALITY: "incomplete or low-quality features",
                UncertaintySource.PREDICTION_COMPLEXITY: "complex prediction scenario"
            }
            
            uncertainty_list = [uncertainty_descriptions.get(source, str(source)) 
                              for source in uncertainty_sources[:3]]  # Top 3
            
            if uncertainty_list:
                base_explanation += f" Main concerns: {', '.join(uncertainty_list)}."
        
        return base_explanation
    
    def _update_calibration_data(self, confidence: float, features: Dict[str, Any],
                               context: Dict[str, Any]):
        """Update calibration data for future reference"""
        # This would typically store the prediction for later validation
        # For now, we'll just track the confidence scores
        pass
    
    def _default_confidence_score(self) -> ConfidenceScore:
        """Return default confidence score for error cases"""
        return ConfidenceScore(
            overall_score=0.5,
            level=ConfidenceLevel.MEDIUM,
            data_quality_score=0.5,
            model_uncertainty_score=0.5,
            temporal_reliability_score=0.5,
            prediction_complexity_score=0.5,
            uncertainty_sources=[UncertaintySource.MODEL_UNCERTAINTY],
            confidence_interval=(0.0, 1.0),
            explanation="Default confidence due to error in confidence calculation."
        )
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get overall confidence statistics"""
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        confidence_scores = [p.get('confidence', 0.5) for p in recent_predictions]
        
        return {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'confidence_trend': self._calculate_confidence_trend(confidence_scores),
            'calibration_metrics': self.get_model_calibration()
        }
    
    def _calculate_confidence_trend(self, scores: List[float]) -> str:
        """Calculate trend in confidence scores"""
        if len(scores) < 10:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
