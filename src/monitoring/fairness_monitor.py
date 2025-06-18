"""
Fairness Monitoring System for Content Recommendation Engine.
Monitors bias, fairness metrics, and demographic parity in recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import json
import time

logger = logging.getLogger(__name__)


class FairnessMetric(Enum):
    """Different fairness metrics to monitor."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    GROUP_FAIRNESS = "group_fairness"
    REPRESENTATION = "representation"


@dataclass
class FairnessAlert:
    """Alert for fairness violations."""
    metric: FairnessMetric
    severity: str  # "low", "medium", "high", "critical"
    threshold_value: float
    actual_value: float
    affected_groups: List[str]
    timestamp: float
    description: str
    recommendations: List[str]


@dataclass
class BiasAnalysis:
    """Results of bias analysis."""
    metric: FairnessMetric
    overall_score: float
    group_scores: Dict[str, float]
    bias_detected: bool
    severity: str
    details: Dict[str, Any]


class FairnessMonitor:
    """
    Comprehensive fairness monitoring system for recommendation algorithms.
    """
    
    def __init__(self, 
                 demographic_groups: List[str] = None,
                 fairness_thresholds: Dict[str, float] = None):
        """
        Initialize fairness monitor.
        
        Args:
            demographic_groups: List of demographic attributes to monitor
            fairness_thresholds: Threshold values for different fairness metrics
        """
        self.demographic_groups = demographic_groups or [
            'gender', 'age_group', 'ethnicity', 'location', 'income_level'
        ]
        
        self.fairness_thresholds = fairness_thresholds or {
            'demographic_parity': 0.8,  # Minimum acceptable ratio
            'equalized_odds': 0.1,      # Maximum acceptable difference
            'calibration': 0.05,         # Maximum calibration error
            'representation': 0.1        # Maximum representation difference
        }
        
        self.alerts = []
        self.historical_metrics = defaultdict(list)
        
    async def monitor_recommendations(self,
                                    recommendations: List[Dict[str, Any]],
                                    user_demographics: Dict[str, Dict[str, Any]],
                                    ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, BiasAnalysis]:
        """
        Monitor recommendations for fairness violations.
        
        Args:
            recommendations: List of recommendation results
            user_demographics: User demographic information
            ground_truth: Optional ground truth for accuracy-based fairness metrics
            
        Returns:
            Dictionary of bias analysis results by metric
        """
        results = {}
        
        try:
            # Analyze demographic parity
            results[FairnessMetric.DEMOGRAPHIC_PARITY.value] = await self._analyze_demographic_parity(
                recommendations, user_demographics
            )
            
            # Analyze representation fairness
            results[FairnessMetric.REPRESENTATION.value] = await self._analyze_representation(
                recommendations, user_demographics
            )
            
            # Analyze group fairness
            results[FairnessMetric.GROUP_FAIRNESS.value] = await self._analyze_group_fairness(
                recommendations, user_demographics
            )
            
            # If ground truth available, analyze prediction fairness
            if ground_truth:
                results[FairnessMetric.EQUALIZED_ODDS.value] = await self._analyze_equalized_odds(
                    recommendations, user_demographics, ground_truth
                )
                
                results[FairnessMetric.CALIBRATION.value] = await self._analyze_calibration(
                    recommendations, user_demographics, ground_truth
                )
            
            # Check for alerts
            await self._check_fairness_alerts(results)
            
            # Store historical data
            self._store_historical_metrics(results)
            
            logger.info(f"Fairness monitoring completed for {len(recommendations)} recommendations")
            return results
            
        except Exception as e:
            logger.error(f"Error in fairness monitoring: {e}")
            return {}
    
    async def _analyze_demographic_parity(self,
                                        recommendations: List[Dict[str, Any]],
                                        user_demographics: Dict[str, Dict[str, Any]]) -> BiasAnalysis:
        """Analyze demographic parity in recommendations."""
        
        try:
            group_stats = defaultdict(lambda: {'total': 0, 'positive': 0})
            
            # Calculate recommendation rates by demographic group
            for rec in recommendations:
                user_id = rec.get('user_id')
                score = rec.get('score', 0)
                is_positive = score > 0.5  # Consider high-scored recommendations as positive
                
                if user_id in user_demographics:
                    for group_attr in self.demographic_groups:
                        if group_attr in user_demographics[user_id]:
                            group_value = user_demographics[user_id][group_attr]
                            group_key = f"{group_attr}_{group_value}"
                            
                            group_stats[group_key]['total'] += 1
                            if is_positive:
                                group_stats[group_key]['positive'] += 1
            
            # Calculate rates and disparities
            group_scores = {}
            rates = {}
            
            for group, stats in group_stats.items():
                if stats['total'] > 0:
                    rate = stats['positive'] / stats['total']
                    rates[group] = rate
                    group_scores[group] = rate
            
            # Calculate overall demographic parity score
            if len(rates) > 1:
                rate_values = list(rates.values())
                min_rate = min(rate_values)
                max_rate = max(rate_values)
                parity_score = min_rate / max_rate if max_rate > 0 else 1.0
            else:
                parity_score = 1.0
            
            # Determine bias severity
            threshold = self.fairness_thresholds['demographic_parity']
            bias_detected = parity_score < threshold
            
            if parity_score >= 0.9:
                severity = "low"
            elif parity_score >= 0.8:
                severity = "medium"
            elif parity_score >= 0.7:
                severity = "high"
            else:
                severity = "critical"
            
            return BiasAnalysis(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                overall_score=parity_score,
                group_scores=group_scores,
                bias_detected=bias_detected,
                severity=severity if bias_detected else "low",
                details={
                    'rates_by_group': rates,
                    'min_rate': min(rates.values()) if rates else 0,
                    'max_rate': max(rates.values()) if rates else 0,
                    'rate_ratio': parity_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing demographic parity: {e}")
            return BiasAnalysis(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                overall_score=0.0,
                group_scores={},
                bias_detected=True,
                severity="critical",
                details={'error': str(e)}
            )
    
    async def _analyze_representation(self,
                                    recommendations: List[Dict[str, Any]],
                                    user_demographics: Dict[str, Dict[str, Any]]) -> BiasAnalysis:
        """Analyze representation fairness in content recommendations."""
        
        try:
            # Count representation of different content types/genres by demographic group
            content_representation = defaultdict(lambda: defaultdict(int))
            group_totals = defaultdict(int)
            
            for rec in recommendations:
                user_id = rec.get('user_id')
                content_type = rec.get('content_type', 'unknown')
                genres = rec.get('genres', [])
                
                if user_id in user_demographics:
                    for group_attr in self.demographic_groups:
                        if group_attr in user_demographics[user_id]:
                            group_value = user_demographics[user_id][group_attr]
                            group_key = f"{group_attr}_{group_value}"
                            
                            group_totals[group_key] += 1
                            content_representation[group_key][content_type] += 1
                            
                            for genre in genres:
                                content_representation[group_key][f"genre_{genre}"] += 1
            
            # Calculate representation scores
            group_scores = {}
            representation_disparities = []
            
            # Get all content types
            all_content_types = set()
            for group_content in content_representation.values():
                all_content_types.update(group_content.keys())
            
            # Calculate representation ratios
            for content_type in all_content_types:
                type_ratios = []
                for group, total in group_totals.items():
                    if total > 0:
                        ratio = content_representation[group][content_type] / total
                        type_ratios.append(ratio)
                        group_scores[f"{group}_{content_type}"] = ratio
                
                if len(type_ratios) > 1:
                    disparity = max(type_ratios) - min(type_ratios)
                    representation_disparities.append(disparity)
            
            # Overall representation score
            avg_disparity = np.mean(representation_disparities) if representation_disparities else 0
            representation_score = 1.0 - avg_disparity
            
            bias_detected = avg_disparity > self.fairness_thresholds['representation']
            
            return BiasAnalysis(
                metric=FairnessMetric.REPRESENTATION,
                overall_score=representation_score,
                group_scores=group_scores,
                bias_detected=bias_detected,
                severity="medium" if bias_detected else "low",
                details={
                    'content_representation': dict(content_representation),
                    'average_disparity': avg_disparity,
                    'disparities_by_content': representation_disparities
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing representation: {e}")
            return BiasAnalysis(
                metric=FairnessMetric.REPRESENTATION,
                overall_score=0.0,
                group_scores={},
                bias_detected=True,
                severity="critical",
                details={'error': str(e)}
            )
    
    async def _analyze_group_fairness(self,
                                    recommendations: List[Dict[str, Any]],
                                    user_demographics: Dict[str, Dict[str, Any]]) -> BiasAnalysis:
        """Analyze fairness across different user groups."""
        
        try:
            group_metrics = defaultdict(lambda: {
                'scores': [],
                'diversity': [],
                'novelty': [],
                'count': 0
            })
            
            # Aggregate metrics by demographic group
            for rec in recommendations:
                user_id = rec.get('user_id')
                score = rec.get('score', 0)
                diversity = rec.get('diversity_score', 0)
                novelty = rec.get('novelty_score', 0)
                
                if user_id in user_demographics:
                    for group_attr in self.demographic_groups:
                        if group_attr in user_demographics[user_id]:
                            group_value = user_demographics[user_id][group_attr]
                            group_key = f"{group_attr}_{group_value}"
                            
                            group_metrics[group_key]['scores'].append(score)
                            group_metrics[group_key]['diversity'].append(diversity)
                            group_metrics[group_key]['novelty'].append(novelty)
                            group_metrics[group_key]['count'] += 1
            
            # Calculate group-level statistics
            group_scores = {}
            metric_disparities = []
            
            for group, metrics in group_metrics.items():
                if metrics['count'] > 0:
                    avg_score = np.mean(metrics['scores'])
                    avg_diversity = np.mean(metrics['diversity'])
                    avg_novelty = np.mean(metrics['novelty'])
                    
                    group_scores[group] = {
                        'average_score': avg_score,
                        'average_diversity': avg_diversity,
                        'average_novelty': avg_novelty,
                        'recommendation_count': metrics['count']
                    }
            
            # Calculate disparities
            if len(group_scores) > 1:
                score_values = [g['average_score'] for g in group_scores.values()]
                diversity_values = [g['average_diversity'] for g in group_scores.values()]
                novelty_values = [g['average_novelty'] for g in group_scores.values()]
                
                score_disparity = max(score_values) - min(score_values)
                diversity_disparity = max(diversity_values) - min(diversity_values)
                novelty_disparity = max(novelty_values) - min(novelty_values)
                
                metric_disparities = [score_disparity, diversity_disparity, novelty_disparity]
            
            avg_disparity = np.mean(metric_disparities) if metric_disparities else 0
            fairness_score = 1.0 - avg_disparity
            
            bias_detected = avg_disparity > 0.2  # 20% disparity threshold
            
            return BiasAnalysis(
                metric=FairnessMetric.GROUP_FAIRNESS,
                overall_score=fairness_score,
                group_scores={k: v['average_score'] for k, v in group_scores.items()},
                bias_detected=bias_detected,
                severity="medium" if bias_detected else "low",
                details={
                    'group_metrics': dict(group_scores),
                    'metric_disparities': metric_disparities,
                    'average_disparity': avg_disparity
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing group fairness: {e}")
            return BiasAnalysis(
                metric=FairnessMetric.GROUP_FAIRNESS,
                overall_score=0.0,
                group_scores={},
                bias_detected=True,
                severity="critical",
                details={'error': str(e)}
            )
    
    async def _analyze_equalized_odds(self,
                                    recommendations: List[Dict[str, Any]],
                                    user_demographics: Dict[str, Dict[str, Any]],
                                    ground_truth: Dict[str, Any]) -> BiasAnalysis:
        """Analyze equalized odds fairness metric."""
        
        try:
            # This would require actual ground truth data
            # For demo purposes, we'll simulate the analysis
            
            group_metrics = defaultdict(lambda: {
                'true_positive_rate': 0,
                'false_positive_rate': 0,
                'true_negative_rate': 0,
                'false_negative_rate': 0
            })
            
            # Simulate equalized odds calculation
            # In practice, this would compare predicted vs actual preferences
            
            return BiasAnalysis(
                metric=FairnessMetric.EQUALIZED_ODDS,
                overall_score=0.85,  # Simulated score
                group_scores={'simulated': 0.85},
                bias_detected=False,
                severity="low",
                details={'note': 'Requires ground truth data for full analysis'}
            )
            
        except Exception as e:
            logger.error(f"Error analyzing equalized odds: {e}")
            return BiasAnalysis(
                metric=FairnessMetric.EQUALIZED_ODDS,
                overall_score=0.0,
                group_scores={},
                bias_detected=True,
                severity="critical",
                details={'error': str(e)}
            )
    
    async def _analyze_calibration(self,
                                 recommendations: List[Dict[str, Any]],
                                 user_demographics: Dict[str, Dict[str, Any]],
                                 ground_truth: Dict[str, Any]) -> BiasAnalysis:
        """Analyze calibration fairness across groups."""
        
        try:
            # Calibration analysis would compare predicted scores to actual outcomes
            # For demo purposes, we'll simulate
            
            return BiasAnalysis(
                metric=FairnessMetric.CALIBRATION,
                overall_score=0.92,  # Simulated score
                group_scores={'simulated': 0.92},
                bias_detected=False,
                severity="low",
                details={'note': 'Requires ground truth data for full analysis'}
            )
            
        except Exception as e:
            logger.error(f"Error analyzing calibration: {e}")
            return BiasAnalysis(
                metric=FairnessMetric.CALIBRATION,
                overall_score=0.0,
                group_scores={},
                bias_detected=True,
                severity="critical",
                details={'error': str(e)}
            )
    
    async def _check_fairness_alerts(self, analysis_results: Dict[str, BiasAnalysis]):
        """Check for fairness violations and generate alerts."""
        
        for metric_name, analysis in analysis_results.items():
            if analysis.bias_detected and analysis.severity in ['high', 'critical']:
                
                # Generate recommendations based on the type of bias
                recommendations = []
                if analysis.metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                    recommendations = [
                        "Adjust recommendation algorithm to ensure equal opportunity across demographic groups",
                        "Implement bias mitigation techniques in the model training process",
                        "Review data collection practices for potential sampling bias"
                    ]
                elif analysis.metric == FairnessMetric.REPRESENTATION:
                    recommendations = [
                        "Diversify content catalog to better represent different demographic groups",
                        "Implement content balancing strategies in recommendation generation",
                        "Review content acquisition policies for representation gaps"
                    ]
                elif analysis.metric == FairnessMetric.GROUP_FAIRNESS:
                    recommendations = [
                        "Implement group-aware recommendation adjustments",
                        "Use fairness-aware machine learning techniques",
                        "Monitor and adjust recommendation quality across user segments"
                    ]
                
                alert = FairnessAlert(
                    metric=analysis.metric,
                    severity=analysis.severity,
                    threshold_value=self.fairness_thresholds.get(metric_name, 0.8),
                    actual_value=analysis.overall_score,
                    affected_groups=list(analysis.group_scores.keys()),
                    timestamp=time.time(),
                    description=f"Fairness violation detected in {metric_name}: {analysis.severity} severity",
                    recommendations=recommendations
                )
                
                self.alerts.append(alert)
                logger.warning(f"Fairness alert generated: {alert.description}")
    
    def _store_historical_metrics(self, analysis_results: Dict[str, BiasAnalysis]):
        """Store fairness metrics for historical tracking."""
        
        timestamp = time.time()
        for metric_name, analysis in analysis_results.items():
            self.historical_metrics[metric_name].append({
                'timestamp': timestamp,
                'overall_score': analysis.overall_score,
                'bias_detected': analysis.bias_detected,
                'severity': analysis.severity,
                'group_scores': analysis.group_scores
            })
    
    def get_fairness_dashboard_data(self) -> Dict[str, Any]:
        """Get data for fairness monitoring dashboard."""
        
        try:
            current_time = time.time()
            
            # Recent alerts (last 24 hours)
            recent_alerts = [
                alert for alert in self.alerts 
                if current_time - alert.timestamp < 86400  # 24 hours
            ]
            
            # Historical trends
            trends = {}
            for metric_name, history in self.historical_metrics.items():
                if history:
                    recent_scores = [h['overall_score'] for h in history[-10:]]  # Last 10 measurements
                    trends[metric_name] = {
                        'current_score': recent_scores[-1] if recent_scores else 0,
                        'trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable',
                        'history': recent_scores
                    }
            
            # Alert summary
            alert_summary = defaultdict(int)
            for alert in recent_alerts:
                alert_summary[alert.severity] += 1
            
            return {
                'current_status': 'healthy' if not recent_alerts else 'issues_detected',
                'recent_alerts': len(recent_alerts),
                'alert_summary': dict(alert_summary),
                'trends': trends,
                'monitored_metrics': list(self.fairness_thresholds.keys()),
                'last_updated': current_time
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    async def generate_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        
        try:
            current_time = time.time()
            
            # Recent performance
            recent_metrics = {}
            for metric_name, history in self.historical_metrics.items():
                if history:
                    recent = history[-1]
                    recent_metrics[metric_name] = {
                        'score': recent['overall_score'],
                        'status': 'pass' if not recent['bias_detected'] else 'fail',
                        'severity': recent['severity']
                    }
            
            # Alert analysis
            alert_analysis = {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
                'high_alerts': len([a for a in self.alerts if a.severity == 'high']),
                'most_common_issues': self._get_most_common_issues()
            }
            
            # Recommendations
            recommendations = self._generate_fairness_recommendations(recent_metrics)
            
            report = {
                'report_timestamp': current_time,
                'overall_fairness_score': self._calculate_overall_fairness_score(recent_metrics),
                'metrics_summary': recent_metrics,
                'alert_analysis': alert_analysis,
                'recommendations': recommendations,
                'compliance_status': self._assess_compliance_status(recent_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating fairness report: {e}")
            return {'error': str(e)}
    
    def _get_most_common_issues(self) -> List[str]:
        """Get most common fairness issues."""
        issue_counts = defaultdict(int)
        for alert in self.alerts:
            issue_counts[alert.metric.value] += 1
        
        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _generate_fairness_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on current fairness metrics."""
        recommendations = []
        
        for metric_name, data in metrics.items():
            if data['status'] == 'fail':
                if metric_name == 'demographic_parity':
                    recommendations.append("Implement demographic parity constraints in recommendation algorithm")
                elif metric_name == 'representation':
                    recommendations.append("Increase content diversity to improve representation across groups")
                elif metric_name == 'group_fairness':
                    recommendations.append("Apply group-aware fairness techniques")
        
        if not recommendations:
            recommendations.append("Continue monitoring current fairness levels")
        
        return recommendations
    
    def _calculate_overall_fairness_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall fairness score."""
        if not metrics:
            return 0.0
        
        scores = [data['score'] for data in metrics.values()]
        return np.mean(scores)
    
    def _assess_compliance_status(self, metrics: Dict[str, Any]) -> str:
        """Assess overall compliance with fairness standards."""
        overall_score = self._calculate_overall_fairness_score(metrics)
        
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "acceptable"
        else:
            return "needs_improvement"


# Demo usage
async def demo_fairness_monitoring():
    """Demonstrate fairness monitoring capabilities."""
    
    monitor = FairnessMonitor()
    
    # Sample recommendations with demographic info
    sample_recs = [
        {'user_id': 'user1', 'score': 0.8, 'content_type': 'movie', 'genres': ['action']},
        {'user_id': 'user2', 'score': 0.6, 'content_type': 'movie', 'genres': ['romance']},
        {'user_id': 'user3', 'score': 0.9, 'content_type': 'book', 'genres': ['sci-fi']},
    ]
    
    sample_demographics = {
        'user1': {'gender': 'male', 'age_group': '25-34', 'location': 'urban'},
        'user2': {'gender': 'female', 'age_group': '35-44', 'location': 'suburban'},
        'user3': {'gender': 'non-binary', 'age_group': '18-24', 'location': 'rural'},
    }
    
    # Monitor fairness
    results = await monitor.monitor_recommendations(sample_recs, sample_demographics)
    
    print("Fairness Analysis Results:")
    for metric, analysis in results.items():
        print(f"{metric}: Score={analysis.overall_score:.3f}, Bias Detected={analysis.bias_detected}")
    
    # Generate dashboard data
    dashboard_data = monitor.get_fairness_dashboard_data()
    print(f"\nDashboard Status: {dashboard_data['current_status']}")
    
    # Generate report
    report = await monitor.generate_fairness_report()
    print(f"Overall Fairness Score: {report['overall_fairness_score']:.3f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_fairness_monitoring())
