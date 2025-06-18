"""
A/B Testing Framework
Comprehensive testing infrastructure for recommendation systems
"""

import asyncio
import logging
import json
import hashlib
import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class TestType(Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    BANDIT = "bandit"
    INTERLEAVED = "interleaved"
    CANARY = "canary"

@dataclass
class Variant:
    """Experiment variant definition"""
    id: str
    name: str
    description: str
    traffic_percentage: float
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    id: str
    name: str
    description: str
    test_type: TestType
    variants: List[Variant]
    start_date: datetime
    end_date: datetime
    success_metrics: List[str]
    guardrail_metrics: List[str] = field(default_factory=list)
    minimum_sample_size: int = 1000
    statistical_power: float = 0.8
    significance_level: float = 0.05
    traffic_allocation: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Experiment result data"""
    experiment_id: str
    variant_id: str
    user_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class StatisticalAnalyzer:
    """Statistical analysis for A/B tests"""
    
    @staticmethod
    def welch_t_test(control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Perform Welch's t-test for unequal variances"""
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'confidence_interval': (0.0, 0.0)
            }
        
        statistic, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        # Calculate confidence interval for difference in means
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        se_diff = np.sqrt(np.var(treatment_data)/len(treatment_data) + np.var(control_data)/len(control_data))
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, min(len(treatment_data), len(control_data)) - 1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': mean_diff / np.sqrt((np.var(treatment_data) + np.var(control_data)) / 2)
        }
    
    @staticmethod
    def mann_whitney_u_test(control_data: List[float], treatment_data: List[float]) -> Dict[str, float]:
        """Non-parametric Mann-Whitney U test"""
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def chi_square_test(control_conversions: int, control_total: int, 
                       treatment_conversions: int, treatment_total: int) -> Dict[str, float]:
        """Chi-square test for conversion rates"""
        # Create contingency table
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions]
        ])
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        # Calculate conversion rates
        control_rate = control_conversions / control_total if control_total > 0 else 0
        treatment_rate = treatment_conversions / treatment_total if treatment_total > 0 else 0
        
        return {
            'chi2_statistic': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        }
    
    @staticmethod
    def sequential_probability_ratio_test(control_data: List[float], treatment_data: List[float],
                                        effect_size: float = 0.1, alpha: float = 0.05, 
                                        beta: float = 0.2) -> Dict[str, Any]:
        """Sequential Probability Ratio Test for early stopping"""
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        
        if n_control < 10 or n_treatment < 10:
            return {
                'decision': 'continue',
                'evidence': 0.0,
                'recommendation': 'Need more data'
            }
        
        # Calculate log likelihood ratio
        mean_control = np.mean(control_data)
        mean_treatment = np.mean(treatment_data)
        
        # Simplified SPRT calculation
        delta = mean_treatment - mean_control
        pooled_std = np.sqrt((np.var(control_data) + np.var(treatment_data)) / 2)
        
        if pooled_std == 0:
            return {
                'decision': 'continue',
                'evidence': 0.0,
                'recommendation': 'No variance in data'
            }
        
        z_score = delta / (pooled_std * np.sqrt(1/n_control + 1/n_treatment))
        
        # Decision boundaries
        upper_boundary = np.log((1 - beta) / alpha)
        lower_boundary = np.log(beta / (1 - alpha))
        
        evidence = z_score * effect_size / pooled_std
        
        if evidence >= upper_boundary:
            decision = 'stop_treatment_wins'
        elif evidence <= lower_boundary:
            decision = 'stop_control_wins'
        else:
            decision = 'continue'
        
        return {
            'decision': decision,
            'evidence': float(evidence),
            'z_score': float(z_score),
            'recommendation': f'Evidence: {evidence:.3f}, Z-score: {z_score:.3f}'
        }

class TrafficSplitter:
    """Handle traffic allocation for experiments"""
    
    @staticmethod
    def hash_user_to_variant(user_id: str, experiment_id: str, variants: List[Variant]) -> str:
        """Hash-based deterministic user assignment"""
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to [0, 1]
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Find variant based on cumulative traffic allocation
        cumulative_percentage = 0.0
        for variant in variants:
            cumulative_percentage += variant.traffic_percentage
            if normalized_hash <= cumulative_percentage:
                return variant.id
        
        # Fallback to control
        return variants[0].id if variants else "control"
    
    @staticmethod
    def validate_traffic_allocation(variants: List[Variant]) -> bool:
        """Validate that traffic allocation sums to 100%"""
        total_traffic = sum(variant.traffic_percentage for variant in variants)
        return abs(total_traffic - 1.0) < 0.001  # Allow for floating point precision

class ExperimentManager:
    """Manage A/B testing experiments"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results_buffer: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.analyzer = StatisticalAnalyzer()
        self.traffic_splitter = TrafficSplitter()
    
    async def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create new experiment"""
        try:
            # Validate configuration
            if not self.traffic_splitter.validate_traffic_allocation(config.variants):
                raise ValueError("Traffic allocation must sum to 100%")
            
            # Store experiment configuration
            self.experiments[config.id] = config
            
            if self.redis_client:
                await self.redis_client.hset(
                    "experiments",
                    config.id,
                    json.dumps(config.__dict__, default=str)
                )
            
            logger.info(f"Created experiment {config.id}: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment {config.id}: {e}")
            return False
    
    async def assign_user_to_variant(self, user_id: str, experiment_id: str) -> Optional[str]:
        """Assign user to experiment variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is active
        now = datetime.now()
        if now < experiment.start_date or now > experiment.end_date:
            return None
        
        # Assign variant
        variant_id = self.traffic_splitter.hash_user_to_variant(
            user_id, experiment_id, experiment.variants
        )
        
        # Store assignment in Redis for consistency
        if self.redis_client:
            await self.redis_client.hset(
                f"experiment_assignments:{experiment_id}",
                user_id,
                variant_id
            )
        
        return variant_id
    
    async def record_result(self, result: ExperimentResult):
        """Record experiment result"""
        self.results_buffer[result.experiment_id].append(result)
        
        # Persist to Redis
        if self.redis_client:
            await self.redis_client.lpush(
                f"experiment_results:{result.experiment_id}",
                json.dumps(result.__dict__, default=str)
            )
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        results = self.results_buffer.get(experiment_id, [])
        
        if len(results) < experiment.minimum_sample_size:
            return {
                "status": "insufficient_data",
                "current_sample_size": len(results),
                "required_sample_size": experiment.minimum_sample_size
            }
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant_id].append(result)
        
        # Find control variant
        control_variant = next((v for v in experiment.variants if v.is_control), experiment.variants[0])
        control_results = variant_results[control_variant.id]
        
        analysis = {
            "experiment_id": experiment_id,
            "status": "analyzed",
            "sample_sizes": {variant_id: len(results) for variant_id, results in variant_results.items()},
            "variants": {}
        }
        
        # Analyze each success metric
        for metric in experiment.success_metrics:
            control_values = [r.metrics.get(metric, 0) for r in control_results if metric in r.metrics]
            
            for variant_id, variant_data in variant_results.items():
                if variant_id == control_variant.id:
                    continue
                
                treatment_values = [r.metrics.get(metric, 0) for r in variant_data if metric in r.metrics]
                
                if not control_values or not treatment_values:
                    continue
                
                # Perform statistical tests
                t_test_result = self.analyzer.welch_t_test(control_values, treatment_values)
                mw_test_result = self.analyzer.mann_whitney_u_test(control_values, treatment_values)
                sprt_result = self.analyzer.sequential_probability_ratio_test(control_values, treatment_values)
                
                if variant_id not in analysis["variants"]:
                    analysis["variants"][variant_id] = {}
                
                analysis["variants"][variant_id][metric] = {
                    "control_mean": float(np.mean(control_values)),
                    "treatment_mean": float(np.mean(treatment_values)),
                    "t_test": t_test_result,
                    "mann_whitney": mw_test_result,
                    "sequential_test": sprt_result,
                    "relative_lift": (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values) if np.mean(control_values) > 0 else 0
                }
        
        return analysis
    
    async def get_recommendation(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment recommendation based on analysis"""
        analysis = await self.analyze_experiment(experiment_id)
        
        if "error" in analysis or analysis.get("status") != "analyzed":
            return {"recommendation": "continue", "reason": "Insufficient data or analysis error"}
        
        # Simple recommendation logic
        significant_improvements = 0
        significant_degradations = 0
        
        for variant_id, variant_analysis in analysis["variants"].items():
            for metric, metric_analysis in variant_analysis.items():
                if metric_analysis["t_test"]["significant"]:
                    if metric_analysis["relative_lift"] > 0:
                        significant_improvements += 1
                    else:
                        significant_degradations += 1
        
        if significant_degradations > 0:
            return {
                "recommendation": "stop_experiment",
                "reason": "Significant degradation detected",
                "winning_variant": None
            }
        elif significant_improvements > 0:
            # Find best performing variant
            best_variant = None
            best_lift = -float('inf')
            
            for variant_id, variant_analysis in analysis["variants"].items():
                for metric, metric_analysis in variant_analysis.items():
                    if metric_analysis["relative_lift"] > best_lift:
                        best_lift = metric_analysis["relative_lift"]
                        best_variant = variant_id
            
            return {
                "recommendation": "declare_winner",
                "reason": f"Significant improvement detected (lift: {best_lift:.1%})",
                "winning_variant": best_variant
            }
        else:
            return {
                "recommendation": "continue",
                "reason": "No significant difference detected yet"
            }

class CanaryDeployment:
    """Canary deployment for gradual rollouts"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.canary_configs = {}
    
    async def start_canary(self, deployment_id: str, canary_percentage: float = 0.05,
                          success_criteria: Dict[str, float] = None) -> bool:
        """Start canary deployment"""
        config = {
            "deployment_id": deployment_id,
            "canary_percentage": canary_percentage,
            "success_criteria": success_criteria or {"error_rate": 0.01, "latency_p95": 200},
            "start_time": datetime.now(),
            "status": "running"
        }
        
        self.canary_configs[deployment_id] = config
        
        if self.redis_client:
            await self.redis_client.hset(
                "canary_deployments",
                deployment_id,
                json.dumps(config, default=str)
            )
        
        logger.info(f"Started canary deployment {deployment_id} with {canary_percentage:.1%} traffic")
        return True
    
    async def should_serve_canary(self, user_id: str, deployment_id: str) -> bool:
        """Determine if user should receive canary version"""
        if deployment_id not in self.canary_configs:
            return False
        
        config = self.canary_configs[deployment_id]
        
        # Hash-based assignment
        hash_input = f"{user_id}:{deployment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        return normalized_hash < config["canary_percentage"]
    
    async def evaluate_canary(self, deployment_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate canary deployment health"""
        if deployment_id not in self.canary_configs:
            return {"error": "Deployment not found"}
        
        config = self.canary_configs[deployment_id]
        success_criteria = config["success_criteria"]
        
        health_status = "healthy"
        issues = []
        
        for metric, threshold in success_criteria.items():
            if metric in metrics:
                if metrics[metric] > threshold:
                    health_status = "unhealthy"
                    issues.append(f"{metric}: {metrics[metric]:.3f} > {threshold:.3f}")
        
        recommendation = "continue" if health_status == "healthy" else "rollback"
        
        return {
            "deployment_id": deployment_id,
            "health_status": health_status,
            "recommendation": recommendation,
            "issues": issues,
            "metrics": metrics,
            "evaluation_time": datetime.now().isoformat()
        }
