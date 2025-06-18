"""
Configuration Management Module

This module handles all configuration settings for the recommendation engine,
including algorithm parameters, API settings, and feature flags.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
from datetime import timedelta

logger = logging.getLogger(__name__)


@dataclass
class CollaborativeFilteringConfig:
    """Configuration for collaborative filtering algorithms."""
    # Matrix Factorization
    mf_factors: int = 50
    mf_regularization: float = 0.01
    mf_iterations: int = 100
    mf_learning_rate: float = 0.01
    mf_use_bias: bool = True
    
    # User/Item Similarity
    similarity_metric: str = 'cosine'  # cosine, pearson, jaccard
    min_common_items: int = 5
    neighborhood_size: int = 50
    
    # Cold Start
    cold_start_method: str = 'popularity'  # popularity, content, hybrid
    min_interactions_user: int = 5
    min_interactions_item: int = 3


@dataclass
class ContentBasedConfig:
    """Configuration for content-based filtering."""
    # Embedding
    embedding_dim: int = 384
    embedding_model: str = 'all-MiniLM-L6-v2'
    use_gemini_embeddings: bool = True
    embedding_cache_size: int = 10000
    
    # Feature Weights
    genre_weight: float = 0.4
    director_weight: float = 0.2
    actor_weight: float = 0.2
    year_weight: float = 0.1
    description_weight: float = 0.1
    
    # Cross-domain
    enable_cross_domain: bool = True
    domain_similarity_threshold: float = 0.6
    
    # Diversity
    diversity_lambda: float = 0.1
    max_same_genre: int = 3


@dataclass
class KnowledgeBasedConfig:
    """Configuration for knowledge-based recommendations."""
    # Trending
    trending_window_days: int = 7
    trending_min_interactions: int = 10
    trending_decay_factor: float = 0.9
    
    # Quality Scoring
    min_rating_count: int = 5
    quality_threshold: float = 3.5
    recency_bonus_months: int = 6
    
    # Context Rules
    enable_time_context: bool = True
    enable_mood_context: bool = True
    context_boost_factor: float = 1.5


@dataclass
class HybridConfig:
    """Configuration for hybrid recommendations."""
    # Algorithm Weights
    collaborative_weight: float = 0.4
    content_weight: float = 0.3
    knowledge_weight: float = 0.3
    
    # Ensemble Methods
    ensemble_method: str = 'weighted'  # weighted, rank_fusion, learned
    auto_tune_weights: bool = True
    tuning_frequency_days: int = 7
    
    # Diversity and Re-ranking
    enable_diversity_rerank: bool = True
    diversity_weight: float = 0.2
    novelty_weight: float = 0.1
    
    # Explanation
    enable_explanations: bool = True
    explanation_method: str = 'feature_importance'  # template, feature_importance, llm


@dataclass
class PersonalizationConfig:
    """Configuration for personalization features."""
    # Short-term preferences
    short_term_window_days: int = 7
    session_timeout_minutes: int = 30
    recency_decay_factor: float = 0.8
    
    # Long-term preferences
    long_term_window_days: int = 180
    profile_update_frequency_days: int = 1
    preference_stability_threshold: float = 0.1
    
    # Sequential patterns
    enable_sequential_mining: bool = True
    max_sequence_length: int = 10
    min_sequence_support: float = 0.1
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_detection_window_days: int = 30
    drift_threshold: float = 0.3


@dataclass
class GroupRecommendationConfig:
    """Configuration for group recommendations."""
    # Aggregation methods
    default_aggregation: str = 'average'  # average, least_misery, most_pleasure, fairness
    enable_fairness_constraint: bool = True
    fairness_threshold: float = 0.1
    
    # Group dynamics
    max_group_size: int = 10
    min_group_agreement: float = 0.3
    group_preference_weight: float = 0.7


@dataclass
class APIConfig:
    """Configuration for API settings."""
    # Server
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # Caching
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    # Timeouts
    request_timeout_seconds: int = 30
    recommendation_timeout_seconds: int = 10


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    # SQLite (default)
    sqlite_path: str = 'data/recommendation_engine.db'
    sqlite_timeout: int = 30
    
    # Connection pooling
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    # Performance
    enable_wal_mode: bool = True
    cache_size_mb: int = 100


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Files
    log_dir: str = 'logs'
    log_file: str = 'app.log'
    error_file: str = 'errors.log'
    
    # Rotation
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Structured logging
    enable_structured: bool = True
    structured_file: str = 'structured.log'


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing and experiments."""
    # A/B Testing
    enable_ab_testing: bool = False
    default_variant: str = 'control'
    experiment_config_path: str = 'config/experiments.json'
    
    # Metrics tracking
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30
    
    # ML Flow integration
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = 'sqlite:///mlruns.db'


@dataclass
class SystemConfig:
    """Main system configuration containing all sub-configurations."""
    collaborative: CollaborativeFilteringConfig = field(default_factory=CollaborativeFilteringConfig)
    content_based: ContentBasedConfig = field(default_factory=ContentBasedConfig)
    knowledge_based: KnowledgeBasedConfig = field(default_factory=KnowledgeBasedConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    group_recommendation: GroupRecommendationConfig = field(default_factory=GroupRecommendationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Global settings
    environment: str = 'development'  # development, staging, production
    debug: bool = True
    enable_profiling: bool = False
    max_recommendations: int = 100
    
    # Directory paths
    data_directory: str = 'data'
    cache_directory: str = 'cache'
    log_directory: str = 'logs'
    output_directory: str = 'output'
    
    # Feature flags
    enable_real_time_updates: bool = True
    enable_cross_domain: bool = True
    enable_multi_language: bool = True
    enable_group_recommendations: bool = True
    enable_explanations: bool = True
    
    # External services
    gemini_api_key: Optional[str] = None
    gemini_model: str = 'gemini-pro'
    redis_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None
    
    # Legacy support
    supported_content_types: List[str] = field(default_factory=lambda: ["movie", "book", "music", "tv_show"])


class ConfigManager:
    """Manages configuration loading, validation, and updates."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or 'config/config.yaml'
        self.config = SystemConfig()
        self._load_config()
        self._setup_environment_overrides()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    if config_file.suffix.lower() == '.json':
                        config_data = json.load(f)
                    elif config_file.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f)
                    else:
                        logger.warning(f"Unsupported config file format: {config_file.suffix}")
                        return
                
                # Update configuration with loaded data
                self._update_config_from_dict(config_data)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.info("Configuration file not found, using defaults")
                # Create default config file
                self.save_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        try:
            # Update each configuration section
            for section_name, section_config in config_dict.items():
                if hasattr(self.config, section_name):
                    section_obj = getattr(self.config, section_name)
                    
                    if hasattr(section_obj, '__dataclass_fields__'):
                        # Update dataclass fields
                        for field_name, field_value in section_config.items():
                            if hasattr(section_obj, field_name):
                                setattr(section_obj, field_name, field_value)
                    else:
                        # Direct attribute update
                        setattr(self.config, section_name, section_config)
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def _setup_environment_overrides(self):
        """Setup environment variable overrides."""
        try:
            # API configuration from environment
            if os.getenv('API_HOST'):
                self.config.api.host = os.getenv('API_HOST')
            if os.getenv('API_PORT'):
                self.config.api.port = int(os.getenv('API_PORT'))
            if os.getenv('DEBUG'):
                self.config.debug = os.getenv('DEBUG').lower() == 'true'
            
            # Database configuration
            if os.getenv('DATABASE_URL'):
                self.config.database.sqlite_path = os.getenv('DATABASE_URL')
            
            # External services
            if os.getenv('GEMINI_API_KEY'):
                self.config.gemini_api_key = os.getenv('GEMINI_API_KEY')
            if os.getenv('REDIS_URL'):
                self.config.redis_url = os.getenv('REDIS_URL')
            
            # Environment
            if os.getenv('ENVIRONMENT'):
                self.config.environment = os.getenv('ENVIRONMENT')
                
        except Exception as e:
            logger.error(f"Error setting up environment overrides: {e}")
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        try:
            self._update_config_from_dict(updates)
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        try:
            save_path = path or self.config_path
            config_dict = asdict(self.config)
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate weights sum to 1.0
            total_weight = (self.config.hybrid.collaborative_weight + 
                          self.config.hybrid.content_weight + 
                          self.config.hybrid.knowledge_weight)
            
            if abs(total_weight - 1.0) > 0.01:
                validation_results['warnings'].append(
                    f"Hybrid weights sum to {total_weight:.3f}, not 1.0"
                )
            
            # Validate positive values
            if self.config.collaborative.mf_factors <= 0:
                validation_results['errors'].append("Matrix factorization factors must be positive")
            
            if self.config.content_based.embedding_dim <= 0:
                validation_results['errors'].append("Embedding dimension must be positive")
            
            # Validate file paths
            if not Path(self.config.database.sqlite_path).parent.exists():
                Path(self.config.database.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
                validation_results['warnings'].append("Created database directory")
            
            # Validate API settings
            if not (1 <= self.config.api.port <= 65535):
                validation_results['errors'].append("API port must be between 1 and 65535")
            
            if validation_results['errors']:
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get current feature flags."""
        return {
            'real_time_updates': self.config.enable_real_time_updates,
            'cross_domain': self.config.enable_cross_domain,
            'multi_language': self.config.enable_multi_language,
            'group_recommendations': self.config.enable_group_recommendations,
            'explanations': self.config.enable_explanations,
            'ab_testing': self.config.experiment.enable_ab_testing,
            'metrics_collection': self.config.experiment.enable_metrics_collection,
            'drift_detection': self.config.personalization.enable_drift_detection,
            'sequential_mining': self.config.personalization.enable_sequential_mining
        }
    
    def get_algorithm_config(self, algorithm_name: str) -> Optional[Any]:
        """Get configuration for a specific algorithm."""
        algorithm_mapping = {
            'collaborative': self.config.collaborative,
            'content_based': self.config.content_based,
            'knowledge_based': self.config.knowledge_based,
            'hybrid': self.config.hybrid,
            'personalization': self.config.personalization,
            'group': self.config.group_recommendation
        }
        
        return algorithm_mapping.get(algorithm_name)


# Global configuration instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> SystemConfig:
    """Get the current system configuration."""
    return get_config_manager().get_config()

def update_config(updates: Dict[str, Any]):
    """Update the system configuration."""
    get_config_manager().update_config(updates)

def save_config(path: Optional[str] = None):
    """Save the current configuration."""
    get_config_manager().save_config(path)

# Legacy compatibility
settings = get_config()

# Data Schema Constants for backward compatibility
class DataSchemas:
    """Constants for data schema definitions"""
    
    USER_PROFILE_SCHEMA = {
        "user_id": "string",
        "age": "integer",
        "gender": "string",
        "location": "string",
        "occupation": "string",
        "registration_date": "datetime",
        "activity_status": "string",
        "preference_history": "json",
        "cross_platform_ids": "json"
    }
    
    CONTENT_METADATA_SCHEMA = {
        "content_id": "string",
        "content_type": "string",
        "title": "string",
        "description": "text",
        "genres": "json",
        "release_date": "datetime",
        "duration": "integer",
        "creator_info": "json",
        "quality_metrics": "json",
        "language": "string",
        "region": "string"
    }


class FeatureConstants:
    """Constants for feature engineering"""
    
    TEMPORAL_FEATURES = [
        "hour_of_day", "day_of_week", "month", "season",
        "is_weekend", "is_holiday", "time_since_last_interaction"
    ]
    
    CONTENT_FEATURES = [
        "genre_vector", "release_year", "duration_normalized",
        "popularity_score", "quality_score", "content_age",
        "language_encoded", "creator_popularity"
    ]


class QualityMetrics:
    """Quality metrics and thresholds"""
    
    DATA_QUALITY_CHECKS = {
        "completeness": 0.85,
        "consistency": 0.90,
        "accuracy": 0.85,
        "timeliness": 0.80,
        "validity": 0.90
    }


# Alias for backward compatibility
Config = ConfigManager
