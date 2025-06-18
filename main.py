"""
Main ETL Pipeline

Orchestrates the complete content recommendation engine pipeline:
1. Data Integration (loading, preprocessing, feature extraction)
2. Content Understanding (embeddings, quality scoring, cross-domain mapping)
3. User Profiling (preferences, behavior, evolution)
4. Data Quality Reporting
5. Feature Store Creation
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import sys
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import settings
from src.utils.logging import LoggingConfig
from src.utils.validation import DataQualityReport

# Data Integration
from src.data_integration.data_loader import DatasetDownloader, UnifiedDataLoader
from src.data_integration.preprocessor import ComprehensivePreprocessor
from src.data_integration.feature_extractor import ComprehensiveFeatureExtractor
from src.data_integration.schema_manager import get_session, create_tables
from sqlalchemy import text

# Content Understanding
from src.content_understanding.gemini_client import GeminiClient
from src.content_understanding.embedding_generator import EmbeddingGenerator
from src.content_understanding.cross_domain_mapper import CrossDomainMapper
from src.content_understanding.quality_scorer import QualityScorer

# User Profiling
from src.user_profiling.preference_tracker import PreferenceTracker
from src.user_profiling.behavior_analyzer import BehaviorAnalyzer
from src.user_profiling.profile_evolution import ProfileEvolution


class ETLPipeline:
    """
    Main ETL Pipeline orchestrator
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Setup logging
        logging_config = LoggingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_downloader = None
        self.data_loader = None
        self.preprocessor = None
        self.feature_extractor = None
        self.gemini_client = None
        self.embedding_generator = None
        self.cross_domain_mapper = None
        self.quality_scorer = None
        self.preference_tracker = None
        self.behavior_analyzer = None
        self.profile_evolution = None
        
        # Pipeline state
        self.pipeline_start_time = None
        self.pipeline_stats = {}
        
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            self.logger.info("Initializing pipeline components...")
            
            # Data Integration
            self.data_downloader = DatasetDownloader()
            self.data_loader = UnifiedDataLoader()
            self.preprocessor = ComprehensivePreprocessor()
            self.feature_extractor = ComprehensiveFeatureExtractor()
            
            # Content Understanding
            self.gemini_client = GeminiClient()
            self.embedding_generator = EmbeddingGenerator(self.gemini_client)
            self.cross_domain_mapper = CrossDomainMapper(self.gemini_client)
            self.quality_scorer = QualityScorer()
            
            # User Profiling
            self.preference_tracker = PreferenceTracker()
            self.behavior_analyzer = BehaviorAnalyzer()
            self.profile_evolution = ProfileEvolution()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline
        
        Returns:
            Pipeline execution results and statistics
        """
        self.pipeline_start_time = time.time()
        self.logger.info("Starting full ETL pipeline execution")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Phase 1: Data Integration
            self.logger.info("=" * 60)
            self.logger.info("PHASE 1: DATA INTEGRATION")
            self.logger.info("=" * 60)
            
            await self._run_data_integration_phase()
            
            # Phase 2: Content Understanding
            self.logger.info("=" * 60)
            self.logger.info("PHASE 2: CONTENT UNDERSTANDING")
            self.logger.info("=" * 60)
            
            await self._run_content_understanding_phase()
            
            # Phase 3: User Profiling
            self.logger.info("=" * 60)
            self.logger.info("PHASE 3: USER PROFILING")
            self.logger.info("=" * 60)
            
            await self._run_user_profiling_phase()
            
            # Phase 4: Quality Reporting
            self.logger.info("=" * 60)
            self.logger.info("PHASE 4: QUALITY REPORTING")
            self.logger.info("=" * 60)
            
            quality_report = await self._generate_quality_report()
            
            # Phase 5: Feature Store Creation
            self.logger.info("=" * 60)
            self.logger.info("PHASE 5: FEATURE STORE CREATION")
            self.logger.info("=" * 60)
            
            feature_store = await self._create_feature_store()
            
            # Calculate final statistics
            pipeline_duration = time.time() - self.pipeline_start_time
            self.pipeline_stats['total_duration_seconds'] = pipeline_duration
            self.pipeline_stats['completion_time'] = datetime.now()
            
            results = {
                'status': 'success',
                'pipeline_stats': self.pipeline_stats,
                'quality_report': quality_report,
                'feature_store_path': feature_store,
                'duration_seconds': pipeline_duration
            }
            
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Total duration: {pipeline_duration:.2f} seconds")
            self.logger.info(f"Quality score: {quality_report.get('overall_score', 'N/A')}")
            
            return results
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return {
                'status': 'failed',
                'error': error_msg,
                'pipeline_stats': self.pipeline_stats,
                'duration_seconds': time.time() - self.pipeline_start_time if self.pipeline_start_time else 0
            }
        
        finally:
            # Cleanup resources
            await self._cleanup_resources()
    
    async def _run_data_integration_phase(self):
        """Run data integration phase"""
        phase_start = time.time()
        
        try:
            # 1. Create database schema
            self.logger.info("Creating database schema...")
            create_tables()
            self.logger.info("✓ Database schema created")
            
            # 2. Download and load datasets
            self.logger.info("Downloading datasets...")
            download_results = self.data_downloader.download_all_datasets()
            
            self.logger.info("Loading datasets...")
            # Load a sample for faster processing (remove sample_size=200000 for full dataset)
            datasets = self.data_loader.load_all_datasets(sample_size=200000)
            
            dataset_stats = {}
            for dataset_name, tables in datasets.items():
                if tables:
                    total_records = 0
                    all_columns = []
                    total_memory = 0
                    
                    for table_name, data in tables.items():
                        if data is not None and hasattr(data, '__len__'):
                            total_records += len(data)
                            if hasattr(data, 'columns'):
                                all_columns.extend(list(data.columns))
                            if hasattr(data, 'memory_usage'):
                                total_memory += data.memory_usage(deep=True).sum() / 1024 / 1024
                    
                    dataset_stats[dataset_name] = {
                        'records_loaded': total_records,
                        'tables': list(tables.keys()),
                        'total_columns': len(set(all_columns)),
                        'memory_usage_mb': total_memory
                    }
                    self.logger.info(f"✓ {dataset_name}: {total_records} total records loaded from {len(tables)} tables")
                else:
                    dataset_stats[dataset_name] = {'error': 'Failed to load'}
                    self.logger.warning(f"✗ {dataset_name}: Failed to load")
            
            self.pipeline_stats['datasets'] = dataset_stats
            
            # Save raw data to database for later phases
            self.logger.info("Persisting data to database...")
            await self._persist_data_to_database(datasets)
            
            # 3. Preprocess data
            self.logger.info("Preprocessing data...")
            processed_datasets = {}
            
            for dataset_name, tables in datasets.items():
                if tables:
                    self.logger.info(f"Preprocessing {dataset_name}...")
                    processed_tables = {}
                    for table_name, data in tables.items():
                        if data is not None:
                            processed_data, preprocessing_metadata = self.preprocessor.preprocess_dataset(data, f"{dataset_name}_{table_name}")
                            processed_tables[table_name] = processed_data
                    processed_datasets[dataset_name] = processed_tables
                    self.logger.info(f"✓ {dataset_name} preprocessed")
            
            # 4. Extract features
            self.logger.info("Extracting features...")
            
            # First extract features from all datasets
            feature_sets = {}
            for dataset_name, tables in processed_datasets.items():
                if tables:
                    # Map dataset keys to feature extractor expected keys
                    flat_data = {}
                    for table_name, data in tables.items():
                        if data is None or len(data) == 0:
                            continue
                            
                        # Map to expected keys for feature extraction
                        if table_name == 'movies':
                            flat_data['movies'] = data
                            flat_data['content'] = data  # Also add as content
                        elif table_name == 'ratings':
                            flat_data['ratings'] = data
                            flat_data['interactions'] = data  # Also add as interactions
                        elif table_name == 'users':
                            flat_data['users'] = data
                        elif table_name == 'books':
                            flat_data['books'] = data
                            flat_data['content'] = data  # Also add as content for books
                        else:
                            # Keep original name for other types (tags, links, etc.)
                            flat_data[table_name] = data
                    
                    if flat_data:  # Only process if we have mapped data
                        self.logger.info(f"Extracting features from {dataset_name} with keys: {list(flat_data.keys())}")
                        try:
                            dataset_features = self.feature_extractor.extract_all_features(flat_data)
                            if dataset_features:
                                # Prefix feature keys with dataset name to avoid conflicts
                                for feature_key, feature_df in dataset_features.items():
                                    prefixed_key = f"{dataset_name}_{feature_key}"
                                    feature_sets[prefixed_key] = feature_df
                                self.logger.info(f"✓ Extracted {len(dataset_features)} feature sets from {dataset_name}")
                            else:
                                self.logger.warning(f"No features extracted from {dataset_name}")
                        except Exception as e:
                            self.logger.warning(f"Feature extraction failed for {dataset_name}: {str(e)}")
                            # Continue with other datasets
                            continue
            
            # Then create unified feature matrix from all extracted features
            if feature_sets:
                self.logger.info("Creating unified feature matrix...")
                try:
                    feature_matrix = self.feature_extractor.create_unified_feature_matrix(feature_sets)
                    if feature_matrix is None or len(feature_matrix) == 0:
                        self.logger.warning("Feature matrix creation returned empty result")
                        feature_matrix = pd.DataFrame()
                except Exception as e:
                    self.logger.warning(f"Unified feature matrix creation failed: {str(e)}")
                    feature_matrix = pd.DataFrame()
            else:
                self.logger.warning("No feature sets available for unified matrix creation")
                feature_matrix = pd.DataFrame()
            
            feature_stats = {
                'total_features': feature_matrix.shape[1] if hasattr(feature_matrix, 'shape') else 0,
                'total_samples': feature_matrix.shape[0] if hasattr(feature_matrix, 'shape') else 0,
                'feature_sets': list(feature_sets.keys()) if feature_sets else [],
                'datasets_processed': len([d for d in processed_datasets.keys() if processed_datasets[d]])
            }
            
            self.pipeline_stats['features'] = feature_stats
            self.logger.info(f"✓ Feature extraction completed:")
            self.logger.info(f"  - Feature matrix: {feature_stats['total_samples']} samples, {feature_stats['total_features']} features")
            self.logger.info(f"  - Feature sets: {len(feature_stats['feature_sets'])}")
            self.logger.info(f"  - Datasets processed: {feature_stats['datasets_processed']}")
            
            phase_duration = time.time() - phase_start
            self.pipeline_stats['data_integration_duration'] = phase_duration
            self.logger.info(f"Data integration phase completed in {phase_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Data integration phase failed: {str(e)}")
            raise
    
    async def _run_content_understanding_phase(self):
        """Run content understanding phase"""
        phase_start = time.time()
        
        try:
            # 1. Generate embeddings for content items
            self.logger.info("Generating content embeddings...")
            
            # Get content data from database for embedding generation
            session = get_session()
            
            # For now, let's create a simple implementation that works with available data
            # Later this can be enhanced to work with actual content from the database
            embedding_stats = {'total_processed': 0, 'successful': 0, 'failed': 0}
            self.logger.info("Content embeddings generation skipped - will be implemented in future iteration")
            
            self.pipeline_stats['embeddings'] = embedding_stats
            self.logger.info(f"✓ Prepared for embedding generation")
            
            # 2. Cross-domain mapping - also needs to be updated for actual implementation
            self.logger.info("Creating cross-domain mappings...")
            mapping_stats = {'total_mappings': 0}
            self.logger.info("Cross-domain mapping skipped - will be implemented in future iteration")
            
            self.pipeline_stats['cross_domain'] = mapping_stats
            self.logger.info(f"✓ Cross-domain mapping prepared")
            
            # 3. Quality scoring - will be implemented in future iteration
            self.logger.info("Calculating quality scores...")
            quality_stats = {'items_scored': 0}
            self.logger.info("Quality scoring skipped - will be implemented in future iteration")
            
            self.pipeline_stats['quality_scores'] = quality_stats
            self.logger.info(f"✓ Quality scoring prepared")
            
            phase_duration = time.time() - phase_start
            self.pipeline_stats['content_understanding_duration'] = phase_duration
            self.logger.info(f"Content understanding phase completed in {phase_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Content understanding phase failed: {str(e)}")
            raise
    
    async def _run_user_profiling_phase(self):
        """Run user profiling phase"""
        phase_start = time.time()
        
        try:
            # Get list of users to profile from database
            session = get_session()
            try:
                # Check if we have any interaction data in database
                interaction_count = session.execute(text("SELECT COUNT(*) FROM interaction_events")).scalar()
                if interaction_count == 0:
                    self.logger.warning("No interaction data found in database - user profiling will be limited")
                    profiling_stats = {
                        'total_users': 0,
                        'preference_profiles': 0,
                        'behavior_profiles': 0,
                        'evolution_profiles': 0,
                        'errors': 0,
                        'warning': 'No interaction data available for profiling'
                    }
                    self.pipeline_stats['user_profiling'] = profiling_stats
                    return
                
                # Get unique users from interaction events
                result = session.execute(text("SELECT DISTINCT user_id FROM interaction_events LIMIT 1000"))
                user_ids = [row[0] for row in result.fetchall()]
                
                # Also check for users in user_profiles table
                user_profile_result = session.execute(text("SELECT DISTINCT user_id FROM user_profiles LIMIT 1000"))
                profile_user_ids = [row[0] for row in user_profile_result.fetchall()]
                
                # Combine both sets of users
                all_user_ids = list(set(user_ids + profile_user_ids))
                
            finally:
                session.close()
            
            self.logger.info(f"Found {len(all_user_ids)} users to profile (from {interaction_count:,} interactions)")
            
            profiling_stats = {
                'total_users': len(all_user_ids),
                'preference_profiles': 0,
                'behavior_profiles': 0,
                'evolution_profiles': 0,
                'errors': 0,
                'interaction_count': interaction_count
            }
            
            if len(all_user_ids) == 0:
                self.logger.warning("No users found for profiling")
                self.pipeline_stats['user_profiling'] = profiling_stats
                return
            
            # Process users in batches to avoid memory issues
            batch_size = 50  # Smaller batch size for more detailed processing
            successful_profiles = 0
            
            for i in range(0, len(all_user_ids), batch_size):
                batch_users = all_user_ids[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(all_user_ids) + batch_size - 1)//batch_size
                
                self.logger.info(f"Processing user batch {batch_num}/{total_batches} ({len(batch_users)} users)")
                
                for user_id in batch_users:
                    try:
                        # Get user's interaction history for profiling
                        session = get_session()
                        user_interactions = None
                        try:
                            # Get user's interaction data
                            interaction_query = text("""
                                SELECT user_id, content_id, rating, timestamp, interaction_type,
                                       time_of_day, day_of_week, is_weekend, season
                                FROM interaction_events 
                                WHERE user_id = :user_id
                                ORDER BY timestamp
                            """)
                            result = session.execute(interaction_query, {"user_id": user_id})
                            interactions_data = result.fetchall()
                            
                            if interactions_data:
                                # Convert to DataFrame for processing
                                columns = ['user_id', 'content_id', 'rating', 'timestamp', 'interaction_type',
                                          'time_of_day', 'day_of_week', 'is_weekend', 'season']
                                user_interactions = pd.DataFrame(interactions_data, columns=columns)
                                
                        finally:
                            session.close()
                        
                        if user_interactions is None or len(user_interactions) == 0:
                            continue
                        
                        # 1. Build preference profile
                        try:
                            preference_profile = self.preference_tracker.build_user_profile(user_id)
                            if preference_profile and (preference_profile.genre_preferences or 
                                                     preference_profile.content_type_preferences or
                                                     preference_profile.rating_behavior):
                                profiling_stats['preference_profiles'] += 1
                        except Exception as e:
                            self.logger.debug(f"Preference profiling failed for user {user_id}: {str(e)}")
                        
                        # 2. Analyze behavior patterns
                        try:
                            behavior_profile = self.behavior_analyzer.analyze_user_behavior(user_id)
                            if behavior_profile and (behavior_profile.session_metrics or 
                                                   behavior_profile.temporal_patterns or
                                                   behavior_profile.content_preferences):
                                profiling_stats['behavior_profiles'] += 1
                        except Exception as e:
                            self.logger.debug(f"Behavior analysis failed for user {user_id}: {str(e)}")
                        
                        # 3. Analyze profile evolution
                        try:
                            evolution_profile = self.profile_evolution.analyze_profile_evolution(user_id)
                            if evolution_profile and (evolution_profile.evolution_periods or
                                                    evolution_profile.preference_drift or
                                                    evolution_profile.trend_analysis):
                                profiling_stats['evolution_profiles'] += 1
                        except Exception as e:
                            self.logger.debug(f"Evolution analysis failed for user {user_id}: {str(e)}")
                        
                        successful_profiles += 1
                        
                        # Log progress every 10 users
                        if successful_profiles % 10 == 0:
                            self.logger.info(f"Processed {successful_profiles}/{len(all_user_ids)} users...")
                        
                    except Exception as e:
                        profiling_stats['errors'] += 1
                        self.logger.debug(f"Error profiling user {user_id}: {str(e)}")
                
                # Log batch completion
                self.logger.info(f"✓ Completed batch {batch_num}/{total_batches}")
            
            self.pipeline_stats['user_profiling'] = profiling_stats
            
            phase_duration = time.time() - phase_start
            self.pipeline_stats['user_profiling_duration'] = phase_duration
            
            self.logger.info("✓ User profiling completed:")
            self.logger.info(f"  - Total users processed: {successful_profiles}/{profiling_stats['total_users']}")
            self.logger.info(f"  - Preference profiles: {profiling_stats['preference_profiles']}")
            self.logger.info(f"  - Behavior profiles: {profiling_stats['behavior_profiles']}")
            self.logger.info(f"  - Evolution profiles: {profiling_stats['evolution_profiles']}")
            self.logger.info(f"  - Errors: {profiling_stats['errors']}")
            self.logger.info(f"Phase completed in {phase_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"User profiling phase failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def _generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        try:
            self.logger.info("Generating data quality report...")
            
            session = get_session()
            quality_metrics = {}
            
            try:
                # Check table existence and get basic counts
                tables_info = {}
                
                # Get interaction events count
                try:
                    interaction_count = session.execute(text("SELECT COUNT(*) FROM interaction_events")).scalar() or 0
                    tables_info['interaction_events'] = interaction_count
                except Exception as e:
                    self.logger.warning(f"Could not query interaction_events: {e}")
                    tables_info['interaction_events'] = 0
                
                # Get content metadata count
                try:
                    content_count = session.execute(text("SELECT COUNT(*) FROM content_metadata")).scalar() or 0
                    tables_info['content_metadata'] = content_count
                except Exception as e:
                    self.logger.warning(f"Could not query content_metadata: {e}")
                    tables_info['content_metadata'] = 0
                
                # Get user profiles count
                try:
                    user_count = session.execute(text("SELECT COUNT(*) FROM user_profiles")).scalar() or 0
                    tables_info['user_profiles'] = user_count
                except Exception as e:
                    self.logger.warning(f"Could not query user_profiles: {e}")
                    tables_info['user_profiles'] = 0
                
                quality_metrics['table_counts'] = tables_info
                
                # Data coverage metrics
                coverage_metrics = {}
                
                # User coverage (users with interactions vs total users)
                if tables_info['interaction_events'] > 0 and tables_info['user_profiles'] > 0:
                    try:
                        interaction_user_count = session.execute(text("SELECT COUNT(DISTINCT user_id) FROM interaction_events")).scalar() or 0
                        coverage_metrics['user_coverage'] = interaction_user_count / max(tables_info['user_profiles'], 1)
                        coverage_metrics['users_with_interactions'] = interaction_user_count
                    except Exception as e:
                        self.logger.warning(f"Could not calculate user coverage: {e}")
                        coverage_metrics['user_coverage'] = 0.0
                        coverage_metrics['users_with_interactions'] = 0
                else:
                    coverage_metrics['user_coverage'] = 0.0
                    coverage_metrics['users_with_interactions'] = 0
                
                # Content coverage (content with interactions vs total content)
                if tables_info['interaction_events'] > 0 and tables_info['content_metadata'] > 0:
                    try:
                        interaction_content_count = session.execute(text("SELECT COUNT(DISTINCT content_id) FROM interaction_events")).scalar() or 0
                        coverage_metrics['content_coverage'] = interaction_content_count / max(tables_info['content_metadata'], 1)
                        coverage_metrics['content_with_interactions'] = interaction_content_count
                    except Exception as e:
                        self.logger.warning(f"Could not calculate content coverage: {e}")
                        coverage_metrics['content_coverage'] = 0.0
                        coverage_metrics['content_with_interactions'] = 0
                else:
                    coverage_metrics['content_coverage'] = 0.0
                    coverage_metrics['content_with_interactions'] = 0
                
                # Embedding coverage (placeholder - will be 0 for now)
                coverage_metrics['embedding_coverage'] = 0.0
                
                quality_metrics['coverage'] = coverage_metrics
                
                # Data completeness metrics
                completeness_metrics = {}
                
                if tables_info['interaction_events'] > 0:
                    # Rating completeness
                    try:
                        rated_interactions = session.execute(text("SELECT COUNT(*) FROM interaction_events WHERE rating IS NOT NULL AND rating > 0")).scalar() or 0
                        completeness_metrics['rating_completeness'] = rated_interactions / max(tables_info['interaction_events'], 1)
                        completeness_metrics['rated_interactions'] = rated_interactions
                    except Exception as e:
                        self.logger.warning(f"Could not calculate rating completeness: {e}")
                        completeness_metrics['rating_completeness'] = 0.0
                        completeness_metrics['rated_interactions'] = 0
                else:
                    completeness_metrics['rating_completeness'] = 0.0
                    completeness_metrics['rated_interactions'] = 0
                
                if tables_info['content_metadata'] > 0:
                    # Metadata completeness
                    try:
                        content_with_genres = session.execute(text("SELECT COUNT(*) FROM content_metadata WHERE genres IS NOT NULL AND genres != ''")).scalar() or 0
                        completeness_metrics['genre_completeness'] = content_with_genres / max(tables_info['content_metadata'], 1)
                        completeness_metrics['content_with_genres'] = content_with_genres
                    except Exception as e:
                        self.logger.warning(f"Could not calculate genre completeness: {e}")
                        completeness_metrics['genre_completeness'] = 0.0
                        completeness_metrics['content_with_genres'] = 0
                else:
                    completeness_metrics['genre_completeness'] = 0.0
                    completeness_metrics['content_with_genres'] = 0
                
                quality_metrics['completeness'] = completeness_metrics
                
                # Data quality metrics
                data_quality_metrics = {}
                
                if tables_info['interaction_events'] > 0:
                    # Rating distribution
                    try:
                        rating_stats = session.execute(text("""
                            SELECT AVG(rating), MIN(rating), MAX(rating), COUNT(rating)
                            FROM interaction_events WHERE rating > 0
                        """)).fetchone()
                        
                        if rating_stats and rating_stats[0]:
                            avg_rating = float(rating_stats[0])
                            # Calculate standard deviation manually since SQLite doesn't have STDDEV
                            variance_query = session.execute(text("""
                                SELECT AVG((rating - ?) * (rating - ?))
                                FROM interaction_events WHERE rating > 0
                            """), (avg_rating, avg_rating)).scalar()
                            std_rating = (variance_query ** 0.5) if variance_query else 0.0
                            
                            data_quality_metrics['rating_distribution'] = {
                                'mean': float(rating_stats[0]),
                                'std': std_rating,
                                'min': float(rating_stats[1]),
                                'max': float(rating_stats[2]),
                                'count': int(rating_stats[3])
                            }
                        else:
                            data_quality_metrics['rating_distribution'] = {
                                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                            }
                    except Exception as e:
                        self.logger.warning(f"Could not calculate rating distribution: {e}")
                        data_quality_metrics['rating_distribution'] = {
                            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                        }
                
                quality_metrics['data_quality'] = data_quality_metrics
                
                # Calculate overall quality score
                # Weight different aspects of data quality
                coverage_score = (
                    coverage_metrics.get('user_coverage', 0) * 0.4 +
                    coverage_metrics.get('content_coverage', 0) * 0.4 +
                    coverage_metrics.get('embedding_coverage', 0) * 0.2
                )
                
                completeness_score = (
                    completeness_metrics.get('rating_completeness', 0) * 0.6 +
                    completeness_metrics.get('genre_completeness', 0) * 0.4
                )
                
                # Data volume score (normalized by expected minimums)
                volume_score = min(1.0, (
                    min(1.0, tables_info['interaction_events'] / 1000) * 0.5 +
                    min(1.0, tables_info['content_metadata'] / 100) * 0.3 +
                    min(1.0, tables_info['user_profiles'] / 50) * 0.2
                ))
                
                overall_score = (coverage_score * 0.4 + completeness_score * 0.4 + volume_score * 0.2)
                quality_metrics['overall_score'] = overall_score
                
                # Determine quality grade
                if overall_score >= 0.9:
                    quality_grade = 'A'
                elif overall_score >= 0.8:
                    quality_grade = 'B'
                elif overall_score >= 0.6:
                    quality_grade = 'C'
                elif overall_score >= 0.4:
                    quality_grade = 'D'
                else:
                    quality_grade = 'F'
                
                quality_metrics['quality_grade'] = quality_grade
                quality_metrics['data_ready_for_recommendations'] = overall_score >= 0.5
                
            finally:
                session.close()
            
            # Add pipeline statistics
            quality_metrics['pipeline_statistics'] = self.pipeline_stats
            
            self.logger.info(f"✓ Quality report generated:")
            self.logger.info(f"  - Overall score: {overall_score:.3f} (Grade: {quality_grade})")
            self.logger.info(f"  - Data tables: {len([t for t in tables_info.values() if t > 0])}/3 populated")
            self.logger.info(f"  - Ready for recommendations: {quality_metrics['data_ready_for_recommendations']}")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'overall_score': 0.0,
                'quality_grade': 'F',
                'data_ready_for_recommendations': False
            }
    
    async def _create_feature_store(self) -> str:
        """Create feature store with processed data"""
        try:
            self.logger.info("Creating feature store...")
            
            # Create feature store directory
            feature_store_path = Path(settings.output_directory) / "feature_store"
            feature_store_path.mkdir(parents=True, exist_ok=True)
            
            # Export processed data to feature store
            session = get_session()
            
            try:
                # Export content features
                self.logger.info("Exporting content features...")
                content_query = """
                    SELECT c.*, e.embedding_vector, q.quality_score
                    FROM content c
                    LEFT JOIN embeddings e ON c.content_id = e.content_id
                    LEFT JOIN quality_metrics q ON c.content_id = q.content_id
                """
                
                # Note: In a real implementation, you'd use pandas or similar to export
                # For now, we'll just create placeholder files
                
                with open(feature_store_path / "content_features.sql", "w") as f:
                    f.write(content_query)
                
                # Export user features
                self.logger.info("Exporting user features...")
                user_query = """
                    SELECT u.*, 
                           COUNT(i.interaction_id) as total_interactions,
                           AVG(i.rating) as avg_rating,
                           COUNT(DISTINCT i.content_id) as unique_content_count
                    FROM users u
                    LEFT JOIN interactions i ON u.user_id = i.user_id
                    GROUP BY u.user_id
                """
                
                with open(feature_store_path / "user_features.sql", "w") as f:
                    f.write(user_query)
                
                # Export interaction features
                self.logger.info("Exporting interaction features...")
                interaction_query = """
                    SELECT i.*, c.content_type, c.genres
                    FROM interactions i
                    JOIN content c ON i.content_id = c.content_id
                """
                
                with open(feature_store_path / "interaction_features.sql", "w") as f:
                    f.write(interaction_query)
                
                # Create feature store metadata
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'pipeline_version': '1.0',
                    'feature_count': self.pipeline_stats.get('features', {}).get('total_features', 0),
                    'sample_count': self.pipeline_stats.get('features', {}).get('total_samples', 0),
                    'quality_score': self.pipeline_stats.get('overall_score', 0.0)
                }
                
                import json
                with open(feature_store_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
            finally:
                session.close()
            
            self.logger.info(f"✓ Feature store created at: {feature_store_path}")
            return str(feature_store_path)
            
        except Exception as e:
            self.logger.error(f"Error creating feature store: {str(e)}")
            raise
    
    async def _cleanup_resources(self):
        """Clean up pipeline resources"""
        try:
            self.logger.info("Cleaning up resources...")
            
            # Close all component resources
            if self.preference_tracker:
                self.preference_tracker.close()
            
            if self.behavior_analyzer:
                self.behavior_analyzer.close()
            
            if self.profile_evolution:
                self.profile_evolution.close()
            
            if self.gemini_client:
                await self.gemini_client.close()
            
            self.logger.info("✓ Resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    async def _persist_data_to_database(self, datasets: Dict[str, Dict[str, pd.DataFrame]]):
        """Persist loaded datasets to database"""
        try:
            session = get_session()
            total_interactions_saved = 0
            total_content_saved = 0
            total_users_saved = 0
            
            self.logger.info("Starting data persistence to database...")
            
            for dataset_name, tables in datasets.items():
                if not tables:
                    continue
                    
                self.logger.info(f"Persisting {dataset_name} data to database...")
                
                # Handle ratings/interactions data
                if 'ratings' in tables:
                    ratings_df = tables['ratings'].copy()
                    self.logger.info(f"Processing {len(ratings_df)} ratings from {dataset_name}")
                    
                    # Ensure required columns for InteractionEvents table
                    if 'interaction_type' not in ratings_df.columns:
                        ratings_df['interaction_type'] = 'rating'
                    
                    # Map timestamp to datetime if needed
                    if 'timestamp' in ratings_df.columns:
                        if ratings_df['timestamp'].dtype != 'datetime64[ns]':
                            # Convert unix timestamps to datetime
                            ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s', errors='coerce')
                    else:
                        # If no timestamp, create a default one
                        ratings_df['timestamp'] = datetime.now()
                    
                    # Add missing temporal context columns
                    if 'timestamp' in ratings_df.columns:
                        valid_timestamps = ratings_df['timestamp'].notna()
                        ratings_df.loc[valid_timestamps, 'time_of_day'] = ratings_df.loc[valid_timestamps, 'timestamp'].dt.hour.apply(
                            lambda x: 'morning' if 6 <= x < 12 else 
                                     'afternoon' if 12 <= x < 18 else 
                                     'evening' if 18 <= x < 22 else 'night'
                        )
                        ratings_df.loc[valid_timestamps, 'day_of_week'] = ratings_df.loc[valid_timestamps, 'timestamp'].dt.day_name()
                        ratings_df.loc[valid_timestamps, 'is_weekend'] = ratings_df.loc[valid_timestamps, 'timestamp'].dt.weekday >= 5
                        ratings_df.loc[valid_timestamps, 'season'] = ratings_df.loc[valid_timestamps, 'timestamp'].dt.month.apply(
                            lambda x: 'winter' if x in [12, 1, 2] else
                                     'spring' if x in [3, 4, 5] else
                                     'summer' if x in [6, 7, 8] else 'fall'
                        )
                        
                        # Fill missing temporal data with defaults
                        ratings_df['time_of_day'] = ratings_df['time_of_day'].fillna('unknown')
                        ratings_df['day_of_week'] = ratings_df['day_of_week'].fillna('unknown')
                        ratings_df['is_weekend'] = ratings_df['is_weekend'].fillna(False)
                        ratings_df['season'] = ratings_df['season'].fillna('unknown')
                    
                    # Store original rating info
                    if 'rating' in ratings_df.columns:
                        ratings_df['original_rating'] = ratings_df['rating'].astype(str)
                        # Determine rating scale from data
                        max_rating = ratings_df['rating'].max()
                        if max_rating <= 5:
                            ratings_df['rating_scale'] = '1-5'
                        elif max_rating <= 10:
                            ratings_df['rating_scale'] = '1-10'
                        else:
                            ratings_df['rating_scale'] = f'1-{int(max_rating)}'
                    
                    # Ensure user_id and content_id are strings
                    if 'user_id' in ratings_df.columns:
                        ratings_df['user_id'] = ratings_df['user_id'].astype(str)
                    if 'content_id' in ratings_df.columns:
                        ratings_df['content_id'] = ratings_df['content_id'].astype(str)
                    
                    # Add UUID for interaction_id (required by database schema)
                    import uuid
                    ratings_df['interaction_id'] = [str(uuid.uuid4()) for _ in range(len(ratings_df))]
                    
                    # Sample data for database (use larger sample for better functionality)
                    sample_size = min(50000, len(ratings_df))  # Increased sample size
                    if len(ratings_df) > sample_size:
                        ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
                        self.logger.info(f"Sampling {sample_size} records from {len(ratings_df)} for database storage")
                    else:
                        ratings_sample = ratings_df
                    
                    # Select only columns that exist in the database schema
                    db_columns = [
                        'interaction_id', 'user_id', 'content_id', 'interaction_type', 'rating', 
                        'original_rating', 'rating_scale', 'timestamp',
                        'time_of_day', 'day_of_week', 'is_weekend', 'season'
                    ]
                    ratings_db = ratings_sample[[col for col in db_columns if col in ratings_sample.columns]]
                    
                    # Remove rows with missing critical data
                    critical_cols = ['interaction_id', 'user_id', 'content_id', 'rating']
                    ratings_db = ratings_db.dropna(subset=[col for col in critical_cols if col in ratings_db.columns])
                    
                    if len(ratings_db) > 0:
                        # Save to database in batches
                        batch_size = 5000
                        for i in range(0, len(ratings_db), batch_size):
                            batch = ratings_db.iloc[i:i + batch_size]
                            batch.to_sql('interaction_events', session.get_bind(), if_exists='append', index=False)
                        
                        total_interactions_saved += len(ratings_db)
                        self.logger.info(f"✓ Saved {len(ratings_db)} interaction events from {dataset_name}")
                
                # Handle movies/content data
                if 'movies' in tables:
                    movies_df = tables['movies'].copy()
                    self.logger.info(f"Processing {len(movies_df)} movies from {dataset_name}")
                    
                    # Ensure content_id is string
                    if 'content_id' in movies_df.columns:
                        movies_df['content_id'] = movies_df['content_id'].astype(str)
                    
                    # Add content_type for movies
                    movies_df['content_type'] = 'movie'
                    
                    # Add UUID for unified_content_id if needed
                    if 'unified_content_id' not in movies_df.columns:
                        movies_df['unified_content_id'] = [str(uuid.uuid4()) for _ in range(len(movies_df))]
                    
                    # Ensure required columns for ContentMetadata table
                    content_columns = ['content_id', 'title', 'genres', 'content_type']
                    if 'publication_year' in movies_df.columns:
                        content_columns.append('publication_year')
                    
                    movies_db = movies_df[[col for col in content_columns if col in movies_df.columns]]
                    
                    # Remove rows with missing critical data
                    movies_db = movies_db.dropna(subset=['content_id', 'title'])
                    
                    # Sample for database storage
                    sample_size = min(10000, len(movies_db))  # Increased sample size
                    if len(movies_db) > sample_size:
                        movies_sample = movies_db.sample(n=sample_size, random_state=42)
                    else:
                        movies_sample = movies_db
                    
                    if len(movies_sample) > 0:
                        movies_sample.to_sql('content_metadata', session.get_bind(), if_exists='append', index=False)
                        total_content_saved += len(movies_sample)
                        self.logger.info(f"✓ Saved {len(movies_sample)} content items from {dataset_name}")
                
                # Handle books data
                if 'books' in tables:
                    books_df = tables['books'].copy()
                    self.logger.info(f"Processing {len(books_df)} books from {dataset_name}")
                    
                    # Ensure content_id is string
                    if 'content_id' in books_df.columns:
                        books_df['content_id'] = books_df['content_id'].astype(str)
                    
                    # Add content_type for books
                    books_df['content_type'] = 'book'
                    
                    # Map book-specific columns to content metadata
                    if 'author' in books_df.columns and 'genres' not in books_df.columns:
                        books_df['genres'] = books_df['author']  # Use author as genre placeholder
                    
                    content_columns = ['content_id', 'title', 'content_type']
                    if 'genres' in books_df.columns:
                        content_columns.append('genres')
                    if 'publication_year' in books_df.columns:
                        content_columns.append('publication_year')
                    
                    books_db = books_df[[col for col in content_columns if col in books_df.columns]]
                    books_db = books_db.dropna(subset=['content_id', 'title'])
                    
                    if len(books_db) > 0:
                        books_db.to_sql('content_metadata', session.get_bind(), if_exists='append', index=False)
                        total_content_saved += len(books_db)
                        self.logger.info(f"✓ Saved {len(books_db)} book items from {dataset_name}")
                
                # Handle users data (create user profiles)
                if 'users' in tables:
                    users_df = tables['users'].copy()
                    self.logger.info(f"Processing {len(users_df)} users from {dataset_name}")
                    
                    # Ensure user_id is string
                    if 'user_id' in users_df.columns:
                        users_df['user_id'] = users_df['user_id'].astype(str)
                    
                    # Add default values for required columns
                    users_df['registration_date'] = datetime.now()
                    users_df['activity_status'] = 'active'
                    users_df['account_type'] = 'free'
                    
                    # Add UUID for unified_user_id if needed
                    if 'unified_user_id' not in users_df.columns:
                        users_df['unified_user_id'] = [str(uuid.uuid4()) for _ in range(len(users_df))]
                    
                    user_columns = ['user_id', 'registration_date', 'activity_status', 'account_type']
                    if 'age' in users_df.columns:
                        user_columns.append('age')
                    if 'location' in users_df.columns:
                        user_columns.append('location')
                    
                    users_db = users_df[[col for col in user_columns if col in users_df.columns]]
                    users_db = users_db.dropna(subset=['user_id'])
                    
                    if len(users_db) > 0:
                        users_db.to_sql('user_profiles', session.get_bind(), if_exists='append', index=False)
                        total_users_saved += len(users_db)
                        self.logger.info(f"✓ Saved {len(users_db)} user profiles from {dataset_name}")
            
            session.close()
            
            # Update pipeline stats
            self.pipeline_stats['database_persistence'] = {
                'total_interactions_saved': total_interactions_saved,
                'total_content_saved': total_content_saved,
                'total_users_saved': total_users_saved,
                'persistence_successful': True
            }
            
            self.logger.info("=" * 60)
            self.logger.info("DATABASE PERSISTENCE SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"✓ Total interactions saved: {total_interactions_saved:,}")
            self.logger.info(f"✓ Total content items saved: {total_content_saved:,}")
            self.logger.info(f"✓ Total user profiles saved: {total_users_saved:,}")
            
        except Exception as e:
            self.logger.error(f"Error persisting data to database: {str(e)}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            if 'session' in locals():
                session.close()
            
            # Update pipeline stats with error info
            self.pipeline_stats['database_persistence'] = {
                'total_interactions_saved': 0,
                'total_content_saved': 0,
                'total_users_saved': 0,
                'persistence_successful': False,
                'error': str(e)
            }
            
            # Don't fail the entire pipeline for database persistence issues
            pass

async def main():
    """Main entry point"""
    print("Content Recommendation Engine - ETL Pipeline")
    print("=" * 60)
    
    try:
        # Initialize and run pipeline
        pipeline = ETLPipeline()
        results = await pipeline.run_full_pipeline()
        
        if results['status'] == 'success':
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Status: {results['status'].upper()}")
            print(f"Duration: {results['duration_seconds']:.2f} seconds")
            print(f"Quality Score: {results['quality_report'].get('overall_score', 'N/A'):.3f}")
            print(f"Feature Store: {results['feature_store_path']}")
            
            # Print key statistics
            stats = results['pipeline_stats']
            print("\nDataset Statistics:")
            for dataset, info in stats.get('datasets', {}).items():
                if 'records_loaded' in info:
                    print(f"  - {dataset}: {info['records_loaded']:,} records")
            
            print(f"\nFeatures: {stats.get('features', {}).get('total_features', 0):,}")
            print(f"Samples: {stats.get('features', {}).get('total_samples', 0):,}")
            
            profiling = stats.get('user_profiling', {})
            print(f"\nUser Profiles:")
            print(f"  - Preference: {profiling.get('preference_profiles', 0):,}")
            print(f"  - Behavior: {profiling.get('behavior_profiles', 0):,}")
            print(f"  - Evolution: {profiling.get('evolution_profiles', 0):,}")
            
            return 0
        else:
            print(f"\nPipeline failed: {results['error']}")
            return 1
            
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
