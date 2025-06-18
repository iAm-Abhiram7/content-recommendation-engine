"""
Unified Schema Manager for Content Recommendation Engine
Handles schema definitions, validation, and cross-dataset compatibility
"""
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy import create_engine, MetaData
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime
import json
from sqlalchemy.orm import sessionmaker
from ..utils.config import settings

Base = declarative_base()


class UserProfile(Base):
    """User profile table with comprehensive user information"""
    __tablename__ = 'user_profiles'
    
    user_id = sa.Column(sa.String(50), primary_key=True)
    unified_user_id = sa.Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    
    # Demographics
    age = sa.Column(sa.Integer)
    gender = sa.Column(sa.String(20))
    location = sa.Column(sa.String(100))
    occupation = sa.Column(sa.String(100))
    education_level = sa.Column(sa.String(50))
    income_bracket = sa.Column(sa.String(30))
    
    # Registration and Activity
    registration_date = sa.Column(sa.DateTime, default=datetime.utcnow)
    last_active_date = sa.Column(sa.DateTime)
    activity_status = sa.Column(sa.String(20), default='active')  # active, inactive, suspended
    account_type = sa.Column(sa.String(20), default='free')  # free, premium, trial
    
    # Cross-platform identifiers
    cross_platform_ids = sa.Column(JSON)  # {"movielens": "123", "amazon": "456", etc.}
    external_ids = sa.Column(JSON)  # Additional external identifiers
    
    # Preference tracking
    preference_history = sa.Column(JSON)  # Historical preference changes
    explicit_preferences = sa.Column(JSON)  # Explicitly stated preferences
    privacy_settings = sa.Column(JSON)  # User privacy and sharing preferences
    
    # Computed metrics
    profile_completeness = sa.Column(sa.Float, default=0.0)
    engagement_score = sa.Column(sa.Float, default=0.0)
    preference_diversity = sa.Column(sa.Float, default=0.0)
    
    # Metadata
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_quality_score = sa.Column(sa.Float, default=0.0)
    source_datasets = sa.Column(JSON)  # Which datasets this user appears in


class ContentMetadata(Base):
    """Content metadata table supporting multiple content types"""
    __tablename__ = 'content_metadata'
    
    content_id = sa.Column(sa.String(50), primary_key=True)
    unified_content_id = sa.Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    
    # Basic Information
    title = sa.Column(sa.String(500), nullable=False)
    original_title = sa.Column(sa.String(500))
    description = sa.Column(sa.Text)
    plot_summary = sa.Column(sa.Text)
    content_type = sa.Column(sa.String(20), nullable=False)  # movie, book, music, tv_show
    
    # Genre and Classification
    genres = sa.Column(JSON)  # ["Action", "Comedy", "Drama"]
    categories = sa.Column(JSON)  # More detailed categorization
    tags = sa.Column(JSON)  # User-generated or system tags
    content_rating = sa.Column(sa.String(10))  # PG, R, etc.
    
    # Temporal Information
    release_date = sa.Column(sa.Date)
    publication_year = sa.Column(sa.Integer)
    duration = sa.Column(sa.Integer)  # Duration in minutes
    episode_count = sa.Column(sa.Integer)  # For TV shows
    
    # Creator Information
    creator_info = sa.Column(JSON)  # Directors, authors, artists, etc.
    cast_info = sa.Column(JSON)  # Cast and crew information
    studio_publisher = sa.Column(sa.String(200))
    production_countries = sa.Column(JSON)
    
    # Content Characteristics
    language = sa.Column(sa.String(10))
    original_language = sa.Column(sa.String(10))
    region = sa.Column(sa.String(50))
    format_info = sa.Column(JSON)  # HD, 4K, audiobook format, etc.
    accessibility_features = sa.Column(JSON)  # Subtitles, audio descriptions
    
    # Quality and Popularity Metrics
    quality_metrics = sa.Column(JSON)  # Various quality scores
    popularity_score = sa.Column(sa.Float, default=0.0)
    critical_score = sa.Column(sa.Float)
    user_score = sa.Column(sa.Float)
    awards_info = sa.Column(JSON)
    
    # AI-Generated Features
    content_embedding = sa.Column(JSON)  # Gemini-generated embedding
    mood_profile = sa.Column(JSON)  # Emotional characteristics
    complexity_score = sa.Column(sa.Float)
    style_indicators = sa.Column(JSON)
    themes_extracted = sa.Column(JSON)  # AI-extracted themes
    
    # Cross-domain mappings
    similar_content_ids = sa.Column(JSON)  # Related content across domains
    adaptation_links = sa.Column(JSON)  # Book->Movie mappings, etc.
    franchise_info = sa.Column(JSON)  # Series, franchise relationships
    
    # Metadata
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_quality_score = sa.Column(sa.Float, default=0.0)
    source_datasets = sa.Column(JSON)
    last_embedding_update = sa.Column(sa.DateTime)


class InteractionEvents(Base):
    """User-content interaction events with comprehensive context"""
    __tablename__ = 'interaction_events'
    
    interaction_id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = sa.Column(sa.String(50), sa.ForeignKey('user_profiles.user_id'), nullable=False)
    content_id = sa.Column(sa.String(50), sa.ForeignKey('content_metadata.content_id'), nullable=False)
    
    # Interaction Details
    interaction_type = sa.Column(sa.String(30), nullable=False)  # rating, view, purchase, etc.
    rating = sa.Column(sa.Float)  # Normalized rating (0-1)
    original_rating = sa.Column(sa.String(10))  # Original rating format
    rating_scale = sa.Column(sa.String(10))  # 1-5, 1-10, thumbs, etc.
    
    # Temporal Context
    timestamp = sa.Column(sa.DateTime, nullable=False)
    session_id = sa.Column(sa.String(50))
    interaction_sequence = sa.Column(sa.Integer)  # Order within session
    time_of_day = sa.Column(sa.String(10))  # morning, afternoon, evening, night
    day_of_week = sa.Column(sa.String(10))
    is_weekend = sa.Column(sa.Boolean)
    season = sa.Column(sa.String(10))
    
    # Device and Platform Context
    device_info = sa.Column(JSON)  # Device type, OS, browser, etc.
    platform = sa.Column(sa.String(50))  # netflix, amazon, spotify, etc.
    app_version = sa.Column(sa.String(20))
    screen_size = sa.Column(sa.String(20))  # mobile, tablet, desktop, tv
    
    # Location Context
    location_context = sa.Column(sa.String(30))  # home, work, travel, unknown
    geo_location = sa.Column(JSON)  # Country, state, city if available
    timezone = sa.Column(sa.String(50))
    
    # Social Context
    social_context = sa.Column(sa.String(30))  # alone, family, friends, unknown
    social_influence = sa.Column(JSON)  # Recommendations from friends, etc.
    
    # Interaction Behavior
    interaction_duration = sa.Column(sa.Integer)  # Duration in seconds
    completion_status = sa.Column(sa.Float)  # 0-1, how much was consumed
    skip_reasons = sa.Column(JSON)  # Why content was skipped
    replay_count = sa.Column(sa.Integer, default=0)
    
    # Recommendation Context
    recommendation_source = sa.Column(sa.String(50))  # algorithm, trending, friend, etc.
    recommendation_rank = sa.Column(sa.Integer)  # Position in recommendation list
    recommendation_context = sa.Column(JSON)  # Additional recommendation metadata
    
    # Mood and Environmental Context
    mood_indicators = sa.Column(JSON)  # Inferred or explicit mood data
    environmental_factors = sa.Column(JSON)  # Weather, events, etc.
    activity_context = sa.Column(sa.String(50))  # commuting, relaxing, working
    
    # Metadata
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    data_source = sa.Column(sa.String(50))  # movielens, amazon, etc.
    data_quality_score = sa.Column(sa.Float, default=0.0)
    processed_features = sa.Column(JSON)  # Extracted features for ML


class ContextualFeatures(Base):
    """Contextual features for personalization"""
    __tablename__ = 'contextual_features'
    
    context_id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = sa.Column(sa.String(50), sa.ForeignKey('user_profiles.user_id'), nullable=False)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    
    # Temporal Context
    temporal_context = sa.Column(JSON)  # Detailed time-based features
    temporal_patterns = sa.Column(JSON)  # User's historical patterns
    
    # Environmental Context
    device_context = sa.Column(JSON)  # Device capabilities and state
    network_context = sa.Column(JSON)  # Connection quality, type
    location_context = sa.Column(JSON)  # Current location details
    
    # Social Context
    social_context = sa.Column(JSON)  # Social situation and influences
    social_network_activity = sa.Column(JSON)  # Friends' recent activity
    
    # Personal Context
    mood_indicators = sa.Column(JSON)  # Current mood estimation
    activity_level = sa.Column(sa.String(20))  # active, passive, browsing
    attention_span = sa.Column(sa.Float)  # Estimated available attention
    stress_indicators = sa.Column(JSON)  # Work hours, deadlines, etc.
    
    # Content Context
    recent_interactions = sa.Column(JSON)  # Recent viewing/rating history
    search_history = sa.Column(JSON)  # Recent searches and queries
    browsing_behavior = sa.Column(JSON)  # Current session behavior
    
    # Derived Features
    recommendation_readiness = sa.Column(sa.Float)  # How ready for recommendations
    exploration_vs_exploitation = sa.Column(sa.Float)  # User's current preference
    novelty_preference = sa.Column(sa.Float)  # Preference for new vs familiar
    
    # Metadata
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    context_window_minutes = sa.Column(sa.Integer, default=60)  # Context validity window


class ContentEmbeddings(Base):
    """AI-generated content embeddings and features"""
    __tablename__ = 'content_embeddings'
    
    embedding_id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = sa.Column(sa.String(50), sa.ForeignKey('content_metadata.content_id'), nullable=False)
    
    # Embeddings
    semantic_embedding = sa.Column(JSON)  # Gemini-generated semantic vector
    visual_embedding = sa.Column(JSON)  # For visual content (movie posters, etc.)
    audio_embedding = sa.Column(JSON)  # For music and audio content
    text_embedding = sa.Column(JSON)  # For text descriptions
    
    # AI-Generated Features
    sentiment_scores = sa.Column(JSON)  # Emotional sentiment analysis
    theme_vectors = sa.Column(JSON)  # Extracted themes and topics
    style_vectors = sa.Column(JSON)  # Style and artistic elements
    complexity_metrics = sa.Column(JSON)  # Content complexity analysis
    
    # Cross-Domain Features
    cross_domain_similarities = sa.Column(JSON)  # Similarities to other content types
    genre_probabilities = sa.Column(JSON)  # Probability of belonging to genres
    mood_classification = sa.Column(JSON)  # Mood and atmosphere classification
    
    # Quality and Metadata
    embedding_model = sa.Column(sa.String(50))  # Model used for generation
    embedding_version = sa.Column(sa.String(20))  # Version of embeddings
    generation_timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    quality_score = sa.Column(sa.Float)  # Embedding quality assessment
    
    # Processing Metadata
    processing_time_seconds = sa.Column(sa.Float)
    token_count = sa.Column(sa.Integer)  # Tokens used for generation
    api_cost = sa.Column(sa.Float)  # Cost of generating embedding


class UserProfiles(Base):
    """Comprehensive user profiles with preferences and behavior patterns"""
    __tablename__ = 'user_profiles_detailed'
    
    profile_id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = sa.Column(sa.String(50), sa.ForeignKey('user_profiles.user_id'), nullable=False)
    
    # Explicit Preferences
    favorite_genres = sa.Column(JSON)  # Genre preferences with weights
    favorite_creators = sa.Column(JSON)  # Preferred directors, authors, etc.
    content_type_preferences = sa.Column(JSON)  # Movie vs book vs music preferences
    rating_history = sa.Column(JSON)  # Historical ratings with timestamps
    
    # Blacklists and Filters
    blacklisted_content = sa.Column(JSON)  # Explicitly rejected content
    content_filters = sa.Column(JSON)  # Content filtering preferences
    trigger_warnings = sa.Column(JSON)  # Content to avoid
    
    # Implicit Behavioral Patterns
    viewing_patterns = sa.Column(JSON)  # Time-based viewing habits
    session_patterns = sa.Column(JSON)  # Typical session characteristics
    completion_patterns = sa.Column(JSON)  # Content completion behavior
    search_patterns = sa.Column(JSON)  # Search and discovery behavior
    
    # Preference Evolution
    preference_trajectory = sa.Column(JSON)  # How preferences change over time
    trend_following = sa.Column(sa.Float)  # Tendency to follow trends
    novelty_seeking = sa.Column(sa.Float)  # Preference for new content
    genre_exploration = sa.Column(sa.Float)  # Willingness to try new genres
    
    # Social Preferences
    social_influence_sensitivity = sa.Column(sa.Float)  # How much friends' opinions matter
    review_reliance = sa.Column(sa.Float)  # How much reviews influence decisions
    popularity_bias = sa.Column(sa.Float)  # Preference for popular content
    
    # Contextual Preferences
    mood_content_mapping = sa.Column(JSON)  # What content for which moods
    time_content_preferences = sa.Column(JSON)  # Time-based content preferences
    device_content_preferences = sa.Column(JSON)  # Device-specific preferences
    
    # Computed Metrics
    preference_consistency = sa.Column(sa.Float)  # How consistent preferences are
    profile_stability = sa.Column(sa.Float)  # How stable the profile is
    engagement_predictability = sa.Column(sa.Float)  # How predictable engagement is
    
    # Metadata
    last_updated = sa.Column(sa.DateTime, default=datetime.utcnow)
    update_frequency = sa.Column(sa.String(20))  # daily, weekly, monthly
    profile_version = sa.Column(sa.Integer, default=1)


class DataQualityMetrics(Base):
    """Track data quality metrics across all datasets"""
    __tablename__ = 'data_quality_metrics'
    
    metric_id = sa.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name = sa.Column(sa.String(50), nullable=False)
    table_name = sa.Column(sa.String(50), nullable=False)
    
    # Quality Scores
    completeness_score = sa.Column(sa.Float)  # Percentage of non-null values
    consistency_score = sa.Column(sa.Float)  # Data consistency across fields
    accuracy_score = sa.Column(sa.Float)  # Data accuracy assessment
    timeliness_score = sa.Column(sa.Float)  # How current the data is
    validity_score = sa.Column(sa.Float)  # Data format and range validity
    
    # Coverage Metrics
    total_records = sa.Column(sa.Integer)
    valid_records = sa.Column(sa.Integer)
    records_with_issues = sa.Column(sa.Integer)
    coverage_percentage = sa.Column(sa.Float)
    
    # Issue Tracking
    critical_issues = sa.Column(JSON)  # Critical data issues
    error_issues = sa.Column(JSON)  # Error-level issues
    warning_issues = sa.Column(JSON)  # Warning-level issues
    recommendations = sa.Column(JSON)  # Suggested improvements
    
    # Processing Metadata
    measurement_timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    processing_duration = sa.Column(sa.Float)  # Time to compute metrics
    data_lineage = sa.Column(JSON)  # Source and transformation history


class SchemaManager:
    """Manages database schema operations and validations"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.metadata = MetaData()
        
    def create_all_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self.engine)
        
    def drop_all_tables(self):
        """Drop all tables (use with caution)"""
        Base.metadata.drop_all(self.engine)
        
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table"""
        table = Base.metadata.tables.get(table_name)
        if not table:
            return {}
            
        return {
            'name': table.name,
            'columns': [
                {
                    'name': col.name,
                    'type': str(col.type),
                    'nullable': col.nullable,
                    'primary_key': col.primary_key,
                    'foreign_keys': [str(fk) for fk in col.foreign_keys]
                }
                for col in table.columns
            ]
        }
    
    def validate_schema_compatibility(self, data_dict: Dict[str, Any], table_name: str) -> List[str]:
        """Validate if data dictionary is compatible with table schema"""
        issues = []
        table = Base.metadata.tables.get(table_name)
        
        if not table:
            issues.append(f"Table {table_name} not found in schema")
            return issues
            
        # Check for missing required columns
        required_columns = [col.name for col in table.columns if not col.nullable and not col.default]
        missing_columns = [col for col in required_columns if col not in data_dict]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            
        # Check for extra columns
        extra_columns = [key for key in data_dict.keys() if key not in [col.name for col in table.columns]]
        if extra_columns:
            issues.append(f"Extra columns not in schema: {extra_columns}")
            
        return issues
    
    def get_unified_schema_mapping(self) -> Dict[str, Dict[str, str]]:
        """Get mapping between different dataset schemas and unified schema"""
        return {
            'movielens': {
                'userId': 'user_id',
                'movieId': 'content_id',
                'rating': 'rating',
                'timestamp': 'timestamp',
                'title': 'title',
                'genres': 'genres'
            },
            'amazon': {
                'reviewerID': 'user_id',
                'asin': 'content_id',
                'overall': 'rating',
                'unixReviewTime': 'timestamp',
                'summary': 'title',
                'reviewText': 'description'
            },
            'book_crossing': {
                'User-ID': 'user_id',
                'ISBN': 'content_id',
                'Book-Rating': 'rating',
                'Book-Title': 'title',
                'Book-Author': 'creator_info',
                'Year-Of-Publication': 'publication_year'
            },
            'netflix': {
                'Cust_Id': 'user_id',
                'Movie_Id': 'content_id',
                'Rating': 'rating',
                'Date': 'timestamp'
            }
        }
    
    def generate_ddl_scripts(self) -> Dict[str, str]:
        """Generate DDL scripts for all tables"""
        from sqlalchemy.schema import CreateTable
        
        ddl_scripts = {}
        for table_name, table in Base.metadata.tables.items():
            ddl = str(CreateTable(table).compile(self.engine))
            ddl_scripts[table_name] = ddl
            
        return ddl_scripts


# Database session management
from sqlalchemy.orm import sessionmaker
from ..utils.config import settings

# Create engine and session factory
# Handle both complete URLs and file paths
if settings.database.sqlite_path.startswith('sqlite:'):
    database_url = settings.database.sqlite_path
else:
    database_url = f"sqlite:///{settings.database.sqlite_path}"
_engine = create_engine(database_url)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

def get_session():
    """Get database session"""
    return _SessionLocal()

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=_engine)
