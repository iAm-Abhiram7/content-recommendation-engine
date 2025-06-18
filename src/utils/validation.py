"""
Comprehensive data validation utilities for the Content Recommendation Engine
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    level: ValidationResult
    field: str
    message: str
    value: Any = None
    suggestion: Optional[str] = None
    count: int = 1


@dataclass
class DataQualityReport:
    """Data quality report for processed datasets"""
    dataset_name: str
    total_records: int
    valid_records: int
    error_count: int
    warning_count: int
    quality_score: float
    issues: List[ValidationIssue]
    processing_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'dataset_name': self.dataset_name,
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'quality_score': self.quality_score,
            'issues': [
                {
                    'level': issue.level.value,
                    'field': issue.field,
                    'message': issue.message,
                    'count': issue.count
                }
                for issue in self.issues
            ],
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }


class DataValidator:
    """Comprehensive data validation for recommendation engine datasets"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.quality_scores: Dict[str, float] = {}
    
    def validate_user_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate user profile data"""
        self.issues.clear()
        
        # Check required fields
        required_fields = ['user_id', 'age', 'gender']
        self._check_required_fields(df, required_fields, 'user_profile')
        
        # Validate user_id uniqueness
        if 'user_id' in df.columns:
            duplicate_count = df['user_id'].duplicated().sum()
            if duplicate_count > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.ERROR,
                    'user_id',
                    f'{duplicate_count} duplicate user IDs found',
                    count=duplicate_count,
                    suggestion='Remove or merge duplicate users'
                ))
        
        # Validate age ranges
        if 'age' in df.columns:
            self._validate_age_field(df['age'])
        
        # Validate gender values
        if 'gender' in df.columns:
            self._validate_gender_field(df['gender'])
        
        # Validate registration dates
        if 'registration_date' in df.columns:
            self._validate_date_field(df['registration_date'], 'registration_date')
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, 'user_profile')
        
        return {
            'issues': self.issues,
            'quality_score': quality_score,
            'total_records': len(df),
            'valid_records': self._count_valid_records(df)
        }
    
    def validate_content_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate content metadata"""
        self.issues.clear()
        
        # Check required fields
        required_fields = ['content_id', 'title', 'content_type']
        self._check_required_fields(df, required_fields, 'content_metadata')
        
        # Validate content_id uniqueness
        if 'content_id' in df.columns:
            duplicate_count = df['content_id'].duplicated().sum()
            if duplicate_count > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.ERROR,
                    'content_id',
                    f'{duplicate_count} duplicate content IDs found',
                    count=duplicate_count
                ))
        
        # Validate content types
        if 'content_type' in df.columns:
            self._validate_content_type_field(df['content_type'])
        
        # Validate release dates
        if 'release_date' in df.columns:
            self._validate_date_field(df['release_date'], 'release_date')
        
        # Validate duration values
        if 'duration' in df.columns:
            self._validate_duration_field(df['duration'])
        
        # Validate genre data
        if 'genres' in df.columns:
            self._validate_genre_field(df['genres'])
        
        quality_score = self._calculate_quality_score(df, 'content_metadata')
        
        return {
            'issues': self.issues,
            'quality_score': quality_score,
            'total_records': len(df),
            'valid_records': self._count_valid_records(df)
        }
    
    def validate_interaction_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate interaction events data"""
        self.issues.clear()
        
        print(f"Starting validation of {len(df):,} interaction events...")
        
        # Check required fields
        print("Checking required fields...")
        required_fields = ['user_id', 'content_id', 'interaction_type', 'timestamp']
        self._check_required_fields(df, required_fields, 'interaction_events')
        
        # Validate timestamp format and range
        if 'timestamp' in df.columns:
            print("Validating timestamp field...")
            self._validate_timestamp_field(df['timestamp'])
        
        # Validate interaction types
        if 'interaction_type' in df.columns:
            print("Validating interaction types...")
            self._validate_interaction_type_field(df['interaction_type'])
        
        # Validate rating values
        if 'rating' in df.columns:
            print("Validating rating values...")
            self._validate_rating_field(df['rating'])
        
        # Validate completion status
        if 'completion_status' in df.columns:
            print("Validating completion status...")
            self._validate_completion_status_field(df['completion_status'])
        
        # Check for temporal consistency
        print("Checking temporal consistency...")
        self._validate_temporal_consistency(df)
        
        print("Calculating quality score...")
        quality_score = self._calculate_quality_score(df, 'interaction_events')
        
        print(f"Validation complete. Quality score: {quality_score:.3f}")
        
        return {
            'issues': self.issues,
            'quality_score': quality_score,
            'total_records': len(df),
            'valid_records': self._count_valid_records(df)
        }
    
    def _check_required_fields(self, df: pd.DataFrame, required_fields: List[str], dataset_name: str):
        """Check for required fields"""
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            self.issues.append(ValidationIssue(
                ValidationResult.CRITICAL,
                'schema',
                f'Missing required fields in {dataset_name}: {missing_fields}',
                value=missing_fields
            ))
        
        # Check for null values in required fields
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    null_percentage = (null_count / len(df)) * 100
                    level = ValidationResult.CRITICAL if null_percentage > 10 else ValidationResult.ERROR
                    self.issues.append(ValidationIssue(
                        level,
                        field,
                        f'{null_count} null values ({null_percentage:.1f}%) in required field',
                        count=null_count,
                        suggestion='Implement data imputation or removal strategy'
                    ))
    
    def _validate_age_field(self, age_series: pd.Series):
        """Validate age field"""
        # Check for valid age range
        invalid_ages = age_series[(age_series < 0) | (age_series > 120)]
        if len(invalid_ages) > 0:
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                'age',
                f'{len(invalid_ages)} invalid age values (outside 0-120 range)',
                count=len(invalid_ages),
                suggestion='Cap ages to reasonable range or mark as missing'
            ))
        
        # Check for suspicious age patterns
        age_counts = age_series.value_counts()
        if len(age_counts) < 10:  # Too few unique ages
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'age',
                f'Only {len(age_counts)} unique age values found',
                suggestion='Verify age data granularity'
            ))
    
    def _validate_gender_field(self, gender_series: pd.Series):
        """Validate gender field"""
        valid_genders = {'M', 'F', 'Male', 'Female', 'male', 'female', 'Other', 'other', 'Unknown', 'unknown'}
        invalid_genders = gender_series[~gender_series.isin(valid_genders) & gender_series.notna()]
        
        if len(invalid_genders) > 0:
            unique_invalid = invalid_genders.unique()
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'gender',
                f'{len(invalid_genders)} records with non-standard gender values: {list(unique_invalid)[:5]}',
                count=len(invalid_genders),
                suggestion='Standardize gender values to M/F/Other'
            ))
    
    def _validate_content_type_field(self, content_type_series: pd.Series):
        """Validate content type field"""
        valid_types = {'movie', 'book', 'music', 'tv_show', 'podcast', 'game'}
        invalid_types = content_type_series[~content_type_series.isin(valid_types) & content_type_series.notna()]
        
        if len(invalid_types) > 0:
            unique_invalid = invalid_types.unique()
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                'content_type',
                f'{len(invalid_types)} records with invalid content types: {list(unique_invalid)}',
                count=len(invalid_types),
                suggestion=f'Use only valid content types: {valid_types}'
            ))
    
    def _validate_date_field(self, date_series: pd.Series, field_name: str):
        """Validate date field"""
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(date_series):
                converted_dates = pd.to_datetime(date_series, errors='coerce')
            else:
                converted_dates = date_series
            
            # Check for invalid dates (NaT after conversion)
            invalid_dates = converted_dates.isna() & date_series.notna()
            if invalid_dates.sum() > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.ERROR,
                    field_name,
                    f'{invalid_dates.sum()} invalid date formats',
                    count=invalid_dates.sum(),
                    suggestion='Standardize date format to YYYY-MM-DD'
                ))
            
            # Check for future dates (if inappropriate)
            if field_name == 'registration_date':
                future_dates = converted_dates > datetime.now()
                if future_dates.sum() > 0:
                    self.issues.append(ValidationIssue(
                        ValidationResult.ERROR,
                        field_name,
                        f'{future_dates.sum()} future dates found',
                        count=future_dates.sum()
                    ))
            
        except Exception as e:
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                field_name,
                f'Error validating dates: {str(e)}'
            ))
    
    def _validate_duration_field(self, duration_series: pd.Series):
        """Validate duration field"""
        # Check for negative durations
        negative_durations = duration_series[duration_series < 0]
        if len(negative_durations) > 0:
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                'duration',
                f'{len(negative_durations)} negative duration values',
                count=len(negative_durations)
            ))
        
        # Check for extremely long durations (> 24 hours = 1440 minutes)
        extreme_durations = duration_series[duration_series > 1440]
        if len(extreme_durations) > 0:
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'duration',
                f'{len(extreme_durations)} extremely long durations (>24 hours)',
                count=len(extreme_durations),
                suggestion='Verify duration units (minutes vs seconds vs hours)'
            ))
    
    def _validate_genre_field(self, genre_series: pd.Series):
        """Validate genre field"""
        # Check if genres are properly formatted (should be JSON-like)
        invalid_genre_format = 0
        for idx, genre_value in genre_series.items():
            if pd.notna(genre_value):
                try:
                    if isinstance(genre_value, str):
                        # Try to parse as JSON
                        json.loads(genre_value)
                    elif not isinstance(genre_value, (list, dict)):
                        invalid_genre_format += 1
                except (json.JSONDecodeError, TypeError):
                    invalid_genre_format += 1
        
        if invalid_genre_format > 0:
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'genres',
                f'{invalid_genre_format} records with invalid genre format',
                count=invalid_genre_format,
                suggestion='Store genres as JSON array or comma-separated string'
            ))
    
    def _validate_timestamp_field(self, timestamp_series: pd.Series):
        """Validate timestamp field"""
        try:
            if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
                converted_timestamps = pd.to_datetime(timestamp_series, errors='coerce')
            else:
                converted_timestamps = timestamp_series
            
            # Check for invalid timestamps
            invalid_timestamps = converted_timestamps.isna() & timestamp_series.notna()
            if invalid_timestamps.sum() > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.ERROR,
                    'timestamp',
                    f'{invalid_timestamps.sum()} invalid timestamp formats',
                    count=invalid_timestamps.sum()
                ))
            
            # Check timestamp range
            current_time = datetime.now()
            future_timestamps = converted_timestamps > current_time
            if future_timestamps.sum() > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.WARNING,
                    'timestamp',
                    f'{future_timestamps.sum()} future timestamps found',
                    count=future_timestamps.sum()
                ))
            
            # Check for very old timestamps (before 1990)
            old_threshold = datetime(1990, 1, 1)
            very_old_timestamps = converted_timestamps < old_threshold
            if very_old_timestamps.sum() > 0:
                self.issues.append(ValidationIssue(
                    ValidationResult.WARNING,
                    'timestamp',
                    f'{very_old_timestamps.sum()} very old timestamps (before 1990)',
                    count=very_old_timestamps.sum()
                ))
                
        except Exception as e:
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                'timestamp',
                f'Error validating timestamps: {str(e)}'
            ))
    
    def _validate_interaction_type_field(self, interaction_type_series: pd.Series):
        """Validate interaction type field"""
        valid_types = {
            'rating', 'view', 'purchase', 'skip', 'save', 'share', 'like', 'dislike',
            'comment', 'bookmark', 'download', 'stream', 'click', 'hover'
        }
        
        invalid_types = interaction_type_series[
            ~interaction_type_series.isin(valid_types) & interaction_type_series.notna()
        ]
        
        if len(invalid_types) > 0:
            unique_invalid = invalid_types.unique()
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'interaction_type',
                f'{len(invalid_types)} records with non-standard interaction types: {list(unique_invalid)[:5]}',
                count=len(invalid_types),
                suggestion=f'Consider using standard interaction types: {valid_types}'
            ))
    
    def _validate_rating_field(self, rating_series: pd.Series):
        """Validate rating field"""
        # Check rating range (assuming 1-5 scale, but flexible)
        min_rating = rating_series.min()
        max_rating = rating_series.max()
        
        if min_rating < 0 or max_rating > 10:
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'rating',
                f'Rating range {min_rating}-{max_rating} outside typical bounds',
                suggestion='Verify rating scale and normalize if needed'
            ))
        
        # Check for decimal ratings where integers expected
        has_decimals = (rating_series % 1 != 0).any()
        if has_decimals and max_rating <= 5:
            self.issues.append(ValidationIssue(
                ValidationResult.WARNING,
                'rating',
                'Decimal ratings found in apparent integer scale',
                suggestion='Verify if decimal ratings are intended'
            ))
    
    def _validate_completion_status_field(self, completion_series: pd.Series):
        """Validate completion status field"""
        # Should be between 0 and 1
        invalid_completion = completion_series[
            (completion_series < 0) | (completion_series > 1)
        ]
        
        if len(invalid_completion) > 0:
            self.issues.append(ValidationIssue(
                ValidationResult.ERROR,
                'completion_status',
                f'{len(invalid_completion)} completion status values outside 0-1 range',
                count=len(invalid_completion),
                suggestion='Normalize completion status to 0-1 range'
            ))
    
    def _validate_temporal_consistency(self, df: pd.DataFrame):
        """Validate temporal consistency in interaction data"""
        if 'timestamp' in df.columns and 'user_id' in df.columns:
            # Optimized check for temporal consistency - sample check instead of full scan
            unique_users = df['user_id'].unique()
            
            # For large datasets, sample a subset of users for performance
            if len(unique_users) > 1000:
                import random
                sample_size = min(1000, len(unique_users))
                sampled_users = random.sample(list(unique_users), sample_size)
                print(f"Sampling {sample_size} users out of {len(unique_users)} for temporal consistency check")
            else:
                sampled_users = unique_users
            
            user_timeline_issues = 0
            
            for user_id in sampled_users:
                user_data = df[df['user_id'] == user_id].sort_values('timestamp')
                # Check if timestamps are monotonically increasing
                if len(user_data) > 1:
                    timestamps = pd.to_datetime(user_data['timestamp'], errors='coerce')
                    if not timestamps.is_monotonic_increasing:
                        user_timeline_issues += 1
            
            if user_timeline_issues > 0:
                total_users_checked = len(sampled_users)
                estimated_issues = int((user_timeline_issues / total_users_checked) * len(unique_users))
                self.issues.append(ValidationIssue(
                    ValidationResult.WARNING,
                    'temporal_consistency',
                    f'~{estimated_issues} users (estimated) with non-chronological interactions',
                    count=user_timeline_issues,
                    suggestion='Sort interactions by timestamp for each user'
                ))
    
    def _calculate_quality_score(self, df: pd.DataFrame, dataset_type: str) -> float:
        """Calculate overall data quality score"""
        total_score = 0.0
        weight_sum = 0.0
        
        # Completeness score (weight: 0.3)
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        total_score += completeness * 0.3
        weight_sum += 0.3
        
        # Validity score based on issues (weight: 0.4)
        critical_issues = sum(1 for issue in self.issues if issue.level == ValidationResult.CRITICAL)
        error_issues = sum(1 for issue in self.issues if issue.level == ValidationResult.ERROR)
        warning_issues = sum(1 for issue in self.issues if issue.level == ValidationResult.WARNING)
        
        total_issues = critical_issues * 3 + error_issues * 2 + warning_issues * 1
        max_possible_issues = len(df) * 0.1  # Assume 10% issues as maximum
        validity = max(0.0, 1.0 - (total_issues / max_possible_issues))
        total_score += validity * 0.4
        weight_sum += 0.4
        
        # Consistency score (weight: 0.3)
        # Simple consistency check based on data types and ranges
        consistency = 0.8  # Base consistency score
        if dataset_type == 'user_profile':
            # Check age consistency
            if 'age' in df.columns:
                age_consistency = 1.0 - (df['age'][(df['age'] < 0) | (df['age'] > 120)].count() / len(df))
                consistency = min(consistency, age_consistency)
        
        total_score += consistency * 0.3
        weight_sum += 0.3
        
        return total_score / weight_sum if weight_sum > 0 else 0.0
    
    def _count_valid_records(self, df: pd.DataFrame) -> int:
        """Count records without critical validation issues"""
        # Simple implementation: count records without any null values in required fields
        return len(df.dropna())
    
    def generate_quality_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        total_records = sum(result['total_records'] for result in validation_results)
        total_valid = sum(result['valid_records'] for result in validation_results)
        avg_quality_score = sum(result['quality_score'] for result in validation_results) / len(validation_results)
        
        all_issues = []
        for result in validation_results:
            all_issues.extend(result['issues'])
        
        # Group issues by severity
        issue_summary = {
            'critical': [issue for issue in all_issues if issue.level == ValidationResult.CRITICAL],
            'error': [issue for issue in all_issues if issue.level == ValidationResult.ERROR],
            'warning': [issue for issue in all_issues if issue.level == ValidationResult.WARNING]
        }
        
        return {
            'overall_quality_score': avg_quality_score,
            'total_records_processed': total_records,
            'valid_records': total_valid,
            'data_coverage': total_valid / total_records if total_records > 0 else 0,
            'issue_summary': {
                'critical_count': len(issue_summary['critical']),
                'error_count': len(issue_summary['error']),
                'warning_count': len(issue_summary['warning'])
            },
            'detailed_issues': issue_summary,
            'recommendations': self._generate_recommendations(issue_summary),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, issue_summary: Dict[str, List[ValidationIssue]]) -> List[str]:
        """Generate actionable recommendations based on validation issues"""
        recommendations = []
        
        if issue_summary['critical']:
            recommendations.append("CRITICAL: Address missing required fields before proceeding")
        
        if issue_summary['error']:
            recommendations.append("Fix data format and range errors to improve data quality")
        
        if len(issue_summary['warning']) > 10:
            recommendations.append("Consider standardizing data formats to reduce warnings")
        
        # Add specific recommendations based on common patterns
        field_issues = {}
        for issues in issue_summary.values():
            for issue in issues:
                if issue.field not in field_issues:
                    field_issues[issue.field] = []
                field_issues[issue.field].append(issue)
        
        for field, issues in field_issues.items():
            if len(issues) > 3:
                recommendations.append(f"Field '{field}' has multiple issues - consider data source review")
        
        return recommendations
