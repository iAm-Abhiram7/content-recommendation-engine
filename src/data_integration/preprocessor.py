"""
Comprehensive Data Preprocessor for Recommendation Engine
Handles missing values, outliers, normalization, and data quality improvement
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import settings, FeatureConstants
from ..utils.logging import get_data_processing_logger
from ..utils.validation import DataValidator


class MissingValueHandler:
    """Advanced missing value handling with multiple strategies"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("MissingValueHandler")
        self.imputation_strategies = {}
        self.imputed_columns = {}
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns in dataset"""
        self.logger.log_processing_start("analyze_missing_patterns", len(df))
        
        missing_analysis = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': {},
            'missing_patterns': {},
            'recommendations': []
        }
        
        # Analyze each column
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                missing_analysis['columns_with_missing'][column] = {
                    'count': missing_count,
                    'percentage': missing_percentage,
                    'data_type': str(df[column].dtype)
                }
        
        # Identify missing patterns
        missing_patterns = df.isnull().value_counts()
        for pattern, count in missing_patterns.head(10).items():
            pattern_str = ', '.join([col for col, is_missing in zip(df.columns, pattern) if is_missing])
            if pattern_str:  # Only non-empty patterns
                missing_analysis['missing_patterns'][pattern_str] = count
        
        # Generate recommendations
        missing_analysis['recommendations'] = self._generate_missing_value_recommendations(
            missing_analysis['columns_with_missing']
        )
        
        self.logger.log_data_quality_metrics("missing_value_analysis", missing_analysis)
        return missing_analysis
    
    def handle_missing_values(self, df: pd.DataFrame, strategy_config: Dict[str, str] = None) -> pd.DataFrame:
        """Handle missing values using specified strategies"""
        df_processed = df.copy()
        
        # Default strategy configuration
        if strategy_config is None:
            strategy_config = self._generate_default_strategies(df)
        
        for column, strategy in strategy_config.items():
            if column in df.columns and df[column].isnull().any():
                self.logger.logger.info(f"Handling missing values in {column} using {strategy}")
                
                try:
                    if strategy == 'drop':
                        df_processed = df_processed.dropna(subset=[column])
                    elif strategy == 'mean':
                        mean_value = df[column].mean()
                        df_processed[column].fillna(mean_value, inplace=True)
                        self.imputed_columns[column] = {'strategy': 'mean', 'value': mean_value}
                    elif strategy == 'median':
                        median_value = df[column].median()
                        df_processed[column].fillna(median_value, inplace=True)
                        self.imputed_columns[column] = {'strategy': 'median', 'value': median_value}
                    elif strategy == 'mode':
                        mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                        df_processed[column].fillna(mode_value, inplace=True)
                        self.imputed_columns[column] = {'strategy': 'mode', 'value': mode_value}
                    elif strategy == 'forward_fill':
                        df_processed[column].fillna(method='ffill', inplace=True)
                    elif strategy == 'backward_fill':
                        df_processed[column].fillna(method='bfill', inplace=True)
                    elif strategy == 'interpolate':
                        df_processed[column].interpolate(inplace=True)
                    elif strategy == 'knn':
                        df_processed = self._knn_imputation(df_processed, column)
                    elif strategy == 'domain_specific':
                        df_processed = self._domain_specific_imputation(df_processed, column)
                    else:
                        # Custom value
                        df_processed[column].fillna(strategy, inplace=True)
                        self.imputed_columns[column] = {'strategy': 'custom', 'value': strategy}
                
                except Exception as e:
                    self.logger.log_error_with_context(
                        f"handle_missing_{column}",
                        e,
                        {"strategy": strategy, "column": column}
                    )
        
        return df_processed
    
    def _generate_default_strategies(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate default imputation strategies based on data characteristics"""
        strategies = {}
        
        for column in df.columns:
            if df[column].isnull().any():
                dtype = df[column].dtype
                missing_percentage = (df[column].isnull().sum() / len(df)) * 100
                
                if missing_percentage > 50:
                    strategies[column] = 'drop'
                elif pd.api.types.is_numeric_dtype(dtype):
                    if df[column].skew() > 1:  # Highly skewed
                        strategies[column] = 'median'
                    else:
                        strategies[column] = 'mean'
                elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
                    strategies[column] = 'mode'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    strategies[column] = 'interpolate'
                else:
                    strategies[column] = 'Unknown'
        
        return strategies
    
    def _knn_imputation(self, df: pd.DataFrame, target_column: str, n_neighbors: int = 5) -> pd.DataFrame:
        """Perform KNN imputation for missing values"""
        # Select numeric columns for KNN
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column in numeric_columns and len(numeric_columns) > 1:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            
            self.imputed_columns[target_column] = {
                'strategy': 'knn',
                'n_neighbors': n_neighbors,
                'features_used': numeric_columns
            }
        
        return df
    
    def _domain_specific_imputation(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply domain-specific imputation rules"""
        domain_rules = {
            'age': lambda x: x.fillna(x.median()) if x.median() > 0 else x.fillna(30),
            'gender': lambda x: x.fillna('Unknown'),
            'location': lambda x: x.fillna('Unknown'),
            'occupation': lambda x: x.fillna('Other'),
            'rating': lambda x: x.fillna(x.mean()) if not x.empty else x.fillna(3.0),
            'genres': lambda x: x.fillna('[]'),  # Empty genre list
            'duration': lambda x: x.fillna(x.median()) if x.median() > 0 else x.fillna(120),
            'publication_year': lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 2000)
        }
        
        for pattern, rule in domain_rules.items():
            if pattern in column.lower():
                try:
                    df[column] = rule(df[column])
                    self.imputed_columns[column] = {'strategy': 'domain_specific', 'pattern': pattern}
                    break
                except Exception as e:
                    self.logger.logger.warning(f"Domain-specific imputation failed for {column}: {e}")
        
        return df
    
    def _generate_missing_value_recommendations(self, missing_columns: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for handling missing values"""
        recommendations = []
        
        for column, info in missing_columns.items():
            percentage = info['percentage']
            
            if percentage > 80:
                recommendations.append(f"Consider dropping column '{column}' (>{percentage:.1f}% missing)")
            elif percentage > 50:
                recommendations.append(f"Column '{column}' has high missingness ({percentage:.1f}%) - review data collection")
            elif percentage > 20:
                recommendations.append(f"Column '{column}' needs attention ({percentage:.1f}% missing)")
            elif info['data_type'] == 'object' and percentage > 10:
                recommendations.append(f"Categorical column '{column}' may need domain-specific imputation")
        
        if len(missing_columns) > len(recommendations):
            recommendations.append("Consider using advanced imputation methods (KNN, model-based) for remaining columns")
        
        return recommendations


class OutlierDetector:
    """Comprehensive outlier detection and handling"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("OutlierDetector")
        self.outlier_bounds = {}
        self.outlier_methods = {}
    
    def detect_outliers(self, df: pd.DataFrame, methods: List[str] = None) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        if methods is None:
            methods = ['iqr', 'zscore', 'isolation_forest']
        
        outlier_results = {
            'total_outliers': 0,
            'outlier_columns': {},
            'outlier_methods': {},
            'recommendations': []
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            column_outliers = {}
            
            for method in methods:
                try:
                    if method == 'iqr':
                        outliers = self._detect_iqr_outliers(df[column])
                    elif method == 'zscore':
                        outliers = self._detect_zscore_outliers(df[column])
                    elif method == 'isolation_forest':
                        outliers = self._detect_isolation_forest_outliers(df[[column]])
                    elif method == 'domain_specific':
                        outliers = self._detect_domain_specific_outliers(df[column], column)
                    else:
                        continue
                    
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        column_outliers[method] = {
                            'count': outlier_count,
                            'percentage': (outlier_count / len(df)) * 100,
                            'indices': df[outliers].index.tolist()
                        }
                
                except Exception as e:
                    self.logger.logger.warning(f"Outlier detection failed for {column} using {method}: {e}")
            
            if column_outliers:
                outlier_results['outlier_columns'][column] = column_outliers
                outlier_results['total_outliers'] += sum(info['count'] for info in column_outliers.values())
        
        # Generate recommendations
        outlier_results['recommendations'] = self._generate_outlier_recommendations(outlier_results)
        
        self.logger.log_data_quality_metrics("outlier_detection", outlier_results)
        return outlier_results
    
    def handle_outliers(self, df: pd.DataFrame, treatment_config: Dict[str, str] = None) -> pd.DataFrame:
        """Handle outliers using specified treatment methods"""
        df_processed = df.copy()
        
        if treatment_config is None:
            treatment_config = self._generate_default_treatments(df)
        
        for column, treatment in treatment_config.items():
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                self.logger.logger.info(f"Treating outliers in {column} using {treatment}")
                
                try:
                    if treatment == 'remove':
                        outliers = self._detect_iqr_outliers(df[column])
                        df_processed = df_processed[~outliers]
                    elif treatment == 'cap':
                        df_processed[column] = self._cap_outliers(df_processed[column])
                    elif treatment == 'winsorize':
                        df_processed[column] = self._winsorize_outliers(df_processed[column])
                    elif treatment == 'log_transform':
                        df_processed[column] = self._log_transform(df_processed[column])
                    elif treatment == 'box_cox':
                        df_processed[column] = self._box_cox_transform(df_processed[column])
                    elif treatment == 'quantile_transform':
                        df_processed[column] = self._quantile_transform(df_processed[column])
                
                except Exception as e:
                    self.logger.log_error_with_context(
                        f"treat_outliers_{column}",
                        e,
                        {"treatment": treatment, "column": column}
                    )
        
        return df_processed
    
    def _detect_iqr_outliers(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        self.outlier_bounds[series.name] = {'lower': lower_bound, 'upper': upper_bound}
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        
        # Create boolean series matching original length
        outliers = pd.Series(False, index=series.index)
        outliers[series.dropna().index] = z_scores > threshold
        
        return outliers
    
    def _detect_isolation_forest_outliers(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """Detect outliers using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_predictions = iso_forest.fit_predict(df.dropna())
        
        # Create boolean series matching original length
        outliers = pd.Series(False, index=df.index)
        outliers[df.dropna().index] = outlier_predictions == -1
        
        return outliers
    
    def _detect_domain_specific_outliers(self, series: pd.Series, column_name: str) -> pd.Series:
        """Detect outliers using domain-specific rules"""
        outliers = pd.Series(False, index=series.index)
        
        # Domain-specific rules
        if 'age' in column_name.lower():
            outliers = (series < 0) | (series > 120)
        elif 'rating' in column_name.lower():
            outliers = (series < 0) | (series > 10)  # Assume max rating is 10
        elif 'duration' in column_name.lower():
            outliers = (series < 0) | (series > 1440)  # More than 24 hours
        elif 'year' in column_name.lower():
            current_year = datetime.now().year
            outliers = (series < 1800) | (series > current_year + 10)
        
        return outliers
    
    def _cap_outliers(self, series: pd.Series, percentiles: Tuple[float, float] = (5, 95)) -> pd.Series:
        """Cap outliers at specified percentiles"""
        lower_cap = series.quantile(percentiles[0] / 100)
        upper_cap = series.quantile(percentiles[1] / 100)
        
        return series.clip(lower=lower_cap, upper=upper_cap)
    
    def _winsorize_outliers(self, series: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
        """Winsorize outliers"""
        from scipy.stats import mstats
        
        winsorized = mstats.winsorize(series.dropna(), limits=limits)
        result = series.copy()
        result[series.dropna().index] = winsorized
        
        return result
    
    def _log_transform(self, series: pd.Series) -> pd.Series:
        """Apply log transformation (for positive values only)"""
        if (series <= 0).any():
            # Add small constant to handle zeros/negatives
            series = series + abs(series.min()) + 1
        
        return np.log(series)
    
    def _box_cox_transform(self, series: pd.Series) -> pd.Series:
        """Apply Box-Cox transformation"""
        from scipy.stats import boxcox
        
        if (series <= 0).any():
            series = series + abs(series.min()) + 1
        
        transformed, _ = boxcox(series.dropna())
        result = series.copy()
        result[series.dropna().index] = transformed
        
        return result
    
    def _quantile_transform(self, series: pd.Series) -> pd.Series:
        """Apply quantile transformation"""
        from sklearn.preprocessing import QuantileTransformer
        
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        transformed = transformer.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        return pd.Series(transformed, index=series.index)
    
    def _generate_default_treatments(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate default outlier treatment strategies"""
        treatments = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            outlier_percentage = self._detect_iqr_outliers(df[column]).sum() / len(df) * 100
            
            if outlier_percentage > 20:
                treatments[column] = 'cap'
            elif outlier_percentage > 10:
                treatments[column] = 'winsorize'
            elif 'rating' in column.lower():
                treatments[column] = 'cap'  # Preserve rating ranges
            elif df[column].skew() > 2:
                treatments[column] = 'log_transform'
            else:
                treatments[column] = 'cap'
        
        return treatments
    
    def _generate_outlier_recommendations(self, outlier_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for outlier treatment"""
        recommendations = []
        
        for column, methods in outlier_results['outlier_columns'].items():
            total_outliers = sum(info['count'] for info in methods.values())
            percentage = (total_outliers / sum(info['count'] for method_data in methods.values() for info in [method_data])) * 100
            
            if percentage > 25:
                recommendations.append(f"Column '{column}' has high outlier percentage - consider data review")
            elif percentage > 10:
                recommendations.append(f"Column '{column}' may benefit from transformation or capping")
        
        if outlier_results['total_outliers'] > 0:
            recommendations.append("Consider domain expertise when choosing outlier treatment methods")
        
        return recommendations


class DataNormalizer:
    """Comprehensive data normalization and standardization"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("DataNormalizer")
        self.scalers = {}
        self.encoders = {}
        
    def normalize_ratings(self, df: pd.DataFrame, rating_column: str = 'rating') -> pd.DataFrame:
        """Normalize ratings to 0-1 scale across different rating systems"""
        df_normalized = df.copy()
        
        if rating_column not in df.columns:
            return df_normalized
        
        # Detect rating scale
        min_rating = df[rating_column].min()
        max_rating = df[rating_column].max()
        
        self.logger.logger.info(f"Normalizing ratings from {min_rating}-{max_rating} scale to 0-1")
        
        # Store original rating info
        df_normalized['original_rating'] = df[rating_column]
        df_normalized['rating_scale'] = f"{min_rating}-{max_rating}"
        
        # Normalize to 0-1 scale
        if max_rating > min_rating:
            df_normalized[rating_column] = (df[rating_column] - min_rating) / (max_rating - min_rating)
        else:
            df_normalized[rating_column] = 0.5  # Default for single-value ratings
        
        return df_normalized
    
    def standardize_features(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'standard') -> pd.DataFrame:
        """Standardize numerical features using specified method"""
        df_standardized = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column in df.columns:
                self.logger.logger.info(f"Standardizing {column} using {method}")
                
                try:
                    if method == 'standard':
                        scaler = StandardScaler()
                    elif method == 'minmax':
                        scaler = MinMaxScaler()
                    elif method == 'robust':
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                    else:
                        continue
                    
                    # Fit and transform
                    values = df[column].values.reshape(-1, 1)
                    df_standardized[column] = scaler.fit_transform(values).flatten()
                    
                    # Store scaler for future use
                    self.scalers[column] = scaler
                
                except Exception as e:
                    self.logger.log_error_with_context(
                        f"standardize_{column}",
                        e,
                        {"method": method, "column": column}
                    )
        
        return df_standardized
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'label') -> pd.DataFrame:
        """Encode categorical features using specified method"""
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in columns:
            if column in df.columns:
                self.logger.logger.info(f"Encoding {column} using {method}")
                
                try:
                    if method == 'label':
                        encoder = LabelEncoder()
                        df_encoded[column] = encoder.fit_transform(df[column].astype(str))
                        self.encoders[column] = encoder
                    
                    elif method == 'onehot':
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        encoded_values = encoder.fit_transform(df[column].values.reshape(-1, 1))
                        
                        # Create new columns for one-hot encoded values
                        feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                        encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=df.index)
                        
                        # Drop original column and add encoded columns
                        df_encoded = df_encoded.drop(columns=[column])
                        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                        
                        self.encoders[column] = encoder
                    
                    elif method == 'target':
                        # Target encoding (requires target variable)
                        if 'rating' in df.columns:
                            target_means = df.groupby(column)['rating'].mean()
                            df_encoded[column] = df[column].map(target_means)
                            self.encoders[column] = target_means
                
                except Exception as e:
                    self.logger.log_error_with_context(
                        f"encode_{column}",
                        e,
                        {"method": method, "column": column}
                    )
        
        return df_encoded
    
    def normalize_temporal_features(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """Extract and normalize temporal features"""
        df_temporal = df.copy()
        
        if timestamp_column not in df.columns:
            return df_temporal
        
        self.logger.logger.info(f"Extracting temporal features from {timestamp_column}")
        
        try:
            # Ensure datetime format
            df_temporal[timestamp_column] = pd.to_datetime(df_temporal[timestamp_column])
            
            # Extract temporal components
            df_temporal['hour'] = df_temporal[timestamp_column].dt.hour
            df_temporal['day_of_week'] = df_temporal[timestamp_column].dt.dayofweek
            df_temporal['month'] = df_temporal[timestamp_column].dt.month
            df_temporal['year'] = df_temporal[timestamp_column].dt.year
            df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical encoding for temporal features
            df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['hour'] / 24)
            df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['hour'] / 24)
            df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
            df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
            df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
            df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
            
            # Time-based features
            df_temporal['time_of_day'] = pd.cut(
                df_temporal['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            )
            
            # Recency features (time since last interaction)
            max_timestamp = df_temporal[timestamp_column].max()
            df_temporal['days_since_interaction'] = (max_timestamp - df_temporal[timestamp_column]).dt.days
            df_temporal['recency_score'] = 1 / (1 + df_temporal['days_since_interaction'])
            
        except Exception as e:
            self.logger.log_error_with_context(
                "normalize_temporal_features",
                e,
                {"timestamp_column": timestamp_column}
            )
        
        return df_temporal


class ComprehensivePreprocessor:
    """Main preprocessor orchestrating all preprocessing steps"""
    
    def __init__(self):
        self.logger = get_data_processing_logger("ComprehensivePreprocessor")
        self.missing_handler = MissingValueHandler()
        self.outlier_detector = OutlierDetector()
        self.normalizer = DataNormalizer()
        self.validator = DataValidator()
        
        self.preprocessing_report = {}
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str, config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline for a dataset"""
        self.logger.log_processing_start(f"preprocess_{dataset_name}", len(df))
        
        preprocessing_steps = []
        df_processed = df.copy()
        
        # Default configuration
        if config is None:
            config = {
                'handle_missing': True,
                'detect_outliers': True,
                'normalize_ratings': True,
                'standardize_features': True,
                'encode_categorical': True,
                'extract_temporal': True
            }
        
        try:
            # Step 1: Initial validation
            preprocessing_steps.append("initial_validation")
            initial_validation = self.validator.validate_interaction_events(df_processed)
            
            # Step 2: Handle missing values
            if config.get('handle_missing', True):
                preprocessing_steps.append("missing_value_handling")
                missing_analysis = self.missing_handler.analyze_missing_patterns(df_processed)
                df_processed = self.missing_handler.handle_missing_values(df_processed)
            
            # Step 3: Detect and handle outliers
            if config.get('detect_outliers', True):
                preprocessing_steps.append("outlier_detection")
                outlier_analysis = self.outlier_detector.detect_outliers(df_processed)
                df_processed = self.outlier_detector.handle_outliers(df_processed)
            
            # Step 4: Normalize ratings
            if config.get('normalize_ratings', True) and 'rating' in df_processed.columns:
                preprocessing_steps.append("rating_normalization")
                df_processed = self.normalizer.normalize_ratings(df_processed)
            
            # Step 5: Extract temporal features
            if config.get('extract_temporal', True) and 'timestamp' in df_processed.columns:
                preprocessing_steps.append("temporal_feature_extraction")
                df_processed = self.normalizer.normalize_temporal_features(df_processed)
            
            # Step 6: Encode categorical features
            if config.get('encode_categorical', True):
                preprocessing_steps.append("categorical_encoding")
                categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_columns:
                    df_processed = self.normalizer.encode_categorical_features(df_processed, categorical_columns)
            
            # Step 7: Standardize numerical features
            if config.get('standardize_features', True):
                preprocessing_steps.append("feature_standardization")
                numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude already normalized ratings and temporal features
                exclude_columns = ['rating', 'recency_score'] + [col for col in numerical_columns if '_sin' in col or '_cos' in col]
                standardize_columns = [col for col in numerical_columns if col not in exclude_columns]
                if standardize_columns:
                    df_processed = self.normalizer.standardize_features(df_processed, standardize_columns)
            
            # Step 8: Final validation
            preprocessing_steps.append("final_validation")
            final_validation = self.validator.validate_interaction_events(df_processed)
            
            # Generate preprocessing report
            report = {
                'dataset_name': dataset_name,
                'original_shape': df.shape,
                'processed_shape': df_processed.shape,
                'preprocessing_steps': preprocessing_steps,
                'initial_validation': initial_validation,
                'final_validation': final_validation,
                'data_quality_improvement': final_validation['quality_score'] - initial_validation['quality_score'],
                'missing_value_analysis': missing_analysis if 'missing_analysis' in locals() else {},
                'outlier_analysis': outlier_analysis if 'outlier_analysis' in locals() else {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            self.preprocessing_report[dataset_name] = report
            
            self.logger.log_processing_complete(f"preprocess_{dataset_name}", time.time(), 1, 0)
            self.logger.log_data_quality_metrics(f"{dataset_name}_preprocessing", report)
            
            return df_processed, report
            
        except Exception as e:
            self.logger.log_error_with_context(
                f"preprocess_{dataset_name}",
                e,
                {"config": config, "steps_completed": preprocessing_steps}
            )
            return df, {"error": str(e), "steps_completed": preprocessing_steps}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive preprocessing report for all datasets"""
        if not self.preprocessing_report:
            return {"error": "No preprocessing reports available"}
        
        overall_report = {
            'total_datasets_processed': len(self.preprocessing_report),
            'overall_quality_improvement': 0,
            'common_issues': {},
            'recommendations': [],
            'dataset_summaries': {}
        }
        
        # Calculate overall metrics
        quality_improvements = []
        for dataset_name, report in self.preprocessing_report.items():
            if 'data_quality_improvement' in report:
                quality_improvements.append(report['data_quality_improvement'])
            
            overall_report['dataset_summaries'][dataset_name] = {
                'shape_change': f"{report['original_shape']} → {report['processed_shape']}",
                'quality_score': report['final_validation']['quality_score'],
                'steps_completed': len(report['preprocessing_steps'])
            }
        
        if quality_improvements:
            overall_report['overall_quality_improvement'] = np.mean(quality_improvements)
        
        # Generate overall recommendations
        overall_report['recommendations'] = [
            "Regularly monitor data quality metrics",
            "Implement automated preprocessing pipelines",
            "Review and update preprocessing strategies based on new data patterns",
            "Maintain preprocessing artifact logs for reproducibility"
        ]
        
        return overall_report
