"""
Comprehensive Data Loader for Multiple Recommendation Engine Datasets
Supports MovieLens, Amazon Reviews, Book-Crossing, Netflix Prize, and custom datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import zipfile
import requests
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..utils.config import settings, DataSchemas
from ..utils.logging import get_data_processing_logger
from ..utils.validation import DataValidator


class DatasetDownloader:
    """Download and manage datasets for recommendation engine"""
    
    def __init__(self, data_directory: str = None):
        self.data_dir = Path(data_directory or settings.data_directory)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = get_data_processing_logger("DatasetDownloader")
        
        # Dataset URLs and information
        self.dataset_info = {
            'movielens_20m': {
                'url': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip',
                'filename': 'ml-20m.zip',
                'extracted_dir': 'ml-20m',
                'files': ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv'],
                'size_mb': 190
            },
            'movielens_25m': {
                'url': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
                'filename': 'ml-25m.zip',
                'extracted_dir': 'ml-25m',
                'files': ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv', 'genome-scores.csv', 'genome-tags.csv'],
                'size_mb': 250
            },
            'book_crossing': {
                'url': 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/',
                'files': ['BX-Users.csv', 'BX-Books.csv', 'BX-Book-Ratings.csv'],
                'manual_download': True,
                'size_mb': 25
            }
        }
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset"""
        if dataset_name not in self.dataset_info:
            self.logger.log_error_with_context(
                "download_dataset",
                ValueError(f"Unknown dataset: {dataset_name}"),
                {"available_datasets": list(self.dataset_info.keys())}
            )
            return False
        
        dataset = self.dataset_info[dataset_name]
        
        if dataset.get('manual_download', False):
            self.logger.logger.warning(f"Dataset {dataset_name} requires manual download from {dataset['url']}")
            return self._check_manual_dataset_exists(dataset_name)
        
        filepath = self.data_dir / dataset['filename']
        
        # Check if already exists
        if filepath.exists() and not force_redownload:
            self.logger.logger.info(f"Dataset {dataset_name} already exists")
            return True
        
        # Download dataset
        self.logger.log_processing_start(f"download_{dataset_name}", dataset['size_mb'])
        
        try:
            response = requests.get(dataset['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) < 8192:
                            self.logger.log_processing_progress(
                                f"download_{dataset_name}",
                                downloaded // (1024 * 1024),
                                total_size // (1024 * 1024)
                            )
            
            # Extract if it's a zip file
            if filepath.suffix == '.zip':
                self._extract_dataset(filepath, dataset['extracted_dir'])
            
            self.logger.log_processing_complete(f"download_{dataset_name}", time.time(), 1, 0)
            return True
            
        except Exception as e:
            self.logger.log_error_with_context(
                f"download_{dataset_name}",
                e,
                {"url": dataset['url'], "filepath": str(filepath)}
            )
            return False
    
    def _extract_dataset(self, zip_path: Path, extract_dir: str):
        """Extract dataset zip file"""
        extract_path = self.data_dir / extract_dir
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
            
        self.logger.logger.info(f"Extracted {zip_path.name} to {extract_path}")
    
    def _check_manual_dataset_exists(self, dataset_name: str) -> bool:
        """Check if manually downloaded dataset exists"""
        dataset = self.dataset_info[dataset_name]
        
        for filename in dataset['files']:
            filepath = self.data_dir / filename
            if not filepath.exists():
                self.logger.logger.warning(f"Missing file: {filepath}")
                return False
        
        return True
    
    def download_all_datasets(self) -> Dict[str, bool]:
        """Download all supported datasets"""
        results = {}
        
        for dataset_name in self.dataset_info.keys():
            if not self.dataset_info[dataset_name].get('manual_download', False):
                results[dataset_name] = self.download_dataset(dataset_name)
            else:
                results[dataset_name] = self._check_manual_dataset_exists(dataset_name)
        
        return results


class UnifiedDataLoader:
    """Load and unify data from multiple recommendation engine datasets"""
    
    def __init__(self, data_directory: str = None, database_url: str = None):
        self.data_dir = Path(data_directory or settings.data_directory)
        # Handle both complete URLs and file paths
        if database_url:
            self.db_url = database_url
        elif settings.database.sqlite_path.startswith('sqlite:'):
            self.db_url = settings.database.sqlite_path
        else:
            self.db_url = f"sqlite:///{settings.database.sqlite_path}"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        self.logger = get_data_processing_logger("UnifiedDataLoader")
        self.validator = DataValidator()
        
        # Schema mappings for different datasets
        self.schema_mappings = self._initialize_schema_mappings()
        
        # Data type conversions
        self.data_type_mappings = self._initialize_data_type_mappings()
    
    def _initialize_schema_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize schema mappings for different datasets"""
        return {
            'movielens': {
                'users': {
                    'userId': 'user_id',
                    'age': 'age',
                    'gender': 'gender',
                    'occupation': 'occupation',
                    'zip-code': 'location'
                },
                'movies': {
                    'movieId': 'content_id',
                    'title': 'title',
                    'genres': 'genres'
                },
                'ratings': {
                    'userId': 'user_id',
                    'movieId': 'content_id',
                    'rating': 'rating',
                    'timestamp': 'timestamp'
                },
                'tags': {
                    'userId': 'user_id',
                    'movieId': 'content_id',
                    'tag': 'tag',
                    'timestamp': 'timestamp'
                }
            },
            'amazon': {
                'reviews': {
                    'reviewerID': 'user_id',
                    'asin': 'content_id',
                    'reviewerName': 'user_name',
                    'helpful': 'helpful_votes',
                    'reviewText': 'review_text',
                    'overall': 'rating',
                    'summary': 'review_summary',
                    'unixReviewTime': 'timestamp',
                    'reviewTime': 'review_date'
                },
                'metadata': {
                    'asin': 'content_id',
                    'title': 'title',
                    'price': 'price',
                    'brand': 'brand',
                    'categories': 'categories',
                    'description': 'description'
                }
            },
            'book_crossing': {
                'users': {
                    'User-ID': 'user_id',
                    'Location': 'location',
                    'Age': 'age'
                },
                'books': {
                    'ISBN': 'content_id',
                    'Book-Title': 'title',
                    'Book-Author': 'author',
                    'Year-Of-Publication': 'publication_year',
                    'Publisher': 'publisher',
                    'Image-URL-S': 'image_url_small',
                    'Image-URL-M': 'image_url_medium',
                    'Image-URL-L': 'image_url_large'
                },
                'ratings': {
                    'User-ID': 'user_id',
                    'ISBN': 'content_id',
                    'Book-Rating': 'rating'
                }
            },
            'netflix': {
                'ratings': {
                    # Netflix data is already in standardized format
                    # No mapping needed as fields are: content_id, user_id, rating, timestamp
                }
            }
        }
    
    def _initialize_data_type_mappings(self) -> Dict[str, str]:
        """Initialize data type mappings for different fields"""
        return {
            'user_id': 'string',
            'content_id': 'string',
            'rating': 'float',
            'timestamp': 'datetime',
            'interaction_type': 'category',
            'age': 'int',
            'gender': 'category',
            'location': 'string',
            'title': 'string',
            'genres': 'string',  # Will be processed as JSON
            'description': 'string',
            'publication_year': 'int',
            'duration': 'int',
            'price': 'float'
        }
    
    def load_movielens_data(self, version: str = '20m', sample_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load MovieLens dataset
        
        Args:
            version: Dataset version ('20m' or '25m')
            sample_size: If provided, load only this many ratings for faster processing
        """
        self.logger.log_processing_start(f"load_movielens_{version}", 0)
        
        if version == '20m':
            base_path = self.data_dir / 'ml-20m'
        elif version == '25m':
            base_path = self.data_dir / 'ml-25m'
        else:
            raise ValueError(f"Unsupported MovieLens version: {version}")
        
        datasets = {}
        
        try:
            # Load ratings
            ratings_path = base_path / 'ratings.csv'
            if ratings_path.exists():
                file_size_mb = ratings_path.stat().st_size / (1024 * 1024)
                
                if sample_size:
                    self.logger.logger.info(f"Loading MovieLens ratings sample ({sample_size:,} records from {file_size_mb:.1f}MB file)...")
                    # Read only the header first to get column names
                    header_df = pd.read_csv(ratings_path, nrows=0)
                    # Then read the specified number of rows
                    ratings_df = pd.read_csv(
                        ratings_path,
                        nrows=sample_size,
                        dtype={
                            'userId': 'int32',
                            'movieId': 'int32', 
                            'rating': 'float32',
                            'timestamp': 'int32'
                        },
                        engine='c'
                    )
                else:
                    self.logger.logger.info(f"Loading MovieLens ratings ({file_size_mb:.1f}MB)...")
                    # Use efficient pandas options for large files
                    ratings_df = pd.read_csv(
                        ratings_path,
                        dtype={
                            'userId': 'int32',
                            'movieId': 'int32', 
                            'rating': 'float32',
                            'timestamp': 'int32'
                        },
                        engine='c'  # Use faster C engine
                    )
                
                self.logger.logger.info(f"✓ Loaded {len(ratings_df):,} ratings")
                ratings_df = self._apply_schema_mapping(ratings_df, 'movielens', 'ratings')
                ratings_df = self._convert_data_types(ratings_df)
                datasets['ratings'] = ratings_df
                
                # Validate ratings data
                validation_result = self.validator.validate_interaction_events(ratings_df)
                self.logger.log_data_quality_metrics('movielens_ratings', validation_result)
            
            # Load movies
            movies_path = base_path / 'movies.csv'
            if movies_path.exists():
                file_size_mb = movies_path.stat().st_size / (1024 * 1024)
                self.logger.logger.info(f"Loading MovieLens movies ({file_size_mb:.1f}MB)...")
                movies_df = pd.read_csv(movies_path, dtype={'movieId': 'int32'}, engine='c')
                self.logger.logger.info(f"✓ Loaded {len(movies_df):,} movies")
                movies_df = self._apply_schema_mapping(movies_df, 'movielens', 'movies')
                movies_df = self._process_movie_genres(movies_df)
                movies_df = self._extract_movie_year(movies_df)
                datasets['movies'] = movies_df
                
                # Validate movies data
                validation_result = self.validator.validate_content_metadata(movies_df)
                self.logger.log_data_quality_metrics('movielens_movies', validation_result)
            
            # Load tags
            tags_path = base_path / 'tags.csv'
            if tags_path.exists():
                file_size_mb = tags_path.stat().st_size / (1024 * 1024)
                self.logger.logger.info(f"Loading MovieLens tags ({file_size_mb:.1f}MB)...")
                tags_df = pd.read_csv(
                    tags_path,
                    dtype={'userId': 'int32', 'movieId': 'int32', 'timestamp': 'int32'},
                    engine='c'
                )
                self.logger.logger.info(f"✓ Loaded {len(tags_df):,} tags")
                tags_df = self._apply_schema_mapping(tags_df, 'movielens', 'tags')
                datasets['tags'] = tags_df
            
            # Load links (IMDb, TMDb)
            links_path = base_path / 'links.csv'
            if links_path.exists():
                self.logger.logger.info("Loading MovieLens links...")
                links_df = pd.read_csv(links_path, dtype={'movieId': 'int32'}, engine='c')
                self.logger.logger.info(f"✓ Loaded {len(links_df):,} links")
                datasets['links'] = links_df
            
            # Load genome data (if available in 25m)
            if version == '25m':
                genome_scores_path = base_path / 'genome-scores.csv'
                genome_tags_path = base_path / 'genome-tags.csv'
                
                if genome_scores_path.exists():
                    datasets['genome_scores'] = pd.read_csv(genome_scores_path)
                if genome_tags_path.exists():
                    datasets['genome_tags'] = pd.read_csv(genome_tags_path)
            
            self.logger.log_processing_complete(f"load_movielens_{version}", time.time(), len(datasets), 0)
            return datasets
            
        except Exception as e:
            self.logger.log_error_with_context(
                f"load_movielens_{version}",
                e,
                {"base_path": str(base_path), "version": version}
            )
            return {}
    
    def load_amazon_reviews_data(self, category: str = 'Books') -> Dict[str, pd.DataFrame]:
        """Load Amazon Reviews dataset"""
        self.logger.log_processing_start(f"load_amazon_{category}", 0)
        
        # Note: Amazon dataset requires manual download and specific category selection
        base_path = self.data_dir / 'amazon' / category.lower()
        
        datasets = {}
        
        try:
            # Load reviews
            reviews_files = list(base_path.glob('*reviews*.json.gz'))
            if reviews_files:
                self.logger.logger.info(f"Loading Amazon {category} reviews...")
                reviews_df = self._load_amazon_json_gz(reviews_files[0])
                reviews_df = self._apply_schema_mapping(reviews_df, 'amazon', 'reviews')
                reviews_df = self._convert_data_types(reviews_df)
                datasets['reviews'] = reviews_df
                
                # Validate reviews data
                validation_result = self.validator.validate_interaction_events(reviews_df)
                self.logger.log_data_quality_metrics(f'amazon_{category}_reviews', validation_result)
            
            # Load metadata
            metadata_files = list(base_path.glob('*metadata*.json.gz'))
            if metadata_files:
                self.logger.logger.info(f"Loading Amazon {category} metadata...")
                metadata_df = self._load_amazon_json_gz(metadata_files[0])
                metadata_df = self._apply_schema_mapping(metadata_df, 'amazon', 'metadata')
                datasets['metadata'] = metadata_df
                
                # Validate metadata
                validation_result = self.validator.validate_content_metadata(metadata_df)
                self.logger.log_data_quality_metrics(f'amazon_{category}_metadata', validation_result)
            
            self.logger.log_processing_complete(f"load_amazon_{category}", time.time(), len(datasets), 0)
            return datasets
            
        except Exception as e:
            self.logger.log_error_with_context(
                f"load_amazon_{category}",
                e,
                {"base_path": str(base_path), "category": category}
            )
            return {}
    
    def load_book_crossing_data(self) -> Dict[str, pd.DataFrame]:
        """Load Book-Crossing dataset"""
        self.logger.log_processing_start("load_book_crossing", 0)
        
        datasets = {}
        
        try:
            # Load users
            users_path = self.data_dir / 'BX-Users.csv'
            if users_path.exists():
                self.logger.logger.info("Loading Book-Crossing users...")
                users_df = pd.read_csv(users_path, sep=';', encoding='latin-1', error_bad_lines=False)
                users_df = self._apply_schema_mapping(users_df, 'book_crossing', 'users')
                users_df = self._convert_data_types(users_df)
                datasets['users'] = users_df
                
                # Validate users data
                validation_result = self.validator.validate_user_profile(users_df)
                self.logger.log_data_quality_metrics('book_crossing_users', validation_result)
            
            # Load books
            books_path = self.data_dir / 'BX-Books.csv'
            if books_path.exists():
                self.logger.logger.info("Loading Book-Crossing books...")
                books_df = pd.read_csv(books_path, sep=';', encoding='latin-1', error_bad_lines=False)
                books_df = self._apply_schema_mapping(books_df, 'book_crossing', 'books')
                books_df = self._convert_data_types(books_df)
                datasets['books'] = books_df
                
                # Validate books data
                validation_result = self.validator.validate_content_metadata(books_df)
                self.logger.log_data_quality_metrics('book_crossing_books', validation_result)
            
            # Load ratings
            ratings_path = self.data_dir / 'BX-Book-Ratings.csv'
            if ratings_path.exists():
                self.logger.logger.info("Loading Book-Crossing ratings...")
                ratings_df = pd.read_csv(ratings_path, sep=';', encoding='latin-1', error_bad_lines=False)
                ratings_df = self._apply_schema_mapping(ratings_df, 'book_crossing', 'ratings')
                ratings_df = self._convert_data_types(ratings_df)
                datasets['ratings'] = ratings_df
                
                # Validate ratings data
                validation_result = self.validator.validate_interaction_events(ratings_df)
                self.logger.log_data_quality_metrics('book_crossing_ratings', validation_result)
            
            self.logger.log_processing_complete("load_book_crossing", time.time(), len(datasets), 0)
            return datasets
            
        except Exception as e:
            self.logger.log_error_with_context(
                "load_book_crossing",
                e,
                {"data_dir": str(self.data_dir)}
            )
            return {}
    
    def load_netflix_data(self, sample_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load Netflix Prize dataset
        
        Args:
            sample_size: If provided, load only this many ratings for faster processing
        """
        self.logger.log_processing_start("load_netflix", 0)
        
        # Note: Netflix dataset requires special handling due to format
        netflix_path = self.data_dir / 'netflix'
        
        datasets = {}
        
        try:
            # Load combined dataset or training data
            combined_files = list(netflix_path.glob('combined_data*.txt'))
            training_files = list(netflix_path.glob('training_set*.tar'))
            
            if combined_files:
                self.logger.logger.info("Loading Netflix combined data...")
                ratings_df = self._load_netflix_combined_data(combined_files, sample_size)
            elif training_files:
                self.logger.logger.info("Loading Netflix training data...")
                ratings_df = self._load_netflix_training_data(training_files[0], sample_size)
            else:
                self.logger.logger.warning("No Netflix data files found")
                return datasets
            
            self.logger.logger.info("Applying schema mapping...")
            ratings_df = self._apply_schema_mapping(ratings_df, 'netflix', 'ratings')
            
            # Add interaction_type field for Netflix ratings
            ratings_df['interaction_type'] = 'rating'
            
            self.logger.logger.info("Converting data types...")
            ratings_df = self._convert_data_types(ratings_df)
            datasets['ratings'] = ratings_df
            
            self.logger.logger.info("Validating data quality...")
            # Validate ratings data
            validation_result = self.validator.validate_interaction_events(ratings_df)
            self.logger.log_data_quality_metrics('netflix_ratings', validation_result)
            
            # Load movie titles if available
            movie_titles_path = netflix_path / 'movie_titles.csv'
            if movie_titles_path.exists():
                movies_df = pd.read_csv(movie_titles_path, encoding='latin-1', header=None)
                movies_df.columns = ['content_id', 'publication_year', 'title']
                datasets['movies'] = movies_df
            
            self.logger.log_processing_complete("load_netflix", time.time(), len(datasets), 0)
            return datasets
            
        except Exception as e:
            self.logger.log_error_with_context(
                "load_netflix",
                e,
                {"netflix_path": str(netflix_path)}
            )
            return {}
    
    def _apply_schema_mapping(self, df: pd.DataFrame, dataset: str, table: str) -> pd.DataFrame:
        """Apply schema mapping to standardize column names"""
        if dataset in self.schema_mappings and table in self.schema_mappings[dataset]:
            mapping = self.schema_mappings[dataset][table]
            df = df.rename(columns=mapping)
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types according to schema"""
        for column in df.columns:
            if column in self.data_type_mappings:
                target_type = self.data_type_mappings[column]
                
                try:
                    self.logger.logger.info(f"Converting column '{column}' to {target_type}")
                    
                    if target_type == 'datetime':
                        # Handle different timestamp formats
                        if df[column].dtype == 'int64':
                            # Unix timestamp
                            df[column] = pd.to_datetime(df[column], unit='s')
                        else:
                            # For Netflix dates in YYYY-MM-DD format, be more specific
                            if column == 'timestamp' and df[column].dtype == 'object':
                                # Netflix timestamp format is typically YYYY-MM-DD
                                df[column] = pd.to_datetime(df[column], format='%Y-%m-%d', errors='coerce')
                            else:
                                df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif target_type == 'int':
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif target_type == 'float':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif target_type == 'category':
                        df[column] = df[column].astype('category')
                    elif target_type == 'string':
                        df[column] = df[column].astype('string')
                        
                    self.logger.logger.info(f"✓ Successfully converted column '{column}' to {target_type}")
                        
                except Exception as e:
                    self.logger.logger.warning(f"Could not convert {column} to {target_type}: {e}")
        
        return df
    
    def _process_movie_genres(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Process movie genres into structured format"""
        if 'genres' in movies_df.columns:
            # Convert pipe-separated genres to JSON list
            movies_df['genres'] = movies_df['genres'].apply(
                lambda x: json.dumps(x.split('|')) if pd.notna(x) and x != '(no genres listed)' else json.dumps([])
            )
        
        return movies_df
    
    def _extract_movie_year(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Extract year from movie title"""
        if 'title' in movies_df.columns:
            # Extract year from title (format: "Title (YEAR)")
            year_extracted = movies_df['title'].str.extract(r'\(([0-9]{4})\)$')
            movies_df['publication_year'] = pd.to_numeric(year_extracted[0], errors='coerce')
            
            # Clean title by removing year
            movies_df['title'] = movies_df['title'].str.replace(r'\s*\([0-9]{4}\)$', '', regex=True)
        
        return movies_df
    
    def _load_amazon_json_gz(self, filepath: Path) -> pd.DataFrame:
        """Load Amazon dataset from JSON.gz format"""
        import gzip
        
        data = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return pd.DataFrame(data)
    
    def _load_netflix_combined_data(self, filepaths: List[Path], sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Netflix combined data format
        
        Args:
            filepaths: List of combined_data_*.txt files
            sample_size: If provided, stop loading after this many ratings
        """
        all_ratings = []
        total_loaded = 0
        
        for filepath in filepaths:
            self.logger.logger.info(f"Processing {filepath.name}...")
            
            with open(filepath, 'r') as f:
                current_movie_id = None
                
                for line in f:
                    line = line.strip()
                    
                    if line.endswith(':'):
                        # Movie ID line
                        current_movie_id = line[:-1]
                    else:
                        # Rating line: customer_id,rating,date
                        parts = line.split(',')
                        if len(parts) == 3:
                            all_ratings.append({
                                'content_id': current_movie_id,
                                'user_id': parts[0],
                                'rating': float(parts[1]),
                                'timestamp': parts[2]
                            })
                            total_loaded += 1
                            
                            # Check if we've reached the sample size
                            if sample_size and total_loaded >= sample_size:
                                self.logger.logger.info(f"✓ Loaded {total_loaded:,} Netflix ratings (sample)")
                                return pd.DataFrame(all_ratings)
        
        self.logger.logger.info(f"✓ Loaded {total_loaded:,} Netflix ratings (complete)")
        return pd.DataFrame(all_ratings)
    
    def _load_netflix_training_data(self, tar_filepath: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load Netflix training data from tar format
        
        Args:
            tar_filepath: Path to the training set tar file
            sample_size: If provided, stop loading after this many ratings
        """
        import tarfile
        
        all_ratings = []
        total_loaded = 0
        
        with tarfile.open(tar_filepath, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    movie_id = member.name.split('/')[-1].replace('.txt', '')
                    
                    for line in f:
                        line = line.decode('utf-8').strip()
                        parts = line.split(',')
                        
                        if len(parts) == 3:
                            all_ratings.append({
                                'content_id': movie_id,
                                'user_id': parts[0],
                                'rating': float(parts[1]),
                                'timestamp': parts[2]
                            })
                            total_loaded += 1
                            
                            # Check if we've reached the sample size
                            if sample_size and total_loaded >= sample_size:
                                self.logger.logger.info(f"✓ Loaded {total_loaded:,} Netflix ratings (sample)")
                                return pd.DataFrame(all_ratings)
        
        self.logger.logger.info(f"✓ Loaded {total_loaded:,} Netflix ratings (complete)")
        return pd.DataFrame(all_ratings)
    
    def load_all_datasets(self, sample_size: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all available datasets
        
        Args:
            sample_size: If provided, load only this many ratings for faster processing
        """
        all_datasets = {}
        
        # Load MovieLens
        try:
            movielens_data = self.load_movielens_data('20m', sample_size=sample_size)
            if movielens_data:
                all_datasets['movielens'] = movielens_data
        except Exception as e:
            self.logger.logger.error(f"Failed to load MovieLens: {e}")
        
        # Load Book-Crossing
        try:
            book_crossing_data = self.load_book_crossing_data()
            if book_crossing_data:
                all_datasets['book_crossing'] = book_crossing_data
        except Exception as e:
            self.logger.logger.error(f"Failed to load Book-Crossing: {e}")
        
        # Load Amazon (if available)
        try:
            amazon_data = self.load_amazon_reviews_data('Books')
            if amazon_data:
                all_datasets['amazon_books'] = amazon_data
        except Exception as e:
            self.logger.logger.error(f"Failed to load Amazon Books: {e}")
        
        # Load Netflix (if available)
        try:
            netflix_data = self.load_netflix_data(sample_size)
            if netflix_data:
                all_datasets['netflix'] = netflix_data
        except Exception as e:
            self.logger.logger.error(f"Failed to load Netflix: {e}")
        
        return all_datasets
    
    def save_to_database(self, datasets: Dict[str, Dict[str, pd.DataFrame]], batch_size: int = 10000):
        """Save all datasets to database"""
        self.logger.log_processing_start("save_to_database", sum(len(tables) for tables in datasets.values()))
        
        try:
            for dataset_name, tables in datasets.items():
                for table_name, df in tables.items():
                    table_full_name = f"{dataset_name}_{table_name}"
                    
                    self.logger.logger.info(f"Saving {table_full_name} ({len(df)} records)")
                    
                    # Save in batches
                    for i in range(0, len(df), batch_size):
                        batch_df = df.iloc[i:i + batch_size]
                        batch_df.to_sql(
                            table_full_name,
                            self.engine,
                            if_exists='append' if i > 0 else 'replace',
                            index=False,
                            method='multi'
                        )
                        
                        self.logger.log_processing_progress(
                            f"save_{table_full_name}",
                            i + len(batch_df),
                            len(df)
                        )
            
            self.logger.log_processing_complete("save_to_database", time.time(), 1, 0)
            
        except Exception as e:
            self.logger.log_error_with_context(
                "save_to_database",
                e,
                {"datasets": list(datasets.keys())}
            )


# Alias for backward compatibility
DataLoader = UnifiedDataLoader
