"""
Content Embedding Generator using Google Gemini
Handles batch embedding generation, quality assessment, and storage
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

from .gemini_client import GeminiClient
from ..utils.config import settings
from ..utils.logging import get_data_processing_logger
from ..utils.validation import DataValidator


class EmbeddingGenerator:
    """Generate and manage content embeddings using Gemini"""
    
    def __init__(self, gemini_client: GeminiClient = None):
        self.gemini_client = gemini_client or GeminiClient()
        self.logger = get_data_processing_logger("EmbeddingGenerator")
        self.validator = DataValidator()
        
        # Embedding storage
        self.embedding_cache_dir = Path(settings.cache_directory) / "embeddings"
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics
        self.embedding_quality_metrics = {
            'total_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_generation_time': 0.0,
            'quality_scores': []
        }
    
    def generate_content_embeddings(self, content_df: pd.DataFrame, 
                                  text_column: str = 'description',
                                  batch_size: int = 50) -> pd.DataFrame:
        """Generate embeddings for content descriptions"""
        
        self.logger.log_processing_start("generate_content_embeddings", len(content_df))
        
        # Prepare content for embedding generation
        content_for_embedding = []
        for idx, row in content_df.iterrows():
            text_content = self._prepare_text_for_embedding(row, text_column)
            if text_content:
                content_for_embedding.append({
                    'content_id': row.get('content_id', idx),
                    'text': text_content,
                    'title': row.get('title', ''),
                    'genres': row.get('genres', ''),
                    'index': idx
                })
        
        self.logger.logger.info(f"Prepared {len(content_for_embedding)} items for embedding generation")
        
        # Generate embeddings in batches
        all_embeddings = []
        successful_count = 0
        failed_count = 0
        
        for i in range(0, len(content_for_embedding), batch_size):
            batch = content_for_embedding[i:i + batch_size]
            
            self.logger.logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(content_for_embedding) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self._generate_batch_embeddings(batch)
            
            for embedding_result in batch_embeddings:
                if embedding_result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
                
                all_embeddings.append(embedding_result)
            
            # Progress logging
            self.logger.log_processing_progress(
                "generate_content_embeddings",
                i + len(batch),
                len(content_for_embedding)
            )
            
            # Rate limiting
            time.sleep(1)
        
        # Create embeddings dataframe
        embeddings_df = self._create_embeddings_dataframe(all_embeddings, content_df)
        
        # Update quality metrics
        self.embedding_quality_metrics['total_generated'] = len(content_for_embedding)
        self.embedding_quality_metrics['successful_generations'] = successful_count
        self.embedding_quality_metrics['failed_generations'] = failed_count
        
        self.logger.log_processing_complete(
            "generate_content_embeddings",
            time.time(),
            successful_count,
            failed_count
        )
        
        # Log quality metrics
        success_rate = successful_count / len(content_for_embedding) * 100 if content_for_embedding else 0
        self.logger.log_data_quality_metrics(
            "embedding_generation",
            {
                'success_rate': success_rate,
                'total_processed': len(content_for_embedding),
                'successful': successful_count,
                'failed': failed_count
            }
        )
        
        return embeddings_df
    
    def generate_embeddings(self, content_df: pd.DataFrame, 
                          text_column: str = 'description',
                          batch_size: int = 50) -> pd.DataFrame:
        """Alias for generate_content_embeddings for backward compatibility"""
        return self.generate_content_embeddings(content_df, text_column, batch_size)
    
    async def generate_embeddings_from_text(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings from a list of text strings"""
        embeddings = []
        for text in texts:
            try:
                embedding = self.gemini_client.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.logger.error(f"Failed to generate embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append(np.zeros(768))
        return embeddings
    
    def _prepare_text_for_embedding(self, content_row: pd.Series, text_column: str) -> str:
        """Prepare text content for embedding generation"""
        
        # Primary text source
        text_parts = []
        
        # Title
        if 'title' in content_row and pd.notna(content_row['title']):
            text_parts.append(f"Title: {content_row['title']}")
        
        # Main text content
        if text_column in content_row and pd.notna(content_row[text_column]):
            text_parts.append(f"Description: {content_row[text_column]}")
        
        # Genres
        if 'genres' in content_row and pd.notna(content_row['genres']):
            genres_text = content_row['genres']
            if isinstance(genres_text, str):
                try:
                    # Handle JSON format
                    if genres_text.startswith('['):
                        genres_list = json.loads(genres_text)
                        genres_text = ', '.join(genres_list)
                    text_parts.append(f"Genres: {genres_text}")
                except:
                    text_parts.append(f"Genres: {genres_text}")
        
        # Additional metadata
        metadata_fields = ['director', 'author', 'creator_info', 'plot_summary']
        for field in metadata_fields:
            if field in content_row and pd.notna(content_row[field]):
                text_parts.append(f"{field.title()}: {content_row[field]}")
        
        # Combine all text
        combined_text = ". ".join(text_parts)
        
        # Ensure minimum text length
        if len(combined_text.strip()) < 20:
            return None
        
        # Truncate if too long
        max_length = 8000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        return combined_text
    
    def _generate_batch_embeddings(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of content items"""
        
        embeddings = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_content = {}
            
            for content_item in batch:
                future = executor.submit(
                    self._generate_single_embedding,
                    content_item
                )
                future_to_content[future] = content_item
            
            # Collect results
            for future in as_completed(future_to_content):
                content_item = future_to_content[future]
                try:
                    embedding_result = future.result()
                    embeddings.append(embedding_result)
                except Exception as e:
                    embeddings.append({
                        'content_id': content_item['content_id'],
                        'embedding': None,
                        'success': False,
                        'error': str(e),
                        'generation_time': 0.0
                    })
        
        return embeddings
    
    def _generate_single_embedding(self, content_item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embedding for a single content item"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"embedding_{content_item['content_id']}"
            cached_embedding = self._load_from_cache(cache_key)
            
            if cached_embedding is not None:
                return {
                    'content_id': content_item['content_id'],
                    'embedding': cached_embedding,
                    'success': True,
                    'cached': True,
                    'generation_time': time.time() - start_time
                }
            
            # Generate new embedding
            embedding = self.gemini_client.generate_embedding(content_item['text'])
            
            generation_time = time.time() - start_time
            
            if embedding is not None:
                # Validate embedding quality
                quality_score = self._assess_embedding_quality(embedding, content_item)
                
                # Cache the embedding
                self._save_to_cache(cache_key, embedding)
                
                return {
                    'content_id': content_item['content_id'],
                    'embedding': embedding,
                    'success': True,
                    'cached': False,
                    'generation_time': generation_time,
                    'quality_score': quality_score
                }
            else:
                return {
                    'content_id': content_item['content_id'],
                    'embedding': None,
                    'success': False,
                    'error': 'Failed to generate embedding',
                    'generation_time': generation_time
                }
        
        except Exception as e:
            return {
                'content_id': content_item['content_id'],
                'embedding': None,
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    def _assess_embedding_quality(self, embedding: List[float], content_item: Dict[str, Any]) -> float:
        """Assess the quality of generated embedding"""
        
        if not embedding or len(embedding) == 0:
            return 0.0
        
        quality_score = 1.0
        
        # Check embedding dimension
        expected_dimension = settings.embedding_dimension
        if len(embedding) != expected_dimension:
            quality_score *= 0.5
        
        # Check for all-zero embeddings
        if all(abs(val) < 1e-6 for val in embedding):
            quality_score *= 0.1
        
        # Check embedding distribution
        embedding_array = np.array(embedding)
        
        # Standard deviation check (should not be too low)
        std_dev = np.std(embedding_array)
        if std_dev < 0.01:
            quality_score *= 0.3
        
        # Range check (values should be diverse)
        value_range = np.max(embedding_array) - np.min(embedding_array)
        if value_range < 0.1:
            quality_score *= 0.4
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
            quality_score = 0.0
        
        return quality_score
    
    def _create_embeddings_dataframe(self, embeddings_results: List[Dict[str, Any]], 
                                   original_df: pd.DataFrame) -> pd.DataFrame:
        """Create a dataframe with embeddings and metadata"""
        
        embeddings_data = []
        
        for result in embeddings_results:
            embedding_record = {
                'content_id': result['content_id'],
                'has_embedding': result['success'],
                'embedding_dimension': len(result['embedding']) if result['embedding'] else 0,
                'generation_time': result.get('generation_time', 0.0),
                'quality_score': result.get('quality_score', 0.0),
                'cached': result.get('cached', False),
                'error': result.get('error', None)
            }
            
            # Add embedding as JSON (for database storage)
            if result['embedding']:
                embedding_record['semantic_embedding'] = json.dumps(result['embedding'])
            else:
                embedding_record['semantic_embedding'] = None
            
            embeddings_data.append(embedding_record)
        
        embeddings_df = pd.DataFrame(embeddings_data)
        
        # Merge with original content data
        result_df = original_df.merge(embeddings_df, on='content_id', how='left')
        
        # Fill missing values
        result_df['has_embedding'] = result_df['has_embedding'].fillna(False)
        result_df['embedding_dimension'] = result_df['embedding_dimension'].fillna(0)
        result_df['quality_score'] = result_df['quality_score'].fillna(0.0)
        
        return result_df
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from local cache"""
        cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to local cache"""
        cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.logger.warning(f"Failed to save to cache: {e}")
    
    def enhance_content_with_ai_analysis(self, content_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance content with AI-generated analysis"""
        
        self.logger.log_processing_start("enhance_content_with_ai", len(content_df))
        
        enhanced_df = content_df.copy()
        
        # Batch process for sentiment analysis
        self.logger.logger.info("Generating sentiment analysis...")
        sentiment_results = self._batch_analyze_sentiment(content_df)
        enhanced_df = self._merge_sentiment_results(enhanced_df, sentiment_results)
        
        # Batch process for theme extraction
        self.logger.logger.info("Extracting themes and topics...")
        theme_results = self._batch_extract_themes(content_df)
        enhanced_df = self._merge_theme_results(enhanced_df, theme_results)
        
        # Generate content summaries
        self.logger.logger.info("Generating enhanced summaries...")
        enhanced_df = self._generate_enhanced_summaries(enhanced_df)
        
        self.logger.log_processing_complete("enhance_content_with_ai", time.time(), len(content_df), 0)
        
        return enhanced_df
    
    def _batch_analyze_sentiment(self, content_df: pd.DataFrame, batch_size: int = 20) -> List[Dict[str, Any]]:
        """Batch analyze sentiment for content descriptions"""
        
        sentiment_results = []
        
        for i in range(0, len(content_df), batch_size):
            batch_df = content_df.iloc[i:i + batch_size]
            
            for idx, row in batch_df.iterrows():
                text = self._prepare_text_for_embedding(row, 'description')
                if text:
                    try:
                        sentiment = self.gemini_client.analyze_sentiment(text)
                        sentiment_results.append({
                            'content_id': row.get('content_id', idx),
                            'sentiment_analysis': sentiment
                        })
                    except Exception as e:
                        sentiment_results.append({
                            'content_id': row.get('content_id', idx),
                            'sentiment_analysis': None,
                            'error': str(e)
                        })
            
            # Rate limiting
            time.sleep(1)
        
        return sentiment_results
    
    def _batch_extract_themes(self, content_df: pd.DataFrame, batch_size: int = 20) -> List[Dict[str, Any]]:
        """Batch extract themes from content"""
        
        theme_results = []
        
        for i in range(0, len(content_df), batch_size):
            batch_df = content_df.iloc[i:i + batch_size]
            
            for idx, row in batch_df.iterrows():
                text = self._prepare_text_for_embedding(row, 'description')
                if text:
                    try:
                        themes = self.gemini_client.extract_themes(text)
                        theme_results.append({
                            'content_id': row.get('content_id', idx),
                            'theme_analysis': themes
                        })
                    except Exception as e:
                        theme_results.append({
                            'content_id': row.get('content_id', idx),
                            'theme_analysis': None,
                            'error': str(e)
                        })
            
            # Rate limiting
            time.sleep(1)
        
        return theme_results
    
    def _merge_sentiment_results(self, df: pd.DataFrame, sentiment_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge sentiment analysis results with content dataframe"""
        
        sentiment_data = []
        for result in sentiment_results:
            sentiment_analysis = result.get('sentiment_analysis', {})
            
            sentiment_record = {
                'content_id': result['content_id'],
                'sentiment_score': sentiment_analysis.get('sentiment_score', 0.5),
                'overall_sentiment': sentiment_analysis.get('overall_sentiment', 'neutral'),
                'emotions': json.dumps(sentiment_analysis.get('emotions', [])),
                'mood': sentiment_analysis.get('mood', 'unknown'),
                'tone': sentiment_analysis.get('tone', 'neutral')
            }
            sentiment_data.append(sentiment_record)
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        return df.merge(sentiment_df, on='content_id', how='left')
    
    def _merge_theme_results(self, df: pd.DataFrame, theme_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge theme analysis results with content dataframe"""
        
        theme_data = []
        for result in theme_results:
            theme_analysis = result.get('theme_analysis', {})
            
            theme_record = {
                'content_id': result['content_id'],
                'main_themes': json.dumps(theme_analysis.get('main_themes', [])),
                'ai_topics': json.dumps(theme_analysis.get('topics', [])),
                'ai_concepts': json.dumps(theme_analysis.get('concepts', [])),
                'ai_genre_indicators': json.dumps(theme_analysis.get('genre_indicators', [])),
                'complexity_level': theme_analysis.get('complexity_level', 'moderate'),
                'target_audience': theme_analysis.get('target_audience', 'general')
            }
            theme_data.append(theme_record)
        
        theme_df = pd.DataFrame(theme_data)
        
        return df.merge(theme_df, on='content_id', how='left')
    
    def _generate_enhanced_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced summaries for content"""
        
        enhanced_summaries = []
        
        for idx, row in df.iterrows():
            try:
                if pd.notna(row.get('title')) and pd.notna(row.get('description')):
                    enhanced_summary = self.gemini_client.generate_content_summary(
                        row['title'], 
                        row['description']
                    )
                    enhanced_summaries.append(enhanced_summary)
                else:
                    enhanced_summaries.append(row.get('description', ''))
            except Exception as e:
                enhanced_summaries.append(row.get('description', ''))
                self.logger.logger.warning(f"Failed to generate summary for {row.get('content_id')}: {e}")
        
        df['ai_enhanced_summary'] = enhanced_summaries
        
        return df
    
    def calculate_embedding_statistics(self, embeddings_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive embedding statistics"""
        
        stats = {
            'total_content_items': len(embeddings_df),
            'successful_embeddings': embeddings_df['has_embedding'].sum(),
            'embedding_success_rate': embeddings_df['has_embedding'].mean() * 100,
            'average_quality_score': embeddings_df['quality_score'].mean(),
            'average_generation_time': embeddings_df['generation_time'].mean(),
            'cached_embeddings': embeddings_df['cached'].sum() if 'cached' in embeddings_df.columns else 0,
            'embedding_dimension_consistency': (embeddings_df['embedding_dimension'] == settings.embedding_dimension).mean() * 100
        }
        
        # Quality distribution
        quality_scores = embeddings_df[embeddings_df['has_embedding']]['quality_score']
        if not quality_scores.empty:
            stats['quality_distribution'] = {
                'min': quality_scores.min(),
                'max': quality_scores.max(),
                'median': quality_scores.median(),
                'std': quality_scores.std()
            }
        
        # Generation time distribution
        generation_times = embeddings_df[embeddings_df['has_embedding']]['generation_time']
        if not generation_times.empty:
            stats['generation_time_distribution'] = {
                'min': generation_times.min(),
                'max': generation_times.max(),
                'median': generation_times.median(),
                'std': generation_times.std()
            }
        
        return stats
