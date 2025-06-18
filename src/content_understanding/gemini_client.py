"""
Google Gemini API Client for Content Understanding
Handles embedding generation, content analysis, and cross-domain mapping
"""
import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
import asyncio
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff

from ..utils.config import settings
from ..utils.redis_client import get_redis_client, RedisConfig
from ..utils.logging import get_data_processing_logger, get_api_logger


class GeminiClient:
    """Google Gemini API client with rate limiting and error handling"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        try:
            self.api_key = api_key or settings.gemini_api_key
            if not self.api_key:
                raise ValueError("Gemini API key is required")
                
            self.model_name = model_name
            self.logger = get_api_logger()
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            
            # Rate limiting
            self.requests_per_minute = 60
            self.requests_per_day = 1500
            self.last_request_time = 0
            self.daily_request_count = 0
            self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Caching
            self.cache_enabled = True
            try:
                self.redis_client = get_redis_client()
                if self.redis_client and self.redis_client.is_healthy():
                    self.cache = self.redis_client.cache
                    self.logger.logger.info("Redis caching enabled")
                else:
                    self.cache_enabled = False
                    self.cache = None
                    self.logger.logger.warning("Redis not available, caching disabled")
            except Exception as e:
                self.cache_enabled = False
                self.cache = None
                self.logger.logger.warning(f"Redis not available, caching disabled: {e}")
        except Exception as e:
            self.logger.logger.error(f"Failed to initialize GeminiClient: {e}")
            raise
        
        # Batch processing
        self.batch_size = 10
        self.max_retries = 3
        
    def _check_rate_limits(self):
        """Check and enforce rate limits"""
        now = time.time()
        
        # Reset daily counter if needed
        if datetime.now() >= self.daily_reset_time + timedelta(days=1):
            self.daily_request_count = 0
            self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check daily limit
        if self.daily_request_count >= self.requests_per_day:
            raise Exception("Daily API rate limit exceeded")
        
        # Check per-minute limit
        time_since_last = now - self.last_request_time
        if time_since_last < 60 / self.requests_per_minute:
            sleep_time = (60 / self.requests_per_minute) - time_since_last
            time.sleep(sleep_time)
    
    def _get_cache_key(self, content: str, operation: str) -> str:
        """Generate cache key for content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"gemini:{operation}:{content_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache"""
        if not self.cache_enabled or not self.cache:
            return None
        
        try:
            cached_result = self.cache.get(cache_key)
            return cached_result
        except Exception as e:
            self.logger.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Any, ttl: int = 86400):
        """Save result to cache (24 hour TTL by default)"""
        if not self.cache_enabled or not self.cache:
            return
        
        try:
            self.cache.set(cache_key, result, ttl)
        except Exception as e:
            self.logger.logger.warning(f"Cache save failed: {e}")
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _make_api_call(self, prompt: str, operation: str) -> Dict[str, Any]:
        """Make API call to Gemini with retry logic"""
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, operation)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Check rate limits
        self._check_rate_limits()
        
        start_time = time.time()
        
        try:
            response = self.model.generate_content(prompt)
            
            duration = time.time() - start_time
            token_count = len(prompt.split())  # Approximate token count
            
            # Log API call
            self.logger.log_gemini_api_call(operation, token_count, duration, True)
            
            # Update counters
            self.last_request_time = time.time()
            self.daily_request_count += 1
            
            # Parse response
            result = {
                'success': True,
                'content': response.text,
                'token_count': token_count,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_gemini_api_call(operation, 0, duration, False)
            
            result = {
                'success': False,
                'error': str(e),
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate semantic embedding for text content"""
        if not text or len(text.strip()) == 0:
            return None
        
        # Truncate text if too long (Gemini has token limits)
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length]
        
        prompt = f"""
        Please analyze the following content and provide a detailed semantic understanding:
        
        Content: {text}
        
        Provide a comprehensive analysis covering:
        1. Main themes and topics
        2. Emotional tone and sentiment
        3. Genre characteristics
        4. Style and complexity
        5. Target audience
        6. Key concepts and entities
        
        Format your response as a structured analysis that captures the semantic essence of the content.
        """
        
        result = self._make_api_call(prompt, "embedding")
        
        if result['success']:
            # Convert text analysis to numerical embedding
            # This is a simplified approach - in production, you'd use proper embedding models
            analysis_text = result['content']
            embedding = self._text_to_embedding(analysis_text)
            return embedding
        
        return None
    
    def _text_to_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """Convert text analysis to numerical embedding vector"""
        # Simple hash-based embedding generation
        # In production, use proper embedding models or Gemini's embedding API when available
        
        words = text.lower().split()
        
        # Create a pseudo-embedding based on text characteristics
        embedding = np.zeros(dimension)
        
        # Text length features
        embedding[0] = min(len(text) / 1000, 1.0)
        embedding[1] = min(len(words) / 500, 1.0)
        
        # Sentiment indicators (simple keyword-based)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'sad', 'angry', 'hate', 'terrible']
        
        positive_score = sum(1 for word in words if word in positive_words) / max(len(words), 1)
        negative_score = sum(1 for word in words if word in negative_words) / max(len(words), 1)
        
        embedding[2] = positive_score
        embedding[3] = negative_score
        
        # Genre/category indicators
        genre_keywords = {
            'action': ['action', 'fight', 'battle', 'adventure', 'hero'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh', 'joke'],
            'drama': ['drama', 'emotion', 'character', 'relationship', 'life'],
            'romance': ['romance', 'love', 'relationship', 'romantic', 'heart'],
            'thriller': ['thriller', 'suspense', 'mystery', 'tension', 'danger'],
            'horror': ['horror', 'scary', 'fear', 'terror', 'frightening']
        }
        
        for i, (genre, keywords) in enumerate(genre_keywords.items()):
            if i + 4 < dimension:
                score = sum(1 for word in words if word in keywords) / max(len(words), 1)
                embedding[i + 4] = score
        
        # Fill remaining dimensions with text-based features
        remaining_start = 10
        text_hash = hashlib.md5(text.encode()).digest()
        
        for i in range(remaining_start, dimension):
            if i - remaining_start < len(text_hash):
                # Normalize hash bytes to [-1, 1]
                embedding[i] = (text_hash[i - remaining_start] - 128) / 128
        
        return embedding.tolist()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotional content"""
        prompt = f"""
        Analyze the sentiment and emotional content of the following text:
        
        Text: {text}
        
        Provide analysis in the following JSON format:
        {{
            "overall_sentiment": "positive/negative/neutral",
            "sentiment_score": 0.0-1.0,
            "emotions": ["emotion1", "emotion2", ...],
            "mood": "mood_description",
            "tone": "tone_description"
        }}
        """
        
        result = self._make_api_call(prompt, "sentiment_analysis")
        
        if result['success']:
            try:
                # Extract JSON from response
                content = result['content']
                # Find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass
        
        # Fallback response
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.5,
            "emotions": [],
            "mood": "unknown",
            "tone": "neutral"
        }
    
    def extract_themes(self, text: str) -> Dict[str, Any]:
        """Extract themes and topics from content"""
        prompt = f"""
        Extract and analyze the main themes, topics, and concepts from the following content:
        
        Content: {text}
        
        Provide analysis in the following JSON format:
        {{
            "main_themes": ["theme1", "theme2", ...],
            "topics": ["topic1", "topic2", ...],
            "concepts": ["concept1", "concept2", ...],
            "genre_indicators": ["genre1", "genre2", ...],
            "complexity_level": "simple/moderate/complex",
            "target_audience": "audience_description"
        }}
        """
        
        result = self._make_api_call(prompt, "theme_extraction")
        
        if result['success']:
            try:
                content = result['content']
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                pass
        
        return {
            "main_themes": [],
            "topics": [],
            "concepts": [],
            "genre_indicators": [],
            "complexity_level": "moderate",
            "target_audience": "general"
        }
    
    def generate_content_summary(self, title: str, description: str) -> str:
        """Generate enhanced content summary"""
        prompt = f"""
        Create a comprehensive and engaging summary for the following content:
        
        Title: {title}
        Description: {description}
        
        Generate a summary that:
        1. Captures the essence and main appeal
        2. Highlights key themes and elements
        3. Indicates the target audience
        4. Mentions genre and style characteristics
        5. Is engaging and informative (2-3 sentences)
        """
        
        result = self._make_api_call(prompt, "content_summary")
        
        if result['success']:
            return result['content'].strip()
        
        # Fallback to original description
        return description[:200] + "..." if len(description) > 200 else description
    
    def compare_content_similarity(self, content1: Dict[str, str], content2: Dict[str, str]) -> float:
        """Compare similarity between two pieces of content"""
        prompt = f"""
        Compare the similarity between these two pieces of content:
        
        Content 1:
        Title: {content1.get('title', 'N/A')}
        Description: {content1.get('description', 'N/A')}
        Genre: {content1.get('genres', 'N/A')}
        
        Content 2:
        Title: {content2.get('title', 'N/A')}
        Description: {content2.get('description', 'N/A')}
        Genre: {content2.get('genres', 'N/A')}
        
        Rate their similarity on a scale of 0.0 to 1.0, considering:
        - Thematic similarity
        - Genre overlap
        - Style and tone
        - Target audience
        - Narrative elements
        
        Respond with just the numerical similarity score (0.0-1.0).
        """
        
        result = self._make_api_call(prompt, "content_similarity")
        
        if result['success']:
            try:
                # Extract numerical score
                content = result['content'].strip()
                score = float(content)
                return max(0.0, min(1.0, score))
            except:
                pass
        
        return 0.5  # Default neutral similarity
    
    def batch_process_content(self, content_list: List[Dict[str, Any]], 
                            operation: str = "embedding") -> List[Dict[str, Any]]:
        """Process multiple content items in batches"""
        
        self.logger.logger.info(f"Starting batch processing of {len(content_list)} items for {operation}")
        
        results = []
        
        # Process in batches
        for i in range(0, len(content_list), self.batch_size):
            batch = content_list[i:i + self.batch_size]
            batch_results = []
            
            # Process batch items with threading
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_content = {}
                
                for content_item in batch:
                    if operation == "embedding":
                        text = content_item.get('description', '') or content_item.get('title', '')
                        future = executor.submit(self.generate_embedding, text)
                    elif operation == "sentiment":
                        text = content_item.get('description', '') or content_item.get('title', '')
                        future = executor.submit(self.analyze_sentiment, text)
                    elif operation == "themes":
                        text = content_item.get('description', '') or content_item.get('title', '')
                        future = executor.submit(self.extract_themes, text)
                    else:
                        continue
                    
                    future_to_content[future] = content_item
                
                # Collect results
                for future in as_completed(future_to_content):
                    content_item = future_to_content[future]
                    try:
                        result = future.result()
                        batch_results.append({
                            'content_id': content_item.get('content_id'),
                            'result': result,
                            'success': True
                        })
                    except Exception as e:
                        batch_results.append({
                            'content_id': content_item.get('content_id'),
                            'result': None,
                            'success': False,
                            'error': str(e)
                        })
            
            results.extend(batch_results)
            
            # Progress logging
            self.logger.logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(content_list) + self.batch_size - 1)//self.batch_size}")
            
            # Rate limiting between batches
            time.sleep(2)
        
        self.logger.logger.info(f"Completed batch processing. Success rate: {sum(1 for r in results if r['success'])/len(results)*100:.1f}%")
        
        return results
    
    async def close(self):
        """Close any open connections"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client is not None:
                self.redis_client.close()
            self.logger.info("GeminiClient connections closed")
        except Exception as e:
            self.logger.warning(f"Error while closing GeminiClient connections: {str(e)}")
