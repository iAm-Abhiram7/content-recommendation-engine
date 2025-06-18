"""
Cross-Domain Content Mapper
Maps and finds relationships between different content types (movies, books, music)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict, Counter
import re
import community.community_louvain as community

from .gemini_client import GeminiClient
from ..utils.config import settings
from ..utils.logging import get_data_processing_logger


class CrossDomainMapper:
    """Map content across different domains (movies, books, music, etc.)"""
    
    def __init__(self, gemini_client: GeminiClient = None):
        self.gemini_client = gemini_client or GeminiClient()
        self.logger = get_data_processing_logger("CrossDomainMapper")
        
        # Domain mappings
        self.domain_mappings = {}
        self.similarity_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Genre mappings across domains
        self.cross_domain_genres = self._initialize_genre_mappings()
        
    def _initialize_genre_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize genre mappings across different content domains"""
        return {
            'action': {
                'movie': ['Action', 'Adventure', 'Thriller'],
                'book': ['Adventure', 'Thriller', 'Action & Adventure', 'Military Fiction'],
                'music': ['Rock', 'Metal', 'Electronic', 'Rap', 'Hip-Hop'],
                'tv': ['Action', 'Adventure', 'Crime', 'Thriller']
            },
            'romance': {
                'movie': ['Romance', 'Romantic Comedy', 'Drama'],
                'book': ['Romance', 'Contemporary Romance', 'Historical Romance'],
                'music': ['Pop', 'R&B', 'Soul', 'Ballad'],
                'tv': ['Romance', 'Drama', 'Soap Opera']
            },
            'comedy': {
                'movie': ['Comedy', 'Romantic Comedy', 'Comedy-Drama'],
                'book': ['Humor', 'Comedy', 'Satirical Fiction'],
                'music': ['Comedy', 'Novelty', 'Parody'],
                'tv': ['Comedy', 'Sitcom', 'Comedy-Drama']
            },
            'drama': {
                'movie': ['Drama', 'Biography', 'Historical Drama'],
                'book': ['Literary Fiction', 'Contemporary Fiction', 'Historical Fiction'],
                'music': ['Folk', 'Singer-Songwriter', 'Classical'],
                'tv': ['Drama', 'Period Drama', 'Family Drama']
            },
            'horror': {
                'movie': ['Horror', 'Thriller', 'Supernatural'],
                'book': ['Horror', 'Supernatural', 'Gothic', 'Dark Fantasy'],
                'music': ['Metal', 'Gothic', 'Industrial', 'Dark Ambient'],
                'tv': ['Horror', 'Supernatural', 'Thriller']
            },
            'fantasy': {
                'movie': ['Fantasy', 'Adventure', 'Family'],
                'book': ['Fantasy', 'Epic Fantasy', 'Urban Fantasy', 'Magical Realism'],
                'music': ['Symphonic Metal', 'Folk', 'New Age', 'Soundtrack'],
                'tv': ['Fantasy', 'Adventure', 'Family']
            },
            'science_fiction': {
                'movie': ['Science Fiction', 'Sci-Fi', 'Futuristic'],
                'book': ['Science Fiction', 'Dystopian', 'Space Opera', 'Cyberpunk'],
                'music': ['Electronic', 'Synthwave', 'Ambient', 'Industrial'],
                'tv': ['Science Fiction', 'Dystopian', 'Futuristic']
            },
            'mystery': {
                'movie': ['Mystery', 'Crime', 'Detective', 'Noir'],
                'book': ['Mystery', 'Crime Fiction', 'Detective Fiction', 'Cozy Mystery'],
                'music': ['Jazz', 'Blues', 'Dark Ambient'],
                'tv': ['Mystery', 'Crime', 'Detective', 'Police Procedural']
            }
        }
    
    def map_movie_to_book_similarities(self, movies_df: pd.DataFrame, 
                                     books_df: pd.DataFrame) -> pd.DataFrame:
        """Find similarities between movies and books"""
        
        self.logger.log_processing_start("map_movie_to_book", len(movies_df) * len(books_df))
        
        # Prepare data
        movie_embeddings = self._extract_embeddings(movies_df)
        book_embeddings = self._extract_embeddings(books_df)
        
        similarities = []
        
        # Calculate similarities using multiple methods
        for i, movie_row in movies_df.iterrows():
            movie_id = movie_row.get('content_id', i)
            
            # Get movie embedding
            movie_embedding = movie_embeddings.get(movie_id)
            
            # Genre-based similarity
            movie_genres = self._parse_genres(movie_row.get('genres', ''))
            
            for j, book_row in books_df.iterrows():
                book_id = book_row.get('content_id', j)
                book_embedding = book_embeddings.get(book_id)
                book_genres = self._parse_genres(book_row.get('genres', ''))
                
                # Calculate composite similarity
                similarity_score = self._calculate_cross_domain_similarity(
                    movie_row, book_row, movie_embedding, book_embedding,
                    movie_genres, book_genres, 'movie', 'book'
                )
                
                if similarity_score > self.similarity_thresholds['low']:
                    similarities.append({
                        'movie_id': movie_id,
                        'book_id': book_id,
                        'similarity_score': similarity_score,
                        'movie_title': movie_row.get('title', ''),
                        'book_title': book_row.get('title', ''),
                        'shared_themes': self._find_shared_themes(movie_genres, book_genres),
                        'mapping_type': 'cross_domain'
                    })
        
        # Create similarity dataframe
        similarity_df = pd.DataFrame(similarities)
        
        # Sort by similarity and keep top matches
        if not similarity_df.empty:
            similarity_df = similarity_df.sort_values('similarity_score', ascending=False)
            # Keep top 3 matches per movie
            similarity_df = similarity_df.groupby('movie_id').head(3).reset_index(drop=True)
        
        self.logger.log_processing_complete("map_movie_to_book", time.time(), len(similarities), 0)
        
        return similarity_df
    
    def create_unified_genre_taxonomy(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create a unified genre taxonomy across all content types"""
        
        self.logger.log_processing_start("create_unified_taxonomy", sum(len(df) for df in datasets.values()))
        
        # Collect all genres from all datasets
        all_genres = defaultdict(set)
        
        for dataset_name, df in datasets.items():
            content_type = self._infer_content_type(dataset_name)
            
            if 'genres' in df.columns:
                for genres_str in df['genres'].dropna():
                    genres = self._parse_genres(genres_str)
                    all_genres[content_type].update(genres)
        
        # Create mappings using cross-domain genre knowledge
        unified_taxonomy = {
            'meta_genres': {},
            'domain_specific_genres': all_genres,
            'cross_domain_mappings': {},
            'genre_hierarchy': {}
        }
        
        # Map to meta-genres
        for meta_genre, domain_mapping in self.cross_domain_genres.items():
            unified_taxonomy['meta_genres'][meta_genre] = domain_mapping
            
            # Create reverse mappings
            for domain, genres in domain_mapping.items():
                if domain not in unified_taxonomy['cross_domain_mappings']:
                    unified_taxonomy['cross_domain_mappings'][domain] = {}
                
                for genre in genres:
                    unified_taxonomy['cross_domain_mappings'][domain][genre.lower()] = meta_genre
        
        # Create genre hierarchy
        unified_taxonomy['genre_hierarchy'] = self._create_genre_hierarchy(all_genres)
        
        # Add AI-enhanced genre analysis
        if self.gemini_client:
            unified_taxonomy['ai_enhanced_mappings'] = self._enhance_genre_mappings_with_ai(all_genres)
        
        self.logger.log_processing_complete("create_unified_taxonomy", time.time(), 1, 0)
        
        return unified_taxonomy
    
    def find_adaptation_relationships(self, movies_df: pd.DataFrame, 
                                    books_df: pd.DataFrame) -> pd.DataFrame:
        """Find book-to-movie adaptation relationships"""
        
        adaptations = []
        
        # Title-based matching (direct adaptations)
        for _, movie_row in movies_df.iterrows():
            movie_title = movie_row.get('title', '').lower()
            movie_year = movie_row.get('publication_year', 0)
            
            for _, book_row in books_df.iterrows():
                book_title = book_row.get('title', '').lower()
                book_year = book_row.get('publication_year', 0)
                
                # Direct title match
                title_similarity = self._calculate_title_similarity(movie_title, book_title)
                
                # Year constraint (movie should be after book)
                year_valid = book_year < movie_year if book_year and movie_year else True
                
                if title_similarity > 0.8 and year_valid:
                    adaptations.append({
                        'movie_id': movie_row.get('content_id'),
                        'book_id': book_row.get('content_id'),
                        'movie_title': movie_row.get('title'),
                        'book_title': book_row.get('title'),
                        'title_similarity': title_similarity,
                        'movie_year': movie_year,
                        'book_year': book_year,
                        'adaptation_type': 'direct',
                        'confidence': title_similarity
                    })
        
        # AI-enhanced adaptation detection
        if self.gemini_client:
            ai_adaptations = self._detect_adaptations_with_ai(movies_df, books_df)
            adaptations.extend(ai_adaptations)
        
        adaptation_df = pd.DataFrame(adaptations)
        
        # Remove duplicates and sort by confidence
        if not adaptation_df.empty:
            adaptation_df = adaptation_df.drop_duplicates(subset=['movie_id', 'book_id'])
            adaptation_df = adaptation_df.sort_values('confidence', ascending=False)
        
        return adaptation_df
    
    def create_content_similarity_network(self, content_df: pd.DataFrame, 
                                        similarity_threshold: float = 0.6) -> nx.Graph:
        """Create a network graph of content similarities"""
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for _, row in content_df.iterrows():
            content_id = row.get('content_id')
            G.add_node(content_id, **{
                'title': row.get('title', ''),
                'content_type': row.get('content_type', ''),
                'genres': self._parse_genres(row.get('genres', '')),
                'popularity': row.get('popularity_score', 0.0)
            })
        
        # Extract embeddings
        embeddings = self._extract_embeddings(content_df)
        content_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[cid] for cid in content_ids])
        
        # Calculate pairwise similarities
        if len(embedding_matrix) > 0:
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            # Add edges for similar content
            for i, content_id_1 in enumerate(content_ids):
                for j, content_id_2 in enumerate(content_ids):
                    if i < j and similarity_matrix[i][j] > similarity_threshold:
                        G.add_edge(content_id_1, content_id_2, 
                                 similarity=similarity_matrix[i][j])
        
        # Add community detection
        try:
            communities = community.best_partition(G)
            nx.set_node_attributes(G, communities, 'community')
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Community detection failed: {str(e)}")
        
        return G
    
    def analyze_cross_domain_preferences(self, user_interactions: pd.DataFrame,
                                       content_metadata: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user preferences across different content domains"""
        
        # Merge interactions with content metadata
        interaction_content = user_interactions.merge(
            content_metadata[['content_id', 'content_type', 'genres']], 
            on='content_id', 
            how='left'
        )
        
        cross_domain_analysis = {}
        
        for user_id in user_interactions['user_id'].unique():
            user_data = interaction_content[interaction_content['user_id'] == user_id]
            
            user_analysis = {
                'user_id': user_id,
                'domain_preferences': {},
                'cross_domain_patterns': {},
                'genre_consistency': 0.0,
                'domain_diversity': 0.0
            }
            
            # Analyze preferences by domain
            domain_ratings = user_data.groupby('content_type')['rating'].agg(['mean', 'count'])
            
            for domain in domain_ratings.index:
                user_analysis['domain_preferences'][domain] = {
                    'avg_rating': domain_ratings.loc[domain, 'mean'],
                    'interaction_count': domain_ratings.loc[domain, 'count'],
                    'preference_strength': domain_ratings.loc[domain, 'mean'] * np.log(domain_ratings.loc[domain, 'count'] + 1)
                }
            
            # Analyze genre consistency across domains
            user_genres = []
            for genres_str in user_data['genres'].dropna():
                user_genres.extend(self._parse_genres(genres_str))
            
            if user_genres:
                genre_counter = Counter(user_genres)
                # Calculate genre diversity
                unique_genres = len(set(user_genres))
                total_genres = len(user_genres)
                user_analysis['genre_consistency'] = 1 - (unique_genres / total_genres) if total_genres > 0 else 0
            
            # Calculate domain diversity
            domains_used = user_data['content_type'].nunique()
            total_domains = content_metadata['content_type'].nunique()
            user_analysis['domain_diversity'] = domains_used / total_domains if total_domains > 0 else 0
            
            cross_domain_analysis[user_id] = user_analysis
        
        return cross_domain_analysis
    
    def _extract_embeddings(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract embeddings from dataframe"""
        embeddings = {}
        
        for _, row in df.iterrows():
            content_id = row.get('content_id')
            
            if 'semantic_embedding' in row and pd.notna(row['semantic_embedding']):
                try:
                    embedding = json.loads(row['semantic_embedding'])
                    embeddings[content_id] = np.array(embedding)
                except:
                    # Generate a fallback embedding based on text features
                    embeddings[content_id] = self._generate_fallback_embedding(row)
            else:
                embeddings[content_id] = self._generate_fallback_embedding(row)
        
        return embeddings
    
    def _generate_fallback_embedding(self, content_row: pd.Series, dimension: int = 768) -> np.ndarray:
        """Generate a fallback embedding based on available content features"""
        embedding = np.zeros(dimension)
        
        # Use title and description to create basic embedding
        text_features = []
        if 'title' in content_row and pd.notna(content_row['title']):
            text_features.extend(content_row['title'].lower().split())
        
        if 'description' in content_row and pd.notna(content_row['description']):
            text_features.extend(content_row['description'].lower().split()[:50])  # Limit words
        
        # Create simple hash-based embedding
        if text_features:
            for i, word in enumerate(text_features[:dimension//4]):
                word_hash = hash(word) % dimension
                embedding[word_hash] += 1.0 / (i + 1)  # Diminishing weight
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _calculate_cross_domain_similarity(self, content1: pd.Series, content2: pd.Series,
                                         embedding1: Optional[np.ndarray], embedding2: Optional[np.ndarray],
                                         genres1: List[str], genres2: List[str],
                                         domain1: str, domain2: str) -> float:
        """Calculate similarity between content from different domains"""
        
        similarity_components = []
        
        # Embedding similarity
        if embedding1 is not None and embedding2 is not None:
            embedding_sim = cosine_similarity([embedding1], [embedding2])[0][0]
            similarity_components.append(('embedding', embedding_sim, 0.4))
        
        # Genre similarity
        genre_sim = self._calculate_genre_similarity(genres1, genres2, domain1, domain2)
        similarity_components.append(('genre', genre_sim, 0.3))
        
        # Title similarity
        title1 = content1.get('title', '').lower()
        title2 = content2.get('title', '').lower()
        title_sim = self._calculate_title_similarity(title1, title2)
        similarity_components.append(('title', title_sim, 0.2))
        
        # Temporal similarity (publication years)
        year1 = content1.get('publication_year', 0)
        year2 = content2.get('publication_year', 0)
        if year1 and year2:
            year_diff = abs(year1 - year2)
            temporal_sim = max(0, 1 - year_diff / 50)  # 50-year window
            similarity_components.append(('temporal', temporal_sim, 0.1))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in similarity_components)
        weighted_similarity = sum(score * weight for _, score, weight in similarity_components) / total_weight
        
        return weighted_similarity
    
    def _calculate_genre_similarity(self, genres1: List[str], genres2: List[str],
                                  domain1: str, domain2: str) -> float:
        """Calculate genre similarity across domains"""
        
        if not genres1 or not genres2:
            return 0.0
        
        # Map genres to meta-genres
        meta_genres1 = set()
        meta_genres2 = set()
        
        for genre in genres1:
            genre_lower = genre.lower()
            for meta_genre, domain_mapping in self.cross_domain_genres.items():
                if domain1 in domain_mapping:
                    if genre_lower in [g.lower() for g in domain_mapping[domain1]]:
                        meta_genres1.add(meta_genre)
        
        for genre in genres2:
            genre_lower = genre.lower()
            for meta_genre, domain_mapping in self.cross_domain_genres.items():
                if domain2 in domain_mapping:
                    if genre_lower in [g.lower() for g in domain_mapping[domain2]]:
                        meta_genres2.add(meta_genre)
        
        # Calculate Jaccard similarity
        if meta_genres1 and meta_genres2:
            intersection = len(meta_genres1.intersection(meta_genres2))
            union = len(meta_genres1.union(meta_genres2))
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        
        if not title1 or not title2:
            return 0.0
        
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words1 = set(re.findall(r'\w+', title1.lower())) - stop_words
        words2 = set(re.findall(r'\w+', title2.lower())) - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _parse_genres(self, genres_str: str) -> List[str]:
        """Parse genres from string format"""
        if pd.isna(genres_str) or not genres_str:
            return []
        
        try:
            # Try JSON format first
            if genres_str.startswith('['):
                return json.loads(genres_str)
            else:
                # Handle pipe or comma separated
                return [g.strip() for g in genres_str.replace('|', ',').split(',') if g.strip()]
        except:
            return [genres_str.strip()] if genres_str.strip() else []
    
    def _infer_content_type(self, dataset_name: str) -> str:
        """Infer content type from dataset name"""
        dataset_lower = dataset_name.lower()
        
        if 'movie' in dataset_lower or 'film' in dataset_lower:
            return 'movie'
        elif 'book' in dataset_lower:
            return 'book'
        elif 'music' in dataset_lower or 'song' in dataset_lower:
            return 'music'
        elif 'tv' in dataset_lower or 'show' in dataset_lower:
            return 'tv'
        else:
            return 'unknown'
    
    def _find_shared_themes(self, genres1: List[str], genres2: List[str]) -> List[str]:
        """Find shared themes between two sets of genres"""
        shared = []
        
        for meta_genre, domain_mapping in self.cross_domain_genres.items():
            # Check if both genre lists map to this meta-genre
            has_meta_genre_1 = any(g.lower() in [mapped.lower() for mapped_list in domain_mapping.values() for mapped in mapped_list] for g in genres1)
            has_meta_genre_2 = any(g.lower() in [mapped.lower() for mapped_list in domain_mapping.values() for mapped in mapped_list] for g in genres2)
            
            if has_meta_genre_1 and has_meta_genre_2:
                shared.append(meta_genre)
        
        return shared
    
    def _create_genre_hierarchy(self, all_genres: Dict[str, set]) -> Dict[str, Any]:
        """Create a hierarchical genre structure"""
        hierarchy = {
            'primary_categories': list(self.cross_domain_genres.keys()),
            'domain_specific': {},
            'mappings': self.cross_domain_genres
        }
        
        for domain, genres in all_genres.items():
            hierarchy['domain_specific'][domain] = sorted(list(genres))
        
        return hierarchy
    
    def _enhance_genre_mappings_with_ai(self, all_genres: Dict[str, set]) -> Dict[str, Any]:
        """Use AI to enhance genre mappings"""
        # This would use Gemini to analyze genre relationships
        # Simplified implementation for now
        return {
            'ai_suggested_mappings': {},
            'confidence_scores': {},
            'new_meta_genres': []
        }
    
    def _detect_adaptations_with_ai(self, movies_df: pd.DataFrame, 
                                  books_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Use AI to detect potential adaptations"""
        adaptations = []
        
        # Sample a subset for AI analysis (to manage API costs)
        sample_size = min(50, len(movies_df))
        movies_sample = movies_df.sample(n=sample_size) if len(movies_df) > sample_size else movies_df
        
        for _, movie_row in movies_sample.iterrows():
            # Find potential book matches
            potential_books = books_df[
                books_df['publication_year'] < movie_row.get('publication_year', 9999)
            ].head(10)  # Limit to 10 potential matches
            
            for _, book_row in potential_books.iterrows():
                try:
                    similarity = self.gemini_client.compare_content_similarity(
                        {
                            'title': movie_row.get('title', ''),
                            'description': movie_row.get('description', ''),
                            'genres': movie_row.get('genres', '')
                        },
                        {
                            'title': book_row.get('title', ''),
                            'description': book_row.get('description', ''),
                            'genres': book_row.get('genres', '')
                        }
                    )
                    
                    if similarity > 0.7:
                        adaptations.append({
                            'movie_id': movie_row.get('content_id'),
                            'book_id': book_row.get('content_id'),
                            'movie_title': movie_row.get('title'),
                            'book_title': book_row.get('title'),
                            'title_similarity': similarity,
                            'movie_year': movie_row.get('publication_year'),
                            'book_year': book_row.get('publication_year'),
                            'adaptation_type': 'ai_detected',
                            'confidence': similarity
                        })
                
                except Exception as e:
                    self.logger.logger.warning(f"AI adaptation detection failed: {e}")
                    continue
        
        return adaptations
