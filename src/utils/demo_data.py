"""
Demo Data Generator
Generate realistic sample data for the demo interface
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class DemoDataGenerator:
    """Generate realistic demo data for the recommendation system"""
    
    def __init__(self):
        self.genres = {
            'movies': ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation', 'Documentary'],
            'books': ['Fiction', 'Non-Fiction', 'Mystery', 'Romance', 'Sci-Fi', 'Biography', 'History', 'Philosophy', 'Self-Help', 'Fantasy'],
            'music': ['Rock', 'Pop', 'Hip-Hop', 'Electronic', 'Classical', 'Jazz', 'Country', 'R&B', 'Indie', 'Alternative']
        }
        
        self.sample_users = self._generate_sample_users()
        self.content_catalogs = self._generate_content_catalogs()
    
    def _generate_sample_users(self) -> Dict[str, Dict[str, Any]]:
        """Generate sample user profiles"""
        return {
            'user_001': {
                'name': 'Alex Chen',
                'persona': 'Tech Enthusiast',
                'age': 28,
                'demographics': {'location': 'San Francisco', 'education': 'Masters'},
                'preferences': {
                    'action': 0.8,
                    'sci-fi': 0.9,
                    'comedy': 0.6,
                    'drama': 0.4,
                    'horror': 0.2
                },
                'personality': {
                    'openness': 0.8,
                    'exploration_tendency': 0.7,
                    'diversity_preference': 0.8
                },
                'viewing_history': 127,
                'favorite_genres': ['Sci-Fi', 'Action', 'Documentary']
            },
            'user_002': {
                'name': 'Maria Rodriguez',
                'persona': 'Creative Professional',
                'age': 34,
                'demographics': {'location': 'New York', 'education': 'Bachelors'},
                'preferences': {
                    'drama': 0.9,
                    'romance': 0.7,
                    'comedy': 0.8,
                    'action': 0.3,
                    'horror': 0.1
                },
                'personality': {
                    'openness': 0.9,
                    'exploration_tendency': 0.8,
                    'diversity_preference': 0.9
                },
                'viewing_history': 203,
                'favorite_genres': ['Drama', 'Romance', 'Independent']
            },
            'user_003': {
                'name': 'David Kim',
                'persona': 'College Student',
                'age': 22,
                'demographics': {'location': 'Boston', 'education': 'Undergraduate'},
                'preferences': {
                    'action': 0.9,
                    'comedy': 0.8,
                    'horror': 0.7,
                    'drama': 0.5,
                    'romance': 0.3
                },
                'personality': {
                    'openness': 0.6,
                    'exploration_tendency': 0.5,
                    'diversity_preference': 0.6
                },
                'viewing_history': 89,
                'favorite_genres': ['Action', 'Comedy', 'Horror']
            },
            'user_004': {
                'name': 'Sarah Johnson',
                'persona': 'Book Lover',
                'age': 45,
                'demographics': {'location': 'Chicago', 'education': 'PhD'},
                'preferences': {
                    'drama': 0.8,
                    'documentary': 0.9,
                    'biography': 0.8,
                    'action': 0.2,
                    'comedy': 0.6
                },
                'personality': {
                    'openness': 0.9,
                    'exploration_tendency': 0.9,
                    'diversity_preference': 0.8
                },
                'viewing_history': 156,
                'favorite_genres': ['Documentary', 'Biography', 'Drama']
            },
            'user_005': {
                'name': 'Mike Thompson',
                'persona': 'Casual Viewer',
                'age': 38,
                'demographics': {'location': 'Austin', 'education': 'High School'},
                'preferences': {
                    'comedy': 0.8,
                    'action': 0.7,
                    'drama': 0.6,
                    'sci-fi': 0.4,
                    'horror': 0.5
                },
                'personality': {
                    'openness': 0.5,
                    'exploration_tendency': 0.4,
                    'diversity_preference': 0.5
                },
                'viewing_history': 67,
                'favorite_genres': ['Comedy', 'Action', 'Sports']
            }
        }
    
    def _generate_content_catalogs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate content catalogs for different types"""
        catalogs = {}
        
        # Movies catalog
        movie_titles = [
            'Quantum Odyssey', 'The Last Equation', 'Nebula Rising', 'Digital Dreams',
            'Midnight in Tokyo', 'The Art of Deception', 'Parallel Lives', 'Cosmic Journey',
            'The Time Keeper', 'Shadows of Tomorrow', 'Electric Hearts', 'The Final Code',
            'Starlight Express', 'The Memory Palace', 'Infinite Loop', 'The Quantum Garden',
            'City of Echoes', 'The Last Algorithm', 'Beyond the Horizon', 'The Digital Divide'
        ]
        
        catalogs['movies'] = []
        for i, title in enumerate(movie_titles):
            movie = {
                'id': f'movie_{i+1:03d}',
                'title': title,
                'genre': random.choice(self.genres['movies']),
                'year': random.randint(2015, 2024),
                'rating': round(random.uniform(6.5, 9.0), 1),
                'duration': random.randint(90, 180),
                'director': f"Director {random.randint(1, 50)}",
                'mood': random.choice(['uplifting', 'intense', 'thoughtful', 'exciting', 'mysterious']),
                'popularity': random.uniform(0.1, 1.0),
                'score': random.uniform(0.6, 0.95),
                'components': {
                    'collaborative': random.uniform(0.2, 0.4),
                    'content_based': random.uniform(0.3, 0.5),
                    'knowledge_based': random.uniform(0.1, 0.3),
                    'popularity': random.uniform(0.05, 0.15)
                }
            }
            catalogs['movies'].append(movie)
        
        # Books catalog
        book_titles = [
            'The Quantum Chronicles', 'Digital Minds', 'The Last Library', 'Consciousness Code',
            'The Algorithm Wars', 'Neural Networks', 'The Data Detective', 'Artificial Dreams',
            'The Silicon Prophecy', 'Machine Learning', 'The Future Archive', 'Quantum Computing',
            'The Digital Renaissance', 'AI and Society', 'The Technology Paradox', 'Virtual Realities',
            'The Information Age', 'Cybernetic Futures', 'The Robot Revolution', 'Digital Evolution'
        ]
        
        catalogs['books'] = []
        for i, title in enumerate(book_titles):
            book = {
                'id': f'book_{i+1:03d}',
                'title': title,
                'genre': random.choice(self.genres['books']),
                'year': random.randint(2010, 2024),
                'author': f"Author {random.randint(1, 30)}",
                'pages': random.randint(200, 600),
                'rating': round(random.uniform(3.5, 5.0), 1),
                'mood': random.choice(['enlightening', 'gripping', 'contemplative', 'inspiring', 'complex']),
                'popularity': random.uniform(0.1, 1.0),
                'score': random.uniform(0.6, 0.95),
                'components': {
                    'collaborative': random.uniform(0.25, 0.45),
                    'content_based': random.uniform(0.25, 0.45),
                    'knowledge_based': random.uniform(0.15, 0.35),
                    'popularity': random.uniform(0.05, 0.15)
                }
            }
            catalogs['books'].append(book)
        
        # Music catalog
        music_titles = [
            'Electric Sunset', 'Digital Dreams', 'Quantum Beats', 'Synthetic Love',
            'Neon Nights', 'Cyber Symphony', 'Algorithmic Soul', 'Binary Emotions',
            'Virtual Harmony', 'Electronic Meditation', 'Data Stream', 'Code Melody',
            'Silicon Valley Blues', 'AI Lullaby', 'Robotic Rhapsody', 'Digital Serenade',
            'Quantum Frequencies', 'Neural Networks', 'Machine Learning', 'Future Sounds'
        ]
        
        catalogs['music'] = []
        for i, title in enumerate(music_titles):
            track = {
                'id': f'music_{i+1:03d}',
                'title': title,
                'genre': random.choice(self.genres['music']),
                'year': random.randint(2018, 2024),
                'artist': f"Artist {random.randint(1, 25)}",
                'duration': f"{random.randint(2, 6)}:{random.randint(10, 59):02d}",
                'album': f"Album {random.randint(1, 15)}",
                'mood': random.choice(['energetic', 'relaxing', 'upbeat', 'melancholic', 'atmospheric']),
                'popularity': random.uniform(0.1, 1.0),
                'score': random.uniform(0.6, 0.95),
                'components': {
                    'collaborative': random.uniform(0.3, 0.5),
                    'content_based': random.uniform(0.2, 0.4),
                    'knowledge_based': random.uniform(0.1, 0.3),
                    'popularity': random.uniform(0.1, 0.2)
                }
            }
            catalogs['music'].append(track)
        
        return catalogs
    
    def get_sample_users(self) -> Dict[str, Dict[str, Any]]:
        """Get sample user profiles"""
        return self.sample_users
    
    def generate_sample_recommendations(self, content_type: str, num_items: int = 5) -> List[Dict[str, Any]]:
        """Generate sample recommendations for a content type"""
        if content_type not in self.content_catalogs:
            return []
        
        # Select random items and add some realistic scoring
        available_items = self.content_catalogs[content_type].copy()
        selected_items = random.sample(available_items, min(num_items, len(available_items)))
        
        # Sort by score (descending) to simulate ranking
        selected_items.sort(key=lambda x: x['score'], reverse=True)
        
        return selected_items
    
    def generate_user_interaction_history(self, user_id: str, num_interactions: int = 50) -> List[Dict[str, Any]]:
        """Generate user interaction history"""
        interactions = []
        base_time = datetime.now() - timedelta(days=90)
        
        for i in range(num_interactions):
            content_type = random.choice(['movies', 'books', 'music'])
            item = random.choice(self.content_catalogs[content_type])
            
            interaction = {
                'user_id': user_id,
                'item_id': item['id'],
                'content_type': content_type,
                'interaction_type': random.choice(['view', 'like', 'dislike', 'share', 'bookmark']),
                'rating': random.choice([None, random.randint(1, 5)]),
                'timestamp': base_time + timedelta(days=random.randint(0, 90)),
                'duration_minutes': random.randint(5, 120) if content_type == 'movies' else None
            }
            interactions.append(interaction)
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x['timestamp'])
        return interactions
    
    def generate_realtime_metrics(self) -> Dict[str, Any]:
        """Generate realistic real-time metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'active_users': random.randint(500, 2000),
            'requests_per_second': random.randint(50, 200),
            'avg_response_time_ms': random.randint(45, 120),
            'error_rate': random.uniform(0.001, 0.01),
            'cache_hit_rate': random.uniform(0.85, 0.95),
            'recommendation_accuracy': {
                'ndcg_at_10': random.uniform(0.35, 0.45),
                'precision_at_10': random.uniform(0.25, 0.35),
                'diversity_score': random.uniform(0.7, 0.85)
            },
            'system_health': {
                'cpu_usage': random.uniform(30, 70),
                'memory_usage': random.uniform(40, 80),
                'disk_usage': random.uniform(20, 60)
            }
        }
    
    def generate_ab_test_data(self, experiment_name: str) -> Dict[str, Any]:
        """Generate A/B test data"""
        control_users = random.randint(4500, 5500)
        treatment_users = random.randint(4500, 5500)
        
        # Control metrics
        control_ndcg = random.uniform(0.34, 0.36)
        control_ctr = random.uniform(0.08, 0.09)
        
        # Treatment metrics (slightly better)
        treatment_ndcg = control_ndcg + random.uniform(0.01, 0.04)
        treatment_ctr = control_ctr + random.uniform(0.005, 0.015)
        
        # Calculate statistical significance
        p_value = random.uniform(0.01, 0.10)
        significance = p_value < 0.05
        
        return {
            'experiment_name': experiment_name,
            'status': 'running',
            'start_date': (datetime.now() - timedelta(days=random.randint(1, 14))).isoformat(),
            'control': {
                'users': control_users,
                'ndcg_at_10': control_ndcg,
                'ctr': control_ctr,
                'conversion_rate': random.uniform(0.02, 0.04)
            },
            'treatment': {
                'users': treatment_users,
                'ndcg_at_10': treatment_ndcg,
                'ctr': treatment_ctr,
                'conversion_rate': random.uniform(0.02, 0.04)
            },
            'statistical_test': {
                'p_value': p_value,
                'significant': significance,
                'confidence_level': 0.95,
                'lift_percentage': ((treatment_ndcg / control_ndcg) - 1) * 100
            }
        }
    
    def generate_feature_importance(self) -> Dict[str, float]:
        """Generate feature importance scores"""
        features = [
            'user_genre_preference', 'item_popularity', 'user_rating_history',
            'item_release_year', 'user_age', 'item_duration', 'time_of_day',
            'day_of_week', 'user_activity_level', 'item_review_count',
            'user_diversity_preference', 'item_genre_diversity', 'collaborative_score',
            'content_similarity', 'temporal_patterns'
        ]
        
        # Generate importance scores that sum to 1
        scores = [random.uniform(0.1, 1.0) for _ in features]
        total = sum(scores)
        normalized_scores = [score / total for score in scores]
        
        return dict(zip(features, normalized_scores))
    
    def generate_drift_detection_data(self) -> Dict[str, Any]:
        """Generate concept drift detection data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': random.choice([True, False]),
            'drift_type': random.choice(['gradual', 'sudden', 'incremental', 'recurring']),
            'affected_features': random.sample([
                'user_preferences', 'item_popularity', 'seasonal_trends',
                'demographic_shifts', 'content_catalog'
            ], random.randint(1, 3)),
            'drift_magnitude': random.uniform(0.1, 0.8),
            'confidence_score': random.uniform(0.7, 0.95),
            'recommended_action': random.choice([
                'retrain_model', 'update_features', 'adjust_weights', 'monitor_closely'
            ])
        }
    
    def generate_explanation_components(self, item_id: str) -> Dict[str, Any]:
        """Generate explanation components for an item"""
        return {
            'item_id': item_id,
            'explanation_type': 'hybrid',
            'components': {
                'collaborative_filtering': {
                    'weight': random.uniform(0.3, 0.5),
                    'explanation': 'Users with similar taste also enjoyed this item',
                    'similar_users_count': random.randint(50, 200),
                    'confidence': random.uniform(0.7, 0.9)
                },
                'content_based': {
                    'weight': random.uniform(0.2, 0.4),
                    'explanation': 'Matches your preference for similar content features',
                    'matching_features': random.sample(['genre', 'mood', 'style', 'theme'], 2),
                    'confidence': random.uniform(0.6, 0.8)
                },
                'knowledge_graph': {
                    'weight': random.uniform(0.1, 0.3),
                    'explanation': 'Connected through semantic relationships',
                    'connection_path': ['user_interest', 'genre_cluster', 'item_features'],
                    'confidence': random.uniform(0.5, 0.7)
                }
            },
            'overall_confidence': random.uniform(0.7, 0.9),
            'explanation_quality': random.uniform(0.8, 0.95)
        }
