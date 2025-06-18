"""
Unit Tests for the Content Recommendation Engine

This module contains comprehensive tests for all recommendation algorithms,
personalization features, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import os

# Import modules to test
from src.recommenders.collaborative import CollaborativeRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.knowledge_based import KnowledgeBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.group_recommender import GroupRecommender
from src.personalization.short_term import ShortTermPersonalizer
from src.personalization.long_term import LongTermPersonalizer
from src.personalization.sequential import SequentialPatternMiner
from src.personalization.drift_detection import PreferenceDriftDetector
from src.utils.scorer import RecommendationScorer
from src.utils.explainer import RecommendationExplainer
from src.utils.config import ConfigManager, SystemConfig


class TestCollaborativeRecommender:
    """Test cases for collaborative filtering."""
    
    @pytest.fixture
    def recommender(self):
        """Create a test collaborative recommender."""
        return CollaborativeRecommender()
    
    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction data."""
        return [
            {'user_id': 'user1', 'item_id': 'item1', 'rating': 4.0, 'timestamp': '2024-01-01'},
            {'user_id': 'user1', 'item_id': 'item2', 'rating': 5.0, 'timestamp': '2024-01-02'},
            {'user_id': 'user2', 'item_id': 'item1', 'rating': 3.0, 'timestamp': '2024-01-01'},
            {'user_id': 'user2', 'item_id': 'item3', 'rating': 4.0, 'timestamp': '2024-01-03'},
            {'user_id': 'user3', 'item_id': 'item2', 'rating': 5.0, 'timestamp': '2024-01-02'},
            {'user_id': 'user3', 'item_id': 'item3', 'rating': 2.0, 'timestamp': '2024-01-03'},
        ]
    
    def test_fit_model(self, recommender, sample_interactions):
        """Test model fitting."""
        recommender.fit(sample_interactions)
        
        assert recommender.user_item_matrix is not None
        assert len(recommender.user_mapping) == 3
        assert len(recommender.item_mapping) == 3
    
    def test_user_similarity(self, recommender, sample_interactions):
        """Test user similarity computation."""
        recommender.fit(sample_interactions)
        similarity = recommender._compute_user_similarity('user1', 'user2')
        
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1
    
    def test_recommend_for_user(self, recommender, sample_interactions):
        """Test user recommendations."""
        recommender.fit(sample_interactions)
        recommendations = recommender.recommend_for_user('user1', k=2)
        
        assert len(recommendations) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in recommendations)
        assert all(0 <= rec['score'] <= 5 for rec in recommendations)
    
    def test_cold_start_user(self, recommender, sample_interactions):
        """Test cold start user handling."""
        recommender.fit(sample_interactions)
        recommendations = recommender.recommend_for_user('new_user', k=2)
        
        # Should return popular items for cold start
        assert len(recommendations) <= 2
    
    def test_matrix_factorization(self, recommender, sample_interactions):
        """Test matrix factorization method."""
        recommender.fit(sample_interactions)
        recommendations = recommender._matrix_factorization_recommend('user1', k=2)
        
        assert len(recommendations) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in recommendations)


class TestContentBasedRecommender:
    """Test cases for content-based filtering."""
    
    @pytest.fixture
    def recommender(self):
        """Create a test content-based recommender."""
        return ContentBasedRecommender()
    
    @pytest.fixture
    def sample_items(self):
        """Create sample item data."""
        return {
            'item1': {
                'title': 'Action Movie 1',
                'genres': ['Action', 'Adventure'],
                'year': 2020,
                'director': 'Director A',
                'description': 'An exciting action-packed adventure movie'
            },
            'item2': {
                'title': 'Comedy Movie 1',
                'genres': ['Comedy', 'Romance'],
                'year': 2021,
                'director': 'Director B',
                'description': 'A funny romantic comedy about love'
            },
            'item3': {
                'title': 'Action Movie 2',
                'genres': ['Action', 'Sci-Fi'],
                'year': 2022,
                'director': 'Director A',
                'description': 'A futuristic action movie with sci-fi elements'
            }
        }
    
    @pytest.fixture
    def sample_user_profile(self):
        """Create sample user profile."""
        return {
            'user_id': 'user1',
            'preferred_genres': ['Action', 'Adventure'],
            'preferred_directors': ['Director A'],
            'average_rating': 4.2,
            'rated_items': ['item1']
        }
    
    def test_fit_model(self, recommender, sample_items):
        """Test model fitting."""
        recommender.fit(list(sample_items.values()))
        
        assert len(recommender.item_features) > 0
        assert 'item1' in recommender.item_features or 0 in recommender.item_features
    
    def test_genre_similarity(self, recommender, sample_items):
        """Test genre similarity computation."""
        item1 = sample_items['item1']
        item2 = sample_items['item3']  # Both action movies
        
        similarity = recommender._compute_genre_similarity(item1, item2)
        assert similarity > 0.5  # Should be similar (both Action)
    
    def test_recommend_for_user(self, recommender, sample_items, sample_user_profile):
        """Test user recommendations."""
        recommender.fit(list(sample_items.values()))
        recommendations = recommender.recommend_for_user(sample_user_profile, k=2)
        
        assert len(recommendations) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in recommendations)
    
    @patch('src.content_understanding.gemini_client.GeminiClient')
    def test_embedding_similarity(self, mock_gemini, recommender, sample_items):
        """Test embedding-based similarity."""
        # Mock Gemini client
        mock_client = Mock()
        mock_client.generate_embeddings.return_value = {
            'embeddings': [np.random.rand(384) for _ in range(len(sample_items))]
        }
        mock_gemini.return_value = mock_client
        
        recommender.fit(list(sample_items.values()))
        
        # Test that embeddings were generated
        assert hasattr(recommender, 'embeddings') or hasattr(recommender, 'item_embeddings')


class TestKnowledgeBasedRecommender:
    """Test cases for knowledge-based recommendations."""
    
    @pytest.fixture
    def recommender(self):
        """Create a test knowledge-based recommender."""
        return KnowledgeBasedRecommender()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for knowledge-based recommendations."""
        return {
            'interactions': [
                {'user_id': 'user1', 'item_id': 'item1', 'rating': 4.0, 'timestamp': '2024-01-01'},
                {'user_id': 'user2', 'item_id': 'item1', 'rating': 5.0, 'timestamp': '2024-01-02'},
                {'user_id': 'user3', 'item_id': 'item2', 'rating': 3.0, 'timestamp': '2024-01-03'},
            ],
            'items': {
                'item1': {
                    'title': 'Popular Movie',
                    'year': 2023,
                    'average_rating': 4.5,
                    'genres': ['Action'],
                    'awards': ['Oscar Winner']
                },
                'item2': {
                    'title': 'Indie Movie',
                    'year': 2024,
                    'average_rating': 3.8,
                    'genres': ['Drama'],
                    'awards': []
                }
            }
        }
    
    def test_trending_recommendations(self, recommender, sample_data):
        """Test trending recommendations."""
        recommender.fit(sample_data['interactions'], sample_data['items'])
        trending = recommender.get_trending_recommendations(k=2)
        
        assert len(trending) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in trending)
    
    def test_new_releases(self, recommender, sample_data):
        """Test new release recommendations."""
        recommender.fit(sample_data['interactions'], sample_data['items'])
        new_releases = recommender.get_new_releases(k=2)
        
        assert len(new_releases) <= 2
        # Should prefer newer items
        if len(new_releases) > 1:
            years = [sample_data['items'][rec['item_id']]['year'] for rec in new_releases]
            assert years == sorted(years, reverse=True)
    
    def test_acclaimed_content(self, recommender, sample_data):
        """Test acclaimed content recommendations."""
        recommender.fit(sample_data['interactions'], sample_data['items'])
        acclaimed = recommender.get_acclaimed_content(k=2)
        
        assert len(acclaimed) <= 2
        # Should prefer items with awards or high ratings
        if acclaimed:
            for rec in acclaimed:
                item = sample_data['items'][rec['item_id']]
                assert item['average_rating'] >= 4.0 or len(item.get('awards', [])) > 0
    
    def test_contextual_recommendations(self, recommender, sample_data):
        """Test contextual recommendations."""
        recommender.fit(sample_data['interactions'], sample_data['items'])
        
        context = {'time_of_day': 'evening', 'mood': 'relaxed'}
        contextual = recommender.get_contextual_recommendations(context, k=2)
        
        assert len(contextual) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in contextual)


class TestHybridRecommender:
    """Test cases for hybrid recommendations."""
    
    @pytest.fixture
    def hybrid_recommender(self):
        """Create a test hybrid recommender."""
        return HybridRecommender()
    
    @pytest.fixture
    def sample_complete_data(self):
        """Create complete sample data for testing."""
        return {
            'interactions': [
                {'user_id': 'user1', 'item_id': 'item1', 'rating': 4.0, 'timestamp': '2024-01-01'},
                {'user_id': 'user1', 'item_id': 'item2', 'rating': 3.0, 'timestamp': '2024-01-02'},
                {'user_id': 'user2', 'item_id': 'item1', 'rating': 5.0, 'timestamp': '2024-01-01'},
                {'user_id': 'user2', 'item_id': 'item3', 'rating': 4.0, 'timestamp': '2024-01-03'},
            ],
            'items': {
                'item1': {'title': 'Movie 1', 'genres': ['Action'], 'year': 2020},
                'item2': {'title': 'Movie 2', 'genres': ['Comedy'], 'year': 2021},
                'item3': {'title': 'Movie 3', 'genres': ['Action'], 'year': 2022},
            },
            'user_profiles': {
                'user1': {'preferred_genres': ['Action'], 'average_rating': 3.5},
                'user2': {'preferred_genres': ['Action', 'Comedy'], 'average_rating': 4.5}
            }
        }
    
    def test_fit_all_models(self, hybrid_recommender, sample_complete_data):
        """Test fitting all underlying models."""
        hybrid_recommender.fit(
            sample_complete_data['interactions'],
            sample_complete_data['items'],
            sample_complete_data['user_profiles']
        )
        
        assert hybrid_recommender.collaborative_recommender is not None
        assert hybrid_recommender.content_recommender is not None
        assert hybrid_recommender.knowledge_recommender is not None
    
    def test_hybrid_recommendations(self, hybrid_recommender, sample_complete_data):
        """Test hybrid recommendation generation."""
        hybrid_recommender.fit(
            sample_complete_data['interactions'],
            sample_complete_data['items'],
            sample_complete_data['user_profiles']
        )
        
        recommendations = hybrid_recommender.recommend_for_user('user1', k=2)
        
        assert len(recommendations) <= 2
        assert all('item_id' in rec and 'score' in rec for rec in recommendations)
        assert all('method' in rec for rec in recommendations)
        assert all('explanation' in rec for rec in recommendations)
    
    def test_weight_optimization(self, hybrid_recommender, sample_complete_data):
        """Test automatic weight optimization."""
        hybrid_recommender.fit(
            sample_complete_data['interactions'],
            sample_complete_data['items'],
            sample_complete_data['user_profiles']
        )
        
        # Test weight optimization (simplified)
        original_weights = hybrid_recommender.weights.copy()
        hybrid_recommender.optimize_weights(sample_complete_data['interactions'])
        
        # Weights should be different after optimization
        assert hybrid_recommender.weights != original_weights
        # Weights should sum to 1
        assert abs(sum(hybrid_recommender.weights.values()) - 1.0) < 0.01


class TestPersonalizationModules:
    """Test cases for personalization modules."""
    
    @pytest.fixture
    def sample_user_data(self):
        """Create sample user data for personalization."""
        return {
            'user_id': 'user1',
            'interactions': [
                {'item_id': 'item1', 'rating': 4.0, 'timestamp': datetime.now() - timedelta(days=1)},
                {'item_id': 'item2', 'rating': 3.0, 'timestamp': datetime.now() - timedelta(hours=2)},
                {'item_id': 'item3', 'rating': 5.0, 'timestamp': datetime.now() - timedelta(minutes=30)},
            ],
            'context': {'time_of_day': 'evening', 'device': 'mobile'}
        }
    
    def test_short_term_personalizer(self, sample_user_data):
        """Test short-term personalization."""
        personalizer = ShortTermPersonalizer()
        personalizer.update_user_profile(sample_user_data['user_id'], sample_user_data['interactions'])
        
        preferences = personalizer.get_current_preferences(sample_user_data['user_id'])
        
        assert 'recent_items' in preferences
        assert 'session_preferences' in preferences
        assert len(preferences['recent_items']) <= 10
    
    def test_long_term_personalizer(self, sample_user_data):
        """Test long-term personalization."""
        personalizer = LongTermPersonalizer()
        
        # Add historical data
        historical_interactions = [
            {'item_id': f'item{i}', 'rating': 4.0, 'timestamp': datetime.now() - timedelta(days=30+i)}
            for i in range(20)
        ]
        
        personalizer.update_user_profile(sample_user_data['user_id'], historical_interactions)
        profile = personalizer.get_user_profile(sample_user_data['user_id'])
        
        assert 'stable_preferences' in profile
        assert 'preference_evolution' in profile
    
    def test_sequential_pattern_miner(self, sample_user_data):
        """Test sequential pattern mining."""
        miner = SequentialPatternMiner()
        
        # Create sequence data
        sequences = [
            ['item1', 'item2', 'item3'],
            ['item1', 'item3', 'item4'],
            ['item2', 'item3', 'item1']
        ]
        
        miner.fit(sequences)
        patterns = miner.find_patterns(min_support=0.5)
        
        assert isinstance(patterns, list)
        if patterns:
            assert all('pattern' in p and 'support' in p for p in patterns)
    
    def test_drift_detection(self, sample_user_data):
        """Test preference drift detection."""
        detector = PreferenceDriftDetector()
        
        # Add interactions over time
        detector.update_user_profile(sample_user_data['user_id'], sample_user_data['interactions'])
        
        drift_result = detector.detect_preference_drift(sample_user_data['user_id'])
        
        assert 'drift_detected' in drift_result
        assert 'confidence' in drift_result
        assert isinstance(drift_result['drift_detected'], bool)
        assert 0 <= drift_result['confidence'] <= 1


class TestUtilityModules:
    """Test cases for utility modules."""
    
    def test_recommendation_scorer(self):
        """Test recommendation scoring."""
        scorer = RecommendationScorer()
        
        recommendations = [
            {'item_id': 'item1', 'score': 0.8},
            {'item_id': 'item2', 'score': 0.6},
            {'item_id': 'item3', 'score': 0.4}
        ]
        
        ground_truth = [
            {'item_id': 'item1', 'rating': 5.0},
            {'item_id': 'item4', 'rating': 4.0}
        ]
        
        metrics = scorer.compute_accuracy_metrics(recommendations, ground_truth)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_recommendation_explainer(self):
        """Test recommendation explanation."""
        explainer = RecommendationExplainer(use_gemini=False)  # Disable Gemini for testing
        
        recommendation = {
            'item_id': 'item1',
            'score': 0.8,
            'method': 'collaborative'
        }
        
        user_profile = {
            'preferred_genres': ['Action'],
            'average_rating': 4.0
        }
        
        item_features = {
            'title': 'Test Movie',
            'genres': ['Action', 'Adventure'],
            'year': 2020
        }
        
        context = {
            'method': 'collaborative',
            'strategy': 'user_similarity'
        }
        
        explanation = explainer.generate_explanation(
            recommendation, user_profile, item_features, context
        )
        
        assert 'template' in explanation
        assert 'features' in explanation
        assert 'confidence' in explanation
    
    def test_config_manager(self):
        """Test configuration management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            
            manager = ConfigManager(config_path)
            config = manager.get_config()
            
            assert isinstance(config, SystemConfig)
            assert hasattr(config, 'collaborative')
            assert hasattr(config, 'content_based')
            assert hasattr(config, 'hybrid')
            
            # Test configuration validation
            validation = manager.validate_config()
            assert 'valid' in validation
            assert 'errors' in validation
            assert 'warnings' in validation


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def complete_system_data(self):
        """Create complete system data for integration testing."""
        return {
            'interactions': [
                {'user_id': f'user{i}', 'item_id': f'item{j}', 'rating': np.random.uniform(1, 5), 
                 'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 100))).isoformat()}
                for i in range(1, 11) for j in range(1, 21) if np.random.random() > 0.7
            ],
            'items': {
                f'item{i}': {
                    'title': f'Item {i}',
                    'genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], 
                                             size=np.random.randint(1, 3), replace=False).tolist(),
                    'year': np.random.randint(2000, 2024),
                    'director': f'Director {np.random.randint(1, 5)}',
                    'average_rating': np.random.uniform(2, 5)
                }
                for i in range(1, 21)
            },
            'user_profiles': {
                f'user{i}': {
                    'preferred_genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], 
                                                       size=np.random.randint(1, 3), replace=False).tolist(),
                    'average_rating': np.random.uniform(2, 5),
                    'age': np.random.randint(18, 65)
                }
                for i in range(1, 11)
            }
        }
    
    def test_full_recommendation_pipeline(self, complete_system_data):
        """Test the complete recommendation pipeline."""
        # Initialize hybrid recommender
        hybrid = HybridRecommender()
        
        # Fit the model
        hybrid.fit(
            complete_system_data['interactions'],
            complete_system_data['items'],
            complete_system_data['user_profiles']
        )
        
        # Generate recommendations for all users
        all_recommendations = {}
        for user_id in complete_system_data['user_profiles'].keys():
            recommendations = hybrid.recommend_for_user(user_id, k=5)
            all_recommendations[user_id] = recommendations
            
            # Verify recommendation format
            assert len(recommendations) <= 5
            assert all('item_id' in rec and 'score' in rec for rec in recommendations)
        
        # Test recommendation quality
        scorer = RecommendationScorer()
        
        for user_id, recommendations in all_recommendations.items():
            # Get user's actual interactions as ground truth
            user_interactions = [
                i for i in complete_system_data['interactions'] 
                if i['user_id'] == user_id and i['rating'] >= 4.0
            ]
            
            if user_interactions and recommendations:
                metrics = scorer.compute_accuracy_metrics(recommendations, user_interactions)
                assert 'precision' in metrics
                assert 'recall' in metrics
    
    def test_group_recommendation_pipeline(self, complete_system_data):
        """Test group recommendation pipeline."""
        group_recommender = GroupRecommender()
        
        # Create a test group
        group_members = [
            complete_system_data['user_profiles']['user1'],
            complete_system_data['user_profiles']['user2'],
            complete_system_data['user_profiles']['user3']
        ]
        
        # Add user_id to each member
        for i, member in enumerate(group_members, 1):
            member['user_id'] = f'user{i}'
        
        # Generate group recommendations
        group_recommendations = group_recommender.recommend_for_group(
            group_members, 
            complete_system_data['items'], 
            k=5
        )
        
        assert len(group_recommendations) <= 5
        assert all('item_id' in rec and 'score' in rec for rec in group_recommendations)
        assert all('member_scores' in rec for rec in group_recommendations)


# Test fixtures and utilities
@pytest.fixture
def sample_database():
    """Create a temporary database for testing."""
    import sqlite3
    import tempfile
    
    fd, path = tempfile.mkstemp(suffix='.db')
    try:
        conn = sqlite3.connect(path)
        
        # Create test tables
        conn.execute('''
            CREATE TABLE interactions (
                user_id TEXT,
                item_id TEXT,
                rating REAL,
                timestamp TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                title TEXT,
                genres TEXT,
                year INTEGER
            )
        ''')
        
        # Insert test data
        test_interactions = [
            ('user1', 'item1', 4.0, '2024-01-01'),
            ('user1', 'item2', 3.0, '2024-01-02'),
            ('user2', 'item1', 5.0, '2024-01-01')
        ]
        
        conn.executemany(
            'INSERT INTO interactions VALUES (?, ?, ?, ?)',
            test_interactions
        )
        
        test_items = [
            ('item1', 'Test Movie 1', 'Action|Adventure', 2020),
            ('item2', 'Test Movie 2', 'Comedy', 2021),
            ('item3', 'Test Movie 3', 'Drama', 2022)
        ]
        
        conn.executemany(
            'INSERT INTO items VALUES (?, ?, ?, ?)',
            test_items
        )
        
        conn.commit()
        yield path
        
    finally:
        os.close(fd)
        os.unlink(path)


# Performance tests
class TestPerformance:
    """Performance tests for the recommendation system."""
    
    def test_recommendation_speed(self, complete_system_data):
        """Test recommendation generation speed."""
        import time
        
        hybrid = HybridRecommender()
        hybrid.fit(
            complete_system_data['interactions'],
            complete_system_data['items'],
            complete_system_data['user_profiles']
        )
        
        # Time recommendation generation
        start_time = time.time()
        recommendations = hybrid.recommend_for_user('user1', k=10)
        end_time = time.time()
        
        recommendation_time = end_time - start_time
        
        # Should generate recommendations in reasonable time (< 1 second)
        assert recommendation_time < 1.0
        assert len(recommendations) <= 10
    
    def test_batch_recommendation_speed(self, complete_system_data):
        """Test batch recommendation speed."""
        import time
        
        hybrid = HybridRecommender()
        hybrid.fit(
            complete_system_data['interactions'],
            complete_system_data['items'],
            complete_system_data['user_profiles']
        )
        
        user_ids = list(complete_system_data['user_profiles'].keys())
        
        # Time batch recommendations
        start_time = time.time()
        batch_recommendations = {}
        for user_id in user_ids:
            batch_recommendations[user_id] = hybrid.recommend_for_user(user_id, k=5)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_user = total_time / len(user_ids)
        
        # Should handle batch recommendations efficiently
        assert avg_time_per_user < 0.5  # Less than 0.5 seconds per user
        assert len(batch_recommendations) == len(user_ids)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
