#!/usr/bin/env python3
"""
Example Usage Script for Content Recommendation Engine

This script demonstrates how to use the complete recommendation system
with sample data and various features.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import recommendation modules
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.group_recommender import GroupRecommender
from src.personalization.short_term import ShortTermPersonalizer
from src.personalization.long_term import LongTermPersonalizer
from src.personalization.sequential import SequentialPatternMiner
from src.personalization.drift_detection import PreferenceDriftDetector
from src.utils.scorer import RecommendationScorer
from src.utils.explainer import RecommendationExplainer
from src.utils.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """Generate sample data for demonstration."""
    print("🎬 Generating sample movie data...")
    
    # Sample movies
    movies = {
        'movie_1': {
            'title': 'The Action Hero',
            'genres': ['Action', 'Adventure'],
            'year': 2023,
            'director': 'Michael Bay',
            'actors': ['Tom Cruise', 'Emily Blunt'],
            'description': 'An action-packed adventure with spectacular stunts',
            'average_rating': 4.2,
            'awards': ['Best Action Sequence']
        },
        'movie_2': {
            'title': 'Laugh Out Loud',
            'genres': ['Comedy', 'Romance'],
            'year': 2022,
            'director': 'Nancy Meyers',
            'actors': ['Ryan Gosling', 'Emma Stone'],
            'description': 'A hilarious romantic comedy about unexpected love',
            'average_rating': 4.5,
            'awards': []
        },
        'movie_3': {
            'title': 'Space Odyssey 2024',
            'genres': ['Sci-Fi', 'Drama'],
            'year': 2024,
            'director': 'Denis Villeneuve',
            'actors': ['Timothée Chalamet', 'Zendaya'],
            'description': 'An epic space drama exploring humanity\'s future',
            'average_rating': 4.8,
            'awards': ['Best Visual Effects', 'Best Cinematography']
        },
        'movie_4': {
            'title': 'Mystery Manor',
            'genres': ['Mystery', 'Thriller'],
            'year': 2023,
            'director': 'Rian Johnson',
            'actors': ['Daniel Craig', 'Ana de Armas'],
            'description': 'A thrilling mystery set in a grand estate',
            'average_rating': 4.1,
            'awards': []
        },
        'movie_5': {
            'title': 'Animated Adventures',
            'genres': ['Animation', 'Family'],
            'year': 2023,
            'director': 'Pete Docter',
            'actors': ['Tom Hanks', 'Scarlett Johansson'],
            'description': 'A heartwarming animated story for the whole family',
            'average_rating': 4.6,
            'awards': ['Best Animated Feature']
        }
    }
    
    # Sample user interactions
    interactions = [
        # User 1: Action lover
        {'user_id': 'alice', 'item_id': 'movie_1', 'rating': 5.0, 'timestamp': '2024-01-15T19:30:00'},
        {'user_id': 'alice', 'item_id': 'movie_3', 'rating': 4.0, 'timestamp': '2024-01-20T20:15:00'},
        {'user_id': 'alice', 'item_id': 'movie_4', 'rating': 3.5, 'timestamp': '2024-01-25T21:00:00'},
        
        # User 2: Comedy fan
        {'user_id': 'bob', 'item_id': 'movie_2', 'rating': 5.0, 'timestamp': '2024-01-16T18:45:00'},
        {'user_id': 'bob', 'item_id': 'movie_5', 'rating': 4.5, 'timestamp': '2024-01-22T17:30:00'},
        {'user_id': 'bob', 'item_id': 'movie_1', 'rating': 3.0, 'timestamp': '2024-01-28T19:45:00'},
        
        # User 3: Sci-fi enthusiast
        {'user_id': 'charlie', 'item_id': 'movie_3', 'rating': 5.0, 'timestamp': '2024-01-18T20:30:00'},
        {'user_id': 'charlie', 'item_id': 'movie_1', 'rating': 4.0, 'timestamp': '2024-01-24T19:15:00'},
        {'user_id': 'charlie', 'item_id': 'movie_4', 'rating': 4.5, 'timestamp': '2024-01-29T21:30:00'},
        
        # User 4: Family content lover
        {'user_id': 'diana', 'item_id': 'movie_5', 'rating': 5.0, 'timestamp': '2024-01-17T16:00:00'},
        {'user_id': 'diana', 'item_id': 'movie_2', 'rating': 4.0, 'timestamp': '2024-01-23T15:30:00'},
        {'user_id': 'diana', 'item_id': 'movie_3', 'rating': 3.5, 'timestamp': '2024-01-30T16:45:00'},
    ]
    
    # Sample user profiles
    user_profiles = {
        'alice': {
            'user_id': 'alice',
            'age': 28,
            'preferred_genres': ['Action', 'Adventure', 'Sci-Fi'],
            'preferred_directors': ['Michael Bay', 'Denis Villeneuve'],
            'average_rating': 4.2,
            'recent_items': ['movie_1', 'movie_3', 'movie_4']
        },
        'bob': {
            'user_id': 'bob',
            'age': 35,
            'preferred_genres': ['Comedy', 'Romance', 'Animation'],
            'preferred_directors': ['Nancy Meyers', 'Pete Docter'],
            'average_rating': 4.1,
            'recent_items': ['movie_2', 'movie_5', 'movie_1']
        },
        'charlie': {
            'user_id': 'charlie',
            'age': 31,
            'preferred_genres': ['Sci-Fi', 'Mystery', 'Thriller'],
            'preferred_directors': ['Denis Villeneuve', 'Rian Johnson'],
            'average_rating': 4.5,
            'recent_items': ['movie_3', 'movie_1', 'movie_4']
        },
        'diana': {
            'user_id': 'diana',
            'age': 42,
            'preferred_genres': ['Animation', 'Family', 'Comedy'],
            'preferred_directors': ['Pete Docter', 'Nancy Meyers'],
            'average_rating': 4.2,
            'recent_items': ['movie_5', 'movie_2', 'movie_3']
        }
    }
    
    return movies, interactions, user_profiles


def demonstrate_individual_recommendations():
    """Demonstrate individual user recommendations."""
    print("\\n" + "="*60)
    print("🎯 INDIVIDUAL RECOMMENDATIONS DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    # Initialize hybrid recommender
    print("\\n📊 Initializing Hybrid Recommendation System...")
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(interactions, movies, user_profiles)
    
    # Generate recommendations for each user
    for user_id, profile in user_profiles.items():
        print(f"\\n👤 Recommendations for {user_id.title()} (Age: {profile['age']}):")
        print(f"   Preferred genres: {', '.join(profile['preferred_genres'])}")
        
        recommendations = hybrid_recommender.recommend_for_user(user_id, k=3)
        
        for i, rec in enumerate(recommendations, 1):
            movie = movies[rec['item_id']]
            print(f"\\n   {i}. {movie['title']} ({movie['year']})")
            print(f"      Genres: {', '.join(movie['genres'])}")
            print(f"      Score: {rec['score']:.3f}")
            print(f"      Method: {rec.get('method', 'hybrid')}")
            if 'explanation' in rec:
                print(f"      Why: {rec['explanation']}")


def demonstrate_group_recommendations():
    """Demonstrate group recommendations."""
    print("\\n" + "="*60)
    print("👥 GROUP RECOMMENDATIONS DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    # Create a group
    group_members = [
        user_profiles['alice'],
        user_profiles['bob'],
        user_profiles['charlie']
    ]
    
    print("\\n🎬 Group Members:")
    for member in group_members:
        print(f"   • {member['user_id'].title()}: {', '.join(member['preferred_genres'])}")
    
    # Initialize group recommender
    print("\\n📊 Generating Group Recommendations...")
    group_recommender = GroupRecommender()
    
    # Try different aggregation methods
    methods = ['average', 'least_misery', 'most_pleasure', 'fairness']
    
    for method in methods:
        print(f"\\n🔄 Method: {method.replace('_', ' ').title()}")
        recommendations = group_recommender.recommend_for_group(
            group_members, movies, k=2, aggregation_method=method
        )
        
        for i, rec in enumerate(recommendations, 1):
            movie = movies[rec['item_id']]
            print(f"\\n   {i}. {movie['title']}")
            print(f"      Overall Score: {rec['score']:.3f}")
            
            if 'member_scores' in rec:
                print("      Individual Scores:")
                for member_id, score in rec['member_scores'].items():
                    print(f"         {member_id.title()}: {score:.3f}")


def demonstrate_personalization():
    """Demonstrate personalization features."""
    print("\\n" + "="*60)
    print("🎨 PERSONALIZATION FEATURES DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    # Short-term personalization
    print("\\n⚡ Short-term Personalization:")
    short_term = ShortTermPersonalizer()
    
    # Convert timestamp strings to datetime objects
    user_interactions = []
    for interaction in interactions:
        if interaction['user_id'] == 'alice':
            interaction_copy = interaction.copy()
            interaction_copy['timestamp'] = datetime.fromisoformat(interaction['timestamp'].replace('T', ' '))
            user_interactions.append(interaction_copy)
    
    short_term.update_user_profile('alice', user_interactions)
    current_prefs = short_term.get_current_preferences('alice')
    
    print(f"   Recent preferences for Alice:")
    if 'recent_genres' in current_prefs:
        print(f"      Recent genres: {current_prefs['recent_genres']}")
    if 'session_preferences' in current_prefs:
        print(f"      Session activity level: {current_prefs['session_preferences'].get('activity_level', 'normal')}")
    
    # Long-term personalization
    print("\\n📈 Long-term Personalization:")
    long_term = LongTermPersonalizer()
    long_term.update_user_profile('alice', user_interactions)
    stable_profile = long_term.get_user_profile('alice')
    
    if 'stable_preferences' in stable_profile:
        print(f"   Stable preferences for Alice:")
        stable_prefs = stable_profile['stable_preferences']
        if 'top_genres' in stable_prefs:
            print(f"      Top genres: {stable_prefs['top_genres']}")
        if 'preference_strength' in stable_prefs:
            print(f"      Preference strength: {stable_prefs['preference_strength']:.3f}")
    
    # Sequential pattern mining
    print("\\n🔍 Sequential Pattern Mining:")
    sequential = SequentialPatternMiner()
    
    # Create sequences for all users
    user_sequences = {}
    for interaction in interactions:
        user_id = interaction['user_id']
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append(interaction['item_id'])
    
    sequences = list(user_sequences.values())
    sequential.fit(sequences)
    patterns = sequential.find_patterns(min_support=0.3)
    
    print(f"   Found {len(patterns)} sequential patterns:")
    for pattern in patterns[:3]:  # Show top 3
        if 'pattern' in pattern and 'support' in pattern:
            pattern_movies = [movies[item]['title'] for item in pattern['pattern'] if item in movies]
            print(f"      {' → '.join(pattern_movies)} (support: {pattern['support']:.3f})")
    
    # Preference drift detection
    print("\\n📊 Preference Drift Detection:")
    drift_detector = PreferenceDriftDetector()
    drift_detector.update_user_profile('alice', user_interactions)
    drift_result = drift_detector.detect_preference_drift('alice')
    
    print(f"   Drift analysis for Alice:")
    print(f"      Drift detected: {drift_result['drift_detected']}")
    print(f"      Confidence: {drift_result['confidence']:.3f}")


def demonstrate_explanations():
    """Demonstrate recommendation explanations."""
    print("\\n" + "="*60)
    print("💡 RECOMMENDATION EXPLANATIONS DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    # Initialize explainer
    explainer = RecommendationExplainer(use_gemini=False)  # Disable Gemini for demo
    
    # Create a sample recommendation
    recommendation = {
        'item_id': 'movie_3',
        'score': 0.85,
        'method': 'hybrid',
        'confidence': 0.82
    }
    
    user_profile = user_profiles['alice']
    item_features = movies['movie_3']
    
    context = {
        'method': 'hybrid',
        'strategy': 'content_similarity',
        'similar_items': ['movie_1'],
        'content_similarity': 0.75
    }
    
    print(f"\\n🎬 Explaining recommendation: {item_features['title']}")
    print(f"   For user: Alice")
    
    explanation = explainer.generate_explanation(
        recommendation, user_profile, item_features, context
    )
    
    print(f"\\n📝 Explanation:")
    if 'template' in explanation:
        print(f"   Template: {explanation['template']}")
    
    if 'features' in explanation and 'feature_explanations' in explanation['features']:
        print(f"\\n🔍 Feature Analysis:")
        for feature, desc in explanation['features']['feature_explanations'].items():
            print(f"   • {feature.title()}: {desc}")
    
    if 'confidence' in explanation:
        print(f"\\n📊 Confidence: {explanation['confidence']:.3f}")


def demonstrate_evaluation():
    """Demonstrate recommendation evaluation."""
    print("\\n" + "="*60)
    print("📊 RECOMMENDATION EVALUATION DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    # Generate recommendations
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(interactions, movies, user_profiles)
    recommendations = hybrid_recommender.recommend_for_user('alice', k=5)
    
    # Create ground truth (Alice's high ratings)
    ground_truth = [
        {'item_id': 'movie_1', 'rating': 5.0},
        {'item_id': 'movie_3', 'rating': 4.0}
    ]
    
    # Initialize scorer
    scorer = RecommendationScorer()
    
    # Update item popularity
    scorer.update_item_popularity(interactions)
    
    # Compute comprehensive scores
    print("\\n📈 Computing Recommendation Quality Metrics...")
    
    comprehensive_score = scorer.compute_comprehensive_score(
        recommendations, ground_truth, user_profiles['alice'], movies
    )
    
    print(f"\\n🎯 Overall Quality Score: {comprehensive_score['overall_score']:.3f}")
    
    print(f"\\n📊 Category Scores:")
    for category, score in comprehensive_score['category_scores'].items():
        print(f"   • {category.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\\n🔍 Detailed Metrics:")
    for category, metrics in comprehensive_score['detailed_metrics'].items():
        if metrics:
            print(f"   {category.title()}:")
            for metric, value in list(metrics.items())[:3]:  # Show top 3 metrics
                if isinstance(value, (int, float)):
                    print(f"      {metric}: {value:.3f}")


def demonstrate_real_time_features():
    """Demonstrate real-time features."""
    print("\\n" + "="*60)
    print("⚡ REAL-TIME FEATURES DEMO")
    print("="*60)
    
    # Generate sample data
    movies, interactions, user_profiles = generate_sample_data()
    
    print("\\n🔄 Simulating Real-time User Interactions...")
    
    # Initialize components
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(interactions, movies, user_profiles)
    
    short_term = ShortTermPersonalizer()
    
    # Simulate real-time interactions
    new_interactions = [
        {
            'user_id': 'alice',
            'item_id': 'movie_2',
            'rating': 4.5,
            'timestamp': datetime.now() - timedelta(minutes=5),
            'context': {'device': 'mobile', 'time_of_day': 'evening'}
        },
        {
            'user_id': 'alice',
            'item_id': 'movie_5',
            'rating': 3.5,
            'timestamp': datetime.now() - timedelta(minutes=2),
            'context': {'device': 'mobile', 'time_of_day': 'evening'}
        }
    ]
    
    print("\\n📱 New interactions:")
    for interaction in new_interactions:
        movie_title = movies[interaction['item_id']]['title']
        print(f"   • {interaction['user_id'].title()} rated '{movie_title}': {interaction['rating']}/5")
    
    # Update personalization
    short_term.update_user_profile('alice', new_interactions)
    updated_prefs = short_term.get_current_preferences('alice')
    
    print(f"\\n🎯 Updated Preferences for Alice:")
    if 'recent_ratings' in updated_prefs:
        print(f"   Recent rating average: {updated_prefs['recent_ratings'].get('average', 'N/A')}")
    if 'context_preferences' in updated_prefs:
        print(f"   Preferred context: {updated_prefs['context_preferences']}")
    
    # Generate updated recommendations
    print("\\n🔄 Generating Updated Recommendations...")
    updated_recommendations = hybrid_recommender.recommend_for_user('alice', k=3)
    
    for i, rec in enumerate(updated_recommendations, 1):
        movie = movies[rec['item_id']]
        print(f"   {i}. {movie['title']} (Score: {rec['score']:.3f})")


def main():
    """Run all demonstrations."""
    print("🎬" + "="*58 + "🎬")
    print("🎯 CONTENT RECOMMENDATION ENGINE DEMO 🎯")
    print("🎬" + "="*58 + "🎬")
    
    try:
        # Load configuration
        config = settings
        print(f"\\n⚙️  Configuration loaded: {config.environment} environment")
        
        # Run demonstrations
        demonstrate_individual_recommendations()
        demonstrate_group_recommendations()
        demonstrate_personalization()
        demonstrate_explanations()
        demonstrate_evaluation()
        demonstrate_real_time_features()
        
        print("\\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\\n🚀 Next Steps:")
        print("   • Start the API server: python api_server.py")
        print("   • Run tests: python -m pytest tests/")
        print("   • Check logs in: logs/")
        print("   • Modify config in: src/utils/config.py")
        
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        logger.exception("Demo error")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
