#!/usr/bin/env python3
"""
Integration test for the hybrid recommendation system.
Tests the core functionality without requiring a full database.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from src.recommenders.hybrid import HybridRecommender
from src.utils.config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_recommender():
    """Test the hybrid recommender with synthetic data"""
    
    print("🚀 Testing Hybrid Recommendation System")
    print("-" * 50)
    
    # Initialize the hybrid recommender
    try:
        hybrid = HybridRecommender(
            collaborative_weight=0.4,
            content_weight=0.4,
            knowledge_weight=0.2,
            diversity_factor=0.1,
            explanation_enabled=True
        )
        print("✓ HybridRecommender initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize HybridRecommender: {e}")
        return False
    
    # Create synthetic data
    try:
        # User-item interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'] * 20,
            'item_id': [f'item{i}' for i in range(1, 101)],
            'rating': np.random.uniform(1, 5, 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
        })
        
        # Item metadata
        items = pd.DataFrame({
            'item_id': [f'item{i}' for i in range(1, 101)],
            'title': [f'Item {i}' for i in range(1, 101)],
            'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], 100),
            'year': np.random.randint(2000, 2024, 100),
            'rating': np.random.uniform(1, 10, 100),
            'popularity': np.random.uniform(0, 1, 100)
        })
        
        print("✓ Created synthetic data")
        
        # Train the model
        hybrid.fit(interactions, items)
        print("✓ Model training completed")
        
        # Test recommendations
        recommendations = hybrid.recommend(
            user_id='user1',
            n_recommendations=5,
            user_history=['item1', 'item2'],
            exclude_seen=True
        )
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        # Display results
        print("\n📊 Sample Recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            print(f"  {i+1}. Item: {rec.get('item_id', 'Unknown')}")
            print(f"     Score: {rec.get('score', 0):.3f}")
            print(f"     Method: {rec.get('method', 'Unknown')}")
            if 'explanation' in rec:
                print(f"     Explanation: {rec['explanation']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoint imports"""
    try:
        from src.api.endpoints import get_all_routers
        routers = get_all_routers()
        print(f"✓ API endpoints loaded: {len(routers)} routers")
        return True
    except Exception as e:
        print(f"✗ API endpoints failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    try:
        config = Config()
        print("✓ Configuration system loaded")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Content Recommendation Engine - Integration Test")
    print("=" * 60)
    
    tests = [
        ("Configuration System", test_configuration),
        ("API Endpoints", test_api_endpoints), 
        ("Hybrid Recommender", test_hybrid_recommender),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("-" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready for use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
