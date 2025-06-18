#!/usr/bin/env python3
"""
Content Recommendation Engine Demo

This script demonstrates the key features of the recommendation system:
1. User profiling and behavior analysis
2. Different recommendation methods (collaborative, content-based, hybrid)
3. Group recommendations
4. Real-time feedback processing
5. Cross-domain recommendations
6. Performance metrics and analytics
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import random

class RecommendationEngineDemo:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_system_health(self):
        """Check if the API server is running and healthy"""
        print("🔍 Checking system health...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ System is healthy: {health_data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API server: {e}")
            return False
    
    def get_system_status(self):
        """Get detailed system status and metrics"""
        print("\n📊 Getting system status...")
        try:
            response = self.session.get(f"{self.base_url}/system/status")
            if response.status_code == 200:
                status = response.json()
                print(f"System Status: {status.get('system_status', 'Unknown')}")
                print(f"Data Quality Score: {status.get('data_quality_score', 0):.2f}")
                
                db_stats = status.get('database_statistics', {})
                print(f"Total Users: {db_stats.get('total_users', 0)}")
                print(f"Total Content: {db_stats.get('total_content', 0)}")
                print(f"Total Interactions: {db_stats.get('total_interactions', 0)}")
                
                return status
            else:
                print(f"Failed to get status: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting status: {e}")
            return None
    
    def get_user_profile(self, user_id: str):
        """Get detailed user profile"""
        print(f"\n👤 Getting profile for user {user_id}...")
        try:
            # Get user preferences
            response = self.session.get(f"{self.base_url}/users/{user_id}/preferences")
            if response.status_code == 200:
                preferences = response.json()
                print(f"User {user_id} Preferences:")
                print(f"  - Diversity Score: {preferences.get('diversity_score', 0):.2f}")
                print(f"  - Exploration Tendency: {preferences.get('exploration_tendency', 0):.2f}")
                print(f"  - Top Genres: {preferences.get('top_genres', [])[:3]}")
                return preferences
            else:
                print(f"Failed to get user preferences: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    def get_individual_recommendations(self, user_id: str, n_recommendations: int = 5):
        """Get individual recommendations for a user"""
        print(f"\n🎯 Getting {n_recommendations} recommendations for user {user_id}...")
        try:
            payload = {
                "user_id": user_id,
                "n_recommendations": n_recommendations,
                "method": "hybrid"
            }
            
            response = self.session.post(
                f"{self.base_url}/recommendations/individual",
                json=payload
            )
            
            if response.status_code == 200:
                recommendations = response.json()
                print(f"Method used: {recommendations.get('method', 'Unknown')}")
                print(f"Total recommendations: {recommendations.get('total_recommendations', 0)}")
                
                print("Top recommendations:")
                for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
                    print(f"  {i}. Item {rec.get('item_id')} (Score: {rec.get('score', 0):.3f})")
                
                return recommendations
            else:
                print(f"Failed to get recommendations: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
    
    def get_group_recommendations(self, user_ids: List[str], n_recommendations: int = 5):
        """Get group recommendations"""
        print(f"\n👥 Getting group recommendations for users {user_ids}...")
        try:
            payload = {
                "user_ids": user_ids,
                "n_recommendations": n_recommendations,
                "aggregation_strategy": "average"
            }
            
            response = self.session.post(
                f"{self.base_url}/recommendations/group",
                json=payload
            )
            
            if response.status_code == 200:
                recommendations = response.json()
                print(f"Aggregation strategy: {recommendations.get('aggregation_strategy')}")
                print("Group recommendations:")
                for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
                    print(f"  {i}. Item {rec.get('item_id')} (Score: {rec.get('score', 0):.3f})")
                
                return recommendations
            else:
                print(f"Failed to get group recommendations: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting group recommendations: {e}")
            return None
    
    def record_user_feedback(self, user_id: str, item_id: str, interaction_type: str = "view"):
        """Record user interaction/feedback"""
        print(f"\n📝 Recording {interaction_type} interaction for user {user_id} on item {item_id}...")
        try:
            payload = {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.base_url}/feedback/interaction",
                json=payload
            )
            
            if response.status_code == 200:
                feedback = response.json()
                print(f"Feedback recorded: {feedback.get('message', 'Success')}")
                return feedback
            else:
                print(f"Failed to record feedback: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return None
    
    def get_similar_items(self, item_id: str, n_similar: int = 5):
        """Get items similar to a given item"""
        print(f"\n🔗 Getting {n_similar} items similar to item {item_id}...")
        try:
            payload = {
                "item_id": item_id,
                "n_similar": n_similar,
                "use_metadata": True
            }
            
            response = self.session.post(
                f"{self.base_url}/recommendations/similar-items",
                json=payload
            )
            
            if response.status_code == 200:
                similar = response.json()
                print(f"Similarity method: {similar.get('similarity_method')}")
                print("Similar items:")
                for i, item in enumerate(similar.get('similar_items', [])[:3], 1):
                    print(f"  {i}. Item {item.get('item_id')} (Similarity: {item.get('similarity_score', 0):.3f})")
                
                return similar
            else:
                print(f"Failed to get similar items: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting similar items: {e}")
            return None
    
    def get_top_quality_content(self, limit: int = 5):
        """Get top quality content"""
        print(f"\n⭐ Getting top {limit} quality content...")
        try:
            response = self.session.get(
                f"{self.base_url}/content/top-quality",
                params={"limit": limit, "min_score": 0.8}
            )
            
            if response.status_code == 200:
                content = response.json()
                print("Top quality content:")
                for i, item in enumerate(content.get('content', content)[:3], 1):
                    if isinstance(item, dict):
                        print(f"  {i}. Content {item.get('content_id', 'Unknown')} (Quality: {item.get('quality_score', 0):.3f})")
                
                return content
            else:
                print(f"Failed to get quality content: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting quality content: {e}")
            return None
    
    def run_complete_demo(self):
        """Run a complete demonstration of the system"""
        print("🚀 Starting Content Recommendation Engine Demo")
        print("=" * 60)
        
        # Check system health
        if not self.check_system_health():
            print("❌ System is not healthy. Please check the API server.")
            return
        
        # Get system status
        status = self.get_system_status()
        if not status:
            print("❌ Cannot get system status.")
            return
        
        # Demo user IDs (using some from the system)
        demo_users = ["1", "2", "10", "25", "50"]
        
        # Test user profiling
        for user_id in demo_users[:2]:
            self.get_user_profile(user_id)
        
        # Test individual recommendations
        for user_id in demo_users[:2]:
            recommendations = self.get_individual_recommendations(user_id, 5)
            
            # If we got recommendations, record some feedback
            if recommendations and recommendations.get('recommendations'):
                item_id = recommendations['recommendations'][0]['item_id']
                self.record_user_feedback(user_id, item_id, "view")
                time.sleep(0.5)  # Small delay between requests
        
        # Test group recommendations
        self.get_group_recommendations(demo_users[:3], 5)
        
        # Test similar items (using a random item ID)
        self.get_similar_items("1", 5)
        
        # Test top quality content
        self.get_top_quality_content(5)
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("Next steps:")
        print("1. Open http://localhost:8000/docs to see the full API documentation")
        print("2. Try the interactive API documentation in your browser")
        print("3. Integrate the API endpoints into your application")
        print("4. Monitor system performance using the /system/metrics endpoint")

def main():
    """Main function to run the demo"""
    demo = RecommendationEngineDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
