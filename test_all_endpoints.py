#!/usr/bin/env python3
"""
Comprehensive API Endpoint Testing Script

Tests all endpoints mentioned in the documentation:
- Core Recommendations
- User Management  
- Analytics
- System endpoints
- Additional features
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def test_endpoint(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     expected_status: int = 200, timeout: int = 10) -> Dict[str, Any]:
        """Test a single endpoint and return results"""
        start_time = time.time()
        self.total_tests += 1
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=timeout)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response_time = (time.time() - start_time) * 1000
            
            result = {
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time_ms": round(response_time, 2),
                "success": response.status_code == expected_status,
                "response_data": None,
                "error": None
            }
            
            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text[:200] if response.text else None
                
            if result["success"]:
                self.passed_tests += 1
                print(f"✅ PASS {method.upper()} {endpoint} - {response.status_code} ({response_time:.0f}ms)")
            else:
                self.failed_tests += 1
                error_detail = ""
                if result["response_data"]:
                    if isinstance(result["response_data"], dict) and "detail" in result["response_data"]:
                        error_detail = result["response_data"]["detail"]
                    else:
                        error_detail = str(result["response_data"])[:100]
                
                result["error"] = f"Expected {expected_status}, got {response.status_code}: {error_detail}"
                print(f"❌ FAIL {method.upper()} {endpoint} - {response.status_code} ({response_time:.0f}ms)")
                print(f"   Error: {result['error'][:100]}...")
                
        except Exception as e:
            self.failed_tests += 1
            result = {
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": None,
                "expected_status": expected_status,
                "response_time_ms": (time.time() - start_time) * 1000,
                "success": False,
                "response_data": None,
                "error": str(e)
            }
            print(f"❌ FAIL {method.upper()} {endpoint} - Exception: {str(e)[:50]}...")
            
        self.results.append(result)
        return result
    
    def test_core_recommendations(self):
        """Test core recommendation endpoints"""
        print("🎯 Testing Core Recommendation Endpoints...")
        
        # Individual recommendations
        self.test_endpoint("POST", "/recommendations/individual", {
            "user_id": "test_user_123",
            "n_recommendations": 10,
            "method": "hybrid",
            "exclude_seen": True
        })
        
        # Group recommendations
        self.test_endpoint("POST", "/recommendations/group", {
            "user_ids": ["user1", "user2", "user3"],
            "n_recommendations": 10,
            "aggregation_strategy": "fairness"
        })
        
        # Similar items (correct endpoint)
        self.test_endpoint("POST", "/recommendations/similar-items", {
            "item_id": "item_123",
            "n_similar": 5
        })
        
        # Next item prediction
        self.test_endpoint("POST", "/recommendations/next-item", {
            "user_id": "test_user_123",
            "current_session": ["item1", "item2"],
            "n_predictions": 5
        })
        
        # Cross-domain recommendations
        self.test_endpoint("POST", "/recommendations/cross-domain", {
            "source_items": ["item1", "item2"],
            "target_domain": "movies",
            "n_recommendations": 5
        })
        
        # Basic recommendations endpoint
        self.test_endpoint("POST", "/recommendations", {
            "user_id": "test_user_123",
            "num_recommendations": 10,
            "exclude_seen": True
        })
    
    def test_user_management(self):
        """Test user management endpoints"""
        print("\n👤 Testing User Management Endpoints...")
        
        # User preferences
        self.test_endpoint("GET", "/users/test_user_123/preferences")
        
        # User behavior
        self.test_endpoint("GET", "/users/test_user_123/behavior")
        
        # User evolution
        self.test_endpoint("GET", "/users/test_user_123/evolution")
        
        # User profile (direct endpoint)
        self.test_endpoint("GET", "/users/test_user_123/profile")
        
        # Submit feedback (correct endpoint)
        self.test_endpoint("POST", "/feedback/interaction", {
            "user_id": "test_user_123",
            "item_id": "item_123",
            "interaction_type": "rating",
            "rating": 4.5
        })
        
        # Feedback submit (alternative endpoint)
        self.test_endpoint("POST", "/feedback/submit", {
            "user_id": "test_user_123",
            "item_id": "item_123",
            "feedback_type": "explicit",
            "value": 4.5
        })
    
    def test_analytics(self):
        """Test analytics endpoints"""
        print("\n📊 Testing Analytics Endpoints...")
        
        # User profile analytics
        self.test_endpoint("GET", "/analytics/user-profile/test_user_123")
        
        # Content analytics
        self.test_endpoint("GET", "/analytics/content/item_123")
        
        # System metrics
        self.test_endpoint("GET", "/system/metrics")
        
        # Additional analytics endpoints that exist
        self.test_endpoint("GET", "/analytics/user-distribution")
        self.test_endpoint("GET", "/analytics/content-stats")
        self.test_endpoint("GET", "/analytics/data-quality")
    
    def test_system_endpoints(self):
        """Test system endpoints"""
        print("\n🔧 Testing System Endpoints...")
        
        # Health check
        self.test_endpoint("GET", "/health")
        
        # System status
        self.test_endpoint("GET", "/system/status")
        
        # Adaptive health check
        self.test_endpoint("GET", "/adaptive/health")
        
        # Adaptive system status
        self.test_endpoint("GET", "/adaptive/system/status")
        
        # Adaptive system metrics
        self.test_endpoint("GET", "/adaptive/system/metrics")
    
    def test_additional_endpoints(self):
        """Test additional endpoints found in the API"""
        print("\n🔍 Testing Additional Endpoints...")
        
        # Content quality
        self.test_endpoint("GET", "/content/item_123/quality")
        
        # Top quality content
        self.test_endpoint("GET", "/content/top-quality")
        
        # Hybrid recommendations API
        self.test_endpoint("POST", "/api/v1/recommendations/hybrid", {
            "user_id": "test_user_123",
            "num_recommendations": 5,
            "exclude_seen": True
        })
        
        # Implicit feedback (using query parameters)
        self.test_endpoint("POST", "/api/v1/feedback/implicit?user_id=test_user_123&item_id=item_123&interaction_type=view", None)
        
        # Feedback history
        self.test_endpoint("GET", "/feedback/test_user_123/history")
    
    def test_adaptive_learning(self):
        """Test adaptive learning endpoints"""
        print("\n🧠 Testing Adaptive Learning Endpoints...")
        
        # Adaptive recommendations (should work now)
        self.test_endpoint("POST", "/adaptive/recommendations", {
            "user_id": "test_user_123",
            "num_items": 10,
            "context": {},
            "include_explanations": True
        })
        
        # Quick recommendations
        self.test_endpoint("GET", "/adaptive/recommendations/test_user_123/quick")
        
        # User control settings
        self.test_endpoint("GET", "/adaptive/users/test_user_123/control")
        self.test_endpoint("POST", "/adaptive/users/test_user_123/control", {
            "user_id": "test_user_123",
            "control_level": "moderate",
            "auto_adapt": True
        })
        
        # User preferences
        self.test_endpoint("GET", "/adaptive/users/test_user_123/preferences")
        self.test_endpoint("POST", "/adaptive/users/test_user_123/preferences", {
            "user_id": "test_user_123",
            "preferences": {"genres": ["action", "comedy"]},
            "merge_strategy": "update"
        })
        
        # Drift analysis
        self.test_endpoint("POST", "/adaptive/users/test_user_123/drift/analyze", {
            "user_id": "test_user_123",
            "time_range": "7d"
        })
        
        # Adaptation history
        self.test_endpoint("GET", "/adaptive/users/test_user_123/adaptations/history")
        
        # Manual adaptation trigger
        self.test_endpoint("POST", "/adaptive/users/test_user_123/adaptations/trigger")
        
        # Explanations
        self.test_endpoint("GET", "/adaptive/users/test_user_123/explanations/adaptation")
        
        # Visualizations
        self.test_endpoint("GET", "/adaptive/users/test_user_123/visualizations/preferences")
        
        # Batch feedback
        self.test_endpoint("POST", "/adaptive/batch/feedback", [
            {
                "user_id": "test_user_123",
                "item_id": "item_123",
                "feedback_type": "explicit",
                "value": 4.5
            }
        ])
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY REPORT")
        print("="*60)
        
        print(f"🎯 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📊 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        total_time = sum(r.get('response_time_ms', 0) for r in self.results) / 1000
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        # Failed endpoints
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            print(f"\n❌ Failed Endpoints:")
            for result in failed_results:
                endpoint = f"{result['method']} {result['endpoint']}"
                status = f"({result['status_code']})" if result['status_code'] else ""
                error = result['error'][:80] + "..." if len(result['error']) > 80 else result['error']
                print(f"   - {endpoint} {status}")
                print(f"     Error: {error}")
        
        # Status code distribution
        status_codes = {}
        for result in self.results:
            if result['status_code']:
                status_codes[result['status_code']] = status_codes.get(result['status_code'], 0) + 1
        
        print(f"\n📊 Status Code Distribution:")
        for status, count in sorted(status_codes.items()):
            print(f"   {status}: {count} endpoints")
        
        avg_response_time = sum(r.get('response_time_ms', 0) for r in self.results) / len(self.results)
        print(f"\n⚡ Average Response Time: {avg_response_time:.0f}ms")

def main():
    print("🚀 Starting Comprehensive API Endpoint Testing")
    print("📍 Base URL: http://localhost:8000")
    print("="*60)
    
    tester = APITester()
    
    # Test all endpoint categories
    tester.test_core_recommendations()
    tester.test_user_management()
    tester.test_analytics()
    tester.test_system_endpoints()
    tester.test_additional_endpoints()
    tester.test_adaptive_learning()
    
    # Print comprehensive summary
    tester.print_summary()
    
    # Return exit code based on test results
    return 0 if tester.failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
