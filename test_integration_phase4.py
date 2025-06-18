"""
Comprehensive integration tests for Phase 4 production deployment.
Tests the entire pipeline from API to monitoring to demo interface.
"""

import asyncio
import pytest
import requests
import time
import subprocess
import signal
import os
import sys
from pathlib import Path
from typing import Dict, Any
import json
import redis
from sqlalchemy import create_engine
import streamlit as st
from unittest.mock import patch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.deployment.api.main import create_app
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.alerting.alert_manager import AlertManager
from src.evaluation.testing.load_testing import LoadTester
from src.evaluation.testing.ab_testing import ABTester
from src.utils.demo_data import DemoDataGenerator


class IntegrationTestSuite:
    """Comprehensive integration test suite for Phase 4."""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.demo_url = "http://localhost:8501"
        self.redis_client = None
        self.test_results = {}
        self.processes = []
        
    async def setup_environment(self):
        """Set up test environment with all services."""
        print("🚀 Setting up integration test environment...")
        
        # Check if Redis is available
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            print("✅ Redis connection established")
        except Exception as e:
            print(f"❌ Redis not available: {e}")
            print("Please start Redis with: redis-server")
            return False
        
        # Start API server
        print("🔧 Starting API server...")
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.deployment.api.main:create_app",
            "--factory",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=Path(__file__).parent.parent)
        self.processes.append(api_process)
        
        # Wait for API to start
        await self.wait_for_service(self.base_url + "/health", "API")
        
        return True
    
    async def wait_for_service(self, url: str, service_name: str, timeout: int = 30):
        """Wait for a service to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            await asyncio.sleep(1)
        
        print(f"❌ {service_name} failed to start within {timeout} seconds")
        return False
    
    async def test_api_endpoints(self):
        """Test all API endpoints comprehensively."""
        print("\n🧪 Testing API endpoints...")
        
        test_cases = [
            {
                "name": "Health Check",
                "method": "GET",
                "url": f"{self.base_url}/health",
                "expected_status": 200
            },
            {
                "name": "Metrics Endpoint",
                "method": "GET",
                "url": f"{self.base_url}/metrics",
                "expected_status": 200
            },
            {
                "name": "User Recommendations",
                "method": "POST",
                "url": f"{self.base_url}/api/v1/recommendations/user",
                "data": {
                    "user_id": "test_user_123",
                    "num_recommendations": 10,
                    "include_explanations": True
                },
                "expected_status": 200
            },
            {
                "name": "Adaptive Learning Update",
                "method": "POST",
                "url": f"{self.base_url}/api/v1/adaptive/feedback",
                "data": {
                    "user_id": "test_user_123",
                    "item_id": "item_456",
                    "feedback_type": "explicit",
                    "rating": 4.5
                },
                "expected_status": 200
            },
            {
                "name": "Cross-domain Recommendations",
                "method": "POST",
                "url": f"{self.base_url}/api/v1/recommendations/cross-domain",
                "data": {
                    "user_id": "test_user_123",
                    "source_domain": "movies",
                    "target_domains": ["books", "music"],
                    "num_recommendations": 5
                },
                "expected_status": 200
            },
            {
                "name": "Group Recommendations",
                "method": "POST",
                "url": f"{self.base_url}/api/v1/recommendations/group",
                "data": {
                    "user_ids": ["user_1", "user_2", "user_3"],
                    "aggregation_method": "average",
                    "num_recommendations": 10
                },
                "expected_status": 200
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                if test_case["method"] == "GET":
                    response = requests.get(test_case["url"], timeout=10)
                else:
                    response = requests.post(
                        test_case["url"],
                        json=test_case.get("data", {}),
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                
                success = response.status_code == test_case["expected_status"]
                results.append({
                    "test": test_case["name"],
                    "status": "✅ PASS" if success else "❌ FAIL",
                    "response_code": response.status_code,
                    "expected_code": test_case["expected_status"],
                    "response_time": response.elapsed.total_seconds()
                })
                
                if success:
                    print(f"✅ {test_case['name']}: {response.status_code} ({response.elapsed.total_seconds():.3f}s)")
                else:
                    print(f"❌ {test_case['name']}: {response.status_code} (expected {test_case['expected_status']})")
                    
            except Exception as e:
                results.append({
                    "test": test_case["name"],
                    "status": "❌ ERROR",
                    "error": str(e)
                })
                print(f"❌ {test_case['name']}: ERROR - {e}")
        
        self.test_results["api_endpoints"] = results
        return results
    
    async def test_monitoring_system(self):
        """Test monitoring and alerting system."""
        print("\n📊 Testing monitoring system...")
        
        try:
            # Initialize metrics collector
            collector = MetricsCollector()
            
            # Test metrics collection
            await collector.collect_system_metrics()
            await collector.collect_ml_metrics({"ndcg": 0.4, "diversity": 0.75})
            await collector.collect_business_metrics({"revenue": 1000, "engagement": 0.8})
            
            # Test alert manager
            alert_manager = AlertManager()
            
            # Simulate an alert
            await alert_manager.check_and_alert({
                "response_time": 150,  # Above threshold
                "error_rate": 0.02,
                "ndcg": 0.3  # Below threshold
            })
            
            print("✅ Monitoring system functional")
            self.test_results["monitoring"] = {"status": "PASS"}
            return True
            
        except Exception as e:
            print(f"❌ Monitoring system error: {e}")
            self.test_results["monitoring"] = {"status": "FAIL", "error": str(e)}
            return False
    
    async def test_evaluation_framework(self):
        """Test evaluation and A/B testing framework."""
        print("\n📈 Testing evaluation framework...")
        
        try:
            # Test A/B testing
            ab_tester = ABTester()
            
            # Create test experiment
            experiment = await ab_tester.create_experiment(
                name="test_experiment",
                description="Integration test experiment",
                control_algorithm="collaborative_filtering",
                treatment_algorithm="hybrid_deep_learning",
                traffic_split=0.5
            )
            
            # Test load testing
            load_tester = LoadTester()
            load_results = await load_tester.run_load_test(
                endpoint=f"{self.base_url}/api/v1/recommendations/user",
                concurrent_users=10,
                duration=30,
                request_data={"user_id": "test_user", "num_recommendations": 10}
            )
            
            print("✅ Evaluation framework functional")
            self.test_results["evaluation"] = {
                "status": "PASS",
                "load_test_results": load_results
            }
            return True
            
        except Exception as e:
            print(f"❌ Evaluation framework error: {e}")
            self.test_results["evaluation"] = {"status": "FAIL", "error": str(e)}
            return False
    
    async def test_demo_interface(self):
        """Test Streamlit demo interface."""
        print("\n🎨 Testing demo interface...")
        
        try:
            # Start Streamlit app
            demo_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "src/demo/streamlit_app.py",
                "--server.port", "8501",
                "--server.headless", "true"
            ], cwd=Path(__file__).parent.parent)
            self.processes.append(demo_process)
            
            # Wait for demo to start
            if await self.wait_for_service(self.demo_url, "Streamlit Demo"):
                print("✅ Demo interface accessible")
                self.test_results["demo"] = {"status": "PASS"}
                return True
            else:
                print("❌ Demo interface failed to start")
                self.test_results["demo"] = {"status": "FAIL", "error": "Failed to start"}
                return False
                
        except Exception as e:
            print(f"❌ Demo interface error: {e}")
            self.test_results["demo"] = {"status": "FAIL", "error": str(e)}
            return False
    
    async def test_performance_requirements(self):
        """Test performance requirements are met."""
        print("\n⚡ Testing performance requirements...")
        
        performance_tests = []
        
        # Test response time requirement (<100ms)
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/recommendations/user",
                json={"user_id": "test_user", "num_recommendations": 10},
                timeout=1
            )
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            performance_tests.append({
                "test": "Response Time",
                "value": f"{response_time:.2f}ms",
                "requirement": "<100ms",
                "status": "✅ PASS" if response_time < 100 else "❌ FAIL"
            })
            
        except Exception as e:
            performance_tests.append({
                "test": "Response Time",
                "status": "❌ ERROR",
                "error": str(e)
            })
        
        # Test concurrent user handling
        try:
            load_tester = LoadTester()
            concurrent_test = await load_tester.test_concurrent_users(
                endpoint=f"{self.base_url}/api/v1/recommendations/user",
                max_users=100,
                request_data={"user_id": "test_user", "num_recommendations": 10}
            )
            
            performance_tests.append({
                "test": "Concurrent Users",
                "value": f"{concurrent_test['max_successful_users']} users",
                "requirement": ">100 users",
                "status": "✅ PASS" if concurrent_test['max_successful_users'] >= 100 else "❌ FAIL"
            })
            
        except Exception as e:
            performance_tests.append({
                "test": "Concurrent Users",
                "status": "❌ ERROR",
                "error": str(e)
            })
        
        self.test_results["performance"] = performance_tests
        
        for test in performance_tests:
            print(f"{test['status']} {test['test']}: {test.get('value', 'N/A')}")
        
        return performance_tests
    
    async def run_comprehensive_tests(self):
        """Run all integration tests."""
        print("🔬 Starting comprehensive integration tests...")
        
        # Setup environment
        if not await self.setup_environment():
            return False
        
        try:
            # Run all test suites
            await self.test_api_endpoints()
            await self.test_monitoring_system()
            await self.test_evaluation_framework()
            await self.test_demo_interface()
            await self.test_performance_requirements()
            
            # Generate test report
            self.generate_test_report()
            
        finally:
            # Cleanup
            await self.cleanup()
        
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📋 Generating test report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": self.test_results,
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "error_tests": 0
            }
        }
        
        # Calculate summary
        for category, results in self.test_results.items():
            if isinstance(results, list):
                for result in results:
                    report["summary"]["total_tests"] += 1
                    if "PASS" in result.get("status", ""):
                        report["summary"]["passed_tests"] += 1
                    elif "FAIL" in result.get("status", ""):
                        report["summary"]["failed_tests"] += 1
                    elif "ERROR" in result.get("status", ""):
                        report["summary"]["error_tests"] += 1
            else:
                report["summary"]["total_tests"] += 1
                if results.get("status") == "PASS":
                    report["summary"]["passed_tests"] += 1
                elif results.get("status") == "FAIL":
                    report["summary"]["failed_tests"] += 1
                else:
                    report["summary"]["error_tests"] += 1
        
        # Save report
        report_file = Path(__file__).parent.parent / "integration_test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report["summary"]
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed: {summary['passed_tests']}")
        print(f"   ❌ Failed: {summary['failed_tests']}")
        print(f"   🔥 Errors: {summary['error_tests']}")
        print(f"   📁 Report saved to: {report_file}")
        
        success_rate = (summary['passed_tests'] / summary['total_tests']) * 100 if summary['total_tests'] > 0 else 0
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Integration tests PASSED! System is production-ready.")
        elif success_rate >= 70:
            print("⚠️  Integration tests PARTIALLY PASSED. Some issues need attention.")
        else:
            print("🚨 Integration tests FAILED. Critical issues need to be resolved.")
    
    async def cleanup(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        print("✅ Cleanup completed")


async def main():
    """Main test runner."""
    test_suite = IntegrationTestSuite()
    success = await test_suite.run_comprehensive_tests()
    
    if success:
        print("\n🎯 Integration testing completed successfully!")
    else:
        print("\n💥 Integration testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
