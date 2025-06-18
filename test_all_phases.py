#!/usr/bin/env python3
"""
Comprehensive Testing Suite for All Phases of the Content Recommendation Engine

This script tests:
- Phase 1: Core recommendation system (collaborative, content-based, hybrid)
- Phase 2: Content understanding and feature extraction 
- Phase 3: Adaptive learning system with drift detection
- Integration tests and API functionality
- Performance benchmarks

Usage:
    python test_all_phases.py --phase all
    python test_all_phases.py --phase 1
    python test_all_phases.py --phase 2  
    python test_all_phases.py --phase 3
    python test_all_phases.py --quick
"""

import asyncio
import sys
import time
import traceback
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Container for test results"""
    def __init__(self, name: str, success: bool, duration: float, details: str = ""):
        self.name = name
        self.success = success
        self.duration = duration
        self.details = details
        self.timestamp = datetime.now()


class PhaseTestRunner:
    """Main test runner for all phases"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        
    def record_result(self, name: str, success: bool, duration: float, details: str = ""):
        """Record a test result"""
        result = TestResult(name, success, duration, details)
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name} ({duration:.2f}s)")
        if details and not success:
            print(f"   Details: {details}")
    
    async def run_phase_1_tests(self) -> bool:
        """Test Phase 1: Core Recommendation System"""
        print("\n🔧 Testing Phase 1: Core Recommendation System")
        print("-" * 60)
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Import core modules
        total_tests += 1
        start_time = time.time()
        try:
            from src.recommenders.hybrid import HybridRecommender
            from src.recommenders.collaborative import CollaborativeRecommender
            from src.recommenders.content_based import ContentBasedRecommender
            duration = time.time() - start_time
            self.record_result("Phase 1 - Core Module Imports", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 1 - Core Module Imports", False, duration, str(e))
        
        # Test 2: Initialize hybrid recommender
        total_tests += 1
        start_time = time.time()
        try:
            hybrid = HybridRecommender(
                collaborative_weight=0.4,
                content_weight=0.4,
                knowledge_weight=0.2
            )
            duration = time.time() - start_time
            self.record_result("Phase 1 - Hybrid Recommender Init", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 1 - Hybrid Recommender Init", False, duration, str(e))
        
        # Test 3: Data loading and preprocessing
        total_tests += 1
        start_time = time.time()
        try:
            from src.data_integration.data_loader import DataLoader
            loader = DataLoader()
            duration = time.time() - start_time
            self.record_result("Phase 1 - Data Loading", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 1 - Data Loading", False, duration, str(e))
        
        # Test 4: Run integration test
        total_tests += 1
        start_time = time.time()
        try:
            # Import and run the existing integration test
            import subprocess
            result = subprocess.run(
                [sys.executable, "test_integration.py"], 
                capture_output=True, 
                text=True,
                timeout=120
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.record_result("Phase 1 - Integration Test", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 1 - Integration Test", False, duration, 
                                 f"Exit code: {result.returncode}, Error: {result.stderr}")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 1 - Integration Test", False, duration, str(e))
        
        print(f"\nPhase 1 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
    
    async def run_phase_2_tests(self) -> bool:
        """Test Phase 2: Content Understanding"""
        print("\n🧠 Testing Phase 2: Content Understanding")
        print("-" * 60)
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Import content understanding modules
        total_tests += 1
        start_time = time.time()
        try:
            from src.content_understanding.embedding_generator import EmbeddingGenerator
            from src.content_understanding.quality_scorer import QualityScorer
            from src.content_understanding.gemini_client import GeminiClient
            duration = time.time() - start_time
            self.record_result("Phase 2 - Content Understanding Imports", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 2 - Content Understanding Imports", False, duration, str(e))
        
        # Test 2: Embedding generation
        total_tests += 1
        start_time = time.time()
        try:
            from src.content_understanding.embedding_generator import EmbeddingGenerator
            generator = EmbeddingGenerator()
            
            # Test with sample text
            sample_text = "This is a test movie about adventure and drama"
            embeddings = await generator.generate_embeddings_from_text([sample_text])
            
            duration = time.time() - start_time
            if embeddings and len(embeddings) > 0:
                self.record_result("Phase 2 - Embedding Generation", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 2 - Embedding Generation", False, duration, "No embeddings generated")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 2 - Embedding Generation", False, duration, str(e))
        
        # Test 3: Quality scoring
        total_tests += 1
        start_time = time.time()
        try:
            from src.content_understanding.quality_scorer import QualityScorer
            scorer = QualityScorer()
            
            # Test with sample content
            sample_content = {
                'title': 'Test Movie',
                'description': 'A great movie about adventure',
                'genres': ['Action', 'Adventure']
            }
            score = scorer.calculate_single_quality_score(sample_content)
            
            duration = time.time() - start_time
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                self.record_result("Phase 2 - Quality Scoring", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 2 - Quality Scoring", False, duration, f"Invalid score: {score}")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 2 - Quality Scoring", False, duration, str(e))
        
        print(f"\nPhase 2 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
    
    async def run_phase_3_tests(self) -> bool:
        """Test Phase 3: Adaptive Learning System"""
        print("\n🧬 Testing Phase 3: Adaptive Learning System")
        print("-" * 60)
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Import adaptive learning modules
        total_tests += 1
        start_time = time.time()
        try:
            from src.adaptive_learning import (
                FeedbackProcessor, OnlineLearner, DriftDetector, AdaptationEngine,
                FeedbackType, LearningAlgorithm, DriftDetectionMethod, AdaptationStrategy
            )
            duration = time.time() - start_time
            self.record_result("Phase 3 - Adaptive Learning Imports", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Adaptive Learning Imports", False, duration, str(e))
        
        # Test 2: Feedback processing
        total_tests += 1
        start_time = time.time()
        try:
            from src.adaptive_learning import FeedbackProcessor, FeedbackType
            
            processor = FeedbackProcessor(buffer_size=100)
            
            # Test feedback processing
            feedback_data = {
                'user_id': 'test_user',
                'item_id': 'test_item',
                'feedback_type': FeedbackType.EXPLICIT,
                'value': 4.5,
                'timestamp': datetime.now()
            }
            
            result = processor.process_feedback(feedback_data)
            duration = time.time() - start_time
            
            if result:
                self.record_result("Phase 3 - Feedback Processing", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 3 - Feedback Processing", False, duration, "Processing failed")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Feedback Processing", False, duration, str(e))
        
        # Test 3: Drift detection
        total_tests += 1
        start_time = time.time()
        try:
            from src.adaptive_learning import DriftDetector, DriftDetectionMethod
            
            detector = DriftDetector()
            
            # Test with sample data
            test_preferences = {
                'action': 0.8,
                'comedy': 0.6,
                'drama': 0.4
            }
            
            result = await detector.update_user_preferences('test_user', test_preferences)
            duration = time.time() - start_time
            
            # The method can return None if no drift is detected, which is valid
            self.record_result("Phase 3 - Drift Detection", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Drift Detection", False, duration, str(e))
        
        # Test 4: Online learning
        total_tests += 1
        start_time = time.time()
        try:
            from src.adaptive_learning import OnlineLearner, LearningAlgorithm
            
            learner = OnlineLearner()
            
            # Test incremental learning
            sample_data = {
                'user_features': [0.1, 0.2, 0.3],
                'item_features': [0.4, 0.5, 0.6],
                'rating': 4.0
            }
            
            await learner.update_model(sample_data)
            duration = time.time() - start_time
            
            self.record_result("Phase 3 - Online Learning", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Online Learning", False, duration, str(e))
        
        # Test 5: Adaptation engine
        total_tests += 1
        start_time = time.time()
        try:
            from src.adaptive_learning import AdaptationEngine, AdaptationStrategy
            
            engine = AdaptationEngine(strategy=AdaptationStrategy.GRADUAL)
            
            # Test adaptation
            adaptation_data = {
                'drift_detected': True,
                'user_id': 'test_user',
                'severity': 0.7
            }
            
            result = await engine.adapt_to_change('test_user', adaptation_data)
            duration = time.time() - start_time
            
            if result:
                self.record_result("Phase 3 - Adaptation Engine", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 3 - Adaptation Engine", False, duration, "No adaptation result")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Adaptation Engine", False, duration, str(e))
        
        # Test 6: Run adaptive learning demo
        total_tests += 1
        start_time = time.time()
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "demo_adaptive_learning.py", "--quick"], 
                capture_output=True, 
                text=True,
                timeout=60
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.record_result("Phase 3 - Demo Test", True, duration)
                success_count += 1
            else:
                self.record_result("Phase 3 - Demo Test", False, duration, 
                                 f"Exit code: {result.returncode}, Error: {result.stderr}")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Phase 3 - Demo Test", False, duration, str(e))
        
        print(f"\nPhase 3 Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
    
    async def run_integration_tests(self) -> bool:
        """Test integration between all phases"""
        print("\n🔗 Testing System Integration")
        print("-" * 60)
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Pipeline integration
        total_tests += 1
        start_time = time.time()
        try:
            from src.pipeline_integration import AdaptiveLearningPipeline
            
            config = {
                'feedback_buffer_size': 50,
                'learning_rate': 0.01,
                'drift_sensitivity': 0.5
            }
            
            pipeline = AdaptiveLearningPipeline(config)
            await pipeline.start()
            
            # Test basic pipeline functionality
            await asyncio.sleep(1)  # Let it initialize
            await pipeline.stop()
            
            duration = time.time() - start_time
            self.record_result("Integration - Pipeline", True, duration)
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Integration - Pipeline", False, duration, str(e))
        
        # Test 2: API server startup
        total_tests += 1
        start_time = time.time()
        try:
            import subprocess
            import signal
            import time as time_module
            
            # Start API server in background
            proc = subprocess.Popen(
                [sys.executable, "api_server.py", "--test-mode"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            time_module.sleep(3)
            
            # Check if it's running
            if proc.poll() is None:
                # Server is running, terminate it
                proc.terminate()
                proc.wait(timeout=5)
                
                duration = time.time() - start_time
                self.record_result("Integration - API Server", True, duration)
                success_count += 1
            else:
                duration = time.time() - start_time
                stderr = proc.stderr.read().decode()
                self.record_result("Integration - API Server", False, duration, f"Server failed to start: {stderr}")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Integration - API Server", False, duration, str(e))
        
        print(f"\nIntegration Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
    
    async def run_performance_tests(self) -> bool:
        """Run performance benchmarks"""
        print("\n⚡ Testing Performance")
        print("-" * 60)
        
        success_count = 0
        total_tests = 0
        
        # Test 1: Recommendation speed
        total_tests += 1
        start_time = time.time()
        try:
            from src.recommenders.hybrid import HybridRecommender
            import numpy as np
            import pandas as pd
            
            hybrid = HybridRecommender()
            
            # Create sample data
            interactions = pd.DataFrame({
                'user_id': ['user1'] * 50,
                'item_id': [f'item{i}' for i in range(50)],
                'rating': np.random.uniform(1, 5, 50)
            })
            
            # Measure recommendation time
            rec_start = time.time()
            recommendations = hybrid.recommend('user1', n_recommendations=10)
            rec_time = time.time() - rec_start
            
            duration = time.time() - start_time
            
            if rec_time < 1.0:  # Should be fast
                self.record_result("Performance - Recommendation Speed", True, duration, 
                                 f"Recommendation time: {rec_time:.3f}s")
                success_count += 1
            else:
                self.record_result("Performance - Recommendation Speed", False, duration, 
                                 f"Too slow: {rec_time:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            self.record_result("Performance - Recommendation Speed", False, duration, str(e))
        
        print(f"\nPerformance Results: {success_count}/{total_tests} tests passed")
        return success_count == total_tests
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r.duration for r in self.results)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration': total_duration,
                'timestamp': datetime.now().isoformat()
            },
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'duration': r.duration,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        return report
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🏁 TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    print(f"  • {result.name}: {result.details}")
        
        total_duration = sum(r.duration for r in self.results)
        print(f"\nTotal Duration: {total_duration:.2f}s")
        print("="*80)


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test all phases of the recommendation engine')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all',
                       help='Which phase to test (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only')
    parser.add_argument('--report', type=str,
                       help='Save detailed report to file')
    
    args = parser.parse_args()
    
    runner = PhaseTestRunner()
    runner.start_time = time.time()
    
    print("🚀 Content Recommendation Engine - Comprehensive Test Suite")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    try:
        all_passed = True
        
        if args.phase in ['1', 'all']:
            phase1_result = await runner.run_phase_1_tests()
            all_passed = all_passed and phase1_result
        
        if args.phase in ['2', 'all']:
            phase2_result = await runner.run_phase_2_tests()
            all_passed = all_passed and phase2_result
        
        if args.phase in ['3', 'all']:
            phase3_result = await runner.run_phase_3_tests()
            all_passed = all_passed and phase3_result
        
        if args.phase == 'all' and not args.quick:
            integration_result = await runner.run_integration_tests()
            all_passed = all_passed and integration_result
            
            performance_result = await runner.run_performance_tests()
            all_passed = all_passed and performance_result
        
        # Generate and display summary
        runner.print_summary()
        
        # Save detailed report if requested
        if args.report:
            report = runner.generate_report()
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {args.report}")
        
        # Exit with appropriate code
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
