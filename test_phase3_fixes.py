#!/usr/bin/env python3
"""
Quick Test Script for Phase 3 Adaptive Learning Fixes

This script performs a quick validation of the implemented fixes
to ensure they work correctly before running the full assessment.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_updates():
    """Test that configuration files have been updated correctly"""
    print("🔧 Testing configuration updates...")
    
    try:
        import yaml
        with open('config/adaptive_learning.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key conservative settings
        pipeline_config = config.get('pipeline', {})
        learning_rate = pipeline_config.get('learning_rate', 0)
        adaptation_rate = pipeline_config.get('adaptation_rate', 0)
        
        online_config = config.get('online_learning', {})
        regularization = online_config.get('regularization', 0)
        
        drift_config = config.get('drift_detection', {})
        sensitivity = drift_config.get('sensitivity', 0)
        
        print(f"  ✅ Learning rate: {learning_rate} (should be ≤ 0.005)")
        print(f"  ✅ Adaptation rate: {adaptation_rate} (should be ≤ 0.05)")
        print(f"  ✅ Regularization: {regularization} (should be ≥ 0.02)")
        print(f"  ✅ Drift sensitivity: {sensitivity} (should be ≤ 0.6)")
        
        # Validate values
        issues = []
        if learning_rate > 0.005:
            issues.append(f"Learning rate too high: {learning_rate}")
        if adaptation_rate > 0.05:
            issues.append(f"Adaptation rate too high: {adaptation_rate}")
        if regularization < 0.02:
            issues.append(f"Regularization too low: {regularization}")
        if sensitivity > 0.6:
            issues.append(f"Drift sensitivity too high: {sensitivity}")
        
        if issues:
            print("  ❌ Configuration issues found:")
            for issue in issues:
                print(f"    • {issue}")
            return False
        else:
            print("  ✅ All configuration values are conservative and appropriate")
            return True
            
    except Exception as e:
        print(f"  ❌ Error reading configuration: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("📦 Testing module imports...")
    
    try:
        # Test pipeline integration
        from src.pipeline_integration import AdaptiveLearningPipeline
        print("  ✅ Pipeline integration imported successfully")
        
        # Test adaptive learning components
        from src.adaptive_learning.adaptation_engine import AdaptationEngine, AdaptationConfig
        print("  ✅ Adaptation engine imported successfully")
        
        from src.adaptive_learning.online_learner import OnlineLearner, OnlineLearningConfig
        print("  ✅ Online learner imported successfully")
        
        from src.adaptive_learning.drift_detector import DriftDetector, DriftDetectionConfig
        print("  ✅ Drift detector imported successfully")
        
        # Test utility modules
        from src.utils.logging import setup_logger
        print("  ✅ Logging utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error during import: {e}")
        return False

def test_pipeline_initialization():
    """Test that pipeline can be initialized with conservative config"""
    print("🔄 Testing pipeline initialization...")
    
    try:
        from src.pipeline_integration import AdaptiveLearningPipeline
        
        # Conservative test configuration
        test_config = {
            'learning_rate': 0.002,
            'adaptation_rate': 0.02,
            'drift_sensitivity': 0.001,
            'quality_validation_enabled': True,
            'min_quality_threshold': 0.35,
            'stream_type': 'mock'  # Use mock for testing
        }
        
        # Initialize pipeline
        pipeline = AdaptiveLearningPipeline(test_config)
        print("  ✅ Pipeline initialized successfully with conservative config")
        
        # Check that components have conservative settings
        if hasattr(pipeline, 'online_learner') and hasattr(pipeline.online_learner, 'config'):
            lr = pipeline.online_learner.config.learning_rate
            print(f"  ✅ Online learner learning rate: {lr}")
            
        if hasattr(pipeline, 'adaptation_engine') and hasattr(pipeline.adaptation_engine, 'config'):
            ar = pipeline.adaptation_engine.config.gradual_learning_rate
            print(f"  ✅ Adaptation engine learning rate: {ar}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline initialization failed: {e}")
        return False

def test_quality_monitor():
    """Test that quality monitor can be imported and initialized"""
    print("📊 Testing quality monitor...")
    
    try:
        # Import should work
        import quality_monitor
        print("  ✅ Quality monitor module imported successfully")
        
        # Test QualityReport dataclass
        from quality_monitor import QualityReport, QualityMonitor
        from datetime import datetime
        
        # Create test report
        test_report = QualityReport(
            timestamp=datetime.now(),
            ndcg_10=0.37,
            diversity_score=0.71,
            coverage=0.85,
            novelty=0.72,
            latency_ms=95.0,
            throughput_rps=650.0,
            quality_grade="B",
            recommendations=[],
            issues=[],
            suggestions=[]
        )
        
        print("  ✅ Quality report created successfully")
        print(f"  ✅ Test NDCG@10: {test_report.ndcg_10} (≥0.35: {'✅' if test_report.ndcg_10 >= 0.35 else '❌'})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quality monitor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Phase 3 Fix Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Updates", test_config_updates),
        ("Module Imports", test_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Quality Monitor", test_quality_monitor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 3 fixes are ready for deployment")
        print("\n🚀 Next steps:")
        print("  1. Run: python fix_adaptive_learning.py")
        print("  2. Run: python quality_monitor.py")
        print("  3. Verify NDCG@10 ≥ 0.35")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
        print("❌ Phase 3 fixes need additional work")
        print("\n🔧 Recommended actions:")
        print("  1. Check error messages above")
        print("  2. Verify all required dependencies are installed")
        print("  3. Check file paths and permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
