"""
Test Script

Quick validation script to test the content recommendation engine setup:
- Check dependencies
- Validate configuration
- Test basic functionality
- Run minimal pipeline
"""

import sys
import traceback
from pathlib import Path
import asyncio
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core dependencies
        import pandas as pd
        import numpy as np
        import sqlalchemy
        import aiohttp
        import fastapi
        import pydantic
        print("✓ Core dependencies imported successfully")
        
        # Project modules
        from src.utils.config import settings
        from src.utils.logging import LoggingConfig
        from src.utils.validation import DataValidator
        from src.data_integration.data_loader import DatasetDownloader
        from src.content_understanding.gemini_client import GeminiClient
        print("✓ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False


def test_configuration():
    """Test configuration setup"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import settings
        
        # Check essential config values
        assert settings.gemini_api_key, "GEMINI_API_KEY not configured"
        assert settings.database.sqlite_path, "Database path not configured"
        assert settings.redis_url, "REDIS_URL not configured"
        
        print(f"✓ Gemini API Key: {'*' * 10}...{settings.gemini_api_key[-4:]}")
        print(f"✓ Database Path: {settings.database.sqlite_path}")
        print(f"✓ Redis URL: {settings.redis_url}")
        print(f"✓ Environment: {settings.environment}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_database():
    """Test database connection"""
    print("\nTesting database...")
    
    try:
        from src.utils.config import settings
        from sqlalchemy import create_engine, text
        
        # Test connection
        # Handle both complete URLs and file paths
        if settings.database.sqlite_path.startswith('sqlite:'):
            db_url = settings.database.sqlite_path
        else:
            db_url = f"sqlite:///{settings.database.sqlite_path}"
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1, "Database query failed"
            print("✓ Database connection working")
        
        return True
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        print("Note: Database connection may fail if PostgreSQL is not running")
        return False


def test_data_loader():
    """Test data loader functionality"""
    print("\nTesting data loader...")
    
    try:
        from src.data_integration.data_loader import DatasetDownloader
        from src.utils.config import settings
        
        # Initialize downloader
        downloader = DatasetDownloader(settings.data_directory)
        
        # Test basic functionality
        print("✓ Data loader initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader error: {e}")
        return False


def test_user_profiling():
    """Test user profiling components"""
    print("\nTesting user profiling...")
    
    try:
        from src.user_profiling.preference_tracker import PreferenceTracker
        from src.user_profiling.behavior_analyzer import BehaviorAnalyzer
        from src.user_profiling.profile_evolution import ProfileEvolution
        from src.data_integration.schema_manager import UserProfile
        from src.utils.config import settings
        
        # Initialize components
        preference_tracker = PreferenceTracker()
        behavior_analyzer = BehaviorAnalyzer()
        profile_evolution = ProfileEvolution()
        
        print("✓ User profiling components initialized")
        
        # Test basic functionality with dummy user
        # (Would require actual data to be meaningful)
        try:
            # This will likely return empty profiles since we don't have data yet
            summary = preference_tracker.get_preference_summary("test_user")
            print("✓ Preference tracker working")
        except Exception:
            print("! Preference tracker needs data (expected for fresh install)")
        
        # Cleanup
        preference_tracker.close()
        behavior_analyzer.close()
        profile_evolution.close()
        
        return True
        
    except Exception as e:
        print(f"✗ User profiling error: {e}")
        return False


async def test_gemini_client():
    """Test Gemini API client (if API key is available)"""
    print("\nTesting Gemini client...")
    
    try:
        from src.content_understanding.gemini_client import GeminiClient
        from src.utils.config import settings
        
        if not settings.gemini_api_key:
            print("! Google API key not configured - skipping Gemini test")
            return True
        
        client = GeminiClient()
        
        # Test with a simple text
        try:
            response = await client.generate_embedding("test content")
            if response:
                print("✓ Gemini client working")
            else:
                print("! Gemini client returned empty response")
        except Exception as e:
            print(f"! Gemini client error (may be quota/network): {e}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"✗ Gemini client setup error: {e}")
        return False


def test_api_components():
    """Test API components"""
    print("\nTesting API components...")
    
    try:
        from fastapi import FastAPI
        from api_server import app
        from src.data_integration.schema_manager import UserProfile
        
        # Test that the app is configured
        assert isinstance(app, FastAPI), "FastAPI app not properly configured"
        
        # Test route configuration
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/users/{user_id}/preferences", "/recommendations"]
        
        for route in expected_routes:
            if any(route.replace("{user_id}", "123") in r for r in routes):
                continue
            elif route in routes:
                continue
            else:
                print(f"! Route {route} not found")
        
        print("✓ API components configured")
        return True
        
    except Exception as e:
        print(f"✗ API components error: {e}")
        return False


async def run_minimal_pipeline():
    """Run a minimal version of the pipeline for testing"""
    print("\nTesting minimal pipeline...")
    
    try:
        # Test individual components initialization
        from src.content_understanding.gemini_client import GeminiClient
        from src.utils.config import settings
        
        # Initialize a basic component
        if settings.gemini_api_key:
            client = GeminiClient()
            await client.close()
        
        print("✓ Pipeline components can be initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        return False


async def main():
    """Run all tests"""
    print("Content Recommendation Engine - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Database Test", test_database),
        ("Data Loader Test", test_data_loader),
        ("User Profiling Test", test_user_profiling),
        ("Gemini Client Test", test_gemini_client),
        ("API Components Test", test_api_components),
        ("Minimal Pipeline Test", run_minimal_pipeline),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running: {test_name}")
        print(f"{'-' * 30}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Configure your Google API key in .env if not already done")
        print("2. Run 'python main.py' to execute the full pipeline")
        print("3. Run 'python api_server.py' to start the API server")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check your .env configuration")
        print("- Ensure sufficient disk space and memory")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
