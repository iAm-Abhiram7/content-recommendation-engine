"""
Redis Health Check and Testing Script

This script tests Redis connectivity, cache operations, streaming functionality,
and provides health monitoring for the Redis setup.
"""

import time
import json
import redis
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.append('/home/abhi/Desktop/content-recommendation-engine')

from src.utils.redis_client import RedisClient, RedisConfig, get_redis_client


def test_basic_connectivity():
    """Test basic Redis connectivity"""
    print("Testing Redis connectivity...")
    
    try:
        # Test with our Redis client
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ Failed to get Redis client")
            return False
        
        # Test basic operations
        if redis_client.is_healthy():
            print("✅ Redis connection healthy")
        else:
            print("❌ Redis connection unhealthy")
            return False
        
        # Test ping
        result = redis_client.client.ping()
        if result:
            print("✅ Redis ping successful")
        else:
            print("❌ Redis ping failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connectivity test failed: {e}")
        return False


def test_cache_operations():
    """Test cache operations"""
    print("\nTesting cache operations...")
    
    try:
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ No Redis client available")
            return False
        
        cache = redis_client.cache
        
        # Test basic set/get
        test_key = "test_cache_key"
        test_value = {"message": "Hello Redis!", "timestamp": datetime.now().isoformat()}
        
        # Set value
        if cache.set(test_key, test_value, ttl=60):
            print("✅ Cache set operation successful")
        else:
            print("❌ Cache set operation failed")
            return False
        
        # Get value
        retrieved_value = cache.get(test_key)
        if retrieved_value and retrieved_value["message"] == test_value["message"]:
            print("✅ Cache get operation successful")
        else:
            print("❌ Cache get operation failed")
            return False
        
        # Test TTL
        ttl = cache.get_ttl(test_key)
        if ttl > 0:
            print(f"✅ Cache TTL working: {ttl} seconds remaining")
        else:
            print("⚠️  Cache TTL not set or expired")
        
        # Test multiple operations
        test_data = {
            "key1": {"value": 1, "type": "number"},
            "key2": {"value": "test", "type": "string"},
            "key3": {"value": [1, 2, 3], "type": "list"}
        }
        
        if cache.set_many(test_data, ttl=30):
            print("✅ Cache set_many operation successful")
        else:
            print("❌ Cache set_many operation failed")
            return False
        
        retrieved_data = cache.get_many(list(test_data.keys()))
        if len(retrieved_data) == len(test_data):
            print("✅ Cache get_many operation successful")
        else:
            print("❌ Cache get_many operation failed")
            return False
        
        # Clean up
        for key in [test_key] + list(test_data.keys()):
            cache.delete(key)
        
        print("✅ Cache cleanup successful")
        return True
        
    except Exception as e:
        print(f"❌ Cache operations test failed: {e}")
        return False


def test_streams():
    """Test Redis streams functionality"""
    print("\nTesting Redis streams...")
    
    try:
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ No Redis client available")
            return False
        
        streams = redis_client.streams
        
        # Test stream creation and data addition
        stream_name = "test_stream"
        test_data = {
            "event_type": "user_interaction",
            "user_id": "test_user_123",
            "item_id": "test_item_456",
            "action": "click",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to stream
        message_id = streams.add_to_stream(stream_name, test_data, maxlen=1000)
        if message_id:
            print(f"✅ Stream add operation successful: {message_id}")
        else:
            print("❌ Stream add operation failed")
            return False
        
        # Test reading from stream
        consumer_group = "test_consumer_group"
        consumer_name = "test_consumer"
        
        messages = streams.read_from_stream(
            stream_name, 
            consumer_group, 
            consumer_name, 
            count=1
        )
        
        if messages and len(messages) > 0:
            print(f"✅ Stream read operation successful: {len(messages)} messages")
            
            # Acknowledge the message
            message_ids = [msg['id'] for msg in messages]
            ack_count = streams.acknowledge(stream_name, consumer_group, *message_ids)
            if ack_count > 0:
                print("✅ Stream acknowledgment successful")
            else:
                print("⚠️  Stream acknowledgment failed or not needed")
        else:
            print("❌ Stream read operation failed")
            return False
        
        # Clean up stream
        client = redis_client.client
        client.delete(stream_name)
        print("✅ Stream cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Streams test failed: {e}")
        return False


def test_performance():
    """Test Redis performance"""
    print("\nTesting Redis performance...")
    
    try:
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ No Redis client available")
            return False
        
        cache = redis_client.cache
        
        # Test cache performance
        num_operations = 1000
        test_data = {f"perf_key_{i}": {"value": i, "data": f"test_data_{i}"} 
                    for i in range(num_operations)}
        
        # Time set operations
        start_time = time.time()
        cache.set_many(test_data, ttl=60)
        set_time = time.time() - start_time
        
        # Time get operations
        start_time = time.time()
        retrieved_data = cache.get_many(list(test_data.keys()))
        get_time = time.time() - start_time
        
        print(f"✅ Performance test completed:")
        print(f"   Set {num_operations} items: {set_time:.3f}s ({num_operations/set_time:.1f} ops/sec)")
        print(f"   Get {num_operations} items: {get_time:.3f}s ({num_operations/get_time:.1f} ops/sec)")
        print(f"   Retrieved {len(retrieved_data)}/{num_operations} items")
        
        # Clean up
        cache.clear_pattern("perf_key_*")
        print("✅ Performance test cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def get_redis_info():
    """Get Redis server information"""
    print("\nGetting Redis server information...")
    
    try:
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ No Redis client available")
            return False
        
        client = redis_client.client
        info = client.info()
        
        print("📊 Redis Server Information:")
        print(f"   Version: {info.get('redis_version', 'Unknown')}")
        print(f"   Mode: {info.get('redis_mode', 'Unknown')}")
        print(f"   Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        print(f"   Connected clients: {info.get('connected_clients', 0)}")
        print(f"   Used memory: {info.get('used_memory_human', 'Unknown')}")
        print(f"   Max memory: {info.get('maxmemory_human', 'No limit')}")
        print(f"   Database keys: {info.get('db0', {}).get('keys', 0) if 'db0' in info else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get Redis info: {e}")
        return False


def monitor_redis_health():
    """Monitor Redis health continuously"""
    print("\nStarting Redis health monitoring (Press Ctrl+C to stop)...")
    
    try:
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ No Redis client available")
            return
        
        while True:
            try:
                start_time = time.time()
                
                # Test ping
                redis_client.client.ping()
                ping_time = (time.time() - start_time) * 1000
                
                # Get memory usage
                info = redis_client.client.info()
                memory_used = info.get('used_memory_human', 'Unknown')
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ✅ Healthy - Ping: {ping_time:.1f}ms, Memory: {memory_used}")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n👋 Health monitoring stopped")
                break
            except Exception as e:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ❌ Unhealthy - {e}")
                time.sleep(5)
                
    except Exception as e:
        print(f"❌ Health monitoring failed: {e}")


def main():
    """Run all Redis tests"""
    print("🔍 Redis Health Check and Testing")
    print("=" * 50)
    
    tests = [
        ("Basic Connectivity", test_basic_connectivity),
        ("Cache Operations", test_cache_operations),
        ("Streams Functionality", test_streams),
        ("Performance", test_performance),
        ("Server Information", get_redis_info)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Redis is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the Redis configuration.")
    
    # Ask if user wants to start health monitoring
    if passed > 0:
        try:
            response = input("\n🔍 Start continuous health monitoring? (y/n): ").lower().strip()
            if response == 'y' or response == 'yes':
                monitor_redis_health()
        except KeyboardInterrupt:
            pass
    
    print("\n👋 Redis testing completed.")


if __name__ == "__main__":
    main()
