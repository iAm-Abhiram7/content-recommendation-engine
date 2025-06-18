"""
Simple Redis verification script for the content recommendation engine
"""

from src.utils.redis_client import get_redis_client
import sys

def main():
    print("🔍 Redis Setup Verification")
    print("=" * 40)
    
    # Test basic connection
    print("1. Testing Redis connection...")
    try:
        client = get_redis_client()
        if client and client.is_healthy():
            print("   ✅ Connection successful")
        else:
            print("   ❌ Connection failed")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test cache operations
    print("2. Testing cache operations...")
    try:
        client.cache.set('verify_test', {'status': 'ok'}, ttl=30)
        result = client.cache.get('verify_test')
        if result and result.get('status') == 'ok':
            print("   ✅ Cache working")
            client.cache.delete('verify_test')
        else:
            print("   ❌ Cache failed")
            return False
    except Exception as e:
        print(f"   ❌ Cache error: {e}")
        return False
    
    # Test streams
    print("3. Testing Redis streams...")
    try:
        stream_id = client.streams.add_to_stream('verify_stream', {'test': 'data'})
        if stream_id:
            print("   ✅ Streams working")
            client.client.delete('verify_stream')
        else:
            print("   ❌ Streams failed")
            return False
    except Exception as e:
        print(f"   ❌ Streams error: {e}")
        return False
    
    print("\n🎉 Redis setup verification complete!")
    print("   All components are working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
