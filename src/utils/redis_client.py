"""
Simplified Redis client utility for the content recommendation engine.
Provides connection management, caching, and streaming functionality.
"""

import redis
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass

from .logging import setup_logger


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    health_check_interval: int = 30
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True


class RedisConnectionManager:
    """Manages Redis connections with health checking and reconnection"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.logger = setup_logger(__name__)
        self._client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self._last_health_check = 0
        self._is_healthy = False
        
    def get_client(self) -> redis.Redis:
        """Get synchronous Redis client"""
        if not self._client or not self._is_connection_healthy():
            self._create_sync_client()
        return self._client
    
    def _create_sync_client(self):
        """Create synchronous Redis client"""
        try:
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses,
                health_check_interval=self.config.health_check_interval
            )
            
            self._client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            self._client.ping()
            self._is_healthy = True
            self._last_health_check = time.time()
            
            self.logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self._is_healthy = False
            raise
    
    def _is_connection_healthy(self) -> bool:
        """Check if sync connection is healthy"""
        try:
            current_time = time.time()
            if current_time - self._last_health_check > self.config.health_check_interval:
                if self._client:
                    self._client.ping()
                    self._is_healthy = True
                    self._last_health_check = current_time
                else:
                    self._is_healthy = False
            return self._is_healthy
        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            self._is_healthy = False
            return False
    
    def close(self):
        """Close all connections"""
        try:
            if self._connection_pool:
                self._connection_pool.disconnect()
            if self._client:
                self._client.close()
            self.logger.info("Closed Redis connections")
        except Exception as e:
            self.logger.error(f"Error closing Redis connections: {e}")


class RedisCache:
    """Redis-based caching with TTL support"""
    
    def __init__(self, connection_manager: RedisConnectionManager, key_prefix: str = "rec_engine"):
        self.connection_manager = connection_manager
        self.key_prefix = key_prefix
        self.logger = setup_logger(__name__)
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.key_prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            client = self.connection_manager.get_client()
            raw_value = client.get(self._make_key(key))
            if raw_value:
                return json.loads(raw_value)
            return None
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            client = self.connection_manager.get_client()
            json_value = json.dumps(value, default=str)
            if ttl:
                return client.setex(self._make_key(key), ttl, json_value)
            else:
                return client.set(self._make_key(key), json_value)
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            client = self.connection_manager.get_client()
            return bool(client.delete(self._make_key(key)))
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            client = self.connection_manager.get_client()
            return bool(client.exists(self._make_key(key)))
        except Exception as e:
            self.logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            client = self.connection_manager.get_client()
            prefixed_keys = [self._make_key(key) for key in keys]
            values = client.mget(prefixed_keys)
            
            result = {}
            for i, (key, value) in enumerate(zip(keys, values)):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to decode cached value for key: {key}")
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting multiple cache keys: {e}")
            return {}
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        try:
            client = self.connection_manager.get_client()
            
            # Prepare the mapping with prefixed keys and JSON values
            prefixed_mapping = {}
            for key, value in mapping.items():
                prefixed_mapping[self._make_key(key)] = json.dumps(value, default=str)
            
            # Use pipeline for efficiency
            pipe = client.pipeline()
            pipe.mset(prefixed_mapping)
            
            if ttl:
                for prefixed_key in prefixed_mapping.keys():
                    pipe.expire(prefixed_key, ttl)
            
            pipe.execute()
            return True
        except Exception as e:
            self.logger.error(f"Error setting multiple cache keys: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            client = self.connection_manager.get_client()
            keys = client.keys(self._make_key(pattern))
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for a key"""
        try:
            client = self.connection_manager.get_client()
            return client.ttl(self._make_key(key))
        except Exception as e:
            self.logger.error(f"Error getting TTL for key {key}: {e}")
            return -1


class RedisStreams:
    """Redis Streams functionality for real-time data processing"""
    
    def __init__(self, connection_manager: RedisConnectionManager):
        self.connection_manager = connection_manager
        self.logger = setup_logger(__name__)
    
    def add_to_stream(self, stream_name: str, data: Dict[str, Any], 
                     maxlen: Optional[int] = None) -> Optional[str]:
        """Add entry to Redis stream"""
        try:
            client = self.connection_manager.get_client()
            
            # Convert data to string values (Redis requirement)
            str_data = {k: json.dumps(v) if not isinstance(v, str) else v 
                       for k, v in data.items()}
            
            return client.xadd(stream_name, str_data, maxlen=maxlen)
        except Exception as e:
            self.logger.error(f"Error adding to stream {stream_name}: {e}")
            return None
    
    def read_from_stream(self, stream_name: str, consumer_group: str, 
                        consumer_name: str, count: int = 1, 
                        block: Optional[int] = None) -> List[Dict]:
        """Read from Redis stream"""
        try:
            client = self.connection_manager.get_client()
            
            # Ensure consumer group exists
            try:
                client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Read from stream
            messages = client.xreadgroup(
                consumer_group, consumer_name,
                {stream_name: '>'},
                count=count,
                block=block
            )
            
            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Parse JSON fields back to objects
                    parsed_fields = {}
                    for key, value in fields.items():
                        try:
                            parsed_fields[key] = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_fields[key] = value
                    
                    result.append({
                        'id': msg_id,
                        'stream': stream,
                        'data': parsed_fields
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error reading from stream {stream_name}: {e}")
            return []
    
    def acknowledge(self, stream_name: str, consumer_group: str, *message_ids):
        """Acknowledge processing of messages"""
        try:
            client = self.connection_manager.get_client()
            return client.xack(stream_name, consumer_group, *message_ids)
        except Exception as e:
            self.logger.error(f"Error acknowledging messages: {e}")
            return 0


class RedisClient:
    """Main Redis client with all functionality"""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        if config is None:
            config = RedisConfig()
        
        self.connection_manager = RedisConnectionManager(config)
        self.cache = RedisCache(self.connection_manager)
        self.streams = RedisStreams(self.connection_manager)
        self.logger = setup_logger(__name__)
    
    @property
    def client(self) -> redis.Redis:
        """Get the underlying Redis client"""
        return self.connection_manager.get_client()
    
    def is_healthy(self) -> bool:
        """Check if Redis is healthy - force a fresh check"""
        try:
            client = self.connection_manager.get_client()
            if client:
                client.ping()
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Close all connections"""
        self.connection_manager.close()
    
    @contextmanager
    def pipeline(self):
        """Get Redis pipeline for batch operations"""
        client = self.connection_manager.get_client()
        pipe = client.pipeline()
        try:
            yield pipe
        finally:
            pass


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> Optional[RedisClient]:
    """Get global Redis client instance"""
    global _redis_client
    
    if _redis_client is None:
        try:
            config = RedisConfig()
            _redis_client = RedisClient(config)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to initialize Redis client: {e}")
            return None
    
    return _redis_client


def initialize_redis(config: Optional[RedisConfig] = None) -> RedisClient:
    """Initialize Redis client with custom config"""
    global _redis_client
    
    if config is None:
        config = RedisConfig()
    
    _redis_client = RedisClient(config)
    return _redis_client


def close_redis():
    """Close global Redis client"""
    global _redis_client
    
    if _redis_client:
        _redis_client.close()
        _redis_client = None
