"""
Rate Limiting Middleware
Token bucket rate limiting with Redis backend
"""

import time
import json
import logging
from typing import Dict, Tuple
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis
import os

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = None
        self.rate_limits = {
            'default': {'requests': 100, 'window': 60},  # 100 requests per minute
            'recommendations': {'requests': 50, 'window': 60},  # 50 recommendations per minute
            'feedback': {'requests': 200, 'window': 60},  # 200 feedback submissions per minute
            'adaptive': {'requests': 30, 'window': 60},  # 30 adaptive operations per minute
        }
        
    async def get_redis_client(self):
        """Get Redis client for rate limiting"""
        if not self.redis_client:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=1,  # Use different DB for rate limiting
                    encoding='utf-8',
                    decode_responses=True
                )
                await self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis not available for rate limiting: {e}")
                self.redis_client = None
        return self.redis_client
    
    def get_rate_limit_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limit key"""
        return f"rate_limit:{identifier}:{endpoint}"
    
    def get_rate_limit_config(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint"""
        if '/recommendations' in path:
            return self.rate_limits['recommendations']
        elif '/feedback' in path:
            return self.rate_limits['feedback']
        elif '/adaptive' in path:
            return self.rate_limits['adaptive']
        else:
            return self.rate_limits['default']
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Use user ID if authenticated
        if hasattr(request.state, 'user_id') and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # Use IP address as fallback
        client_ip = request.client.host
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        return f"ip:{client_ip}"
    
    async def check_rate_limit_redis(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using Redis sliding window"""
        redis_client = await self.get_redis_client()
        if not redis_client:
            return True, {'remaining': limit, 'reset_time': int(time.time() + window)}
        
        current_time = time.time()
        pipeline = redis_client.pipeline()
        
        try:
            # Remove expired entries
            pipeline.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            current_count = results[1] + 1  # +1 for the request we just added
            
            allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            reset_time = int(current_time + window)
            
            if not allowed:
                # Remove the request we added if limit exceeded
                await redis_client.zrem(key, str(current_time))
            
            return allowed, {
                'remaining': remaining,
                'reset_time': reset_time,
                'current_count': current_count
            }
            
        except Exception as e:
            logging.error(f"Redis rate limiting error: {e}")
            return True, {'remaining': limit, 'reset_time': int(current_time + window)}
    
    async def check_rate_limit_memory(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Fallback in-memory rate limiting"""
        # Simple in-memory implementation for when Redis is not available
        current_time = time.time()
        
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        if key not in self._memory_store:
            self._memory_store[key] = []
        
        # Remove expired entries
        self._memory_store[key] = [
            timestamp for timestamp in self._memory_store[key]
            if timestamp > current_time - window
        ]
        
        # Check limit
        current_count = len(self._memory_store[key]) + 1
        allowed = current_count <= limit
        
        if allowed:
            self._memory_store[key].append(current_time)
        
        remaining = max(0, limit - current_count)
        reset_time = int(current_time + window)
        
        return allowed, {
            'remaining': remaining,
            'reset_time': reset_time,
            'current_count': current_count
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ['/health', '/health/ready', '/health/live', '/metrics']:
            return await call_next(request)
        
        # Get rate limit configuration
        rate_config = self.get_rate_limit_config(request.url.path)
        limit = rate_config['requests']
        window = rate_config['window']
        
        # Get client identifier
        identifier = self.get_client_identifier(request)
        
        # Create rate limit key
        endpoint = request.url.path.split('/')[1:3]  # First two path segments
        endpoint_key = '/'.join(endpoint) if endpoint else 'root'
        key = self.get_rate_limit_key(identifier, endpoint_key)
        
        # Check rate limit
        try:
            allowed, info = await self.check_rate_limit_redis(key, limit, window)
        except Exception:
            allowed, info = await self.check_rate_limit_memory(key, limit, window)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {limit} per {window} seconds",
                    "retry_after": info['reset_time'] - int(time.time())
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": str(info['remaining']),
                    "X-RateLimit-Reset": str(info['reset_time']),
                    "Retry-After": str(info['reset_time'] - int(time.time()))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(info['reset_time'])
        
        return response
