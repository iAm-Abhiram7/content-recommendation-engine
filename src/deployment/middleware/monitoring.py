"""
Monitoring Middleware
Comprehensive request monitoring and metrics collection
"""

import time
import logging
import json
from typing import Dict, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import psutil
import asyncio

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Comprehensive monitoring middleware"""
    
    def __init__(self, app, registry: Optional[CollectorRegistry] = None):
        super().__init__(app)
        self.active_sessions = set()
        self.registry = registry
        
        # Initialize Prometheus metrics with custom registry
        self.REQUEST_COUNT = Counter(
            'monitoring_http_requests_total', 
            'Total HTTP requests', 
            ['method', 'endpoint', 'status_code', 'user_type'],
            registry=self.registry
        )
        self.REQUEST_DURATION = Histogram(
            'monitoring_http_request_duration_seconds', 
            'HTTP request duration', 
            ['endpoint'],
            registry=self.registry
        )
        self.ERROR_COUNT = Counter(
            'monitoring_http_errors_total', 
            'Total HTTP errors', 
            ['endpoint', 'error_type'],
            registry=self.registry
        )
        self.CONCURRENT_REQUESTS = Gauge(
            'monitoring_http_concurrent_requests', 
            'Concurrent HTTP requests',
            registry=self.registry
        )
        self.USER_SESSIONS = Gauge(
            'monitoring_active_user_sessions', 
            'Active user sessions',
            registry=self.registry
        )
        
        # System metrics
        self.CPU_USAGE = Gauge(
            'monitoring_system_cpu_usage_percent', 
            'System CPU usage',
            registry=self.registry
        )
        self.MEMORY_USAGE = Gauge(
            'monitoring_system_memory_usage_bytes', 
            'System memory usage',
            registry=self.registry
        )
        self.MEMORY_PERCENT = Gauge(
            'monitoring_system_memory_usage_percent', 
            'System memory usage percentage',
            registry=self.registry
        )
        
        # Start system metrics collection
        asyncio.create_task(self.collect_system_metrics())
    
    async def collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.CPU_USAGE.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.MEMORY_USAGE.set(memory.used)
                self.MEMORY_PERCENT.set(memory.percent)
                
                # Active sessions
                self.USER_SESSIONS.set(len(self.active_sessions))
                
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    def get_endpoint_label(self, path: str) -> str:
        """Get normalized endpoint label for metrics"""
        # Remove query parameters and normalize path
        path = path.split('?')[0]
        
        # Replace user IDs and other dynamic segments
        path_parts = path.split('/')
        normalized_parts = []
        
        for part in path_parts:
            if part.isdigit() or (len(part) > 10 and '-' in part):
                normalized_parts.append('{id}')
            else:
                normalized_parts.append(part)
        
        return '/'.join(normalized_parts)
    
    def get_user_type(self, request: Request) -> str:
        """Get user type for metrics"""
        if hasattr(request.state, 'user'):
            return 'authenticated'
        elif request.headers.get('X-API-Key'):
            return 'api_key'
        else:
            return 'anonymous'
    
    def log_request_details(self, request: Request, response_status: int, duration: float):
        """Log detailed request information"""
        user_id = getattr(request.state, 'user_id', None)
        user_type = self.get_user_type(request)
        
        log_data = {
            'timestamp': time.time(),
            'method': request.method,
            'path': str(request.url.path),
            'query_params': dict(request.query_params),
            'status_code': response_status,
            'duration_ms': round(duration * 1000, 2),
            'user_id': user_id,
            'user_type': user_type,
            'client_ip': request.client.host,
            'user_agent': request.headers.get('User-Agent', ''),
            'referer': request.headers.get('Referer', ''),
            'content_length': request.headers.get('Content-Length', 0)
        }
        
        # Log request details
        if response_status >= 400:
            logging.warning(f"HTTP {response_status} - {json.dumps(log_data)}")
        else:
            logging.info(f"HTTP {response_status} - {json.dumps(log_data)}")
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        endpoint = self.get_endpoint_label(request.url.path)
        user_type = self.get_user_type(request)
        
        # Track concurrent requests
        self.CONCURRENT_REQUESTS.inc()
        
        # Track user session
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            self.active_sessions.add(user_id)
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            self.REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
                user_type=user_type
            ).inc()
            
            self.REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
            
            # Log request details
            self.log_request_details(request, response.status_code, duration)
            
            # Add monitoring headers
            response.headers["X-Request-Duration"] = str(round(duration * 1000, 2))
            response.headers["X-Request-ID"] = str(id(request))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            # Track errors
            self.ERROR_COUNT.labels(
                endpoint=endpoint,
                error_type=error_type
            ).inc()
            
            # Log error details
            logging.error(f"Request error: {e}", extra={
                'endpoint': endpoint,
                'method': request.method,
                'duration': duration,
                'error_type': error_type,
                'user_id': user_id
            })
            
            raise
            
        finally:
            self.CONCURRENT_REQUESTS.dec()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            'active_sessions': len(self.active_sessions),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            },
            'timestamp': time.time()
        }
