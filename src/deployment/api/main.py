"""
Production FastAPI Application
High-performance, scalable API with comprehensive monitoring and security
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import os
import sys
from pathlib import Path

from src.api.endpoints import get_all_routers
from src.deployment.middleware.monitoring import MonitoringMiddleware
from src.deployment.middleware.auth import AuthMiddleware
from src.deployment.middleware.monitoring import MonitoringMiddleware
from src.deployment.middleware.rate_limiter import RateLimitMiddleware
from src.monitoring.metrics_collector import MetricsCollector
from src.pipeline_integration import AdaptiveLearningPipeline

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Create custom registry to avoid conflicts
metrics_registry = CollectorRegistry()

# Prometheus metrics with custom registry
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'], registry=metrics_registry)
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', registry=metrics_registry)
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections', registry=metrics_registry)
RECOMMENDATION_LATENCY = Histogram('recommendation_latency_seconds', 'Recommendation generation latency', registry=metrics_registry)
ERROR_RATE = Counter('api_errors_total', 'Total API errors', ['error_type'], registry=metrics_registry)

# Global variables
redis_client: Optional[redis.Redis] = None
adaptive_pipeline = None

# Authentication
security = HTTPBearer()

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Track performance metrics for all requests"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(duration)
            response.headers["X-Request-ID"] = str(id(request))
            
            return response
            
        except Exception as e:
            ERROR_RATE.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, metrics_collector, adaptive_pipeline
    
    # Startup
    logging.info("Starting production API server...")
    
    # Initialize Redis connection pool
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        encoding='utf-8',
        decode_responses=True,
        max_connections=100,
        retry_on_timeout=True
    )
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Initialize adaptive pipeline
    try:
        adaptive_pipeline = AdaptiveLearningPipeline()
        await adaptive_pipeline.initialize()
        logging.info("Adaptive learning pipeline initialized")
    except Exception as e:
        logging.error(f"Failed to initialize adaptive pipeline: {e}")
    
    # Health check for dependencies
    try:
        await redis_client.ping()
        logging.info("Redis connection established")
    except Exception as e:
        logging.error(f"Redis connection failed: {e}")
    
    logging.info("Production API server started successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down production API server...")
    
    if redis_client:
        await redis_client.close()
    
    if adaptive_pipeline:
        await adaptive_pipeline.shutdown()
    
    logging.info("Production API server shut down gracefully")

def create_app() -> FastAPI:
    """Create and configure production FastAPI application"""
    
    app = FastAPI(
        title="Content Recommendation Engine - Production API",
        description="High-performance recommendation system with adaptive learning",
        version="1.0.0",
        docs_url="/api/docs" if os.getenv('ENVIRONMENT') == 'development' else None,
        redoc_url="/api/redoc" if os.getenv('ENVIRONMENT') == 'development' else None,
        lifespan=lifespan
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv('ALLOWED_HOSTS', '*').split(',')
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(MonitoringMiddleware)  # Removed because MonitoringMiddleware is not defined
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Include routers
    for router in get_all_routers():
        app.include_router(router, prefix="/api/v1")
    
    # Try to import adaptive routers (optional)
    try:
        from src.api.adaptive_endpoints import adaptive_router
        from src.api.feedback_router import feedback_router
        app.include_router(adaptive_router, prefix="/api/v1/adaptive", tags=["adaptive"])
        app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["feedback"])
    except ImportError as e:
        logging.warning(f"Could not import adaptive endpoints: {e}")
    
    return app

# Create the production app
app = create_app()

# Health check endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Redis health
    try:
        if redis_client:
            await redis_client.ping()
            checks["checks"]["redis"] = "healthy"
        else:
            checks["checks"]["redis"] = "not_initialized"
    except Exception as e:
        checks["checks"]["redis"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    # Adaptive pipeline health
    try:
        if adaptive_pipeline and adaptive_pipeline.is_running:
            checks["checks"]["adaptive_pipeline"] = "healthy"
        else:
            checks["checks"]["adaptive_pipeline"] = "not_running"
    except Exception as e:
        checks["checks"]["adaptive_pipeline"] = f"unhealthy: {str(e)}"
        checks["status"] = "degraded"
    
    return checks

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if redis_client and adaptive_pipeline:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/api/v1/info")
async def api_info():
    """API information and capabilities"""
    return {
        "name": "Content Recommendation Engine",
        "version": "1.0.0",
        "environment": os.getenv('ENVIRONMENT', 'production'),
        "features": [
            "Hybrid Recommendations",
            "Adaptive Learning",
            "Cross-Domain Suggestions",
            "Explainable AI",
            "Real-time Feedback",
            "A/B Testing",
            "Performance Monitoring"
        ],
        "endpoints": {
            "recommendations": "/api/v1/recommendations",
            "adaptive": "/api/v1/adaptive",
            "feedback": "/api/v1/feedback",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging"""
    ERROR_RATE.labels(error_type="http_error").inc()
    
    logging.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    ERROR_RATE.labels(error_type="internal_error").inc()
    
    logging.error(f"Unhandled exception on {request.url}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    # Production server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        workers=int(os.getenv('WORKERS', 4)),
        loop="uvloop",
        http="httptools",
        access_log=True,
        log_level="info",
        reload=False
    )
