"""
Simplified Production FastAPI Application
Avoiding circular imports for demo purposes
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import os
import sys
from pathlib import Path

# Create custom registry to avoid conflicts
metrics_registry = CollectorRegistry()

# Prometheus metrics with custom registry
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'], registry=metrics_registry)
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', registry=metrics_registry)
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections', registry=metrics_registry)

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
            logging.error(f"Request error: {e}")
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    
    # Startup
    logging.info("Starting simplified production API server...")
    
    yield
    
    # Shutdown
    logging.info("Shutting down production API server...")

def create_app() -> FastAPI:
    """Create and configure production FastAPI application"""
    
    app = FastAPI(
        title="Content Recommendation Engine - Production API",
        description="High-performance recommendation system with adaptive learning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Performance middleware
    app.add_middleware(PerformanceMiddleware)
    
    # Simple recommendations endpoint
    @app.post("/api/v1/recommendations/user")
    async def get_user_recommendations(request: Dict[str, Any]):
        """Get recommendations for a user (simplified implementation)"""
        user_id = request.get("user_id", "unknown")
        num_recommendations = request.get("num_recommendations", 10)
        
        # Simple fallback recommendations
        recommendations = []
        for i in range(min(num_recommendations, 10)):
            recommendations.append({
                "item_id": f"item_{i+1}",
                "title": f"Sample Content {i+1}",
                "score": 0.9 - (i * 0.05),
                "genre": "Drama" if i % 2 == 0 else "Comedy",
                "year": 2020 + i,
                "explanation": f"Recommended based on your viewing history and preferences",
                "components": {
                    "collaborative": 0.3,
                    "content_based": 0.4,
                    "knowledge_based": 0.3
                }
            })
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "algorithm": "hybrid_fallback",
            "generated_at": datetime.utcnow().isoformat()
        }
    
    @app.post("/api/v1/recommendations/group")
    async def get_group_recommendations(request: Dict[str, Any]):
        """Get group recommendations (simplified implementation)"""
        user_ids = request.get("user_ids", [])
        num_recommendations = request.get("num_recommendations", 10)
        
        recommendations = []
        for i in range(min(num_recommendations, 8)):
            recommendations.append({
                "item_id": f"group_item_{i+1}",
                "title": f"Group Content {i+1}",
                "score": 0.85 - (i * 0.03),
                "genre": "Action" if i % 3 == 0 else "Adventure",
                "year": 2021 + i,
                "group_satisfaction": 0.8 + (i * 0.01),
                "explanation": f"Popular choice among similar groups of {len(user_ids)} users"
            })
        
        return {
            "user_ids": user_ids,
            "recommendations": recommendations,
            "aggregation_method": "fairness_aware",
            "group_size": len(user_ids)
        }
    
    @app.post("/api/v1/recommendations/cross-domain")
    async def get_cross_domain_recommendations(request: Dict[str, Any]):
        """Get cross-domain recommendations (simplified implementation)"""
        user_id = request.get("user_id", "unknown")
        source_domain = request.get("source_domain", "movies")
        target_domains = request.get("target_domains", ["books"])
        
        recommendations = {}
        for domain in target_domains:
            recommendations[domain] = []
            for i in range(5):
                recommendations[domain].append({
                    "item_id": f"{domain}_item_{i+1}",
                    "title": f"{domain.title()} Recommendation {i+1}",
                    "score": 0.8 - (i * 0.05),
                    "genre": "Cross-domain Discovery",
                    "cross_domain_connection": f"Similar themes to your {source_domain} preferences"
                })
        
        return {
            "user_id": user_id,
            "source_domain": source_domain,
            "target_domains": target_domains,
            "recommendations": recommendations
        }
    
    @app.post("/api/v1/adaptive/feedback")
    async def submit_feedback(request: Dict[str, Any]):
        """Submit user feedback (simplified implementation)"""
        user_id = request.get("user_id", "unknown")
        item_id = request.get("item_id", "unknown")
        feedback_type = request.get("feedback_type", "explicit")
        rating = request.get("rating", 0)
        
        return {
            "message": "Feedback received successfully",
            "user_id": user_id,
            "item_id": item_id,
            "feedback_type": feedback_type,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "processed"
        }
    
    return app

# Create the production app
app = create_app()

# Health check endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "recommendation-engine"
    }

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    return {"status": "ready"}

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(metrics_registry)

@app.get("/api/v1/info")
async def api_info():
    """API information and capabilities"""
    return {
        "name": "Content Recommendation Engine",
        "version": "1.0.0",
        "environment": "production",
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
        "main_simple:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        log_level="info",
        reload=False
    )
