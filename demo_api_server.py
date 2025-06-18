"""
Standalone FastAPI Demo Server
Completely independent to avoid import issues
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import os

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
    logging.info("Starting demo API server...")
    
    yield
    
    # Shutdown
    logging.info("Shutting down demo API server...")

# Create FastAPI app
app = FastAPI(
    title="Content Recommendation Engine - Demo API",
    description="High-performance recommendation system demo",
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

# API Endpoints
@app.post("/api/v1/recommendations/user")
async def get_user_recommendations(request: Dict[str, Any]):
    """Get recommendations for a user"""
    user_id = request.get("user_id", "unknown")
    num_recommendations = request.get("num_recommendations", 10)
    
    # Generate sample recommendations
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
            },
            "diversity_score": 0.75 + (i * 0.02),
            "novelty_score": 0.65 + (i * 0.03)
        })
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "algorithm": "hybrid_demo",
        "generated_at": datetime.utcnow().isoformat(),
        "performance": {
            "latency_ms": 45.2,
            "cache_hit": True
        }
    }

@app.post("/api/v1/recommendations/group")
async def get_group_recommendations(request: Dict[str, Any]):
    """Get group recommendations"""
    user_ids = request.get("user_ids", [])
    num_recommendations = request.get("num_recommendations", 10)
    aggregation_method = request.get("aggregation_method", "fairness_aware")
    
    recommendations = []
    for i in range(min(num_recommendations, 8)):
        recommendations.append({
            "item_id": f"group_item_{i+1}",
            "title": f"Group Content {i+1}",
            "score": 0.85 - (i * 0.03),
            "genre": "Action" if i % 3 == 0 else "Adventure",
            "year": 2021 + i,
            "group_satisfaction": 0.8 + (i * 0.01),
            "explanation": f"Popular choice among similar groups of {len(user_ids)} users",
            "fairness_score": 0.82,
            "individual_scores": {user_id: 0.8 + (i * 0.02) for user_id in user_ids}
        })
    
    return {
        "user_ids": user_ids,
        "recommendations": recommendations,
        "aggregation_method": aggregation_method,
        "group_size": len(user_ids),
        "group_cohesion": 0.75
    }

@app.post("/api/v1/recommendations/cross-domain")
async def get_cross_domain_recommendations(request: Dict[str, Any]):
    """Get cross-domain recommendations"""
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
                "cross_domain_connection": f"Similar themes to your {source_domain} preferences",
                "similarity_score": 0.78
            })
    
    return {
        "user_id": user_id,
        "source_domain": source_domain,
        "target_domains": target_domains,
        "recommendations": recommendations,
        "cross_domain_model": "knowledge_graph_v2"
    }

@app.post("/api/v1/adaptive/feedback")
async def submit_feedback(request: Dict[str, Any]):
    """Submit user feedback"""
    user_id = request.get("user_id", "unknown")
    item_id = request.get("item_id", "unknown")
    feedback_type = request.get("feedback_type", "explicit")
    rating = request.get("rating", 0)
    
    return {
        "message": "Feedback received and processed",
        "user_id": user_id,
        "item_id": item_id,
        "feedback_type": feedback_type,
        "rating": rating,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "processed",
        "model_updated": True,
        "adaptation_score": 0.15
    }

@app.get("/api/v1/recommendations")
async def get_recommendations_info():
    """Get information about available recommendation endpoints"""
    return {
        "available_endpoints": [
            "/api/v1/recommendations/user",
            "/api/v1/recommendations/group", 
            "/api/v1/recommendations/cross-domain"
        ],
        "features": [
            "Hybrid recommendations",
            "Group consensus algorithms",
            "Cross-domain discovery",
            "Real-time adaptation",
            "Fairness monitoring"
        ]
    }

# Health check endpoints
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "recommendation-engine-demo",
        "checks": {
            "api": "healthy",
            "ml_models": "loaded",
            "cache": "operational"
        }
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
        "name": "Content Recommendation Engine Demo",
        "version": "1.0.0",
        "environment": "demo",
        "features": [
            "Hybrid Recommendations",
            "Adaptive Learning", 
            "Cross-Domain Suggestions",
            "Explainable AI",
            "Real-time Feedback",
            "Group Recommendations",
            "Fairness Monitoring"
        ],
        "endpoints": {
            "user_recommendations": "/api/v1/recommendations/user",
            "group_recommendations": "/api/v1/recommendations/group",
            "cross_domain": "/api/v1/recommendations/cross-domain",
            "feedback": "/api/v1/adaptive/feedback",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "performance": {
            "avg_latency_ms": 45,
            "throughput_rps": 1200,
            "accuracy_ndcg": 0.37
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
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
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', 8000)),
        log_level="info"
    )
