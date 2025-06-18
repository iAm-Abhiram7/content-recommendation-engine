# Production Deployment Package
"""
Production deployment components for the Content Recommendation Engine.

This package provides:
- Production-ready FastAPI application
- Authentication and authorization middleware
- Rate limiting and monitoring middleware
- Docker containerization
- Health checks and graceful shutdown
- Prometheus metrics integration
"""

# Avoid circular imports - import these directly when needed
# from .api.main import create_app
from .middleware.auth import AuthMiddleware
from .middleware.rate_limiter import RateLimitMiddleware
from .middleware.monitoring import MonitoringMiddleware

__all__ = [
    # 'create_app',
    'AuthMiddleware',
    'RateLimitMiddleware',
    'MonitoringMiddleware'
]
