"""
Authentication Middleware
JWT-based authentication and authorization
"""

import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import os

# JWT configuration
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Authentication scheme
security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication middleware"""
    
    # Public endpoints that don't require authentication
    PUBLIC_PATHS = {
        '/health',
        '/health/ready',
        '/health/live',
        '/metrics',
        '/api/docs',
        '/api/redoc',
        '/openapi.json',
        '/api/v1/auth/login',
        '/api/v1/auth/register',
        '/api/v1/info'
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Check for API key in development
        if os.getenv('ENVIRONMENT') == 'development':
            api_key = request.headers.get('X-API-Key')
            if api_key == os.getenv('DEV_API_KEY', 'dev-key'):
                return await call_next(request)
        
        # Extract JWT token
        authorization = request.headers.get('Authorization')
        if not authorization:
            return JSONResponse(
                status_code=401,
                content={"error": "Authorization header required"}
            )
        
        try:
            # Parse Bearer token
            scheme, token = authorization.split(' ')
            if scheme.lower() != 'bearer':
                raise ValueError("Invalid authorization scheme")
            
            # Verify JWT token
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Add user info to request state
            request.state.user = payload
            request.state.user_id = payload.get('user_id')
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"error": "Token has expired"}
            )
        except jwt.InvalidTokenError as e:
            return JSONResponse(
                status_code=401,
                content={"error": f"Invalid token: {str(e)}"}
            )
        except ValueError as e:
            return JSONResponse(
                status_code=401,
                content={"error": str(e)}
            )
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication service error"}
            )

def create_access_token(user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
    """Create JWT access token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow(),
        'iss': 'recommendation-engine'
    }
    
    if additional_claims:
        payload.update(additional_claims)
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = security) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    payload = verify_token(credentials.credentials)
    return payload

def require_auth(request: Request) -> Dict[str, Any]:
    """Dependency to ensure user is authenticated"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Authentication required")
    return request.state.user
