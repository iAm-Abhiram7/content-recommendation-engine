# Redis Setup and Configuration Guide

## Overview

This guide covers the complete Redis setup for the Content Recommendation Engine. Redis is used for:

- **Caching**: Fast retrieval of computed recommendations, user profiles, and model predictions
- **Streaming**: Real-time event processing for user interactions and feedback
- **Session Management**: Temporary storage for user sessions and context
- **Distributed Computing**: Coordination between different system components

## Installation Status ✅

- **Redis Server**: Valkey (Redis-compatible) installed and running on Fedora
- **Python Libraries**: `redis>=4.5.0` and `aioredis>=2.0.0` installed
- **Service Status**: Running on localhost:6379
- **System Integration**: Fully integrated with the recommendation engine

## Configuration

### Redis Server Configuration

Redis is running with the following settings:
- **Host**: localhost
- **Port**: 6379
- **Service**: valkey.service (Redis-compatible)
- **Status**: Active and enabled

### Application Configuration

The Redis client is configured in `src/utils/redis_client.py` with:

```python
@dataclass
class RedisConfig:
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
```

### YAML Configuration Files

**config/config.yaml**:
```yaml
redis_url: "redis://localhost:6379/0"
```

**config/adaptive_learning.yaml**:
```yaml
# Streaming configuration
streaming:
  redis:
    host: "localhost"
    port: 6379
    db: 0
    streams:
      - "user_stream"
      - "feedback_stream"
    consumer_group: "adaptive_learning"

# Caching configuration
caching:
  backend: "redis"
  redis_cache:
    host: "localhost"
    port: 6379
    db: 1
    max_connections: 50
```

## Usage Examples

### Basic Connection

```python
from src.utils.redis_client import get_redis_client

# Get Redis client
client = get_redis_client()

# Check health
if client and client.is_healthy():
    print("Redis is ready!")
```

### Caching Operations

```python
# Cache user preferences
user_prefs = {"genres": ["action", "comedy"], "rating_threshold": 4.0}
client.cache.set(f"user_prefs:{user_id}", user_prefs, ttl=3600)

# Retrieve cached data
prefs = client.cache.get(f"user_prefs:{user_id}")

# Batch operations
recommendations = {
    f"rec:{user_id}:movies": movie_recommendations,
    f"rec:{user_id}:books": book_recommendations
}
client.cache.set_many(recommendations, ttl=1800)
```

### Streaming Operations

```python
# Add user interaction to stream
interaction_data = {
    "user_id": "user_123",
    "item_id": "movie_456",
    "action": "click",
    "timestamp": datetime.now().isoformat()
}

stream_id = client.streams.add_to_stream("user_interactions", interaction_data)

# Read from stream
messages = client.streams.read_from_stream(
    "user_interactions", 
    "recommendation_engine", 
    "consumer_1",
    count=10
)

# Process and acknowledge messages
for message in messages:
    # Process message
    process_interaction(message['data'])
    # Acknowledge
    client.streams.acknowledge("user_interactions", "recommendation_engine", message['id'])
```

## System Integration

### Components Using Redis

1. **Content Understanding Module** (`src/content_understanding/gemini_client.py`)
   - Caches Gemini API responses
   - Stores embedding computations
   - TTL: 24 hours for embeddings

2. **Adaptive Learning** (`src/adaptive_learning/feedback_processor.py`)
   - Real-time feedback processing
   - User session tracking
   - Context window management

3. **Streaming System** (`src/streaming/stream_handler.py`)
   - Event routing and processing
   - Consumer group management
   - Message acknowledgment

4. **User Profiling** (Various modules)
   - Profile caching
   - Preference storage
   - Session management

### Database Usage

- **DB 0**: General caching (default)
- **DB 1**: Specialized caching (adaptive learning)
- **Streams**: Named streams for different event types

## Performance Characteristics

Based on testing:
- **Cache Set Operations**: ~33,000 ops/sec
- **Cache Get Operations**: ~101,000 ops/sec
- **Memory Usage**: ~1.31MB (minimal overhead)
- **Connection Pooling**: 50 max connections
- **Health Check**: 30-second intervals

## Monitoring and Health Checks

### Automated Health Monitoring

```python
# Basic health check
client = get_redis_client()
is_healthy = client.is_healthy()

# Detailed monitoring
info = client.client.info()
memory_usage = info['used_memory_human']
connected_clients = info['connected_clients']
```

### Verification Scripts

- `verify_redis.py`: Quick health check
- `test_redis_setup.py`: Comprehensive testing suite

### Logs and Debugging

Redis operations are logged through the structured logging system:
- **Location**: `logs/app.log`
- **Level**: INFO for connections, WARNING for failures
- **Format**: Structured JSON with timestamps

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   sudo systemctl status valkey
   sudo systemctl start valkey
   ```

2. **Memory Issues**
   - Monitor with `client.client.info()['used_memory_human']`
   - Set appropriate TTLs for cached data
   - Use `FLUSHDB` to clear database if needed

3. **Performance Issues**
   - Check connection pool settings
   - Monitor `connected_clients` count
   - Adjust `max_connections` if needed

### Service Management

```bash
# Check service status
sudo systemctl status valkey

# Start/stop service
sudo systemctl start valkey
sudo systemctl stop valkey

# Enable/disable auto-start
sudo systemctl enable valkey
sudo systemctl disable valkey

# View logs
sudo journalctl -u valkey -f
```

## Security Considerations

### Current Setup
- Local-only access (localhost:6379)
- No authentication required
- Development environment configuration

### Production Recommendations
- Enable Redis AUTH with strong password
- Configure firewall rules
- Use TLS encryption for connections
- Set up Redis Sentinel for high availability
- Regular backup of critical data

## Performance Tuning

### Recommended Settings

```yaml
# config/adaptive_learning.yaml adjustments for production
caching:
  user_preferences_ttl: 3600     # 1 hour
  model_predictions_ttl: 300     # 5 minutes
  drift_scores_ttl: 600          # 10 minutes
  explanations_ttl: 1800         # 30 minutes

streaming:
  redis:
    max_connections: 100         # Increase for high load
```

### Memory Optimization

- Use appropriate TTLs for different data types
- Implement data compression for large objects
- Monitor memory usage regularly
- Set Redis `maxmemory` directive if needed

## Integration Examples

### With Gemini Client

```python
# Automatic caching of API responses
gemini_client = GeminiClient()
# Responses automatically cached with 24-hour TTL
result = gemini_client.analyze_content(content)
```

### With Adaptive Learning

```python
# Real-time feedback processing with Redis streams
processor = FeedbackProcessor(enable_real_time=True)
# Automatically uses Redis for session tracking and caching
processor.process_feedback(feedback_data)
```

### With Recommendation Engine

```python
# Cached recommendations with automatic invalidation
recommendations = recommender.get_recommendations(user_id)
# Results cached with appropriate TTL based on user activity
```

## Testing

### Verification Commands

```bash
# Quick verification
python verify_redis.py

# Comprehensive testing
python test_redis_setup.py

# Manual testing
python -c "from src.utils.redis_client import get_redis_client; print(get_redis_client().is_healthy())"
```

### Expected Results
- All connection tests should pass
- Cache operations: set/get/delete working
- Streams: add/read/acknowledge working
- Performance: >10,000 ops/sec for basic operations

## Summary

✅ **Redis Setup Complete**
- Valkey service running and healthy
- Python Redis client configured and tested
- All major components integrated
- Caching, streaming, and session management operational
- Performance optimized for recommendation engine workload

The Redis setup is now fully operational and ready to support the content recommendation engine's caching, streaming, and real-time processing requirements.
