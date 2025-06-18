# Content Recommendation Engine - Complete Documentation

## 🎯 Overview

This is a comprehensive, production-ready hybrid recommendation engine that supports multi-domain content recommendations with advanced personalization, group recommendations, and real-time adaptation capabilities.

## 🚀 Features

### Core Recommendation Algorithms
- **Collaborative Filtering**: Matrix factorization (ALS, SVD, BPR), user/item similarity
- **Content-Based Filtering**: Gemini embeddings, metadata similarity, cross-domain support
- **Knowledge-Based**: Trending content, new releases, acclaimed items, contextual rules
- **Hybrid Ensemble**: Weighted combination with auto-tuning and diversity optimization

### Advanced Personalization
- **Short-term**: Session-based preferences, recency weighting, context awareness
- **Long-term**: Historical preference modeling, lifecycle tracking, stability analysis
- **Sequential Patterns**: Pattern mining, Markov chains, next-item prediction
- **Drift Detection**: Preference change detection, adaptation triggers

### Group Recommendations
- **Aggregation Methods**: Average, least misery, most pleasure, fairness-aware
- **Group Dynamics**: Consensus modeling, member satisfaction optimization
- **Explanations**: Group-level and individual explanations

### Production Features
- **RESTful API**: Complete FastAPI endpoints with rate limiting and caching
- **Real-time Updates**: Live preference adaptation and feedback integration
- **Explanations**: Multi-modal explanations (template, feature-based, natural language)
- **Evaluation**: Comprehensive metrics (accuracy, diversity, novelty, fairness)
- **Cross-domain**: Multi-content-type support with domain mapping
- **Multi-language**: Cross-lingual recommendations with language detection

## 📁 Project Structure

```
content-recommendation-engine/
├── src/
│   ├── recommenders/           # Core recommendation algorithms
│   │   ├── collaborative.py    # Collaborative filtering
│   │   ├── content_based.py    # Content-based filtering
│   │   ├── knowledge_based.py  # Knowledge-based recommendations
│   │   ├── hybrid.py           # Hybrid ensemble
│   │   └── group_recommender.py # Group recommendations
│   ├── personalization/        # Personalization features
│   │   ├── short_term.py       # Short-term preferences
│   │   ├── long_term.py        # Long-term modeling
│   │   ├── sequential.py       # Sequential patterns
│   │   └── drift_detection.py  # Preference drift
│   ├── api/                    # API layer
│   │   ├── schemas.py          # Pydantic models
│   │   └── endpoints.py        # FastAPI endpoints
│   ├── content_understanding/  # Content processing
│   │   ├── gemini_client.py    # Gemini integration
│   │   ├── embedding_generator.py # Embeddings
│   │   └── cross_domain_mapper.py # Cross-domain
│   ├── user_profiling/         # User modeling
│   │   ├── behavior_analyzer.py # Behavior analysis
│   │   └── preference_tracker.py # Preference tracking
│   ├── data_integration/       # Data processing
│   │   ├── data_loader.py      # Data loading
│   │   └── preprocessor.py     # Data preprocessing
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration management
│       ├── scorer.py           # Evaluation metrics
│       ├── explainer.py        # Explanations
│       ├── logging.py          # Logging setup
│       └── validation.py       # Data validation
├── tests/                      # Test suite
│   └── test_recommenders.py    # Comprehensive tests
├── data/                       # Data storage
├── logs/                       # Application logs
├── models/                     # Saved models
├── api_server.py              # Main API server
├── main.py                    # CLI interface
├── example_usage.py           # Usage examples
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA for GPU acceleration

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd content-recommendation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (especially GEMINI_API_KEY)

# Initialize data directories
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data', 'logs', 'models', 'cache']]"
```

### Configuration
Edit `src/utils/config.py` or create a `config/config.yaml` file:

```yaml
hybrid:
  collaborative_weight: 0.4
  content_weight: 0.3
  knowledge_weight: 0.3
  auto_tune_weights: true

api:
  host: "0.0.0.0"
  port: 8000
  rate_limit_per_minute: 60

database:
  sqlite_path: "data/recommendation_engine.db"
```

## 🚀 Quick Start

### 1. Run the Example Demo
```bash
python example_usage.py
```

This demonstrates all features with sample data.

### 2. Start the API Server
```bash
python api_server.py
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### 3. Basic Usage

#### Individual Recommendations
```python
from src.recommenders.hybrid import HybridRecommender

# Initialize recommender
recommender = HybridRecommender()

# Fit with your data
recommender.fit(interactions, items, user_profiles)

# Get recommendations
recommendations = recommender.recommend_for_user('user123', k=10)
```

#### Group Recommendations
```python
from src.recommenders.group_recommender import GroupRecommender

group_recommender = GroupRecommender()
group_recommendations = group_recommender.recommend_for_group(
    group_members, items, k=10, aggregation_method='fairness'
)
```

#### Real-time Personalization
```python
from src.personalization.short_term import ShortTermPersonalizer

personalizer = ShortTermPersonalizer()
personalizer.update_user_profile(user_id, new_interactions)
current_preferences = personalizer.get_current_preferences(user_id)
```

## 📊 API Endpoints

### Core Recommendations
- `POST /recommendations/individual` - Individual user recommendations
- `POST /recommendations/group` - Group recommendations
- `POST /recommendations/similar` - Similar item recommendations
- `POST /recommendations/trending` - Trending content

### User Management
- `POST /users/profile` - Update user profile
- `GET /users/{user_id}/preferences` - Get user preferences
- `POST /users/feedback` - Submit user feedback

### Analytics
- `GET /analytics/metrics` - System performance metrics
- `GET /analytics/user/{user_id}/insights` - User-specific insights
- `GET /analytics/drift/{user_id}` - Preference drift analysis

### System
- `GET /health` - Health check
- `GET /status` - System status
- `POST /admin/retrain` - Trigger model retraining

## 🔧 Configuration Options

### Algorithm Configuration
```python
from src.utils.config import get_config, update_config

# Get current configuration
config = get_config()

# Update algorithm weights
update_config({
    'hybrid': {
        'collaborative_weight': 0.5,
        'content_weight': 0.3,
        'knowledge_weight': 0.2
    }
})
```

### Feature Flags
```python
# Enable/disable features
config.enable_explanations = True
config.enable_real_time_updates = True
config.enable_cross_domain = True
```

## 📈 Evaluation and Metrics

The system provides comprehensive evaluation metrics:

### Accuracy Metrics
- Precision@K, Recall@K, F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
- Mean Reciprocal Rank (MRR)

### Beyond-Accuracy Metrics
- **Diversity**: Intra-list diversity, genre coverage
- **Novelty**: Long-tail item coverage, popularity bias
- **Serendipity**: Unexpectedness while maintaining relevance
- **Fairness**: Demographic parity, equal opportunity

### Usage
```python
from src.utils.scorer import RecommendationScorer

scorer = RecommendationScorer()
metrics = scorer.compute_comprehensive_score(
    recommendations, ground_truth, user_profile, item_features
)
```

## 🎨 Personalization Features

### Short-term Personalization
- Session-based preference tracking
- Recency-weighted recommendations
- Context-aware suggestions (time, device, location)

### Long-term Personalization
- Historical preference modeling
- User lifecycle tracking
- Preference stability analysis

### Sequential Pattern Mining
- User behavior sequence analysis
- Next-item prediction
- Temporal pattern discovery

### Preference Drift Detection
- Automatic detection of changing preferences
- Adaptation triggers for model updates
- Drift analysis and reporting

## 👥 Group Recommendations

### Aggregation Strategies
1. **Average**: Average member preferences
2. **Least Misery**: Maximize minimum satisfaction
3. **Most Pleasure**: Ensure at least one member loves it
4. **Fairness**: Balance all members' preferences

### Usage
```python
group_members = [
    {'user_id': 'user1', 'preferred_genres': ['Action', 'Sci-Fi']},
    {'user_id': 'user2', 'preferred_genres': ['Comedy', 'Romance']},
    {'user_id': 'user3', 'preferred_genres': ['Drama', 'Thriller']}
]

recommendations = group_recommender.recommend_for_group(
    group_members, items, k=10, aggregation_method='fairness'
)
```

## 💡 Explanations

The system provides multiple types of explanations:

### Template-based
```
"Because you liked Action movies like 'The Matrix' and prefer highly-rated content"
```

### Feature-based
```
Matches your preferences for:
• Genres: Action (0.9 similarity)
• Director: Christopher Nolan (you rated 3 of his movies highly)
• Year: 2023 (matches your preference for recent releases)
```

### Natural Language (Gemini-powered)
```
"Since you've consistently enjoyed Christopher Nolan's thought-provoking sci-fi films 
and rated action movies highly, this latest release combines the best elements you love 
with cutting-edge cinematography."
```

## 🔄 Real-time Features

### Live Feedback Integration
```python
# Submit real-time feedback
await feedback_endpoint({
    'user_id': 'user123',
    'item_id': 'movie456',
    'feedback_type': 'rating',
    'value': 4.5,
    'context': {'device': 'mobile', 'time': 'evening'}
})
```

### Streaming Updates
The system supports real-time preference updates and immediate recommendation adaptation.

## 🌐 Cross-domain Support

Support for multiple content types:
- Movies and TV shows
- Books and articles
- Music and podcasts
- Products and services

### Cross-domain Mapping
```python
# Map preferences across domains
movie_preferences = ['Action', 'Sci-Fi']
book_preferences = cross_domain_mapper.map_preferences(
    movie_preferences, source_domain='movies', target_domain='books'
)
# Result: ['Thriller', 'Science Fiction']
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_recommenders.py::TestCollaborativeRecommender -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories
- Unit tests for all algorithms
- Integration tests for complete workflows
- Performance tests for scalability
- API endpoint tests

## 📊 Performance Optimization

### Caching
- Embedding caching for content features
- User profile caching
- Recommendation caching with TTL

### Batch Processing
- Batch recommendation generation
- Parallel model training
- Efficient matrix operations

### Memory Management
- Incremental model updates
- Memory-mapped data loading
- Garbage collection optimization

## 🔒 Security and Privacy

### Data Protection
- User data anonymization
- Secure API authentication
- Privacy-preserving recommendations

### Rate Limiting
- API rate limiting per user
- Request throttling
- DDoS protection

## 📝 Logging and Monitoring

### Structured Logging
```python
# Logs are structured for easy analysis
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "component": "hybrid_recommender",
    "user_id": "user123",
    "action": "generate_recommendations",
    "latency_ms": 45,
    "recommendation_count": 10
}
```

### Monitoring Metrics
- Recommendation latency
- Model performance drift
- User engagement metrics
- System resource usage

## 🔧 Troubleshooting

### Common Issues

1. **Gemini API Key Missing**
   ```
   Error: GEMINI_API_KEY environment variable is required
   Solution: Set GEMINI_API_KEY in your .env file
   ```

2. **Memory Issues with Large Datasets**
   ```
   Solution: Enable batch processing and adjust batch_size in config
   ```

3. **Slow Recommendations**
   ```
   Solution: Enable caching and reduce embedding dimensions
   ```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
python api_server.py
```

## 📚 Advanced Usage

### Custom Algorithm Implementation
```python
class CustomRecommender(BaseRecommender):
    def fit(self, interactions, items, user_profiles):
        # Your custom training logic
        pass
    
    def recommend_for_user(self, user_id, k=10):
        # Your custom recommendation logic
        return recommendations

# Register with hybrid recommender
hybrid.register_algorithm('custom', CustomRecommender())
```

### Custom Evaluation Metrics
```python
def custom_metric(recommendations, ground_truth):
    # Your custom metric logic
    return score

scorer.register_metric('custom_metric', custom_metric)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include logs and configuration details

## 🎯 Roadmap

### Phase 1 (Completed)
- ✅ Core recommendation algorithms
- ✅ Hybrid ensemble system
- ✅ RESTful API
- ✅ Basic personalization

### Phase 2 (Current)
- ✅ Advanced personalization
- ✅ Group recommendations
- ✅ Real-time adaptation
- ✅ Comprehensive explanations

### Phase 3 (Future)
- 🔄 Deep learning models
- 🔄 Federated learning
- 🔄 Advanced A/B testing
- 🔄 Graph-based recommendations

---

**Happy Recommending! 🎉**
