# NL2Q-Analyst-V2 🚀

## Next-Generation Natural Language to Query Platform

**NL2Q-Analyst-V2** is the evolution of natural language database querying, featuring advanced AI integration, microservices architecture, and enterprise-grade capabilities.

## 🆕 What's New in Version 2

### Multi-LLM Support
- **OpenAI GPT-4/4-Turbo**: Industry-leading performance
- **Anthropic Claude**: Enhanced reasoning capabilities  
- **Google Gemini**: Cost-effective scalability
- **Automatic Model Selection**: Best model for each query type

### Enhanced Architecture
- **Microservices Design**: Scalable, maintainable components
- **Event-Driven Communication**: Real-time updates and notifications
- **Advanced Caching**: Redis-powered query and schema caching
- **Streaming Responses**: Real-time query execution feedback

### Enterprise Features
- **Role-Based Access Control (RBAC)**: Granular permissions
- **Multi-Tenant Support**: Isolated workspaces
- **Advanced Security**: OAuth2, SSO, API key management
- **Audit & Compliance**: Comprehensive activity logging

### Extended Database Support
- **Traditional**: PostgreSQL, MySQL, Snowflake, SQL Server
- **Cloud Data Warehouses**: BigQuery, Redshift, Databricks
- **NoSQL**: MongoDB, Elasticsearch (read operations)
- **Modern**: ClickHouse, DuckDB

### Advanced Analytics
- **ML-Powered Query Suggestions**: Learn from usage patterns
- **Predictive Analytics**: Trend analysis and forecasting
- **Advanced Visualizations**: Interactive dashboards
- **Real-time Collaboration**: Multi-user query sessions

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Gateway   │    │   Auth Service  │
│   (React/TS)    │◄──►│   (FastAPI)     │◄──►│   (OAuth2)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼────────┐ ┌────▼─────────┐ ┌──▼──────────────┐
        │  Query Service │ │ Vector Store │ │ Analytics Engine │
        │  (Multi-LLM)   │ │ (Pinecone)   │ │ (ML Pipeline)    │
        └────────────────┘ └──────────────┘ └──────────────────┘
                │               │               │
        ┌───────▼────────┐ ┌────▼─────────┐ ┌──▼──────────────┐
        │ Database Proxy │ │ Cache Layer  │ │ Event Bus       │
        │ (Multi-DB)     │ │ (Redis)      │ │ (Message Queue) │
        └────────────────┘ └──────────────┘ └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Redis (for caching)

### Development Setup
```bash
# Clone repository
git clone https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2.git
cd NL2Q-Analyst-V2

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

# Environment configuration
cp .env.example .env
# Edit .env with your configuration

# Start development servers
docker-compose up -d  # Infrastructure (Redis, DBs)
cd backend && uvicorn main:app --reload
cd ../frontend && npm start
```

## 🔧 Configuration

### Environment Variables
```env
# AI/LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
GOOGLE_API_KEY=your_gemini_key
DEFAULT_LLM_PROVIDER=openai

# Database Connections
DATABASE_URL=postgresql://user:pass@localhost:5432/db
SNOWFLAKE_ACCOUNT=your_account
MONGODB_URI=mongodb://localhost:27017/nl2q

# Authentication
JWT_SECRET_KEY=your_jwt_secret
OAUTH_PROVIDERS=google,github,azure

# Infrastructure
REDIS_URL=redis://localhost:6379
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=nl2q-schema-v2

# Feature Flags
ENABLE_ML_SUGGESTIONS=true
ENABLE_REAL_TIME_COLLAB=true
ENABLE_MULTI_TENANT=true
```

## 📊 API Endpoints

### Core Query API
- `POST /api/v2/query` - Execute natural language query
- `GET /api/v2/query/{query_id}` - Get query status/results
- `POST /api/v2/query/{query_id}/optimize` - Optimize query performance

### Multi-LLM Management
- `GET /api/v2/llm/providers` - List available LLM providers
- `POST /api/v2/llm/select` - Select optimal LLM for query
- `GET /api/v2/llm/usage` - LLM usage analytics

### Advanced Analytics
- `POST /api/v2/analytics/insights` - Generate AI insights
- `GET /api/v2/analytics/suggestions` - Get ML-powered suggestions
- `POST /api/v2/analytics/predict` - Predictive analytics

### Enterprise Features
- `GET /api/v2/tenants` - List tenant workspaces
- `POST /api/v2/rbac/roles` - Manage user roles
- `GET /api/v2/audit/logs` - Access audit trail

## 🎯 Key Features

### Smart Query Processing
```python
# Multi-LLM query processing with automatic optimization
result = await query_service.execute(
    natural_language="Show me sales trends by region",
    llm_strategy="auto",  # Selects best LLM automatically
    optimization_level="aggressive",
    cache_strategy="intelligent"
)
```

### Real-time Collaboration
```javascript
// Real-time query collaboration
const session = new CollaborationSession({
    workspace: 'team-analytics',
    query_id: 'sales-analysis-001'
});

session.onUserJoined((user) => {
    console.log(`${user.name} joined the session`);
});
```

### ML-Powered Suggestions
```python
# Get intelligent query suggestions
suggestions = await ml_service.get_suggestions(
    context="sales data analysis",
    user_history=user.query_history,
    team_patterns=team.common_queries
)
```

## 📈 Performance & Scalability

- **Query Caching**: Intelligent caching reduces response time by up to 90%
- **Streaming Results**: Real-time result streaming for large datasets
- **Horizontal Scaling**: Microservices architecture supports auto-scaling
- **Load Balancing**: Built-in load balancing for high availability

## 🔐 Security & Compliance

- **Zero-Trust Architecture**: Every request is authenticated and authorized
- **Data Encryption**: End-to-end encryption for sensitive data
- **Audit Logging**: Comprehensive activity tracking
- **Compliance**: GDPR, HIPAA, SOX compliance features

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=src

# Frontend tests
cd frontend
npm test -- --coverage

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📖 Documentation

- [API Documentation](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Deployment Guide](./docs/deployment.md)
- [Developer Guide](./docs/development.md)
- [Migration from V1](./docs/migration.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🆙 Migration from V1

Upgrading from NL2Q-Analyst V1? Check our [migration guide](./docs/migration.md) for step-by-step instructions.

---

**Ready to revolutionize your data querying experience? Get started with NL2Q-Analyst-V2 today!** 🚀