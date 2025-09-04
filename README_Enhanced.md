# 🏥 Enhanced Pharma NL2Q Analytics Platform v2.0

An advanced Natural Language to Query (NL2Q) analytics platform specifically designed for pharmaceutical companies, featuring the latest agentic AI approach, comprehensive user management, and Claude Sonnet-inspired interface.

![Platform Preview](docs/platform-preview.png)

## 🚀 Key Features

### 🤖 Latest Agentic AI Approach
- **Advanced Reasoning**: o3-mini model for complex pharmaceutical analysis
- **Multi-step Planning**: Intelligent query orchestration with validation
- **Context Awareness**: Conversation history and therapeutic area context
- **Adaptive Execution**: Self-correcting queries with real-time optimization

### 👥 Comprehensive User Management
- **Role-based Access Control**: Analyst, Data Scientist, Medical Affairs, Commercial, Regulatory, Executive
- **User Profiles**: Personalized preferences, therapeutic area specializations
- **Permission Management**: Granular data access controls
- **Usage Analytics**: Query tracking, cost monitoring, performance metrics

### 💬 Enhanced Chat Interface (Claude Sonnet-inspired)
- **Conversation History**: Persistent chat sessions with search
- **Real-time Collaboration**: Multi-user sessions and sharing
- **Rich Visualizations**: Interactive charts and pharmaceutical metrics
- **Plan Execution Tracking**: Visual progress of AI agent reasoning

### 🔒 Pharma-specific Compliance & Governance
- **Data Privacy**: Automatic PHI/PII detection and masking
- **Regulatory Compliance**: HIPAA, FDA, EMA guidelines adherence
- **Audit Trail**: Comprehensive logging for regulatory requirements
- **Approval Workflows**: Sensitive query review processes

### 📊 Advanced Analytics Capabilities
- **Therapeutic Area Intelligence**: Specialized analysis for Oncology, Diabetes, etc.
- **Clinical Trial Analytics**: Patient outcomes, adverse events, efficacy metrics
- **Commercial Intelligence**: Market share, prescriber behavior, competitive analysis
- **Real-world Evidence**: Healthcare utilization, patient journey analytics

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ Enhanced UI     │ │ Chat History    │ │ User Profile │ │
│  │ Components      │ │ Management      │ │ Management   │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ REST API
                               │
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ Enhanced        │ │ User Profile    │ │ Chat History │ │
│  │ Orchestrator    │ │ Manager         │ │ Manager      │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ Compliance      │ │ Schema Tools    │ │ Visualization│ │
│  │ Validator       │ │ & SQL Runner    │ │ Builder      │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               │
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ Snowflake       │ │ SQLite          │ │ File Storage │ │
│  │ (Pharma Data)   │ │ (Chat/Users)    │ │ (Exports)    │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Access to Snowflake (or other supported database)
- OpenAI API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd "NL2Q Agent"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Environment Configuration
Create a `.env` file with your configuration:
```env
# AI Models
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
REASONING_MODEL=o3-mini
REASONING_MODEL_TEMPERATURE=0.1
USE_REASONING_FOR_PLANNING=true

# Database (Snowflake)
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=COMMERCIAL_AI
SNOWFLAKE_SCHEMA=ENHANCED_NBA

# Security
AUTH_TOKEN=your_secure_token
```

### 3. Start the Enhanced Platform
```bash
# Option 1: Use the enhanced startup script
python start_enhanced_platform.py

# Option 2: Manual startup
# Terminal 1 - Backend
python -m uvicorn backend.enhanced_main:enhanced_app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend && npm start
```

### 4. Access the Platform
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 👤 User Roles & Permissions

### Role Definitions
| Role | Description | Default Permissions |
|------|-------------|-------------------|
| **Analyst** | Data analysts and researchers | ENHANCED_NBA.* tables |
| **Data Scientist** | Advanced analytics and modeling | Advanced access + ML tools |
| **Medical Affairs** | Clinical and medical data analysis | CLINICAL.*, MEDICAL.* tables |
| **Commercial** | Sales and marketing analytics | SALES.*, MARKETING.* tables |
| **Regulatory** | Compliance and regulatory reporting | REGULATORY.*, COMPLIANCE.* |
| **Executive** | High-level reporting and dashboards | Aggregated views, no PHI |
| **Admin** | System administration | Full access to all data |

### Demo Users
The system creates demo users for testing:
- **analyst1** (Sarah Chen) - Commercial Analytics, Oncology/Diabetes
- **medaffairs1** (Dr. Michael Roberts) - Medical Affairs, Oncology/Immunology  
- **datascientist1** (Alex Kumar) - Data Science, All therapeutic areas

## 🧠 Agentic AI Features

### Enhanced Query Processing
1. **Intent Understanding**: Natural language processing with pharma context
2. **Plan Generation**: Multi-step reasoning with o3-mini model
3. **Schema Discovery**: Intelligent table and column identification
4. **Compliance Validation**: Automatic PHI/PII detection and governance
5. **SQL Optimization**: Performance-optimized queries for large datasets
6. **Result Visualization**: Therapeutic area-specific charts and insights

### Reasoning Models
- **GPT-4o-mini**: Fast SQL generation and data grounding
- **o3-mini**: Complex reasoning and multi-step planning
- **Text-embedding-3-large**: High-quality semantic matching

### Example Queries
```
"Show Q4 oncology prescribing trends by physician specialty"
→ Multi-table join, time-series analysis, specialty segmentation

"Compare diabetes drug adherence rates across age groups" 
→ Cohort analysis, adherence calculations, demographic breakdowns

"Analyze adverse events for immunology products in clinical trials"
→ Safety analysis, regulatory compliance, statistical significance
```

## 💬 Chat Interface Features

### Conversation Management
- **Persistent History**: All conversations saved with context
- **Search & Filter**: Find previous analyses quickly
- **Favorites**: Bookmark important conversations
- **Sharing**: Collaborate with team members
- **Export**: Download results in multiple formats

### Real-time Features
- **Plan Execution Tracking**: Watch AI agent progress
- **Cost Monitoring**: Track API usage and costs
- **Performance Metrics**: Query execution time and optimization
- **Error Handling**: Graceful failure recovery with suggestions

### Therapeutic Area Context
- **Auto-detection**: Identify relevant therapeutic areas from queries
- **Specialized Prompts**: Tailored reasoning for different disease areas
- **Compliance Rules**: Area-specific governance and validation
- **Business Metrics**: Relevant KPIs and pharmaceutical measures

## 🔒 Security & Compliance

### Data Privacy
- **PHI Detection**: Pattern matching for sensitive patient data
- **Automatic Masking**: Anonymization of sensitive fields
- **Access Controls**: Role-based data permissions
- **Audit Logging**: Complete trail for regulatory compliance

### Regulatory Features
- **HIPAA Compliance**: Healthcare data protection standards
- **FDA Guidelines**: Clinical trial data handling
- **EMA Requirements**: European regulatory compliance
- **GxP Standards**: Good practice guidelines for pharma

### Governance Controls
- **Approval Workflows**: Sensitive queries require management approval
- **Data Lineage**: Track data sources and transformations
- **Version Control**: Query history and change tracking
- **Risk Assessment**: Automatic classification of query sensitivity

## 📊 Analytics Capabilities

### Commercial Analytics
- Market share analysis and competitive intelligence
- Prescriber behavior and preference analysis
- Brand performance and promotional effectiveness
- Customer segmentation and targeting
- Sales forecasting and territory optimization

### Clinical Analytics
- Patient outcomes and real-world evidence
- Adverse event monitoring and safety signals
- Clinical trial performance and enrollment
- Treatment pathways and care patterns
- Healthcare resource utilization

### Regulatory Analytics
- Compliance reporting and monitoring
- Regulatory submission support
- Post-market surveillance
- Risk-benefit assessment
- Pharmacovigilance analytics

## 🛠️ Development & Customization

### Adding New Therapeutic Areas
1. Update `TherapeuticContext` enum in `enhanced_orchestrator.py`
2. Add specialized reasoning prompts
3. Configure compliance rules and data permissions
4. Update visualization templates

### Custom Compliance Rules
```python
# In enhanced_orchestrator.py
self.compliance_rules = {
    "phi_patterns": [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"custom_pattern_here"      # Add your patterns
    ],
    "restricted_tables": [
        "your_sensitive_table"
    ]
}
```

### API Extensions
The platform uses FastAPI with automatic OpenAPI documentation. Add endpoints in `enhanced_main.py`:

```python
@app.get("/api/custom/endpoint")
async def custom_endpoint():
    return {"message": "Custom functionality"}
```

## 📈 Performance & Scalability

### Optimization Features
- **Query Caching**: Intelligent result caching for repeated queries
- **Schema Optimization**: Fast metadata retrieval and caching
- **Async Processing**: Non-blocking query execution
- **Connection Pooling**: Efficient database connection management

### Scaling Considerations
- **Horizontal Scaling**: Multiple backend instances with load balancing
- **Database Optimization**: Proper indexing and partitioning
- **Caching Strategy**: Redis for session and query caching
- **CDN Integration**: Static asset delivery optimization

## 🔧 Configuration Options

### Model Configuration
```env
# Switch between models based on needs
OPENAI_MODEL=gpt-4o-mini          # Fast, cost-effective
REASONING_MODEL=o3-mini           # Advanced reasoning
USE_REASONING_FOR_PLANNING=true   # Enable complex planning
REASONING_MODEL_TEMPERATURE=0.1   # Consistency vs creativity
```

### Database Options
The platform supports multiple database backends:
- **Snowflake**: Primary recommendation for pharma
- **PostgreSQL**: Open-source alternative
- **MySQL**: Legacy system integration
- **BigQuery**: Google Cloud integration

### Compliance Levels
```python
class PharmaComplianceLevel(Enum):
    PUBLIC = "public"        # No restrictions
    INTERNAL = "internal"    # Aggregated data only
    RESTRICTED = "restricted" # Requires approval
    CONFIDENTIAL = "confidential" # Highest security
```

## 📋 API Reference

### Core Endpoints

#### User Management
- `GET /api/user/profile/{user_id}` - Get user profile
- `PUT /api/user/profile/{user_id}/preferences` - Update preferences
- `GET /api/user/profile/{user_id}/analytics` - Usage analytics

#### Chat Management
- `GET /api/chat/conversations/{user_id}` - Get conversations
- `POST /api/chat/conversation` - Create new conversation
- `GET /api/chat/search/{user_id}?q={query}` - Search conversations

#### Agent Operations
- `POST /api/agent/query` - Send query to AI agent
- `GET /api/agent/plan/{plan_id}/status` - Get execution status
- `POST /api/agent/plan/{plan_id}/approve` - Approve sensitive query

#### Pharma-specific
- `GET /api/pharma/therapeutic-areas` - Available therapeutic areas
- `GET /api/pharma/compliance-templates` - Compliance query templates
- `GET /api/pharma/data-sources` - Available data sources

## 🧪 Testing

### Run Tests
```bash
# Backend tests
pytest tests/ -v

# Frontend tests
cd frontend && npm test

# Integration tests
python -m pytest tests/test_integration.py
```

### Test Coverage
- Unit tests for core components
- Integration tests for API endpoints
- End-to-end tests for user workflows
- Performance tests for query execution

## 📚 Documentation

### Additional Resources
- [User Guide](docs/user-guide.md) - Detailed usage instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API explorer
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Security Guide](docs/security.md) - Security best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **Database Connection**: Check Snowflake credentials and network access
- **API Key Issues**: Verify OpenAI API key and billing status
- **Frontend Build**: Ensure Node.js 16+ and run `npm install`
- **Permissions**: Check user roles and data access permissions

### Getting Help
- 📧 Email: support@pharma-nl2q.com
- 💬 Slack: #pharma-analytics-support
- 📖 Documentation: [docs.pharma-nl2q.com](https://docs.pharma-nl2q.com)
- 🐛 Issues: GitHub Issues tab

---

**Built with ❤️ for Pharmaceutical Analytics Teams**

*Empowering data-driven decisions in healthcare and life sciences through advanced AI and natural language processing.*
