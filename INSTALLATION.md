# NL2Q Analyst - Installation Guide

## Prerequisites

- **Python 3.9+** (Recommended: Python 3.11)
- **Node.js 16+** (for frontend)
- **Git**
- **Visual C++ Build Tools** (Windows only)

## Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hellosandeeptiwari/NL2Q-Analyst.git
cd NL2Q-Analyst
```

### 2. Backend Setup

#### Option A: Standard Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Development Installation
```bash
# For development with all tools
pip install -r requirements-dev.txt
```

#### Option C: Docker Installation
```bash
# For production/container deployment
pip install -r requirements-docker.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Environment Configuration

Create `.env` file in the root directory:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# Database Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# Vector Database
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Azure (Optional)
AZURE_SEARCH_SERVICE_NAME=your_service
AZURE_SEARCH_API_KEY=your_key

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

### 5. Start the Application

#### Backend
```bash
# From root directory
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend
```bash
# From frontend directory
npm start
```

## Package Versions Summary

| Category | Key Packages | Version |
|----------|-------------|---------|
| **Core Framework** | FastAPI, Uvicorn | 0.109.2, 0.27.0 |
| **Database** | psycopg2, snowflake-connector | 2.9.9, 3.7.1 |
| **AI/ML** | OpenAI, Pinecone | 1.12.0+, 3.0.3 |
| **Data Science** | Pandas, NumPy, Plotly | 2.2.0, 1.26.3, 5.18.0 |
| **Visualization** | Matplotlib, Seaborn | 3.8.2, 0.13.2 |

## Troubleshooting

### Common Issues:

1. **Snowflake Connection Issues (Windows)**
   ```bash
   # Install build tools
   pip install --upgrade setuptools wheel
   pip install snowflake-connector-python --no-cache-dir
   ```

2. **Pinecone Import Errors**
   ```bash
   pip install pinecone-client==3.0.3 --force-reinstall
   ```

3. **React/Frontend Issues**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Authentication Issues**
   - Verify `.env` file configuration
   - Check API key validity
   - Ensure proper permissions

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Considerations

- Use `requirements-docker.txt` for minimal production deployment
- Set up proper environment variables
- Configure reverse proxy (nginx) 
- Enable SSL/TLS
- Set up monitoring and logging
- Use container orchestration for scaling

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review environment configuration
3. Create an issue on GitHub

## Latest Updates

- ✅ Enhanced Plotly chart visualization  
- ✅ Fixed JSON serialization for date objects
- ✅ Improved schema-aware planning
- ✅ Enhanced security with safe parsing
- ✅ TypeScript compliance improvements
