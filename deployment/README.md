# NL2Q Analyst V2 - Deployment Guide

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Local Development Setup](#local-development-setup)
- [Docker Deployment](#docker-deployment)
- [Azure Deployment](#azure-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### For Team Members (Local Development)

```powershell
# 1. Clone the repository
git clone https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2.git
cd NL2Q-Analyst-V2

# 2. Run setup script
.\deployment\scripts\setup-local.ps1

# 3. Edit .env with your credentials
notepad .env

# 4. Start the application
.\start_app.ps1  # Backend
cd frontend; npm start  # Frontend (in separate terminal)
```

### For Quick Testing (Docker)

```bash
# 1. Clone and navigate
git clone https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2.git
cd NL2Q-Analyst-V2

# 2. Copy and edit environment file
cp deployment/.env.example deployment/.env
nano deployment/.env  # Edit with your credentials

# 3. Start with Docker Compose
cd deployment/docker
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

---

## 🛠️ Local Development Setup

### Prerequisites

- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **Git** - [Download](https://git-scm.com/)

### Step-by-Step Setup

#### 1. Clone Repository

```powershell
git clone https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2.git
cd NL2Q-Analyst-V2
```

#### 2. Backend Setup

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### 3. Frontend Setup

```powershell
cd frontend
npm install
cd ..
```

#### 4. Environment Configuration

```powershell
# Copy template
cp deployment\.env.example .env

# Edit with your credentials (see Environment Variables section)
notepad .env
```

#### 5. Database Initialization (Optional)

```powershell
python init_db.py
```

#### 6. Start Application

**Backend:**
```powershell
.\start_app.ps1
# OR manually:
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (separate terminal):**
```powershell
cd frontend
npm start
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment

### Prerequisites

- **Docker** - [Download](https://www.docker.com/get-started)
- **Docker Compose** - Usually included with Docker Desktop

### Quick Start

```bash
# Navigate to project
cd NL2Q-Analyst-V2

# Copy and configure environment
cp deployment/.env.example deployment/.env
nano deployment/.env  # Edit credentials

# Start with Docker Compose
cd deployment/docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Docker Commands

```bash
# Build images
docker-compose build

# Start containers (detached)
docker-compose up -d

# Start containers (with logs)
docker-compose up

# Stop containers
docker-compose down

# Restart specific service
docker-compose restart backend

# View logs
docker-compose logs -f [service-name]

# Execute command in container
docker-compose exec backend bash
```

### Docker Configuration

The deployment includes:
- `Dockerfile.backend` - Python FastAPI backend with ODBC drivers
- `Dockerfile.frontend` - React app with nginx web server
- `docker-compose.yml` - Orchestrates both services
- `nginx.conf` - Nginx configuration for frontend

---

## ☁️ Azure Deployment

### Option 1: Azure App Service (Recommended)

#### Automated Deployment

```powershell
# Run deployment script
.\deployment\azure\deploy-app-service.ps1

# Follow prompts to enter:
# - Resource Group name
# - Location (e.g., eastus)
# - App Service Plan name
# - Backend/Frontend app names
# - Environment variables
```

#### Manual Deployment

```bash
# 1. Login to Azure
az login

# 2. Create Resource Group
az group create --name nl2q-analyst-rg --location eastus

# 3. Create App Service Plan
az appservice plan create \
  --name nl2q-analyst-plan \
  --resource-group nl2q-analyst-rg \
  --sku B2 \
  --is-linux

# 4. Create Backend Web App
az webapp create \
  --resource-group nl2q-analyst-rg \
  --plan nl2q-analyst-plan \
  --name nl2q-analyst-backend \
  --runtime "PYTHON:3.9"

# 5. Configure App Settings
az webapp config appsettings set \
  --resource-group nl2q-analyst-rg \
  --name nl2q-analyst-backend \
  --settings \
    DB_HOST="your-db-host" \
    DB_USER="your-db-user" \
    DB_PASSWORD="your-db-password" \
    OPENAI_API_KEY="your-openai-key" \
    PINECONE_API_KEY="your-pinecone-key"

# 6. Deploy Code
az webapp up \
  --name nl2q-analyst-backend \
  --resource-group nl2q-analyst-rg
```

### Option 2: Azure Container Instances

```bash
# 1. Build and push Docker images to Azure Container Registry
az acr create --resource-group nl2q-analyst-rg --name nl2qanalyst --sku Basic
az acr login --name nl2qanalyst

docker build -t nl2qanalyst.azurecr.io/backend:latest -f deployment/docker/Dockerfile.backend .
docker build -t nl2qanalyst.azurecr.io/frontend:latest -f deployment/docker/Dockerfile.frontend .

docker push nl2qanalyst.azurecr.io/backend:latest
docker push nl2qanalyst.azurecr.io/frontend:latest

# 2. Create Container Instances
az container create \
  --resource-group nl2q-analyst-rg \
  --name nl2q-backend \
  --image nl2qanalyst.azurecr.io/backend:latest \
  --cpu 1 \
  --memory 2 \
  --registry-login-server nl2qanalyst.azurecr.io \
  --ip-address public \
  --ports 8000 \
  --environment-variables \
    DB_HOST="your-db-host" \
    DB_USER="your-db-user"
```

### Option 3: Azure Static Web Apps + App Service

**Frontend (Static Web App):**
```bash
az staticwebapp create \
  --name nl2q-frontend \
  --resource-group nl2q-analyst-rg \
  --source https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2 \
  --location eastus \
  --branch main \
  --app-location "/frontend" \
  --output-location "build" \
  --login-with-github
```

**Backend (App Service):**
Use the App Service steps above for the backend.

---

## 🔐 Environment Variables

### Required Variables

```bash
# Database Configuration
DB_HOST=odsproduction.database.windows.net
DB_PORT=1433
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=DWHPRODIBSA

# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000  # Local
# REACT_APP_API_URL=https://your-backend.azurewebsites.net  # Production
```

### Optional Variables

```bash
# Application Settings
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Azure-specific
ENABLE_ORYX_BUILD=true
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

### Where to Set Variables

**Local Development:**
- Create `.env` file in project root
- Use `deployment/.env.example` as template

**Docker:**
- Edit `deployment/.env` file
- Docker Compose will load automatically

**Azure App Service:**
- Azure Portal → Configuration → Application Settings
- OR use `az webapp config appsettings set` command

**Azure Container Instances:**
- Use `--environment-variables` or `--secure-environment-variables` in `az container create`

---

## 🔧 Troubleshooting

### Backend Issues

**Problem: Database connection fails**
```bash
# Check connection string
python -c "from backend.db.engine import test_connection; test_connection()"

# Verify firewall rules (Azure SQL)
# Add your IP in Azure Portal → SQL Server → Firewalls and virtual networks
```

**Problem: Module not found**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"
```

**Problem: Port 8000 already in use**
```bash
# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac:
lsof -i :8000
kill -9 <pid>
```

### Frontend Issues

**Problem: API calls fail (CORS)**
```javascript
// Backend: backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Problem: Build fails**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+
```

### Docker Issues

**Problem: Container fails to start**
```bash
# View logs
docker-compose logs backend
docker-compose logs frontend

# Check container status
docker-compose ps

# Restart containers
docker-compose restart
```

**Problem: Environment variables not loaded**
```bash
# Verify .env file location
ls deployment/.env

# Check if Docker Compose loads it
docker-compose config
```

### Azure Issues

**Problem: Deployment fails**
```bash
# Check deployment logs
az webapp log tail --name nl2q-analyst-backend --resource-group nl2q-analyst-rg

# Check application logs in Azure Portal
# Your App → Monitoring → Log stream
```

**Problem: App doesn't start**
```bash
# Check startup command
az webapp config show --name nl2q-analyst-backend --resource-group nl2q-analyst-rg

# Set correct startup command
az webapp config set \
  --name nl2q-analyst-backend \
  --resource-group nl2q-analyst-rg \
  --startup-file "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
```

---

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review application logs
3. Check GitHub Issues: https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2/issues
4. Contact the development team

---

## 📝 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Docker Documentation](https://docs.docker.com/)
