# 🚀 NL2Q Analyst V2 - Deployment Package

This folder contains all necessary files and configurations for deploying the NL2Q Analyst V2 application.

## 📁 Folder Structure

```
deployment/
├── README.md                    # Complete deployment guide (YOU ARE HERE)
├── .env.example                 # Environment variables template (local/dev)
├── .env.production.example      # Environment variables template (production)
│
├── azure/                       # Azure deployment configurations
│   └── deploy-app-service.ps1  # Automated Azure deployment script
│
├── docker/                      # Docker deployment files
│   ├── docker-compose.yml       # Docker Compose orchestration
│   ├── Dockerfile.backend       # Backend container configuration
│   ├── Dockerfile.frontend      # Frontend container configuration
│   └── nginx.conf               # Nginx web server configuration
│
└── scripts/                     # Deployment automation scripts
    ├── setup-local.ps1          # Local development setup (Windows)
    └── deploy-docker.sh         # Docker deployment script (Linux/Mac)
```

## ⚡ Quick Start Options

### 1️⃣ For Team Members (Local Development)

```powershell
# Run the automated setup script
.\deployment\scripts\setup-local.ps1
```

This will:
- ✅ Check Python and Node.js installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create .env file from template
- ✅ Guide you through database initialization

### 2️⃣ For Testing (Docker)

```bash
# Copy environment file
cp deployment/.env.example deployment/.env

# Edit with your credentials
nano deployment/.env

# Start with Docker Compose
cd deployment/docker
docker-compose up -d
```

Access at:
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000

### 3️⃣ For Production (Azure)

```powershell
# Run automated Azure deployment
.\deployment\azure\deploy-app-service.ps1
```

This will:
- ✅ Create Azure Resource Group
- ✅ Set up App Service Plan
- ✅ Deploy Backend and Frontend
- ✅ Configure environment variables
- ✅ Provide deployment URLs

## 📚 Documentation

**Main Guide:** See [deployment/README.md](./README.md) for:
- Complete step-by-step setup instructions
- Environment variable configuration
- Troubleshooting guide
- Azure deployment options
- Docker deployment guide

## 🔑 Required Credentials

Before deploying, you'll need:

1. **Database Credentials**
   - Azure SQL Server host, user, password
   - Database name (DWHPRODIBSA)

2. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys

3. **Pinecone API Key**
   - Get from: https://www.pinecone.io/
   - Environment and Index name

## 🎯 Deployment Options Summary

| Option | Best For | Complexity | Cost |
|--------|----------|------------|------|
| **Local Development** | Team members testing/developing | ⭐ Easy | Free |
| **Docker** | Quick testing, consistent environments | ⭐⭐ Medium | Free (local) |
| **Azure App Service** | Production deployment | ⭐⭐⭐ Medium | $$ Pay-as-you-go |
| **Azure Container Instances** | Production with containers | ⭐⭐⭐⭐ Advanced | $$ Pay-as-you-go |
| **Azure Static Web Apps** | Frontend production | ⭐⭐ Easy | $ Free tier available |

## 🔧 Quick Commands Reference

### Local Development
```powershell
# Start backend
.\start_app.ps1

# Start frontend (separate terminal)
cd frontend
npm start
```

### Docker
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up --build -d
```

### Azure
```bash
# View logs
az webapp log tail --name your-app-name --resource-group your-rg

# Restart app
az webapp restart --name your-app-name --resource-group your-rg

# Configure settings
az webapp config appsettings set --name your-app-name --settings KEY=VALUE
```

## ⚠️ Important Notes

1. **Never commit .env files** - They contain sensitive credentials
2. **Use .env.example as template** - Copy and fill with your values
3. **Test locally first** - Ensure everything works before deploying to Azure
4. **Check firewall rules** - Azure SQL requires IP whitelisting
5. **Monitor costs** - Azure services incur charges based on usage

## 🆘 Troubleshooting

**Quick Checks:**
1. Environment variables set correctly? → Check `.env` file
2. Database connection failing? → Verify Azure SQL firewall rules
3. Port conflicts? → Check if 3000/8000 are available
4. Module not found? → Reinstall dependencies (`pip install -r requirements.txt`)

**Detailed troubleshooting:** See [deployment/README.md](./README.md#troubleshooting)

## 📞 Support

- **Full Documentation:** [deployment/README.md](./README.md)
- **GitHub Issues:** https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2/issues
- **Contact:** Development team

---

**Ready to deploy?** Choose your option above and follow the instructions in the [main deployment guide](./README.md)! 🚀
