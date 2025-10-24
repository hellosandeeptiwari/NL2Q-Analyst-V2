# Frontend Deployment Guide

## Quick Deploy to Azure Static Web Apps

### Option 1: Using Azure Portal (Recommended)

1. **Go to Azure Portal**: https://portal.azure.com
2. **Create Static Web App**:
   - Click "Create a resource"
   - Search for "Static Web App"
   - Click "Create"

3. **Configure**:
   - **Subscription**: Same as backend
   - **Resource Group**: DSLAI (same as backend)
   - **Name**: `nl2q-analyst-frontend`
   - **Region**: East US (same as backend)
   - **Source**: GitHub
   - **Organization**: hellosandeeptiwari
   - **Repository**: NL2Q-Analyst-V2
   - **Branch**: main
   - **Build Presets**: React
   - **App location**: `/frontend`
   - **Output location**: `build`

4. **Click**: Review + Create → Create

5. **Wait**: 2-3 minutes for deployment

### Option 2: Manual Build and Deploy

If you prefer to build locally first:

```powershell
# Navigate to frontend
cd frontend

# Install dependencies (if not already done)
npm install

# Build for production
npm run build

# The build folder will be created with optimized production build
```

Then deploy the `build` folder to Azure Static Web App or any static hosting service.

## Testing Backend API

Your backend is deployed at:
- **API**: https://l2q-analyst-backend-ayffadegfschjcs.eastus-01.azurewebsites.net
- **Docs**: https://l2q-analyst-backend-ayffadegfschjcs.eastus-01.azurewebsites.net/docs

Test endpoints:
- GET `/health` - Health check
- GET `/docs` - API documentation
- POST `/api/agent/query` - Main query endpoint

## Environment Configuration

The frontend is configured to use:
- **Development** (.env.development): http://localhost:8000
- **Production** (.env.production): Your Azure backend URL

This is automatically selected based on build mode.
