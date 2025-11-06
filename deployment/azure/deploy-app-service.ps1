# Azure App Service Deployment Script
# Run this script to deploy to Azure App Service

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "nl2q-analyst-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$AppServicePlan = "nl2q-analyst-plan",
    
    [Parameter(Mandatory=$false)]
    [string]$BackendAppName = "nl2q-analyst-backend",
    
    [Parameter(Mandatory=$false)]
    [string]$FrontendAppName = "nl2q-analyst-frontend"
)

Write-Host "🚀 Starting Azure deployment..." -ForegroundColor Green

# Step 1: Login to Azure
Write-Host "`n📝 Step 1: Logging into Azure..." -ForegroundColor Cyan
az login

# Step 2: Create Resource Group
Write-Host "`n📝 Step 2: Creating resource group..." -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location

# Step 3: Create App Service Plan
Write-Host "`n📝 Step 3: Creating App Service Plan..." -ForegroundColor Cyan
az appservice plan create `
    --name $AppServicePlan `
    --resource-group $ResourceGroup `
    --sku B2 `
    --is-linux

# Step 4: Create Backend Web App
Write-Host "`n📝 Step 4: Creating Backend Web App..." -ForegroundColor Cyan
az webapp create `
    --resource-group $ResourceGroup `
    --plan $AppServicePlan `
    --name $BackendAppName `
    --runtime "PYTHON:3.9"

# Step 5: Configure Backend App Settings
Write-Host "`n📝 Step 5: Configuring Backend App Settings..." -ForegroundColor Cyan
Write-Host "⚠️  Please enter your environment variables:" -ForegroundColor Yellow

$DB_HOST = Read-Host "Enter DB_HOST (default: odsproduction.database.windows.net)"
if ([string]::IsNullOrWhiteSpace($DB_HOST)) { $DB_HOST = "odsproduction.database.windows.net" }

$DB_USER = Read-Host "Enter DB_USER"
$DB_PASSWORD = Read-Host "Enter DB_PASSWORD" -AsSecureString
$DB_PASSWORD_Plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($DB_PASSWORD))

$DB_NAME = Read-Host "Enter DB_NAME (default: DWHPRODIBSA)"
if ([string]::IsNullOrWhiteSpace($DB_NAME)) { $DB_NAME = "DWHPRODIBSA" }

$OPENAI_API_KEY = Read-Host "Enter OPENAI_API_KEY" -AsSecureString
$OPENAI_API_KEY_Plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($OPENAI_API_KEY))

$PINECONE_API_KEY = Read-Host "Enter PINECONE_API_KEY" -AsSecureString
$PINECONE_API_KEY_Plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($PINECONE_API_KEY))

$PINECONE_ENVIRONMENT = Read-Host "Enter PINECONE_ENVIRONMENT"
$PINECONE_INDEX_NAME = Read-Host "Enter PINECONE_INDEX_NAME"

az webapp config appsettings set `
    --resource-group $ResourceGroup `
    --name $BackendAppName `
    --settings `
        DB_HOST="$DB_HOST" `
        DB_PORT="1433" `
        DB_USER="$DB_USER" `
        DB_PASSWORD="$DB_PASSWORD_Plain" `
        DB_NAME="$DB_NAME" `
        OPENAI_API_KEY="$OPENAI_API_KEY_Plain" `
        PINECONE_API_KEY="$PINECONE_API_KEY_Plain" `
        PINECONE_ENVIRONMENT="$PINECONE_ENVIRONMENT" `
        PINECONE_INDEX_NAME="$PINECONE_INDEX_NAME" `
        SCM_DO_BUILD_DURING_DEPLOYMENT="true" `
        ENABLE_ORYX_BUILD="true"

# Step 6: Configure Startup Command
Write-Host "`n📝 Step 6: Configuring startup command..." -ForegroundColor Cyan
az webapp config set `
    --resource-group $ResourceGroup `
    --name $BackendAppName `
    --startup-file "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

# Step 7: Deploy Backend Code
Write-Host "`n📝 Step 7: Deploying backend code..." -ForegroundColor Cyan
az webapp up --name $BackendAppName --resource-group $ResourceGroup

# Step 8: Create Frontend Static Web App
Write-Host "`n📝 Step 8: Creating Frontend Static Web App..." -ForegroundColor Cyan
Write-Host "⚠️  You'll need to authorize GitHub access" -ForegroundColor Yellow
az staticwebapp create `
    --name $FrontendAppName `
    --resource-group $ResourceGroup `
    --source https://github.com/hellosandeeptiwari/NL2Q-Analyst-V2 `
    --location $Location `
    --branch main `
    --app-location "/frontend" `
    --output-location "build" `
    --login-with-github

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "`n📊 Resource URLs:" -ForegroundColor Cyan
Write-Host "Backend: https://$BackendAppName.azurewebsites.net" -ForegroundColor White
Write-Host "Frontend: Check Azure Portal for Static Web App URL" -ForegroundColor White
Write-Host "`n💡 Next steps:" -ForegroundColor Cyan
Write-Host "1. Update frontend environment variables with backend URL" -ForegroundColor White
Write-Host "2. Configure CORS in backend if needed" -ForegroundColor White
Write-Host "3. Set up custom domain (optional)" -ForegroundColor White
Write-Host "4. Enable Application Insights for monitoring" -ForegroundColor White
