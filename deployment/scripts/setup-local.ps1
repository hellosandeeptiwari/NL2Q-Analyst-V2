# Local Setup Script for Team Members
# Run this script to set up the project for local development

Write-Host "🚀 NL2Q Analyst V2 - Local Setup" -ForegroundColor Green
Write-Host "================================`n" -ForegroundColor Green

# Step 1: Check Python installation
Write-Host "📝 Step 1: Checking Python installation..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python is not installed. Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green

# Step 2: Check Node.js installation
Write-Host "`n📝 Step 2: Checking Node.js installation..." -ForegroundColor Cyan
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Node.js is not installed. Please install Node.js 16 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Found Node.js: $nodeVersion" -ForegroundColor Green

# Step 3: Create virtual environment
Write-Host "`n📝 Step 3: Creating Python virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Step 4: Activate virtual environment and install backend dependencies
Write-Host "`n📝 Step 4: Installing backend dependencies..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Write-Host "✅ Backend dependencies installed" -ForegroundColor Green

# Step 5: Install frontend dependencies
Write-Host "`n📝 Step 5: Installing frontend dependencies..." -ForegroundColor Cyan
cd frontend
npm install
cd ..
Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green

# Step 6: Setup environment variables
Write-Host "`n📝 Step 6: Setting up environment variables..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item "deployment\.env.example" ".env"
    Write-Host "✅ Created .env file from template" -ForegroundColor Green
    Write-Host "⚠️  IMPORTANT: Edit .env file with your actual credentials!" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  .env file already exists. Skipping..." -ForegroundColor Yellow
}

# Step 7: Initialize database
Write-Host "`n📝 Step 7: Initializing database..." -ForegroundColor Cyan
$initDb = Read-Host "Do you want to initialize the database? (y/n)"
if ($initDb -eq "y" -or $initDb -eq "Y") {
    python init_db.py
    Write-Host "✅ Database initialized" -ForegroundColor Green
} else {
    Write-Host "⚠️  Skipping database initialization" -ForegroundColor Yellow
}

# Step 8: Display next steps
Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`n📚 Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your credentials" -ForegroundColor White
Write-Host "2. Start backend: .\start_app.ps1" -ForegroundColor White
Write-Host "3. Start frontend (in separate terminal):" -ForegroundColor White
Write-Host "   cd frontend" -ForegroundColor Gray
Write-Host "   npm start" -ForegroundColor Gray
Write-Host "`n4. Open browser: http://localhost:3000" -ForegroundColor White
Write-Host "`n💡 Tip: Use 'docker-compose up' for easier local development" -ForegroundColor Yellow
