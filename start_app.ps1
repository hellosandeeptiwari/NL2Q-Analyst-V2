# Start the NL2Q-Analyst application (backend and frontend)
Write-Host "NL2Q-Analyst Startup Script" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Create a .env file if it doesn't exist
$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Write-Host "Creating default .env file..." -ForegroundColor Yellow
    @"
DB_ENGINE=sqlite
DB_CONNECTION_STRING=sqlite:///nba.db
OPENAI_API_KEY=your_openai_api_key_here
"@ | Out-File -FilePath $envPath -Encoding utf8
    Write-Host "Default .env file created. Please edit it to set your OpenAI API key." -ForegroundColor Yellow
}

# Function to check if a command exists
function Test-Command {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Check installation of Node.js
if (-not (Test-Command "node")) {
    Write-Host "Node.js is not installed. Please install Node.js to run the frontend." -ForegroundColor Red
    exit 1
}

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Green
Push-Location frontend
npm install
Pop-Location

# Define how to start backend
function Start-Backend {
    Write-Host "Starting Backend Server..." -ForegroundColor Magenta
    try {
        python -m backend.main
    }
    catch {
        Write-Host "Failed to start backend server: $_" -ForegroundColor Red
    }
}

# Define how to start frontend
function Start-Frontend {
    Write-Host "Starting Frontend Development Server..." -ForegroundColor Magenta
    try {
        Push-Location frontend
        npm start
        Pop-Location
    }
    catch {
        Write-Host "Failed to start frontend server: $_" -ForegroundColor Red
    }
}

# Start both servers
Write-Host "Starting NL2Q-Analyst application..." -ForegroundColor Green
Write-Host "Backend will run on http://localhost:8001" -ForegroundColor Yellow
Write-Host "Frontend will run on http://localhost:3000" -ForegroundColor Yellow

# Start each process in its own window
Start-Process powershell -ArgumentList "-NoExit -Command & { Set-Location '$PSScriptRoot'; Start-Backend }"
Start-Process powershell -ArgumentList "-NoExit -Command & { Set-Location '$PSScriptRoot'; Start-Frontend }"

Write-Host "Application startup initiated! Both servers should open in new windows." -ForegroundColor Green
