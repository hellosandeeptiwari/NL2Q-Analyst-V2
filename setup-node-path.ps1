# Auto-Setup Node.js PATH for VS Code Terminal
# This script automatically finds and adds Node.js to PATH

Write-Host "🔍 Checking Node.js setup..." -ForegroundColor Cyan

# Common Node.js installation locations
$nodePaths = @(
    "C:\Program Files\nodejs",
    "C:\Program Files (x86)\nodejs",
    "$env:LOCALAPPDATA\Programs\nodejs",
    "$env:APPDATA\npm",
    "$env:ProgramFiles\nodejs",
    "$env:USERPROFILE\.nvm",
    "$env:NVM_HOME"
)

$foundNode = $false
$nodeLocation = $null

# Check if node is already available
try {
    $currentNode = Get-Command node -ErrorAction SilentlyContinue
    if ($currentNode) {
        Write-Host "✅ Node.js is already available!" -ForegroundColor Green
        Write-Host "   Location: $($currentNode.Source)" -ForegroundColor Gray
        Write-Host "   Version: $(node --version)" -ForegroundColor Gray
        $foundNode = $true
    }
} catch {
    # Node not in PATH, continue searching
}

# If not found, search for Node.js
if (-not $foundNode) {
    Write-Host "⚠️  Node.js not in current PATH, searching..." -ForegroundColor Yellow
    
    foreach ($path in $nodePaths) {
        if (Test-Path "$path\node.exe") {
            $nodeLocation = $path
            Write-Host "✅ Found Node.js at: $nodeLocation" -ForegroundColor Green
            
            # Add to current session PATH
            if ($env:PATH -notlike "*$nodeLocation*") {
                $env:PATH = "$nodeLocation;$env:PATH"
                Write-Host "✅ Added to PATH for this session" -ForegroundColor Green
            }
            
            # Verify it works
            try {
                $version = & node --version
                Write-Host "   Version: $version" -ForegroundColor Gray
                $foundNode = $true
                break
            } catch {
                Write-Host "⚠️  Found node.exe but couldn't execute it" -ForegroundColor Yellow
            }
        }
    }
}

# If still not found, show installation instructions
if (-not $foundNode) {
    Write-Host ""
    Write-Host "❌ Node.js is NOT installed on this system" -ForegroundColor Red
    Write-Host ""
    Write-Host "📥 To install Node.js:" -ForegroundColor Yellow
    Write-Host "   1. Visit: https://nodejs.org/" -ForegroundColor White
    Write-Host "   2. Download the LTS version (recommended)" -ForegroundColor White
    Write-Host "   3. Run the installer (keep 'Add to PATH' checked)" -ForegroundColor White
    Write-Host "   4. Restart VS Code after installation" -ForegroundColor White
    Write-Host ""
    Write-Host "   OR use winget:" -ForegroundColor Yellow
    Write-Host "   winget install OpenJS.NodeJS.LTS" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "🚀 Node.js is ready to use!" -ForegroundColor Green
    Write-Host "   You can now run: npm install, npm start, etc." -ForegroundColor Gray
    Write-Host ""
}

# Show npm version if available
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Host "📦 npm version: $npmVersion" -ForegroundColor Cyan
    }
} catch {
    # npm not available
}
