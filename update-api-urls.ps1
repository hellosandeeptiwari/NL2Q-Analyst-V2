# Script to replace localhost URLs with environment variables

$files = @(
    "frontend/src/components/SettingsPanel.tsx",
    "frontend/src/components/QueryPanelV2.tsx",
    "frontend/src/components/EnterpriseQueryPanel.tsx",
    "frontend/src/components/ConversationalQueryPanel.tsx"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Updating $file..."
        (Get-Content $file -Raw) -replace 'http://localhost:8000', '${API_URL}' | Set-Content $file -NoNewline
    }
}

Write-Host "Done!"
