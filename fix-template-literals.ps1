# Fix template literals in EnhancedPharmaChat.tsx
$file = "frontend/src/components/EnhancedPharmaChat.tsx"
$content = Get-Content $file -Raw

# Fix all malformed template literals
$content = $content -replace "fetch\(`\$\{API_URL\}/([^']+)'\)", 'fetch(`${API_URL}/$1`)'
$content = $content -replace "fetch\('`\$\{API_URL\}/([^']+)'\)", 'fetch(`${API_URL}/$1`)'
$content = $content -replace "WebSocket\('`\$\{WS_URL\}/([^']+)'\)", 'WebSocket(`${WS_URL}/$1`)'
$content = $content -replace "WebSocket\(`\$\{WS_URL\}/([^']+)'\)", 'WebSocket(`${WS_URL}/$1`)'

Set-Content $file $content -NoNewline
Write-Host "Fixed template literals"
