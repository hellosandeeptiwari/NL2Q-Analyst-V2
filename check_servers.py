import requests

print('🔍 Checking server status...')

# Check backend
try:
    response = requests.get('http://localhost:8003/health', timeout=3)
    print(f'✅ Backend (port 8003): {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'   Database connected: {data.get("connected")}')
        print(f'   Latency: {data.get("latency_ms")}ms')
except Exception as e:
    print(f'❌ Backend error: {e}')

# Check frontend
try:
    response = requests.get('http://localhost:3000', timeout=3)
    print(f'✅ Frontend (port 3000): {response.status_code}')
    print(f'   Content length: {len(response.text)} chars')
    if '<div id="root">' in response.text:
        print('   ✅ React app HTML structure found')
    else:
        print('   ❌ React app HTML structure not found')
        print(f'   Sample content: {response.text[:200]}...')
except Exception as e:
    print(f'❌ Frontend error: {e}')

print('\n🔗 URLs:')
print('   Backend API: http://localhost:8003')
print('   Frontend App: http://localhost:3000')
