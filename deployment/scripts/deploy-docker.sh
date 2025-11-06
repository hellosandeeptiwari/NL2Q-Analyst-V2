#!/bin/bash
# Docker Deployment Script
# Run this script to deploy using Docker Compose

echo "🚀 NL2Q Analyst V2 - Docker Deployment"
echo "======================================"
echo ""

# Check if Docker is installed
echo "📝 Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi
echo "✅ Docker is installed"

# Check if Docker Compose is installed
echo ""
echo "📝 Checking Docker Compose installation..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
echo "✅ Docker Compose is installed"

# Check for .env file
echo ""
echo "📝 Checking environment variables..."
if [ ! -f "deployment/.env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp deployment/.env.example deployment/.env
    echo "✅ Created deployment/.env file"
    echo "⚠️  IMPORTANT: Edit deployment/.env with your actual credentials!"
    read -p "Press Enter to continue after editing .env file..."
else
    echo "✅ .env file found"
fi

# Build Docker images
echo ""
echo "📝 Building Docker images..."
cd deployment/docker
docker-compose build
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi
echo "✅ Docker images built successfully"

# Start containers
echo ""
echo "📝 Starting containers..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "❌ Failed to start containers"
    exit 1
fi
echo "✅ Containers started successfully"

# Wait for services to be healthy
echo ""
echo "📝 Waiting for services to be ready..."
sleep 10

# Check container status
echo ""
echo "📊 Container Status:"
docker-compose ps

# Display URLs
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Application URLs:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "💡 Useful commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop:          docker-compose down"
echo "   Restart:       docker-compose restart"
echo "   Rebuild:       docker-compose up --build -d"
