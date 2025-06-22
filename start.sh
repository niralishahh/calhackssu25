#!/bin/bash

# Claude AI Document Summarizer Startup Script

echo "🤖 Starting Claude AI Document Summarizer..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check environment variables
echo "🔍 Checking environment variables..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set"
    echo "   Please set it with: export ANTHROPIC_API_KEY='your-api-key'"
fi

# Start backend server
echo "🚀 Starting backend server..."
cd backend/claude
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🌐 Starting frontend server..."
cd ../../app
python -m http.server 8000 &
FRONTEND_PID=$!

echo ""
echo "✅ Application started successfully!"
echo "=============================================="
echo "🌐 Frontend: http://localhost:8000"
echo "🔧 Backend API: http://localhost:5003"
echo "📊 Health Check: http://localhost:5003/health"
echo "📚 API Documentation: http://localhost:5003/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait 