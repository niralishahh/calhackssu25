#!/bin/bash

echo "🚀 Starting Vertex AI RAG Document Analysis System..."

# Check if we're in the right directory
if [ ! -f "backend/claude/rag_app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY environment variable is required"
    echo "Please set it with: export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "❌ Error: SUPABASE_URL environment variable is required"
    echo "Please set it with: export SUPABASE_URL='your-supabase-url'"
    exit 1
fi

if [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "❌ Error: SUPABASE_SERVICE_ROLE_KEY environment variable is required"
    echo "Please set it with: export SUPABASE_SERVICE_ROLE_KEY='your-service-role-key'"
    exit 1
fi

if [ -z "$GOOGLE_CLOUD_PROJECT_ID" ]; then
    echo "❌ Error: GOOGLE_CLOUD_PROJECT_ID environment variable is required"
    echo "Please set it with: export GOOGLE_CLOUD_PROJECT_ID='your-project-id'"
    exit 1
fi

# Check if Google Cloud credentials are set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "Please set it with: export GOOGLE_APPLICATION_CREDENTIALS='path/to/service-account-key.json'"
    echo "Or use: gcloud auth application-default login"
fi

echo "✅ Environment variables are set"

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -r requirements.txt

# Create uploads directory if it doesn't exist
mkdir -p backend/claude/uploads

# Start the RAG server
echo "🌐 Starting Vertex AI RAG server on port 5005..."
cd backend/claude
python rag_app.py 