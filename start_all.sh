#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

# Activate virtual environment
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Virtual environment activated."
else
    echo "Warning: Virtual environment not found at $VENV_PATH. Make sure to set it up."
fi

# Start the FastAPI backend server
echo "--- Starting FastAPI Backend on port 8000 ---"
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \\"$PROJECT_DIR/mycode/backend\\" && uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"
end tell
EOF

sleep 2 # Give it a moment to start

# Start the frontend file server from the app directory
echo "--- Starting Frontend Server on port 3000 ---"
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \\"$PROJECT_DIR/mycode/app\\" && echo 'Frontend server running at http://localhost:3000' && python3 -m http.server 3000"
end tell
EOF

echo "All services started in new Terminal tabs." 