#!/bin/bash

# Ensure dependencies
echo "Installing Dependencies: Flask, Numpy, Scikit-Learn..."
pip install flask flask-cors numpy scikit-learn

if [ -f "backend/server.py" ]; then
    # Start Python Backend
    echo "Starting Backend..."
    python3 backend/server.py &
    PY_PID=$!
else
    echo "Error: backend/server.py not found!"
    exit 1
fi

sleep 2

# Start Simple Python HTTP for frontend since we might not have Node configured yet or user wants simple
# Actually user had Node before. But to be "quick", Python http.server is fastest if node_modules missing.
# Let's try Node if package_json exists, else Python.
if [ -f "frontend_server.js" ]; then
    echo "Starting Node Frontend..."
    node frontend_server.js &
    NODE_PID=$!
else
    echo "Starting Simple Frontend Server..."
    cd classical_app
    python3 -m http.server 3000 &
    NODE_PID=$!
    cd ..
fi

echo "Systems Online."
echo "API: http://localhost:5001"
echo "UI:  http://localhost:3000"

trap "kill $PY_PID $NODE_PID" EXIT

wait
