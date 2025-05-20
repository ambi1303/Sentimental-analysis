#!/bin/bash

# Start the backend server
echo "Starting backend server..."
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 5

# Start the frontend development server
echo "Starting frontend development server..."
cd frontend
npm start 