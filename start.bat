@echo off
echo Starting backend server...
start cmd /k "python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 5

echo Starting frontend development server...
cd frontend
start cmd /k "npm start" 