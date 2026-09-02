@echo off
title AI Documents Assistant Launcher
echo ==========================================
echo Starting AI Documents Assistant...
echo ==========================================

echo.
echo [1/3] Starting Milvus Vector Database...
docker-compose up -d etcd minio standalone

echo.
echo [2/3] Starting FastAPI Backend...
:: Mở một cửa sổ Terminal mới để chạy Backend
start "Backend Server" cmd /k "cd backend && .\.venv\Scripts\activate && uvicorn app.main:app --reload --port 8001"

echo.
echo [3/3] Starting React Frontend...
:: Mở một cửa sổ Terminal mới để chạy Frontend
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo All services are starting up!
echo Backend is running on: http://localhost:8001
echo Frontend is running on: http://localhost:3000
echo ==========================================
echo You can safely close this window.
pause
