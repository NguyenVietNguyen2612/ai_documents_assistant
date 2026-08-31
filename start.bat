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
start "Backend Server" cmd /k "cd backend && .\.venv\Scripts\activate && uvicorn app.main:app --reload"

echo.
echo [3/3] Starting React Frontend...
:: Dọn dẹp tiến trình cũ (nếu có) đang chạy trên cổng 5173
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173 " ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1
:: Mở một cửa sổ Terminal mới để chạy Frontend
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo All services are starting up!
echo Backend is running on: http://localhost:8000
echo Frontend is running on: http://localhost:5173 (usually)
echo ==========================================
echo You can safely close this window.
pause
