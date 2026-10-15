@echo off
title AI Documents Assistant Stopper
echo ==========================================
echo Stopping AI Documents Assistant containers...
echo ==========================================

docker-compose stop

echo.
echo All containers stopped successfully!
pause
