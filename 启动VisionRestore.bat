@echo off
setlocal
chcp 65001 >nul

for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
set "WEB_URL=http://127.0.0.1:5173"
set "API_URL=http://127.0.0.1:8000/docs"

title VisionRestore Agent Launcher

echo.
echo ==========================================
echo   VisionRestore Agent
echo ==========================================
echo.
echo Project: %PROJECT_DIR%
echo Web:     %WEB_URL%
echo API:     %API_URL%
echo.

if not exist "%PROJECT_DIR%\scripts\start.ps1" (
  echo [ERROR] Cannot find scripts\start.ps1.
  echo Please check whether the project path is correct:
  echo %PROJECT_DIR%
  echo.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$api=(Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue); $web=(Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue); if($api -and $web){ exit 10 } elseif($api -or $web){ exit 11 } else { exit 0 }"

if "%ERRORLEVEL%"=="10" (
  echo [OK] Backend and frontend are already running.
  goto OPEN_WEB
)

if "%ERRORLEVEL%"=="11" (
  echo [WARN] Port 8000 or 5173 is already occupied.
  echo If the page cannot open, close the old terminal/process first, then run this bat again.
  goto OPEN_WEB
)

echo [START] Starting backend and frontend...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%\scripts\start.ps1"

if errorlevel 1 (
  echo.
  echo [ERROR] Startup failed. Check the message above.
  echo Common reasons:
  echo - Port 8000 or 5173 is occupied
  echo - config\models.local.yaml is missing
  echo - .venv or npm dependencies are incomplete
  echo.
  pause
  exit /b 1
)

:OPEN_WEB
echo.
echo [OPEN] Opening VisionRestore Agent...
timeout /t 3 /nobreak >nul
start "" "%WEB_URL%"
echo.
echo If the browser page is blank, wait a few seconds and refresh.
echo Frontend: %WEB_URL%
echo API docs: %API_URL%
echo.
pause
