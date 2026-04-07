@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "NO_WATCH=0"

if /I "%~1"=="--no-watch" set "NO_WATCH=1"
if /I "%~1"=="-NoWatch" set "NO_WATCH=1"

pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%deploy_k8s.ps1" -BuildImages -DockerDesktop
if errorlevel 1 exit /b %errorlevel%

echo.
echo Kubernetes setup complete.
if "%NO_WATCH%"=="1" goto :after_watch

echo Streaming Kubernetes pod status in this terminal.
echo Press Ctrl+C to stop watching and return to the prompt.
echo.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%watch_k8s_status.ps1"
if errorlevel 1 exit /b %errorlevel%

:after_watch
echo Start port-forwards manually in the terminal where you want them to run:
echo   pwsh scripts\port_forward.ps1
echo.
echo Demo endpoints will be reachable on localhost:8001-8004 after port-forwarding starts.
echo Run:
echo   python demo.py --samples 200 --pipeline-id demo-200
