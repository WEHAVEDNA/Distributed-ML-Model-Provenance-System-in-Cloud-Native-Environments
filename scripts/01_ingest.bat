@echo off
set SAMPLES=%~1
set SPLIT=%~2
if "%SAMPLES%"=="" set SAMPLES=500
if "%SPLIT%"=="" set SPLIT=train
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipeline.ps1" -Step ingest -Samples %SAMPLES% -Split %SPLIT%
