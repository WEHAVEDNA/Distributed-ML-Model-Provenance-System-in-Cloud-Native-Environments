@echo off
set SPLIT=%~1
if "%SPLIT%"=="" set SPLIT=train
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipeline.ps1" -Step preprocess -Split %SPLIT%
