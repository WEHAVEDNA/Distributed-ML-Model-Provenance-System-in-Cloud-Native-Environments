@echo off
setlocal

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0port_forward.ps1" -Stop
