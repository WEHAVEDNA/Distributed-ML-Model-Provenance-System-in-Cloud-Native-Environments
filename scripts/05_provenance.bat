@echo off
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0pipeline.ps1" -Step provenance
