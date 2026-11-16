@echo off
rem One click: rebuild the Docker image and roll the running demo onto it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo-publish.ps1"
pause
