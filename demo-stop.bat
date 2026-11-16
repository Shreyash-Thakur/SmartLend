@echo off
rem One click: stop the AWS demo (keeps everything; demo-up.bat resumes it).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo-stop.ps1"
pause
