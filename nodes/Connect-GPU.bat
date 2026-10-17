@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3 with Tkinter is required. Install Python from python.org, then reopen this file.
  pause
  exit /b 1
)
py -3 "%~dp0connector_app.py"
if errorlevel 1 pause
