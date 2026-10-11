@echo off
cd /d "%~dp0"
start "EarthToHoI4" python app.py
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:5000
