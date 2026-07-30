@echo off
title NAVI AI Smart Glasses Dashboard
cd /d "%~dp0"
echo ========================================================
echo        Starting NAVI AI Smart Glasses Server...
echo ========================================================
venv\Scripts\python.exe main.py
pause
