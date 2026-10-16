@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto build
py -3.10 -m venv .venv
if errorlevel 1 py -3.9 -m venv .venv
if errorlevel 1 goto fail
:build
.venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe build_release.py
if errorlevel 1 goto fail
echo Release ZIP is in the releases folder.
pause
exit /b 0
:fail
echo Build failed. Python 3.9 or 3.10 x64 with Tk is required.
pause
exit /b 1
