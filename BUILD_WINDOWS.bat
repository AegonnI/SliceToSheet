@echo off
setlocal
cd /d "%~dp0"
if exist .venv-app\Scripts\python.exe goto build
py -3.12 -m venv .venv-app
if errorlevel 1 goto fail
:build
.venv-app\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto fail
.venv-app\Scripts\python.exe build_release.py
if errorlevel 1 goto fail
echo Release ZIP is in the releases folder.
pause
exit /b 0
:fail
echo Build failed. Python 3.12 x64 is required.
pause
exit /b 1
