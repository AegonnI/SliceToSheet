@echo off
cd /d "%~dp0"
if exist .venv-app\Scripts\python.exe goto packages
py -3.12 -m venv .venv-app
:created
if errorlevel 1 goto fail
:packages
rem Always reconcile requirements, including updates to existing installations.
.venv-app\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
echo installed>.venv-app\installed.ok
:run
.venv-app\Scripts\python.exe app.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Install Python 3.12 x64 and py launcher, then retry.
echo For a failed package install run: .venv-app\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1
