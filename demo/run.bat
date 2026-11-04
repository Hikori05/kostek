@echo off
rem Windows: uruchamia aplikacje.   Uzycie: run.bat
rem Zmiana adresu/portu:  set APP_HOST=0.0.0.0 ^& set APP_PORT=9000 ^& run.bat
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Brak .venv - najpierw uruchom setup.bat
    exit /b 1
)

if not defined APP_HOST set "APP_HOST=127.0.0.1"
if not defined APP_PORT set "APP_PORT=8000"

".venv\Scripts\python.exe" -m uvicorn app:app --host %APP_HOST% --port %APP_PORT%
endlocal
