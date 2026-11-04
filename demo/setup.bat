@echo off
rem Windows: tworzy .venv i instaluje zaleznosci.   Uzycie: setup.bat
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Nie znaleziono Pythona. Zainstaluj Python 3.10 lub nowszy z python.org
    exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Wymagany Python 3.10 lub nowszy.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ^>^> Tworze .venv
    %PY% -m venv .venv
    if errorlevel 1 exit /b 1
) else (
    echo ^>^> Uzywam istniejacego .venv
)

echo ^>^> Instaluje zaleznosci
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Gotowe. Uruchom aplikacje: run.bat
endlocal
