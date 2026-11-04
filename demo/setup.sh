#!/usr/bin/env bash
# Linux / macOS: tworzy .venv i instaluje zależności.   Użycie: bash setup.sh
# Inny interpreter:  PYTHON=python3.12 bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Nie znaleziono Pythona (wymagany 3.10 lub nowszy)." >&2
  exit 1
fi

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Wymagany Python 3.10 lub nowszy (znaleziono: $("$PY" --version 2>&1))." >&2
  exit 1
}

if [ ! -x .venv/bin/python ]; then
  echo ">> Tworzę .venv"
  "$PY" -m venv .venv || {
    echo "Nie udało się utworzyć venv. Na Debian/Ubuntu/Pop!_OS: sudo apt install python3-venv" >&2
    exit 1
  }
else
  echo ">> Używam istniejącego .venv"
fi

echo ">> Instaluję zależności"
.venv/bin/python -m pip install --disable-pip-version-check --upgrade pip
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt

echo
echo "Gotowe. Uruchom aplikację:  bash run.sh"
