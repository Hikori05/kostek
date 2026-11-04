#!/usr/bin/env bash
# Linux / macOS: uruchamia aplikację.   Użycie: bash run.sh
# Zmiana adresu/portu:  APP_HOST=0.0.0.0 APP_PORT=9000 bash run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Brak .venv - najpierw uruchom: bash setup.sh" >&2
  exit 1
fi

exec .venv/bin/python -m uvicorn app:app --host "${APP_HOST:-127.0.0.1}" --port "${APP_PORT:-8000}"
