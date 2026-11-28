#!/usr/bin/env bash
# Uruchamia lokalny panel ML (dane, podgląd, trening na GPU, benchmark, eksport ONNX)
# Użycie: bash dashboard/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "Brak głównego .venv z PyTorchem/Ultralytics w katalogu projektu." >&2
  exit 1
fi

exec .venv/bin/python -m uvicorn dashboard.app:app --host "${DASH_HOST:-127.0.0.1}" --port "${DASH_PORT:-8001}"
