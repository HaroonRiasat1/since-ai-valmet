#!/usr/bin/env bash
# Start the Field-Service Navigator (as in README "Quick start").
#
#   ./start.sh              backend + the app in its own window on the NVIDIA GPU
#   ./start.sh --no-gpu     backend only; then open http://127.0.0.1:8000
#   PORT=8001 ./start.sh    another port
#
# Uses the first Python environment it finds: .venv here, the shared ../.venv, or uv (which installs on first run).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
GPU="--gpu"
[ "${1:-}" = "--no-gpu" ] && GPU=""

if [ -x .venv/Scripts/python.exe ]; then PY=(.venv/Scripts/python.exe)
elif [ -x .venv/bin/python ]; then PY=(.venv/bin/python)
elif [ -x ../.venv/Scripts/python.exe ]; then PY=(../.venv/Scripts/python.exe)
elif [ -x ../.venv/bin/python ]; then PY=(../.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then uv sync --all-extras; PY=(uv run python)
else echo "No Python environment found: install uv (https://docs.astral.sh/uv/) and run this again." >&2; exit 1
fi

[ -f web/data/scene.json ] || { echo "web/data/scene.json is missing: rebuild the scene first (README, 'Rebuilding the data')." >&2; exit 1; }
[ -f .env ] && grep -q "^FEATHERLESS_API_KEY=." .env || echo "Note: no FEATHERLESS_API_KEY in .env, so the assistant can only use a local LM Studio model."

if ! "${PY[@]}" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "The backend packages are missing in this environment: run 'uv sync --all-extras' first." >&2; exit 1
fi

echo "Field-Service Navigator on http://127.0.0.1:${PORT}  (Ctrl+C stops it)"
PORT="$PORT" exec "${PY[@]}" -m backend --port "$PORT" $GPU
