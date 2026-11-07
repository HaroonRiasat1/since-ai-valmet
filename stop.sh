#!/usr/bin/env bash
# Stop the Field-Service Navigator started with ./start.sh: only the server listening on its port, nothing else.
#
#   ./stop.sh              port 8000
#   PORT=8001 ./stop.sh    another port
set -euo pipefail

PORT="${PORT:-8000}"

if command -v netstat >/dev/null 2>&1 && command -v taskkill >/dev/null 2>&1; then   # Windows (Git Bash)
  PIDS=$(netstat -ano | awk -v p=":${PORT}" '$2 ~ p"$" && $4 == "LISTENING" {print $5}' | sort -u)
  for pid in $PIDS; do taskkill //PID "$pid" //T //F >/dev/null && echo "Stopped the server on port ${PORT} (process ${pid})."; done
elif command -v lsof >/dev/null 2>&1; then                                             # macOS / Linux
  PIDS=$(lsof -ti "tcp:${PORT}" -sTCP:LISTEN || true)
  for pid in $PIDS; do kill "$pid" && echo "Stopped the server on port ${PORT} (process ${pid})."; done
else
  echo "Cannot look up the process on port ${PORT} here: stop it with Ctrl+C in its window." >&2; exit 1
fi

[ -n "${PIDS:-}" ] || echo "Nothing is running on port ${PORT}."
