#!/usr/bin/env bash
# ==============================================================================
# Mersivity Dual-App Streamlined Launch & Management Script
# Usage:
#   ./launch_apps.sh start      # Start both apps in background
#   ./launch_apps.sh stop       # Stop both apps
#   ./launch_apps.sh restart    # Restart both apps
#   ./launch_apps.sh status     # Check status of both apps
#   ./launch_apps.sh run        # Run both in foreground (Ctrl+C to stop)
#   ./launch_apps.sh logs       # Tail logs
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

"$PYTHON_CMD" manage_apps.py "$@"
