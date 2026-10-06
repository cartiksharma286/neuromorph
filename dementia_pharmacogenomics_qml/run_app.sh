#!/usr/bin/env bash
# Launcher for Neuromorph QML Pharmacogenomics & Photonic Dementia Suite
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( dirname "$DIR" )"

export PYTHONPATH="$DIR:$ROOT_DIR:$PYTHONPATH"
PORT="${PORT:-5080}"

echo "=========================================================================="
echo " Starting Neuromorph QML Pharmacogenomics & Photonic Dementia Suite"
echo " Nature Preprint: Nature_Preprint_Pharmacogenomics_Photonic_Quantum_Dementia.pdf"
echo " Endpoint: http://127.0.0.1:${PORT}"
echo "=========================================================================="

exec "$ROOT_DIR/.venv/bin/python" "$DIR/server.py"
