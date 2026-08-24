#!/bin/bash

# Pelvic Floor Implant BEM / HPC / NVQLink Design Studio
# Launch Script

echo "Launching Pelvic Floor BEM/HPC/NVQLink Design Studio..."
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

echo ""
echo "===================================================================="
echo "PELVIC FLOOR IMPLANT: BEM / HPC / NVQLINK DESIGN STUDIO"
echo "===================================================================="
echo "Geometry:     Chamfered implant boundary + geometric mesh repair"
echo "Manifold:     Continued-fraction quasi-periodic blending"
echo "Simulation:   Boundary Element Method (Kelvin kernel)"
echo "Compute:      HPC SLURM job scheduler (Amdahl scaling)"
echo "Accelerator:  NVQLink hybrid GPU-QPU interconnect (simulated)"
echo "===================================================================="
echo ""

PORT="${PORT:-5057}"
echo "Starting Flask Server on port $PORT..."
echo "Access the application at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

PORT=$PORT python3 app.py
