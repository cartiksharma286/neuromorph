#!/bin/bash

echo "========================================"
echo "   Neuromorph System Deployment"
echo "========================================"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# 2. Install Dependencies
echo "[*] Installing dependencies..."
pip3 install -r requirements.txt
if [ 0 -ne 0 ]; then
    echo "Warning: Failed to install dependencies. Please check requirements.txt."
fi

# 3. Initialize DHF
echo "[*] Initializing DHF Structure..."
python3 dhf_validator.py --init

# 4. Run Verification
echo "[*] Running System Verification..."
python3 neuromorph_system.py check-compliance
python3 neuromorph_system.py analyze-cbt
python3 neuromorph_system.py analyze-entanglement --correlation 0.9

# 5. Start Web App
echo "[*] Starting Pulse System Web App..."
echo "    Access at http://localhost:5000"
echo "    Press Ctrl+C to stop."
python3 pulse_system/app.py
