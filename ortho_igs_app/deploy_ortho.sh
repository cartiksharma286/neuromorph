#!/bin/bash

# Deployment Script for Orthopedic Surgery Robot Module

echo "============================================="
echo "   Deploying NVQLink Orthopedic Robot"
echo "============================================="

# 1. Environment Setup
echo "[*] Checking Environment..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# 2. Install Dependencies
echo "[*] Installing Dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Warning: Dependency installation had issues."
fi

# 3. Validation
echo "[*] Validating Modules..."
python3 -c "import sys; sys.path.append('.'); from orthopedic_surgery_robot.backend.nvqlink_optimizer import NVQLinkKneeOptimizer; print('   - NVQLink Optimizer: OK')"
python3 -c "import sys; sys.path.append('.'); from orthopedic_surgery_robot.backend.health_economics import HealthEconomicsEngine; print('   - Health Economics : OK')"

# 4. Launch Application
echo "[*] Launching Dashboard..."
echo "    App running at: http://localhost:5000"
echo "    Monitor logs below. Press Ctrl+C to stop."
echo "============================================="

python3 orthopedic_surgery_robot/app.py
