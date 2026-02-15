#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Launching MRI Reconstruction Simulator on Port 5002..."
echo "Access internally at: http://localhost:5002"
echo "Access externally at: http://192.168.2.14:5002"
python3 mri_reconstruction_sim/app_enhanced.py
