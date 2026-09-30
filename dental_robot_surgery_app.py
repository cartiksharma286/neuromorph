#!/usr/bin/env python3
"""
Root entry point for Dental Image Guided Surgery Robot App.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dental_robot_surgery.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5088))
    print(f"Launching Dental Surgery Robot Console on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
