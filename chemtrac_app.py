#!/usr/bin/env python3
"""
Root entry point for ChemTRAC Water Quality Monitor & Treatment Optimization App.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'chemtrac'))

from chemtrac.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5089))
    print(f"Launching ChemTRAC Water Quality Console on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
