#!/usr/bin/env python3
"""Root entrypoint launcher for Quantum-Statistical Traffic Signal Intelligence."""

import os
import sys

# Ensure traffic_cameras directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Quantum-Statistical Traffic Signal Monitoring App")
    parser.add_argument("--port", type=int, default=5005, help="Port to serve Flask application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"=================================================================")
    print(f"🚦 Quantum-Statistical Traffic Signal Intelligence Server Running")
    print(f"📍 Local URL:    http://127.0.0.1:{args.port}")
    print(f"🌐 Network URL:  http://0.0.0.0:{args.port}")
    print(f"⚡ Monitored Cameras: 301 Nodes (City of Toronto)")
    print(f"=================================================================")
    app.run(host=args.host, port=args.port, debug=args.debug)
