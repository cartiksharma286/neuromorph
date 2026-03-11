import numpy as np
import sys
import os

# Patch numpy for compatibility with older libraries if needed
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

import uvicorn
from server import app

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 8082))
    print(f"Starting Quantum Neural Circuitry App on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
