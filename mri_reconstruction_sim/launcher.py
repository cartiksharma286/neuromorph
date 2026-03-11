import numpy as np
import sys

# Patch numpy before any other imports
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

# Now import the actual app
import app_final

if __name__ == '__main__':
    # Force the app to run if it doesn't already
    if hasattr(app_final, 'app'):
        app_final.app.run(host='0.0.0.0', port=5050, debug=True)
