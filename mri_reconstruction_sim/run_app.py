import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

# Patch numpy before any other imports
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int

import app

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5050))
    app.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
