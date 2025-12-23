from flask import Flask, send_from_directory, jsonify, request
import os
from mri_coil_fea_simulation import run_mri_fea_simulation

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/simulate-fea', methods=['POST'])
def simulate_fea():
    try:
        # Get parameters from request
        params = request.get_json() or {} 
        
        # Run simulation with parameters
        stats, image_b64 = run_mri_fea_simulation(params)
        
        return jsonify({
            "success": True,
            "image_url": image_b64, # Direct base64 data URI
            "stats": stats
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Using 0.0.0.0 to be accessible if needed, port 5000
    app.run(host='0.0.0.0', port=5000)
