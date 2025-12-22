from flask import Flask, jsonify, request
from flask_cors import CORS
import data_provider
import ml_engine

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Initialize Training on Startup
print("Initializing EcoGeo Data Science Engine...")
# Train initial model
X_train, y_train = data_provider.data_provider.generate_training_data(1000)
ml_engine.engine.train_classifier(X_train, y_train)
print("Model Initialized (SGD).")

@app.route('/api/lidar/scan', methods=['GET'])
def scan_forest():
    loc = request.args.get('location', 'Banff')
    data = data_provider.data_provider.generate_lidar_scan(loc, size=64)
    return jsonify(data)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    # Allow user to trigger retraining (Online Learning with SGD)
    count = request.json.get('samples', 500)
    X, y = data_provider.data_provider.generate_training_data(count)
    metrics = ml_engine.engine.train_classifier(X, y)
    return jsonify(metrics)

@app.route('/api/model/predict', methods=['POST'])
def predict_risk():
    feats = request.json.get('features', [0.5, 0.5, 0.5, 0.5])
    result = ml_engine.engine.predict_risk(feats)
    
    # Handle the new QKD Security Packet
    if isinstance(result, dict):
        risk = result['value']
        signature = result['qkd_signature']
        cipher = result['cipher_text']
        return jsonify({
            "fire_risk_probability": risk,
            "security": {
                "method": "QKD-BB84 + ChaCha20",
                "signature": signature,
                "ciphertext": cipher
            }
        })
    else:
        # Fallback
        return jsonify({"fire_risk_probability": result})

@app.route('/api/simulation/step', methods=['POST'])
def sim_step():
    # Optimized Cellular Automata Step
    grid = request.json.get('grid')
    if not grid:
        return jsonify({"error": "No grid provided"}), 400
    update = ml_engine.engine.optimize_fire_spread(grid)
    return jsonify(update)

@app.route('/api/nvqlink/status', methods=['GET'])
def nvqlink_status():
    # Simulate NVQLink telemetry
    return jsonify({
        "status": "ONLINE (QUANTUM ENCRYPTED)",
        "bandwidth": "100 Gbps",
        "latency_ms": 0.42,
        "quantum_state": "COHERENT (Bell State Φ+)",
        "nodes_active": 12, # Dynamic later?
        "encryption": "Entangled QKD"
    })

@app.route('/')
def home():
    return "EcoGeo Intelligence API Online [Port 5005]"

if __name__ == '__main__':
    # Run on 5005
    app.run(port=5005, debug=True, use_reloader=False) 
    # use_reloader=False prevents double process in some envs
