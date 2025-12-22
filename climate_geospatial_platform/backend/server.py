from flask import Flask, jsonify, request
from flask_cors import CORS
import data_provider
import ml_engine

app = Flask(__name__)
CORS(app)

# Initialize Training on Startup (simulated "pre-trained")
print("Initializing Data Science Engine...")
X_train, y_train = data_provider.data_provider.generate_training_data(2000)
ml_engine.engine.train_classifier(X_train, y_train)
print("Model Trained.")

@app.route('/api/lidar/scan', methods=['GET'])
def scan_forest():
    loc = request.args.get('location', 'Banff')
    data = data_provider.data_provider.generate_lidar_scan(loc, size=64)
    return jsonify(data)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    # Allow user to trigger retraining
    count = request.json.get('samples', 1000)
    X, y = data_provider.data_provider.generate_training_data(count)
    metrics = ml_engine.engine.train_classifier(X, y)
    return jsonify(metrics)

@app.route('/api/model/predict', methods=['POST'])
def predict_risk():
    feats = request.json.get('features', [0.5, 0.5, 0.5, 0.5])
    risk = ml_engine.engine.predict_risk(feats)
    return jsonify({"fire_risk_probability": risk})

@app.route('/api/simulation/step', methods=['POST'])
def sim_step():
    # Optimized Cellular Automata Step
    grid = request.json['grid']
    update = ml_engine.engine.optimize_fire_spread(grid)
    return jsonify(update)

@app.route('/api/nvqlink/status', methods=['GET'])
def nvqlink_status():
    # Simulate NVQLink telemetry
    return jsonify({
        "status": "ONLINE (QUANTUM ENCRYPTED)",
        "bandwidth": "100 Gbps",
        "latency_ms": 0.523,
        "quantum_state": "COHERENT",
        "nodes_active": 12,
        "encryption": "Entangled"
    })

@app.route('/')
def home():
    return "Climate Data Science API Online"

if __name__ == '__main__':
    app.run(port=5001, debug=True)
