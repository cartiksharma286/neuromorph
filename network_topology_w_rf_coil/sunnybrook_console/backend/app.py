import sys
import os

# Add path to backend_integration to import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend_integration')))

from flask import Flask, jsonify, request, send_from_directory
import random
import time
import threading
import math
from scanner_integration_service import HeadCoilDesignerService

app = Flask(__name__, static_folder='../static')

# Simulation State
SYSTEM_STATE = {
    "network_health": "Optimal",
    "threat_level": "Low",
    "active_nodes": 12,
    "encryption_status": "AES-256-GCM + Kyber-512 (Quantum Safe)",
    "alerts": [],
    "performance": {
        "memory_usage": 45.5, # GB
        "core_frequency": 3.4, # GHz
        "active_threads": 128
    },
    "cluster_matrix": [] 
}

# Generate initial cluster schematic matrix with named groups
rack_configs = [
    {"name": "XNAT Cluster A (Primary)", "size": 8},
    {"name": "XNAT Cluster B (Replica)", "size": 8},
    {"name": "AI Inference Grid", "size": 12},
    {"name": "Quantum Relay Nodes", "size": 6}
]

for config in rack_configs:
    units = []
    for j in range(config["size"]):
        units.append({
            "id": f"U-{j+1:02d}",
            "status": "active", # active, idle, error
            "temp": random.randint(35, 65)
        })
    SYSTEM_STATE["cluster_matrix"].append({
        "name": config["name"],
        "units": units
    })

def simulation_thread():
    """
    Background thread to continuously update system state.
    """
    print("Starting background simulation thread...")
    while True:
        # Simulate Memory Fluctuation
        base_mem = 45.0
        SYSTEM_STATE["performance"]["memory_usage"] = round(base_mem + math.sin(time.time() * 0.5) * 10 + random.uniform(-2, 2), 2)
        
        # Simulate Core Frequency
        base_freq = 3.4
        SYSTEM_STATE["performance"]["core_frequency"] = round(base_freq + random.uniform(-0.2, 0.4), 2)
        
        # Simulate Active Threads
        SYSTEM_STATE["performance"]["active_threads"] = int(128 + random.randint(-10, 50))

        # Randomly update unit temperatures
        # Randomly update unit temperatures
        for rack in SYSTEM_STATE["cluster_matrix"]:
            for unit in rack["units"]:
                unit["temp"] = max(30, min(95, unit["temp"] + random.randint(-1, 2)))
                if unit["temp"] > 85:
                    unit["status"] = "warning"
                else:
                    unit["status"] = "active"

        time.sleep(1)

# Start background thread
t = threading.Thread(target=simulation_thread, daemon=True)
t.start()

@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../static', path)

@app.route('/api/status')
def get_status():
    SYSTEM_STATE["active_nodes"] = random.randint(12, 18)
    return jsonify({
        "network_health": SYSTEM_STATE["network_health"],
        "threat_level": SYSTEM_STATE["threat_level"],
        "active_nodes": SYSTEM_STATE["active_nodes"],
        "encryption_status": SYSTEM_STATE["encryption_status"],
        "alerts": SYSTEM_STATE["alerts"]
    })

@app.route('/api/performance')
def get_performance():
    return jsonify(SYSTEM_STATE["performance"])

@app.route('/api/cluster')
def get_cluster():
    return jsonify(SYSTEM_STATE["cluster_matrix"])

@app.route('/api/topology')
def get_topology():
    # Same topology as before
    nodes = [
        {"id": "MRI Scanners", "group": 1},
        {"id": "CT Scanners", "group": 1},
        {"id": "Digital Pathology", "group": 1},
        {"id": "Secure DICOM Gateway", "group": 2},
        {"id": "Load Balancer (ALB)", "group": 2},
        {"id": "Data Lake (S3)", "group": 3},
        {"id": "Archive (Glacier)", "group": 3},
        {"id": "Neuro Pipeline", "group": 4},
        {"id": "Cardio Pipeline", "group": 4},
        {"id": "Pathology Pipeline", "group": 4},
        {"id": "Mammo Pipeline", "group": 4},
        {"id": "SageMaker Training", "group": 5},
        {"id": "SageMaker Inference", "group": 5},
        {"id": "GCP Vertex AI", "group": 5},
        {"id": "Quantum Simulator", "group": 6},
        {"id": "XNAT Server Cluster", "group": 7}
    ]
    
    links = [
        {"source": "MRI Scanners", "target": "Secure DICOM Gateway", "value": 3},
        {"source": "CT Scanners", "target": "Secure DICOM Gateway", "value": 3},
        {"source": "Digital Pathology", "target": "Secure DICOM Gateway", "value": 3},
        {"source": "Secure DICOM Gateway", "target": "Load Balancer (ALB)", "value": 5},
        {"source": "Load Balancer (ALB)", "target": "Data Lake (S3)", "value": 5},
        {"source": "Data Lake (S3)", "target": "Neuro Pipeline", "value": 2},
        {"source": "Data Lake (S3)", "target": "Cardio Pipeline", "value": 2},
        {"source": "Data Lake (S3)", "target": "Pathology Pipeline", "value": 2},
        {"source": "Data Lake (S3)", "target": "Mammo Pipeline", "value": 2},
        {"source": "Neuro Pipeline", "target": "SageMaker Inference", "value": 4},
        {"source": "Cardio Pipeline", "target": "SageMaker Inference", "value": 4},
        {"source": "Pathology Pipeline", "target": "GCP Vertex AI", "value": 4},
        {"source": "Mammo Pipeline", "target": "SageMaker Inference", "value": 4},
        {"source": "SageMaker Training", "target": "SageMaker Inference", "value": 1},
        {"source": "SageMaker Inference", "target": "Quantum Simulator", "value": 8},
        {"source": "Data Lake (S3)", "target": "Archive (Glacier)", "value": 1},
        {"source": "Load Balancer (ALB)", "target": "XNAT Server Cluster", "value": 5}
    ]
    return jsonify({"nodes": nodes, "links": links})

@app.route('/api/security/rotate-keys', methods=['POST'])
def rotate_keys():
    time.sleep(1) # Simulate delay
    SYSTEM_STATE["alerts"].append({"msg": "KMS Keys Rotated Successfully", "timestamp": time.time()})
    return jsonify({"status": "success", "message": "Master keys rotated."})

@app.route('/api/costs')
def get_costs():
    # Simulate cost accumulation based on active nodes
    # Base rate per node per hour approx $0.50
    nodes = SYSTEM_STATE["active_nodes"]
    hourly_rate = nodes * 0.50
    
    # Simulate Department Usage Split
    departments = {
        "Neuroimaging": 0.35,
        "Cardiovascular": 0.25,
        "Pathology": 0.20,
        "Mammography": 0.15,
        "Research": 0.05
    }
    
    costs = {}
    current_monthly_total = 15000 + (time.time() % 1000) # Mock growing cost
    
    for dept, share in departments.items():
        costs[dept] = round(current_monthly_total * share, 2)
        
    # Simulate Optimization
    on_demand_price = current_monthly_total
    spot_price = current_monthly_total * 0.4 # 60% savings
    savings = on_demand_price - spot_price
    
    return jsonify({
        "current_monthly_total": round(spot_price, 2), # We assume we are optimizing
        "potential_on_demand_cost": round(on_demand_price, 2),
        "savings": round(savings, 2),
        "department_breakdown": costs,
        "spot_instance_usage": 85 # percentage
    })

@app.route('/api/security/lockdown', methods=['POST'])
def lockdown():
    SYSTEM_STATE["threat_level"] = "HIGH"
    SYSTEM_STATE["network_health"] = "RESTRICTED MODE"
    SYSTEM_STATE["alerts"].append({"msg": "SYSTEM LOCKDOWN INITIATED", "timestamp": time.time()})
    return jsonify({"status": "warning", "message": "System is in lockdown."})

@app.route('/api/optimize-coil', methods=['POST'])
def optimize_coil():
    try:
        data = request.json or {}
        field_strength = float(data.get('field_strength', 3.0))
        
        service = HeadCoilDesignerService()
        topology, circuit = service.design_coil_for_scanner(field_strength)
        
        if topology and circuit:
            return jsonify({
                "status": "success",
                "topology": {
                    "turns": topology.turns,
                    "length_mm": topology.length * 1000,
                    "radius_mm": topology.radius * 1000
                },
                "circuit": {
                    "q_factor": circuit.quality_factor,
                    "inductance_uH": circuit.inductance * 1e6,
                    "resonance_MHz": circuit.resonant_frequency / 1e6
                }
            })
        else:
            return jsonify({"status": "error", "message": "Optimization failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting Sunnybrook Cloud Console...")
    # threaded=True is default in recent Flask versions, but good to be explicit or use gunicorn in prod
    app.run(port=3000, debug=True, threaded=True)
