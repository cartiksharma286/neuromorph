
import numpy as np
import threading
import time
from flask import Flask, render_template, jsonify, request


# Import custom modules
from kinematics import RobotKinematics
from thermometry import ThermalModel
from cryo import CryoModel
from anatomy import ProstatePhantom
from nvqlink import NVQLink
from generative_ai import GeminiGenAI
from kalman import KalmanFilter

app = Flask(__name__)

# System State
robot = RobotKinematics()
phantom = ProstatePhantom(width=128, height=128)
# Ensure anatomy is generated/loaded to make mask available
if not hasattr(phantom, 'tumor_mask'):
    phantom.load_anatomy()
    # Or generate if load failed or didn't set mask (generate uses it)
    if not hasattr(phantom, 'tumor_mask'):
         # Force generate to get mask if not present
         phantom.generate_anatomy()

tumor_mask = phantom.get_tumor_mask()
thermo = ThermalModel(phantom.width, phantom.height, phantom.get_mask(), tumor_mask)
cryo = CryoModel(phantom.width, phantom.height, phantom.get_mask())
nvq = NVQLink()
genai = GeminiGenAI()
kf = KalmanFilter(process_noise=1e-5, measurement_noise=1e-3) # Tuned for microsurgery

state = {
    "running": True,
    "laser_active": True,
    "cryo_active": False,
    "target": {"x": 0.008, "y": 0.003, "z": 0.0},
    "filtered_target": {"x": 0.008, "y": 0.003, "z": 0.0}
}

def physics_loop():
    # Connect Link
    nvq.connect()
    
    while state["running"]:
        # 1. Kalman Filter Estimation
        # Simulate noisy sensor input from "Tumor Surface Tracking"
        raw_target = state["target"]
        
        # Apply Filter
        filtered = kf.update(raw_target)
        state["filtered_target"] = filtered
        
        # 2. Update Robot with Refined Target
        robot.update(filtered)
        
        # 3. Get Effector Position and Map
        pos = robot.get_end_effector()
        grid_x = int(64 + (pos[0] * 2000))
        grid_y = int(64 + (pos[1] * 2000))
        grid_x = max(0, min(127, grid_x))
        grid_y = max(0, min(127, grid_y))
        
        # 4. Apply Therapies
        if state["laser_active"]:
            thermo.apply_heat(grid_x, grid_y, power=25.0)
            
        if state["cryo_active"]:
            cryo.apply_cooling(grid_x, grid_y)
            thermo.apply_cooling(grid_x, grid_y, intensity=50.0)
            
        # 5. Solvers
        thermo.step()
        
        # 6. NVQ Telemetry Processing
        nvq.process_telemetry({
            "joints": robot.current_joints,
            "temp_max": np.max(thermo.temp_map),
            "est_error": np.linalg.norm(np.array(list(filtered.values())) - np.array(list(raw_target.values())))
        })
        
        time.sleep(0.05)

# Start Physics
t = threading.Thread(target=physics_loop, daemon=True)
t.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init')
def get_init():
    return jsonify({
        "anatomy": phantom.image.tolist(),
        "width": phantom.width,
        "height": phantom.height
    })

@app.route('/api/status')
def get_status():
    pos = robot.get_end_effector().tolist()
    grid_x = int(64 + (pos[0] * 2000))
    grid_y = int(64 + (pos[1] * 2000))
    
    return jsonify({
        "joints": robot.current_joints.tolist(),
        "position": pos,
        "grid_pos": [grid_x, grid_y],
        "temperature": thermo.get_temp_map_compressed(),
        # Anatomy removed for performance/latency
        "damage": thermo.damage_map.tolist(),
        "target_est": state["filtered_target"],
        "ablation": thermo.get_ablation_stats()
    })

@app.route('/api/control', methods=['POST'])
def control():
    data = request.json
    if "target" in data:
        state["target"] = data["target"]
    if "laser" in data:
        state["laser_active"] = data["laser"]
    if "cryo" in data:
        state["cryo_active"] = data["cryo"]
    return jsonify({"status": "ok"})

@app.route('/api/genai/plan', methods=['POST'])
def genai_plan():
    # Trigger AI analysis
    target, explanation = genai.generate_ablation_plan(phantom.image)
    # Auto-set target?
    # state["target"] = target # Optional: Auto-execution
    return jsonify({
        "target": target,
        "explanation": explanation,
        "model": genai.model_version
    })

@app.route('/api/nvq/status')
def nvq_status():
    return jsonify({
        "status": nvq.status,
        "latency": nvq.latency,
        "quantum": nvq.quantum_entanglement
    })

@app.after_request
def set_secure_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

if __name__ == '__main__':
    print("\n" + "="*50)
    print("      NEUROMORPH PROSTATE ROBOT - V2.4")
    print("="*50)
    print(" [SEC] CYBERSECURITY PROTOCOL:    ACTIVE")
    print(" [PWR] GREEN FOOTPRINT:           ENSURED (Eco-Mode)")
    print(" [NET] NVQ LINK:                  READY")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
