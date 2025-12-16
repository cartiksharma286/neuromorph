from flask import Flask, render_template, jsonify, request
import numpy as np
import threading
import time

from robot_kinematics import Robot6DOF
from thermometry import MRIThermometry
from nvqlink import NVQLink
from game_theory import GameTheoryController
from cryo import CryoModule
from vasculature import VasculatureSpectralAnalyzer

app = Flask(__name__)

# Global Simulation State
robot = Robot6DOF()
thermo = MRIThermometry(width=64, height=64)
cryo = CryoModule(width=64, height=64)
link = NVQLink()
gt_controller = GameTheoryController()
vasculature = VasculatureSpectralAnalyzer(num_nodes=64)

# State variables
simulation_running = True
laser_enabled = False
cryo_enabled = False
target_coords = {'x': 0.5, 'y': 0.0, 'z': 0.5}

def simulation_loop():
    global simulation_running, laser_enabled, cryo_enabled, target_coords
    
    # Connect Link
    link.connect()
    
    while simulation_running:
        # 1. Update Robot with Game Theory
        telemetry = link.process_telemetry({}, None)
        coeffs = telemetry.get('solver_coeffs', [])
        
        current_fk = robot.fk(robot.joints)
        move_vec = gt_controller.get_optimal_move(current_fk, 
                                                 [target_coords['x'], target_coords['y'], target_coords['z']], 
                                                 coeffs)
        
        target_step = current_fk + move_vec
        robot.ik_step(target_step)
        
        joints, end_effector = robot.joints, robot.fk(robot.joints)
        
        # 2. Map Robot Pos to Grid
        grid_x = int((end_effector[0] + 0.5) * 64)
        grid_y = int((end_effector[2]) * 64) 
        grid_x = max(0, min(63, grid_x))
        grid_y = max(0, min(63, grid_y))
        
        # 3. Vasculature Update (Finite Math)
        vasculature.update()
        vascular_spec = vasculature.get_analysis()
        
        # 4. Apply Modalities with Quantum Surface Integral Injection
        # We assume the 'spectral_radius' from vasculature modulates the ablation efficacy
        # representing blood flow cooling or tissue density.
        
        ablation_modulator = 1.0 / (1.0 + vascular_spec['spectral_radius']) 
        
        # QUANTUM SURFACE INTEGRAL (Approximation)
        # S = Integral( Psi * Grad(Phi) . dA )
        # Here we approximate 'Psi' as the thermal map state and 'Grad(Phi)' as the robot motion vector
        # This calculates a "Flux" of ablation energy.
        grad_phi = np.linalg.norm(move_vec)
        psi_avg = np.mean(thermo.get_map())
        quantum_flux = psi_avg * grad_phi * 100.0 # Scaling factor
        
        # Modulate power by this flux (Quantum Control)
        effective_power = 50.0 * ablation_modulator * (1.0 + quantum_flux)
        
        # Heat
        thermo.apply_laser(grid_x, grid_y, power=effective_power, enabled=laser_enabled)
        thermo.update()
        
        # Cold
        cryo.apply_cryo(grid_x, grid_y, pressure_level=3000, enabled=cryo_enabled)
        cryo.update()
        
        # 5. NVQLink Telemetry Processing
        link.process_telemetry(robot_state={'joints': joints.tolist()}, thermal_data=thermo.get_map())
        
        time.sleep(0.05) 

# Start background thread
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def get_telemetry():
    joints, end_effector = robot.joints.tolist(), robot.fk(robot.joints).tolist()
    temp_map = thermo.get_map().tolist()
    cryo_map = cryo.get_map().tolist()
    mr_anatomy = cryo.anatomy_map.tolist()
    temp_hist = thermo.get_history()
    vasc_spec = vasculature.get_analysis()
    
    return jsonify({
        'joints': joints,
        'position': end_effector,
        'temperature_map': temp_map,
        'cryo_map': cryo_map,
        'mr_anatomy': mr_anatomy,
        'temp_history': temp_hist,
        'vasculature': vasc_spec,
        'nvqlink': {
            'status': link.status,
            'latency': link.latency_ms,
            'active': link.active
        }
    })

@app.route('/api/control', methods=['POST'])
def control_robot():
    global target_coords, laser_enabled, cryo_enabled
    data = request.json
    
    if 'target' in data:
        target_coords['x'] = float(data['target']['x'])
        target_coords['z'] = float(data['target']['z'])
        
    if 'laser' in data:
        laser_enabled = bool(data['laser'])
    
    if 'cryo' in data:
        cryo_enabled = bool(data['cryo'])
        
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
