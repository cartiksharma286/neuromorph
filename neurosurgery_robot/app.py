from flask import Flask, render_template, jsonify, request
import numpy as np
import threading
import time
import os

# Try to import quantum-enhanced robot, fallback to classical
try:
    from robot_kinematics_quantum import QuantumEnhancedRobot6DOF
    QUANTUM_AVAILABLE = True
    print("Quantum-enhanced kinematics loaded successfully!")
except ImportError:
    from robot_kinematics import Robot6DOF
    QUANTUM_AVAILABLE = False
    print("Using classical kinematics (quantum module not available)")

from thermometry import MRIThermometry
from nvqlink import NVQLink
from game_theory import GameTheoryController
from cryo import CryoModule
from vasculature import VasculatureSpectralAnalyzer
from generative_heating import GenerativeTissueHeating
from guidance_system import AutomatedGuidanceSystem

app = Flask(__name__)

# Global Simulation State
if QUANTUM_AVAILABLE:
    robot = QuantumEnhancedRobot6DOF()
else:
    robot = Robot6DOF()
    
thermo = MRIThermometry(width=64, height=64)
cryo = CryoModule(width=64, height=64)
link = NVQLink()
gt_controller = GameTheoryController()
vasculature = VasculatureSpectralAnalyzer(num_nodes=64)
gen_heating = GenerativeTissueHeating()
guidance = None 

# State variables
simulation_running = True
laser_enabled = False
cryo_enabled = False
target_coords = {'x': 0.1, 'y': 0.0, 'z': 0.6}
use_quantum_mode = QUANTUM_AVAILABLE

def simulation_loop():
    global simulation_running, laser_enabled, cryo_enabled, target_coords, guidance
    
    # Connect Link
    link.connect()
    
    while simulation_running:
        # Automated Guidance Logic
        if guidance and guidance.active:
            # We pass current robot FK (x, y, z)
            # Robot Z is mapped to Grid Y in guidance logic
            current_pos = robot.fk(robot.joints)
            
            tgt, fire_laser = guidance.get_next_target(current_pos)
            
            if tgt:
                target_coords['x'] = tgt[0]
                target_coords['z'] = tgt[1]
                laser_enabled = fire_laser
            else:
                # Done or inactive
                if guidance.completed:
                    print("Guidance completed.")
                    # Keep laser off
                    laser_enabled = False
        
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
        
        # 4. Apply Modalities with Generative AI & Quantum Surface Integral
        
        # Generative AI Control Action (Quantum Machine Learning)
        # 1. Identify Tumor Target from Anatomy
        tumor_mask = (cryo.anatomy_map > 0.8).astype(float)
        
        # 2. Generate Optimal Heat Pattern (QML)
        # We perform this optimization step
        heat_pattern = gen_heating.generate_heating_pattern(tumor_mask, mode="Standard")
        
        # 3. Statistical Classifier for Efficacy
        chem_prob = gen_heating.statistical_classifier(thermo.get_map(), tumor_mask)
        
        # Get Power Leve
        current_max_temp = np.max(thermo.get_map())
        ai_control = gen_heating.get_control_action(current_max_temp)
        ai_power = ai_control['power'] * (1.0 + chem_prob) # Boost power if probable success
        
        # Vascular modulation
        ablation_modulator = 1.0 / (1.0 + vascular_spec['spectral_radius']) 
        
        # Quantum Flux (Surface Integral Approximation)
        grad_phi = np.linalg.norm(move_vec)
        psi_avg = np.mean(thermo.get_map())
        quantum_flux = psi_avg * grad_phi * 100.0 
        
        # Combine Modalities
        effective_power = ai_power * ablation_modulator * (1.0 + quantum_flux * 0.1)
        
        # Heat with QML Pattern
        thermo.apply_laser(grid_x, grid_y, power=effective_power, enabled=laser_enabled, pattern=heat_pattern)
        thermo.update()
        
        # Cold (Cryo Ablation)
        # Automatically trigger Cryo if Laser is active to create "Thermal Shock" at tumor site
        # or if explicitly enabled
        auto_cryo = False
        if laser_enabled and current_max_temp > 45.0:
            auto_cryo = True # Dual ablation
            
        final_cryo_enabled = cryo_enabled or auto_cryo
        cryo.apply_cryo(grid_x, grid_y, pressure_level=3000, enabled=final_cryo_enabled)
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
    damage_map = thermo.get_damage_map().tolist()
    cryo_map = cryo.get_map().tolist()
    mr_anatomy = cryo.anatomy_map.tolist()
    temp_hist = thermo.get_history()
    vasc_spec = vasculature.get_analysis()
    
    # Get GenAI state
    current_max_temp = np.max(thermo.get_map())
    ai_state = gen_heating.get_control_action(current_max_temp)
    
    # Get quantum metrics if available
    quantum_metrics = {}
    if QUANTUM_AVAILABLE and hasattr(robot, 'get_quantum_metrics'):
        quantum_metrics = robot.get_quantum_metrics()
    
    # Get thermometry performance metrics
    thermo_perf = {}
    if hasattr(thermo, 'get_performance_metrics'):
        thermo_perf = thermo.get_performance_metrics()
    
    return jsonify({
        'joints': joints,
        'position': end_effector,
        'temperature_map': temp_map,
        'damage_map': damage_map,
        'cryo_map': cryo_map,
        'mr_anatomy': mr_anatomy,
        'temp_history': temp_hist,
        'vasculature': vasc_spec,
        'gen_ai': {
            'target_temp': ai_state['target_temp'],
            'model_state': ai_state['model_state'],
            'mode': ai_state['mode'],
            'generated_profile': gen_heating.generated_profile.tolist() if hasattr(gen_heating, 'generated_profile') and isinstance(gen_heating.generated_profile, np.ndarray) else []
        },
        'guidance': {
            'active': guidance.active if guidance else False,
            'completed': guidance.completed if guidance else False
        },
        'nvqlink': {
            'status': link.status,
            'latency': link.latency_ms,
            'active': link.active
        },
        'quantum': {
            'enabled': QUANTUM_AVAILABLE,
            'metrics': quantum_metrics
        },
        'thermometry': {
            'high_performance': True,
            'metrics': thermo_perf
        },
        'laser_enabled': laser_enabled,
        'cryo_enabled': cryo_enabled,
        'laser_pos': {'x': thermo.laser_pos[0], 'y': thermo.laser_pos[1]} if hasattr(thermo, 'laser_pos') else None
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
        
    if 'mode' in data:
        print(f"Switching AI Mode to: {data['mode']}")
        gen_heating.generate_heating_curve(mode=data['mode'])
        
    return jsonify({'status': 'ok'})

@app.route('/api/guidance', methods=['POST'])
def toggle_guidance():
    global guidance, laser_enabled
    data = request.json
    enabled = data.get('enabled', False)
    
    if enabled:
        print("Initializing Automated Guidance...")
        guidance = AutomatedGuidanceSystem(cryo.anatomy_map)
        guidance.start()
        # Ensure Standard heating mode is on for precision
        gen_heating.generate_heating_curve(mode="GENTLE") 
    else:
        if guidance:
            guidance.stop()
        laser_enabled = False
            
    return jsonify({'status': 'ok'})

@app.route('/api/quantum/status')
def quantum_status():
    """Get quantum system status and metrics"""
    if not QUANTUM_AVAILABLE:
        return jsonify({'enabled': False, 'message': 'Quantum mode not available'})
    
    metrics = robot.get_quantum_metrics() if hasattr(robot, 'get_quantum_metrics') else {}
    
    return jsonify({
        'enabled': True,
        'coherence': metrics.get('coherence', 0.0),
        'uncertainty': metrics.get('uncertainty', 0.0),
        'qml_fidelity': metrics.get('qml_fidelity', 0.0),
        'tracking_error': metrics.get('tracking_error', 0.0),
        'avg_tracking_error': metrics.get('avg_tracking_error', 0.0)
    })

@app.route('/api/quantum/train', methods=['POST'])
def train_quantum():
    """Train quantum ML component"""
    if not QUANTUM_AVAILABLE:
        return jsonify({'error': 'Quantum mode not available'}), 400
    
    data = request.json
    num_steps = data.get('steps', 10)
    
    if hasattr(robot, 'train_qml'):
        losses = robot.train_qml(num_steps)
        return jsonify({
            'status': 'training_complete',
            'steps': num_steps,
            'final_loss': losses[-1] if losses else 0.0,
            'loss_history': losses
        })
    else:
        return jsonify({'error': 'Training not supported'}), 400

@app.route('/api/reports/quantum_kalman')
def get_quantum_report():
    """Get path to quantum Kalman technical report"""
    report_path = os.path.join(os.path.dirname(__file__), 
                               'Quantum_Kalman_Surgical_Robotics_Report.tex')
    
    if os.path.exists(report_path):
        return jsonify({
            'available': True,
            'path': report_path,
            'format': 'LaTeX',
            'title': 'Quantum Kalman Operators for Advanced Pose Estimation in Neurosurgical Robotics'
        })
    else:
        return jsonify({'available': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
