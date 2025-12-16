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
from generative_heating import GenerativeTissueHeating
from guidance_system import AutomatedGuidanceSystem

app = Flask(__name__)

# Global Simulation State
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
        
        # Generative AI Control Action
        # Get the optimal heating control based on current max temperature
        current_max_temp = np.max(thermo.get_map())
        ai_control = gen_heating.get_control_action(current_max_temp)
        ai_power = ai_control['power']
        
        # Vascular modulation
        ablation_modulator = 1.0 / (1.0 + vascular_spec['spectral_radius']) 
        
        # Quantum Flux (Surface Integral Approximation)
        grad_phi = np.linalg.norm(move_vec)
        psi_avg = np.mean(thermo.get_map())
        quantum_flux = psi_avg * grad_phi * 100.0 
        
        # Combine Modalities: Generative AI + Quantum Flux + Vasculature
        effective_power = ai_power * ablation_modulator * (1.0 + quantum_flux * 0.1)
        
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
    damage_map = thermo.get_damage_map().tolist()
    cryo_map = cryo.get_map().tolist()
    mr_anatomy = cryo.anatomy_map.tolist()
    temp_hist = thermo.get_history()
    vasc_spec = vasculature.get_analysis()
    
    # Get GenAI state
    current_max_temp = np.max(thermo.get_map())
    ai_state = gen_heating.get_control_action(current_max_temp) # Just to peak at state
    
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
