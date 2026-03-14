from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import threading
import time
import os
import io
import base64

# Set matplotlib to use non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import enhanced surgical components
from precision_kinematics import PrecisionRobot6DOF
from enhanced_thermometry import EnhancedThermometry
from enhanced_cryo import EnhancedCryoModule
from level_set_segmentation import LevelSetSegmentation
from quantum_kalman import QuantumKalmanFilter
from tumor_thermometry_segmentation import MRThermometrySegmenter

# Try to import optional modules

try:
    from game_theory import GameTheoryController
    HAS_GAME_THEORY = True
except:
    HAS_GAME_THEORY = False
    GameTheoryController = None

try:
    from vasculature import VasculatureSpectralAnalyzer
    HAS_VASCULATURE = True
except:
    HAS_VASCULATURE = False
    VasculatureSpectralAnalyzer = None

try:
    from guidance_system import AutomatedGuidanceSystem
    HAS_GUIDANCE = True
except:
    HAS_GUIDANCE = False
    AutomatedGuidanceSystem = None

try:
    from five_g_guidance import FiveGNeuralPathway, LaserDeliveryOptimizer
    HAS_5G_GUIDANCE = True
except:
    HAS_5G_GUIDANCE = False
    FiveGNeuralPathway = None
    LaserDeliveryOptimizer = None

app = Flask(__name__)

# Helper function to convert numpy types to Python native types
def numpy_to_native(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_native(item) for item in obj]
    return obj

# Global Simulation State - Enhanced
robot = PrecisionRobot6DOF()
thermo = EnhancedThermometry(width=128, height=128)
cryo = EnhancedCryoModule(width=128, height=128)
segmentation = LevelSetSegmentation(width=128, height=128)

# Initialize segmentation from anatomy
segmentation.initialize_from_image(thermo.tissue_type)

# Quantum Kalman Filter for end-effector localization (6-DOF state, 3-D measurement)
qkf = QuantumKalmanFilter(state_dim=6, measurement_dim=3)

# MR-Thermometry tumour segmenter
mr_thermo_seg = MRThermometrySegmenter(width=128, height=128)

# Shared QKF telemetry snapshot (written by simulation thread, read by HTTP thread)
_qkf_snapshot = {
    'estimated_position': [0.0, 0.0, 0.0],
    'uncertainty':        0.1,
    'coherence':          1.0,
    'measurement_count':  0,
}
# Shared MR-thermometry segmentation snapshot
_mr_seg_snapshot = {
    'centroid':           [0.5, 0.5],
    'tumor_volume_mm2':   0.0,
    'ablation_coverage':  0.0,
    'necrosis_fraction':  0.0,
}
gt_controller = GameTheoryController() if HAS_GAME_THEORY else None
vasculature = VasculatureSpectralAnalyzer(num_nodes=64) if HAS_VASCULATURE else None
guidance = None
five_g_guidance = None
laser_optimizer = LaserDeliveryOptimizer() if HAS_5G_GUIDANCE else None

# State variables
simulation_running = True
laser_enabled = False
cryo_enabled = False
ablation_active = False
target_pos = np.array([0.5, 0.0, 0.5])
CONTROL_LOOP_HZ = 50.0

# Visualization caching
_viz_cache = {
    'temp_viz': None,
    'cryo_viz': None,
    'cache_time': 0,
    'cache_interval': 0.2  # Update visualizations every 200ms, not every 100ms
}

def _should_update_viz():
    """Check if it's time to update cached visualizations"""
    return (time.time() - _viz_cache['cache_time']) > _viz_cache['cache_interval']

def _update_viz_cache():
    """Update cached visualizations"""
    global _viz_cache
    
    if not _should_update_viz():
        return
    
    try:
        temp_map = thermo.get_map()
        damage_map = thermo.get_damage_map()
        cryo_map = cryo.get_ice_map()
        seg_data = segmentation.get_visualization_data()
        
        # Create thermal visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#0f172a')
        
        ax1 = axes[0]
        im1 = ax1.imshow(temp_map, cmap='hot', vmin=20, vmax=80)
        ax1.set_title('Temperature (°C)', color='white', fontsize=9, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='°C')
        
        ax2 = axes[1]
        ax2.imshow(temp_map, cmap='gray', alpha=0.7)
        ax2.contour(seg_data['tumor_mask'], levels=[0.5], colors='red', linewidths=2)
        ax2.contour(seg_data['safe_zone'], levels=[0.5], colors='yellow', linewidths=1, linestyles='--')
        ax2.set_title('Tumor & Safety', color='white', fontsize=9, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#0f172a', dpi=80)
        buf.seek(0)
        plt.close(fig)
        _viz_cache['temp_viz'] = base64.b64encode(buf.getvalue()).decode()
        
        # Create cryo visualization
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0f172a')
        im = ax.imshow(cryo_map, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Cryo Ice Ball', color='white', fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Ice Fraction')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#0f172a', dpi=80)
        buf.seek(0)
        plt.close(fig)
        _viz_cache['cryo_viz'] = base64.b64encode(buf.getvalue()).decode()
        
        _viz_cache['cache_time'] = time.time()
    except Exception as e:
        print(f"Visualization cache error: {e}")


def get_system_state():
    """Expose runtime state for the UI without NVQLink-specific telemetry."""
    return {
        'status': 'CONNECTED' if simulation_running else 'DISCONNECTED',
        'loop_hz': CONTROL_LOOP_HZ,
        'simulation_running': simulation_running,
        'modules': {
            'robot': True,
            'thermal': True,
            'cryo': True,
            'segmentation': True,
            'guidance': HAS_GUIDANCE,
            'vasculature': HAS_VASCULATURE,
            'game_theory': HAS_GAME_THEORY,
        },
    }


def simulation_loop():
    """Main simulation loop with enhanced surgical physics"""
    global simulation_running, laser_enabled, cryo_enabled, ablation_active, target_pos
    global guidance, five_g_guidance, _qkf_snapshot, _mr_seg_snapshot
    
    loop_count = 0
    
    while simulation_running:
        loop_count += 1
        
        # 1. Robot Control - Update towards target
        robot.update_control(target_pos)
        current_pos, current_T = robot.forward_kinematics(robot.joints)

        # --- 2a. Quantum Kalman Filter: predict then update with FK measurement ---
        qkf.predict()
        qkf_estimated_state, qkf_coherence = qkf.update(current_pos)
        _qkf_snapshot = {
            'estimated_position': qkf_estimated_state[:3].tolist(),
            'uncertainty':        float(qkf.get_uncertainty()),
            'coherence':          float(qkf_coherence),
            'measurement_count':  _qkf_snapshot['measurement_count'] + 1,
        }
        # Use QKF-corrected position for downstream thermal targeting
        qkf_pos = qkf.get_position_estimate()

        # Check for 5G guidance first (highest priority)
        if five_g_guidance and five_g_guidance.active:
            next_target, should_fire = five_g_guidance.get_next_waypoint(current_pos)
            if next_target is not None:
                target_pos = np.array([next_target[0], target_pos[1], next_target[1]])
                laser_enabled = should_fire
                ablation_active = should_fire
            else:
                five_g_guidance.active = False
                laser_enabled = False
                ablation_active = False
        # Then check for regular guidance
        elif guidance and guidance.active:
            guided_target, fire_laser = guidance.get_next_target(current_pos)
            if guided_target is not None:
                target_pos = np.array([guided_target[0], target_pos[1], guided_target[1]])
                laser_enabled = fire_laser
                ablation_active = fire_laser
            elif guidance.completed:
                laser_enabled = False
                ablation_active = False
        
        # 2. Map robot position to thermal grid
        tx = np.clip(current_pos[0], 0, 1.0)
        ty = np.clip(current_pos[1] + 0.5, 0, 1.0)  # Offset for centered coordinate system
        tz = np.clip(current_pos[2], 0, 1.0)
        
        # 3. Update level-set and MR-thermometry segmentation periodically
        if loop_count % 5 == 0:
            segmentation.evolve(thermo.get_map(), iterations=2)
        
        if loop_count % 10 == 0:
            seg_result = mr_thermo_seg.segment_from_thermometry(
                thermo.get_map(), thermo.get_damage_map()
            )
            _mr_seg_snapshot = {
                'centroid':          seg_result['centroid'].tolist(),
                'tumor_volume_mm2':  seg_result['tumor_volume_mm2'],
                'ablation_coverage': seg_result['ablation_coverage'],
                'necrosis_fraction': seg_result['necrosis_fraction'],
            }
            # Auto-guide laser to thermometry-derived tumour centroid when
            # no other guidance system is active
            if not (guidance and guidance.active) and not (five_g_guidance and five_g_guidance.active):
                thermo_target = mr_thermo_seg.get_laser_target()
                if seg_result['tumor_volume_mm2'] > 4:
                    target_pos = np.array([
                        thermo_target[0],
                        target_pos[1],
                        thermo_target[1],
                    ])
        
        # Get tumor location from segmentation
        tumor_center = segmentation.get_center_of_mass()
        
        # 4. Apply laser heating – use QKF-corrected position for precision targeting
        # Map QKF estimated position to thermal grid
        qkf_tx = np.clip(qkf_pos[0], 0, 1.0)
        qkf_tz = np.clip(qkf_pos[2] if len(qkf_pos) > 2 else qkf_pos[1], 0, 1.0)
        if laser_enabled and ablation_active:
            # Dynamic power based on temperature
            current_temp = thermo.T[int(tz * 128), int(tx * 128)]
            
            # Use adaptive laser power if 5G optimizer is available
            if laser_optimizer and five_g_guidance and five_g_guidance.active:
                power_watts = laser_optimizer.compute_optimal_power(current_temp)
            else:
                power_watts = 60.0 if current_temp < 50 else 30.0
            
            # Deliver laser at QKF-estimated position for sub-mm accuracy
            thermo.apply_heat_source(qkf_tx, qkf_tz, power_watts=power_watts, radius_mm=2.5)
        
        # 5. Apply cryo-ablation
        if cryo_enabled:
            cryo.activate_cryoprobe(tx, tz, power_pct=80.0)
        else:
            cryo.deactivate_cryoprobe()
        
        # 6. Update thermal physics
        thermo.update()
        cryo.update()
        
        time.sleep(1.0 / CONTROL_LOOP_HZ)


# Start simulation thread
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def get_telemetry():
    """Get real-time telemetry data - optimized for fast response"""
    # Update visualizations cache (non-blocking)
    _update_viz_cache()
    
    pos, T = robot.forward_kinematics(robot.joints)
    safety = robot.get_safety_status()
    
    temp_map = thermo.get_map()
    damage_map = thermo.get_damage_map()
    cryo_map = cryo.get_ice_map()
    quantum_metrics = {}
    if hasattr(robot, 'get_quantum_metrics'):
        quantum_metrics = robot.get_quantum_metrics()
    guidance_state = {
        'active': guidance.active if guidance else False,
        'completed': guidance.completed if guidance else False,
    }
    vasculature_analysis = vasculature.get_analysis() if vasculature else {}
    
    cryo_metrics = cryo.get_damage_metrics()
    thermo_metrics = thermo.get_performance_metrics()
    
    return jsonify(numpy_to_native({
        'joints': robot.get_joint_angles_degrees().tolist(),
        'position': pos.tolist(),
        'target_pos': target_pos.tolist(),
        'temperature_map': temp_map.tolist(),
        'damage_map': damage_map.tolist(),
        'cryo_map': cryo_map.tolist(),
        'mr_anatomy': thermo.tissue_type.tolist(),
        'temp_history': thermo.get_history()[-100:],
        'laser_enabled': laser_enabled,
        'cryo_enabled': cryo_enabled,
        'ablation_active': ablation_active,
        'laser_pos': [float(np.clip(pos[0], 0, 1.0)), float(np.clip(pos[2], 0, 1.0))],
        'guidance': guidance_state,
        'system': get_system_state(),
        'quantum': {
            'enabled': bool(quantum_metrics),
            'metrics': quantum_metrics,
        },
        'vasculature': vasculature_analysis,
        'robot': {
            'position_error_m': float(robot.position_error),
            'safety': safety,
        },
        'thermal': {
            'max_temperature': float(np.max(temp_map)),
            'mean_temperature': float(np.mean(temp_map)),
            'visualization': _viz_cache['temp_viz'],
            'metrics': thermo_metrics,
        },
        'cryo': {
            'visualization': _viz_cache['cryo_viz'],
            'metrics': cryo_metrics,
            'active': cryo.probe_active,
        },
        'segmentation': {
            'tumor_center': segmentation.get_center_of_mass().tolist(),
            'tumor_volume_pixels': segmentation.get_tumor_volume_pixels(),
            'quality_metrics': segmentation.evaluate_segmentation_quality(),
        },
        'qkf_localization': {
            'estimated_position': _qkf_snapshot['estimated_position'],
            'raw_position':       pos.tolist(),
            'uncertainty':        _qkf_snapshot['uncertainty'],
            'coherence':          _qkf_snapshot['coherence'],
            'measurement_count':  _qkf_snapshot['measurement_count'],
            'residual_norm_mm':   float(np.linalg.norm(
                np.array(_qkf_snapshot['estimated_position']) - pos
            ) * 1000),
        },
        'mr_thermometry_seg': _mr_seg_snapshot,
        'control': {
            'laser_enabled': laser_enabled,
            'cryo_enabled': cryo_enabled,
            'ablation_active': ablation_active,
        },
    }))

@app.route('/api/control', methods=['POST'])
def control():
    """Control robot, laser, and cryo"""
    global laser_enabled, cryo_enabled, ablation_active, target_pos
    
    data = request.json or {}
    
    if 'target_pos' in data:
        target_pos = np.array([
            data['target_pos']['x'],
            data['target_pos']['y'],
            data['target_pos']['z']
        ])
    elif 'target' in data:
        target_pos = np.array([
            float(data['target']['x']),
            float(data['target'].get('y', 0.0)),
            float(data['target']['z'])
        ])
    
    if 'laser' in data:
        laser_enabled = bool(data['laser'])
        if 'ablation' not in data:
            ablation_active = laser_enabled
    
    if 'cryo' in data:
        cryo_enabled = bool(data['cryo'])
    
    if 'ablation' in data:
        ablation_active = bool(data['ablation'])
    
    if 'home' in data and data['home']:
        robot.home()
    
    return jsonify({'status': 'ok', 'updated': True})

@app.route('/api/guidance', methods=['POST'])
def toggle_guidance():
    """Toggle automated guidance for tumor coverage."""
    global guidance, laser_enabled, ablation_active

    if not HAS_GUIDANCE:
        return jsonify({'status': 'unavailable', 'enabled': False}), 501

    data = request.json or {}
    enabled = bool(data.get('enabled', False))

    if enabled:
        guidance_map = segmentation.get_visualization_data()['ablation_region']
        guidance = AutomatedGuidanceSystem(
            guidance_map,
            width=guidance_map.shape[0],
            height=guidance_map.shape[1],
        )
        guidance.start()
    else:
        if guidance:
            guidance.stop()
        laser_enabled = False
        ablation_active = False

    return jsonify({
        'status': 'ok',
        'enabled': enabled,
        'guidance_active': guidance.active if guidance else False,
    })


@app.route('/api/guidance/5g', methods=['POST'])
def toggle_5g_guidance():
    """Toggle 5G neural pathway guided ablation system."""
    global five_g_guidance, laser_enabled, ablation_active
    
    if not HAS_5G_GUIDANCE:
        return jsonify({'status': 'unavailable', 'enabled': False}), 501
    
    data = request.json or {}
    enabled = bool(data.get('enabled', False))
    
    if enabled:
        # Get segmentation and anatomy data
        seg_data = segmentation.get_visualization_data()
        tumor_mask = seg_data.get('tumor_mask', np.zeros((128, 128)))
        anatomy_map = thermo.tissue_type
        
        # Initialize 5G guidance system
        five_g_guidance = FiveGNeuralPathway(
            anatomy_map=anatomy_map,
            tumor_mask=tumor_mask,
            width=anatomy_map.shape[0],
            height=anatomy_map.shape[1],
        )
        five_g_guidance.active = True
        five_g_guidance.current_idx = 0
        
        laser_enabled = False  # Will be controlled by 5G system
        ablation_active = False
    else:
        if five_g_guidance:
            five_g_guidance.active = False
        laser_enabled = False
        ablation_active = False
    
    return jsonify({
        'status': 'ok',
        'enabled': enabled,
        '5g_status': 'active' if five_g_guidance and five_g_guidance.active else 'inactive',
    })


@app.route('/api/guidance/5g/status')
def get_5g_guidance_status():
    """Get 5G guidance system status and visualization."""
    if not five_g_guidance:
        return jsonify({
            'active': False,
            'completed': False,
            'progress': 0.0,
            'current_waypoint': 0,
            'total_waypoints': 0,
            'trajectory': [],
            'neural_pathways': [],
            'safe_corridors': [],
        })
    
    viz_data = five_g_guidance.get_visualization_data()
    
    return jsonify(numpy_to_native({
        'active': bool(five_g_guidance.active),
        'completed': five_g_guidance.completed,
        'progress': viz_data['progress'],
        'current_waypoint': five_g_guidance.current_idx,
        'total_waypoints': len(five_g_guidance.optimal_trajectory),
        'trajectory': viz_data['trajectory'].tolist() if viz_data['trajectory'] is not None else [],
        'neural_pathways': viz_data['neural_pathways'].tolist() if viz_data['neural_pathways'] is not None else [],
        'safe_corridors': viz_data['safe_corridors'].tolist() if viz_data['safe_corridors'] is not None else [],
    }))


@app.route('/api/quantum/status')
def quantum_status():
    """Get quantum metrics exposed by the robot controller."""
    metrics = robot.get_quantum_metrics() if hasattr(robot, 'get_quantum_metrics') else {}
    return jsonify(numpy_to_native({
        'enabled': bool(metrics),
        'coherence': metrics.get('coherence', 0.0),
        'uncertainty': metrics.get('uncertainty', 0.0),
        'qml_fidelity': metrics.get('qml_fidelity', 0.0),
        'tracking_error': metrics.get('tracking_error', 0.0),
        'avg_tracking_error': metrics.get('avg_tracking_error', 0.0),
    }))


@app.route('/api/quantum/train', methods=['POST'])
def train_quantum():
    """Train the robot's quantum model if the implementation supports it."""
    if not hasattr(robot, 'train_qml'):
        return jsonify({'error': 'Training not supported'}), 501

    data = request.json or {}
    num_steps = int(data.get('steps', 10))
    losses = robot.train_qml(num_steps)
    return jsonify(numpy_to_native({
        'status': 'training_complete',
        'steps': num_steps,
        'final_loss': losses[-1] if losses else 0.0,
        'loss_history': losses,
    }))


@app.route('/api/reports/quantum_kalman')
def get_quantum_report():
    """Return metadata for the bundled quantum robotics report."""
    report_path = os.path.join(
        os.path.dirname(__file__),
        'Quantum_Kalman_Surgical_Robotics_Report.tex',
    )
    if not os.path.exists(report_path):
        return jsonify({'available': False})

    return jsonify({
        'available': True,
        'path': report_path,
        'format': 'LaTeX',
        'title': 'Quantum Kalman Operators for Advanced Pose Estimation in Neurosurgical Robotics',
    })

@app.route('/api/trajectory/plan', methods=['POST'])
def plan_trajectory():
    """Plan robot trajectory to position"""
    data = request.json or {}
    target = np.array([data['x'], data['y'], data['z']])
    duration = data.get('duration', 5.0)
    path_type = data.get('path_type', 'linear')
    
    trajectory = robot.plan_trajectory(target, duration=duration, path_type=path_type)
    
    return jsonify({
        'planned': len(trajectory) > 0,
        'waypoints': len(trajectory),
        'duration': duration,
    })

@app.route('/api/segmentation/quality')
def segmentation_quality():
    """Get segmentation quality metrics"""
    metrics = segmentation.evaluate_segmentation_quality()
    
    return jsonify(numpy_to_native({
        'quality': 'excellent' if metrics['circularity'] > 0.7 else 'good',
        'metrics': metrics,
        'tumor_ready_for_ablation': metrics['circularity'] > 0.6,
    }))

@app.route('/api/ablation/plan', methods=['POST'])
def plan_ablation():
    """Plan ablation trajectory for tumor"""
    data = request.json or {}
    method = data.get('method', 'sequential')  # sequential, concentric, spiral
    
    tumor_center = segmentation.get_center_of_mass()
    ablation_region = segmentation.get_ablation_region()
    
    # Get ablation waypoints
    if method == 'sequential':
        # Simple center targeting
        waypoints = [tumor_center]
    elif method == 'concentric':
        # Concentric circles around tumor
        center = tumor_center
        waypoints = []
        for radius in np.linspace(0.02, 0.08, 4):
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x = center[0] + radius * np.cos(angle)
                z = center[1] + radius * np.sin(angle)
                waypoints.append([np.clip(x, 0, 1), np.clip(z, 0, 1)])
    else:  # spiral
        center = tumor_center
        waypoints = []
        for t in np.linspace(0, 2*np.pi, 20):
            r = 0.02 + 0.04 * (t / (2*np.pi))
            x = center[0] + r * np.cos(t)
            z = center[1] + r * np.sin(t)
            waypoints.append([np.clip(x, 0, 1), np.clip(z, 0, 1)])
    
    return jsonify(numpy_to_native({
        'method': method,
        'waypoints': waypoints,
        'tumor_center': tumor_center.tolist(),
        'tumor_center_px': [int(tumor_center[0]*128), int(tumor_center[1]*128)],
    }))

@app.route('/api/quantum/localization')
def quantum_localization():
    """Quantum Kalman Filter end-effector localisation state."""
    pos, _ = robot.forward_kinematics(robot.joints)
    est    = np.array(_qkf_snapshot['estimated_position'])
    residual = pos - est
    return jsonify(numpy_to_native({
        'qkf_state':            qkf.state.tolist(),
        'qkf_covariance_trace': float(np.trace(qkf.P)),
        'qkf_coherence':        float(qkf.coherence),
        'qkf_uncertainty':      float(qkf.get_uncertainty()),
        'estimated_position':   est.tolist(),
        'measured_position':    pos.tolist(),
        'position_residual':    residual.tolist(),
        'residual_norm_mm':     float(np.linalg.norm(residual) * 1000),
        'measurement_count':    _qkf_snapshot['measurement_count'],
    }))


@app.route('/api/thermal/tumor_segmentation')
def thermal_tumor_segmentation():
    """On-demand MR-thermometry tumour segmentation snapshot."""
    seg = mr_thermo_seg.segment_from_thermometry(
        thermo.get_map(), thermo.get_damage_map()
    )
    return jsonify(numpy_to_native({
        'centroid':                   seg['centroid'].tolist(),
        'centroid_px':                seg['centroid_px'].tolist(),
        'tumor_volume_mm2':           seg['tumor_volume_mm2'],
        'ablation_coverage':          seg['ablation_coverage'],
        'necrosis_fraction':          seg['necrosis_fraction'],
        # Down-sampled masks for network efficiency (128→32)
        'tumor_mask_ds':              seg['tumor_mask'][::4, ::4].astype(int).tolist(),
        'ablation_mask_ds':           seg['ablation_mask'][::4, ::4].astype(int).tolist(),
        'necrosis_mask_ds':           seg['necrosis_mask'][::4, ::4].astype(int).tolist(),
        'delta_T_ds':                 seg['delta_T'][::4, ::4].tolist(),
    }))


@app.route('/api/thermal/history')
def thermal_history():
    """Get thermal history for plotting"""
    history = thermo.get_history()
    
    return jsonify(numpy_to_native({
        'max_temps': history[-100:] if len(history) > 100 else history,
        'simulation_time_s': thermo.accumulated_time,
    }))

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    print(f"\n{'='*60}")
    print(f"🚀 NeuroMorph Surgical Robotics Platform")
    print(f"{'='*60}")
    print(f"Starting Server on http://{host}:{port}")
    print(f"Simulation Loop: Active (50Hz)")
    print(f"Components: Thermal | Cryo | Robot | Segmentation")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=False)

