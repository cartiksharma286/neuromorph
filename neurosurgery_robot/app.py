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
from matplotlib.colors import LinearSegmentedColormap

from precision_kinematics import PrecisionRobot6DOF
from slam_kinematics import SlamEnhancedRobot
from enhanced_thermometry import EnhancedThermometry
from enhanced_cryo import EnhancedCryoModule
from level_set_segmentation import LevelSetSegmentation
from quantum_kalman import QuantumKalmanFilter
from tumor_thermometry_segmentation import MRThermometrySegmenter
from thermal_neuro_morphometry import ThermalNeuroMorphometry
from generative_heating import GenerativeTissueHeating
from nvqlink import NVQLinkBridge
from quantum_ml_segmentation import QuantumMLSegmenter

nvqlink = NVQLinkBridge()
qml_segmenter = QuantumMLSegmenter()

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
robot = SlamEnhancedRobot()
thermo = EnhancedThermometry(width=128, height=128)
cryo = EnhancedCryoModule(width=128, height=128)
segmentation = LevelSetSegmentation(width=128, height=128)

# Initialize segmentation from anatomy
segmentation.initialize_from_image(thermo.tissue_type)

# Quantum Kalman Filter for end-effector localization (6-DOF state, 3-D measurement)
qkf = QuantumKalmanFilter(state_dim=6, measurement_dim=3)

# MR-Thermometry tumour segmenter
mr_thermo_seg = MRThermometrySegmenter(width=128, height=128)

# Thermal neuro-morphometry engine
thermal_morphometry = ThermalNeuroMorphometry(width=128, height=128)

# Generative thermometry controller for target profile telemetry
gen_heating = GenerativeTissueHeating(width=128, height=128)
gen_heating.generate_heating_curve(duration_steps=160, mode="STANDARD")

# Shared QKF telemetry snapshot (written by simulation thread, read by HTTP thread)
_qkf_snapshot = {
    'estimated_position': [0.0, 0.0, 0.0],
    'uncertainty':        0.1,
    'coherence':          1.0,
    'measurement_count':  0,
}
# Shared MR-thermometry segmentation snapshot
# NOTE: also stores downsampled masks so HTTP endpoints never re-run computation
_z32  = [[0]   * 32 for _ in range(32)]
_zf32 = [[0.0] * 32 for _ in range(32)]
_mr_seg_snapshot = {
    'centroid':           [0.5, 0.5],
    'centroid_px':        [64.0, 64.0],
    'tumor_volume_mm2':   0.0,
    'ablation_coverage':  0.0,
    'necrosis_fraction':  0.0,
    'effector_norm':      [0.5, 0.5],
    'tumor_mask_ds':      [r[:] for r in _z32],
    'ablation_mask_ds':   [r[:] for r in _z32],
    'necrosis_mask_ds':   [r[:] for r in _z32],
    'delta_T_ds':         [r[:] for r in _zf32],
}
_thermal_morph_snapshot = {
    'centroid':                 [0.5, 0.5],
    'conformal_stability':      1.0,
    'distortion_mean':          0.0,
    'hotspot_area_px':          0,
    'thermal_shape_index':      0.0,
    'continued_fraction_terms': [1, 1, 1, 1],
    'continued_fraction_value': 1.0,
    'continued_fraction_depth': 4,
    'principal_ratio':          1.0,
    'curvature_energy':         0.0,
    'conformal_invariant_ds':   [r[:] for r in _zf32],
    'beltrami_ds':              [r[:] for r in _zf32],
    'hotspot_mask_ds':          [r[:] for r in _z32],
    'delta_t_ds':               [r[:] for r in _zf32],
}
# Shared level-set segmentation snapshot (written by sim thread, read by HTTP)
_ls_snapshot = {
    'boundary_ds':     [r[:] for r in _z32],
    'tumor_mask_ds':   [r[:] for r in _z32],
    'phi_ds':          [r[:] for r in _zf32],
    'safe_zone_ds':    [r[:] for r in _z32],
    'tumor_center':    [0.5, 0.5],
    'tumor_volume_px': 0,
    'circularity':     0.0,
    'solidity':        1.0,
    'boundary_length': 0,
}
gt_controller = GameTheoryController() if HAS_GAME_THEORY else None
vasculature = VasculatureSpectralAnalyzer(num_nodes=64) if HAS_VASCULATURE else None
guidance = None
five_g_guidance = None
laser_optimizer = LaserDeliveryOptimizer() if HAS_5G_GUIDANCE else None

# State variables
simulation_running = True # Default Control States
laser_enabled = False
cryo_enabled = False
ablation_active = False
target_pos = np.array([0.5, 0.5, 0.5])
cryo_lut_name = 'grayscale'
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
        
        # Create cryo visualization with custom Egg-shell LUT
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0f172a')
        
        lut_colors = ['#0f172a', '#312e81', '#3730a3', '#4338ca', '#3b82f6', '#0ea5e9', '#22d3ee', '#67e8f9'] # Deep blue -> Indigo -> Vibrant Cyan
        custom_egg_lut = LinearSegmentedColormap.from_list('egg_lut', lut_colors)
        
        im = ax.imshow(cryo_map, cmap=custom_egg_lut, vmin=0, vmax=1)
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
    global guidance, five_g_guidance, _qkf_snapshot, _mr_seg_snapshot, _thermal_morph_snapshot, _ls_snapshot
    
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
            # Capture level-set snapshot for HTTP serving
            _ls_viz   = segmentation.get_visualization_data()
            _ls_qual  = segmentation.evaluate_segmentation_quality()
            _ls_snapshot = {
                'boundary_ds':     _ls_viz['boundary_map'][::4, ::4].astype(int).tolist(),
                'tumor_mask_ds':   _ls_viz['tumor_mask'][::4, ::4].astype(int).tolist(),
                'phi_ds':          np.clip(_ls_viz['level_set'], -20, 20)[::4, ::4].tolist(),
                'safe_zone_ds':    _ls_viz['safe_zone'][::4, ::4].astype(int).tolist(),
                'tumor_center':    _ls_viz['tumor_center'].tolist(),
                'tumor_volume_px': int(_ls_qual['tumor_volume_pixels']),
                'circularity':     float(_ls_qual['circularity']),
                'solidity':        float(_ls_qual['solidity']),
                'boundary_length': int(_ls_qual['boundary_length']),
            }
        
        if loop_count % 10 == 0:
            seg_result = mr_thermo_seg.segment_from_thermometry(
                thermo.get_map(), thermo.get_damage_map()
            )
            _mr_seg_snapshot = {
                'centroid':          seg_result['centroid'].tolist(),
                'centroid_px':       seg_result['centroid_px'].tolist(),
                'tumor_volume_mm2':  seg_result['tumor_volume_mm2'],
                'ablation_coverage': seg_result['ablation_coverage'],
                'necrosis_fraction': seg_result['necrosis_fraction'],
                'effector_norm':     [float(np.clip(current_pos[0], 0, 1.0)),
                                      float(np.clip(current_pos[2], 0, 1.0))],
                'tumor_mask_ds':     seg_result['tumor_mask'][::4, ::4].astype(int).tolist(),
                'ablation_mask_ds':  seg_result['ablation_mask'][::4, ::4].astype(int).tolist(),
                'necrosis_mask_ds':  seg_result['necrosis_mask'][::4, ::4].astype(int).tolist(),
                'delta_T_ds':        seg_result['delta_T'][::4, ::4].tolist(),
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

            morph_result = thermal_morphometry.analyze(
                thermo.get_map(), thermo.get_damage_map(), thermo.tissue_type
            )
            _thermal_morph_snapshot = {
                'centroid':                 morph_result['centroid'].tolist(),
                'conformal_stability':      morph_result['conformal_stability'],
                'distortion_mean':          morph_result['distortion_mean'],
                'hotspot_area_px':          morph_result['hotspot_area_px'],
                'thermal_shape_index':      morph_result['thermal_shape_index'],
                'continued_fraction_terms': morph_result['continued_fraction_terms'],
                'continued_fraction_value': morph_result['continued_fraction_value'],
                'continued_fraction_depth': morph_result['continued_fraction_depth'],
                'principal_ratio':          morph_result['principal_ratio'],
                'curvature_energy':         morph_result['curvature_energy'],
                'conformal_invariant_ds':   morph_result['conformal_invariant_map'][::4, ::4].tolist(),
                'beltrami_ds':              morph_result['beltrami_map'][::4, ::4].tolist(),
                'hotspot_mask_ds':          morph_result['hotspot_mask'][::4, ::4].astype(int).tolist(),
                'delta_t_ds':               morph_result['delta_t'][::4, ::4].tolist(),
            }
        
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
    
    # Update visualizations cache
    _update_viz_cache()
    
    pos, T = robot.forward_kinematics(robot.joints)
    safety = robot.get_safety_status()
    
    temp_map = thermo.get_map()
    damage_map = thermo.get_damage_map()
    cryo_map = cryo.get_ice_map()
    quantum_metrics = {}
    if hasattr(robot, 'get_quantum_metrics'):
        quantum_metrics = robot.get_quantum_metrics()
    
    # Global state variables (ensure these are accessible)
    # These are defined as globals in app.py
    global laser_enabled, cryo_enabled, ablation_active, target_pos, cryo_lut_name
    
    guidance_state = {
        'active': guidance.active if guidance else False,
        'completed': guidance.completed if guidance else False,
    }
    vasculature_analysis = vasculature.get_analysis() if vasculature else {}
    
    cryo_metrics = cryo.get_damage_metrics()
    thermo_metrics = thermo.get_performance_metrics()
    current_max_temp = float(np.max(temp_map))
    ai_state = gen_heating.get_control_action(current_max_temp)
    generated_profile = gen_heating.generated_profile
    if isinstance(generated_profile, np.ndarray):
        generated_profile = generated_profile.tolist()
    elif generated_profile is None:
        generated_profile = []
        
    # ── Level-Set + QML Pinpoint Geometry ───────────────────────────
    ls_mask = segmentation.tumor_mask # 128x128
    qml_mask = qml_segmenter.refine_geometry(ls_mask, thermo.tissue_type)
    
    # Extract explicit RGB map for Cryo Egg geometries, pinpointed at QML/LS geometry
    # We use the QML mask to modulate the cryo profile visibility at the tumor
    refined_cryo = cryo_map * (0.4 + 0.6 * qml_mask)
    
    # Professional LUT definitions
    if cryo_lut_name == 'spectral_ice':
        lut_colors = ['#0f172a', '#1d4ed8', '#2563eb', '#3b82f6', '#06b6d4', '#22d3ee', '#ec4899', '#ffffff']
    elif cryo_lut_name == 'clinical_prf':
        lut_colors = ['#0f172a', '#0ea5e9', '#0284c7', '#0369a1', '#e0f2fe']
    elif cryo_lut_name == 'deep_frost':
        lut_colors = ['#0f172a', '#172554', '#1e3a8a', '#1e40af', '#3b82f6', '#93c5fd']
    elif cryo_lut_name == 'grayscale':
        lut_colors = ['#000000', '#222222', '#555555', '#888888', '#aaaaaa', '#dddddd', '#ffffff']
    elif cryo_lut_name == 'frost_edge':
        lut_colors = ['#083344', '#0e7490', '#22d3ee', '#a5f3fc', '#ffffff']
    else:
        lut_colors = ['#0f172a', '#312e81', '#3730a3', '#4338ca', '#3b82f6']
        
    custom_egg_lut = LinearSegmentedColormap.from_list('dynamic_lut', lut_colors)
    rgb_map = custom_egg_lut(refined_cryo)[:, :, :3]
    
    # Downsample matrix to prevent heavy JSON overhead while providing pure RGB map representation
    rgb_map_ds = rgb_map[::2, ::2, :].tolist()
    
    # ── Multimodal Reasoning Engine ──
    # Synthesize MRI, Cryo, Thermo, and QML metrics into actionable clinical text.
    reasoning_insights = []
    if cryo.probe_active:
        reasoning_insights.append("🧊 Cryo-Ablation ACTIVE.")
        if cryo_metrics.get('fully_frozen_pixels', 0) > 10:
            reasoning_insights.append(f"Lethal ice core expanding (Radius: {cryo_metrics['ice_ball_radius_mm']:.1f} mm).")
        else:
            reasoning_insights.append("Initial ice crystal formation phase.")
            
        # Check intersection with QML mask
        ice_tumor_overlap = np.sum((cryo_map > 0.5) & (qml_mask > 0.5))
        tumor_size = np.sum(qml_mask > 0.5)
        if tumor_size > 0:
            coverage = ice_tumor_overlap / tumor_size
            reasoning_insights.append(f"Tumor target coverage: {coverage*100:.1f}%.")
            if coverage > 0.95:
                reasoning_insights.append("✅ Optimal ablation margins achieved. Recommend transition to thawing.")
    else:
        if current_max_temp > 45:
            reasoning_insights.append("🔥 Hyperthermic ablation in progress.")
        else:
            reasoning_insights.append("⏸ System in observation mode.")
            
    if qml_segmenter and qml_segmenter.get_qml_metadata()['spectral_coherence'] > 0.98:
        reasoning_insights.append("⚛ QML pinpoint geometry maintains high coherence. Boundaries are locked.")
        
    reasoning_text = " ".join(reasoning_insights)

    # Wrap in NVQLink Packet
    qml_metadata = qml_segmenter.get_qml_metadata()
    cryo_packet = nvqlink.package_rgb_matrix(rgb_map_ds)
    cryo_packet['qml_metrics'] = qml_metadata
    cryo_packet['cryo_lut_colors'] = lut_colors
    cryo_packet['brin_state'] = qml_metadata.get('brin_state', 0)
    
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
        'slam': robot.get_slam_metrics() if hasattr(robot, 'get_slam_metrics') else {},
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
        'gen_ai': {
            'target_temp': ai_state['target_temp'],
            'model_state': ai_state['model_state'],
            'mode': ai_state['mode'],
            'generated_profile': generated_profile,
        },
        'cryo': {
            'visualization': _viz_cache['cryo_viz'],
            'nvqlink_packet': cryo_packet,
            'metrics': cryo_metrics,
            'active': cryo.probe_active,
        },
        'multimodal_reasoning': reasoning_text,
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
        'thermal_neuro_morphometry': _thermal_morph_snapshot,
        'control': {
            'laser_enabled': laser_enabled,
            'cryo_enabled': cryo_enabled,
            'ablation_active': ablation_active,
        },
    }))

@app.route('/api/control', methods=['POST'])
def control():
    """Control robot, laser, and cryo"""
    global laser_enabled, cryo_enabled, ablation_active, target_pos, cryo_lut_name
    
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
        
    if 'cryo_color_profile' in data:
        cryo_lut_name = str(data['cryo_color_profile'])
    
    if 'ablation' in data:
        ablation_active = bool(data['ablation'])

    if 'mode' in data:
        requested_mode = str(data['mode']).upper()
        mode_map = {
            'STANDARD': 'STANDARD',
            'GENTLE': 'GENTLE',
            'RAPID': 'RAPID',
        }
        gen_heating.generate_heating_curve(
            duration_steps=160,
            mode=mode_map.get(requested_mode, 'STANDARD')
        )
    
    if 'home' in data and data['home']:
        robot.home()
    
    return jsonify({'status': 'ok', 'updated': True})

@app.route('/api/laser/precision_target', methods=['POST'])
def precision_target():
    """Snap robot target to MR-thermometry tumour centroid for precision ablation."""
    global target_pos, laser_enabled, ablation_active
    data = request.json or {}
    centroid = _mr_seg_snapshot.get('centroid', [0.5, 0.5])
    cx = float(np.clip(centroid[0], 0.0, 1.0))
    cz = float(np.clip(centroid[1], 0.0, 1.0))
    target_pos = np.array([cx, target_pos[1], cz])
    fire = bool(data.get('fire', False))
    if fire:
        laser_enabled = True
        ablation_active = True
    return jsonify({
        'status': 'ok',
        'target': [cx, cz],
        'tumor_volume_mm2': _mr_seg_snapshot.get('tumor_volume_mm2', 0.0),
        'laser': laser_enabled,
    })


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
        gen_heating.generate_heating_curve(duration_steps=160, mode='GENTLE')
    else:
        if guidance:
            guidance.stop()
        laser_enabled = False
        ablation_active = False
        gen_heating.generate_heating_curve(duration_steps=160, mode='STANDARD')

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


@app.route('/api/segmentation/level_set')
def level_set_segmentation():
    """Level-set tumour segmentation snapshot (served from simulation cache)."""
    return jsonify(numpy_to_native(_ls_snapshot))


@app.route('/api/thermal/tumor_segmentation')
def thermal_tumor_segmentation():
    """MR-thermometry tumour segmentation snapshot (served from simulation cache)."""
    return jsonify(numpy_to_native(_mr_seg_snapshot))


@app.route('/api/thermal/neuro_morphometry')
def thermal_neuro_morphometry():
    """Thermal neuro-morphometry snapshot with conformal invariants (served from simulation cache)."""
    return jsonify(numpy_to_native(_thermal_morph_snapshot))


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
    # Enable threaded mode to handle multiple concurrent telemetry requests from the UI
    app.run(host=host, port=port, debug=False, threaded=True)

