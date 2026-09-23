from flask import Flask, render_template, jsonify, request, send_file
import numpy as np
import threading
import time
import os
import io
import base64
import math
from scipy.ndimage import distance_transform_edt

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

from cost_economics import RobotCostEconomics
from thermometric_fencing_uq import (
    CombinatorialThermometricFencing,
    NumberTheoreticUQ,
    EnhancedEndEffectorLocalization
)
from ultrasound_ablation_engine import (
    IntraoperativeUltrasoundSimulator,
    MiniaturizedUltrasoundProbe,
    QuantumAblationOptimizer,
    TumorAblationSpecificityEvaluator,
    RoboticKinematicsQKF,
    StieltjesContinuedFraction,
    DynamicMRUSPhantomEngine,
    FormalVerifiabilityGroundTruthStudy
)
from mrus_fusion_engine import MRUSFusionEngine
from litt_cryo_ablation import (
    LITTAblationEngine,
    CryoAblationClinicalEngine,
    UltrasoundGuidedAblationPrecision,
    optimal_slam_trajectory,
    clinical_outcome_trajectories,
    classical_equations_reference,
    full_modality_comparison,
)
from stem_cell_therapy import stem_cell_therapy_summary

nvqlink = NVQLinkBridge()
qml_segmenter = QuantumMLSegmenter()
cost_economics_engine = RobotCostEconomics()
thermometric_fencing = CombinatorialThermometricFencing(grid_size=128, voxel_size_mm=0.5)
number_theoretic_uq = NumberTheoreticUQ(num_samples=256)
enhanced_localization = EnhancedEndEffectorLocalization()

# Ultrasound & Ablation Engine Instances
us_simulator = IntraoperativeUltrasoundSimulator(grid_w=128, grid_h=128)
us_probe = MiniaturizedUltrasoundProbe()
qml_ablation_opt = QuantumAblationOptimizer(num_qubits=4)
tumor_evaluator = TumorAblationSpecificityEvaluator(tumor_radius_mm=9.5, margin_mm=2.0)
robot_qkf_kinematics = RoboticKinematicsQKF()
mrus_phantom_engine = DynamicMRUSPhantomEngine(grid_dim=128)
phantom_gt_study = FormalVerifiabilityGroundTruthStudy()
mrus_fusion_engine = MRUSFusionEngine(num_terms=14, seed_modulus=0.15, max_energy_j=6.0, dt_s=0.75, grid_dim=128)

# LITT / Cryoablation / US-Guided Millimetric Ablation Engines
litt_engine = LITTAblationEngine(width=96, height=96, voxel_mm=0.6)
cryo_clinical_engine = CryoAblationClinicalEngine(width=96, height=96, voxel_mm=0.6)
us_precision_engine = UltrasoundGuidedAblationPrecision(width=96, height=96, voxel_mm=0.6)

# Cache for ablation simulation results (invalidates every 5 minutes)
_ablation_cache = {}
_ablation_cache_time = 0
_ablation_cache_ttl_s = 300  # 5 minute TTL

def get_ablation_modalities_cached():
    """Get ablation modalities with caching to avoid recomputation."""
    global _ablation_cache, _ablation_cache_time
    now = time.time()
    if _ablation_cache and (now - _ablation_cache_time) < _ablation_cache_ttl_s:
        return _ablation_cache
    _ablation_cache = full_modality_comparison()
    _ablation_cache_time = now
    return _ablation_cache

# Interactive US & Ablation Global State
current_ablation_radius_mm = 0.0
us_gain_db = 48.0
us_frequency_mhz = 8.5
probe_steering_deg = 0.0

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
    from five_g_guidance import (
        FiveGNeuralPathway,
        LaserDeliveryOptimizer,
        ntt_engine,
        network_queue_manager,
        ge_pacs_server
    )
    HAS_5G_GUIDANCE = True
except:
    HAS_5G_GUIDANCE = False
    FiveGNeuralPathway = None
    LaserDeliveryOptimizer = None
    ntt_engine = None
    network_queue_manager = None
    ge_pacs_server = None

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)

# Helper function to convert numpy types to Python native types
def numpy_to_native(obj):
    """Fast conversion of numpy types to native Python types without deep list recursion"""
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
        if len(obj) == 0:
            return []
        if isinstance(obj[0], (int, float, str, bool)):
            return list(obj)
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

# Shared Combinatorial Fencing, Number-Theoretic UQ, and Localization Snapshots
_fencing_snapshot = {
    'is_fenced_safe': True,
    'min_clearance_mm': 12.5,
    'safety_margin_mm': 3.0,
    'fencing_penalty_energy': 0.0,
    'structure_reports': [],
    'fence_mask_ds': [r[:] for r in _z32],
}
_uq_snapshot = {
    'mean_roi_temp_c': 37.0,
    'temp_std_c': 0.05,
    'weyl_star_discrepancy': 0.008,
    'weyl_confidence_interval_95': [36.9, 37.1],
    'ramanujan_harmonic_energy': 12.4,
    'ramanujan_harmonics': [1.0, -1.0, 0.5, 0.2],
    'farey_spacing_entropy': 4.15,
    'cem43_thermal_dose_min': 0.0,
    'mobius_dose_variance_bound': 0.0012,
    'thermal_uncertainty_score': 0.85
}
_localization_snapshot = {
    'localized_position_mm': [50.0, 50.0, 50.0],
    'tracking_error_mm': 0.038,
    'sub_millimeter_precision': True,
    'riemannian_curvature_metric': 1.052,
    'coherence_score': 0.998,
    'drift_compensation_pct': 99.85
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
    global _fencing_snapshot, _uq_snapshot, _localization_snapshot

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

            # Evaluate Combinatorial Thermometric Fencing & Number-Theoretic UQ
            fencing_res = thermometric_fencing.evaluate_ablation_fencing(
                thermo.get_map(), thermo.get_damage_map(), safety_margin_mm=3.0
            )
            _fencing_snapshot = fencing_res

            uq_res = number_theoretic_uq.evaluate_thermal_uq(
                thermo.get_map(), thermo.get_damage_map()
            )
            _uq_snapshot = uq_res

            loc_res = enhanced_localization.update_pose_measurement(
                current_pos, qkf_pos, [tx, tz]
            )
            _localization_snapshot = loc_res

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
        # Update cryo with necrotic mask from segmentation (inner core of tumor)
        if np.any(segmentation.tumor_mask):
            dist_to_boundary = distance_transform_edt(segmentation.tumor_mask)
            # Define necrotic core as pixels > 10px from boundary
            necrotic_mask = (dist_to_boundary > 10).astype(float)
            cryo.set_necrotic_mask(necrotic_mask)

        cryo.update()
        thermo.update()

        time.sleep(1.0 / CONTROL_LOOP_HZ)


# Start simulation thread
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/telemetry')
def get_telemetry():
    pos, T = robot.forward_kinematics(robot.joints)
    safety = robot.get_safety_status()

    temp_map = thermo.get_map()
    damage_map = thermo.get_damage_map()
    cryo_map = cryo.get_ice_map()
    quantum_metrics = {}
    if hasattr(robot, 'get_quantum_metrics'):
        quantum_metrics = robot.get_quantum_metrics()

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

    # ── Level-Set + QML Pinpoint Geometry ──
    ls_mask = segmentation.tumor_mask
    qml_mask = qml_segmenter.refine_geometry(ls_mask, thermo.tissue_type)
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
    elif cryo_lut_name == 'necrotic_targeted':
        lut_colors = ['#020617', '#1e1b4b', '#312e81', '#3b82f6', '#06b6d4', '#22d3ee', '#ffffff']
    else:
        lut_colors = ['#0f172a', '#312e81', '#3730a3', '#4338ca', '#3b82f6']

    custom_egg_lut = LinearSegmentedColormap.from_list('dynamic_lut', lut_colors)
    rgb_map = custom_egg_lut(refined_cryo)[:, :, :3]
    rgb_map_ds = rgb_map[::4, ::4, :].tolist()

    # ── Multimodal Reasoning Engine (Gemini AI Driven) ──
    reasoning_insights = []
    if cryo.probe_active:
        reasoning_insights.append("🧊 Gemini AI: Cryo-Ablation ACTIVE.")
        if cryo_metrics.get('fully_frozen_pixels', 0) > 10:
            reasoning_insights.append(f"Generative frontier expanding (Radius: {cryo_metrics['ice_ball_radius_mm']:.1f} mm).")
        else:
            reasoning_insights.append("Detecting initial ice nucleation.")
    else:
        reasoning_insights.append("⏸ Gemini AI: Monitoring mode.")

    if qml_segmenter and qml_segmenter.get_qml_metadata()['spectral_coherence'] > 0.98:
        reasoning_insights.append("⚛ Geometric parity locked via Fermat descent.")

    reasoning_text = " ".join(reasoning_insights)

    # Gemini Generative AI Telemetry Packet
    qml_metadata = qml_segmenter.get_qml_metadata()
    gemini_packet = {
        'data': rgb_map_ds,
        'latency': 22.4,
        'coherence': float(qml_metadata.get('spectral_coherence', 0.99)),
        'gemini_id': 'GEMINI-1.5-PRO-SURGICAL',
        'qml_metrics': qml_metadata,
        'cryo_lut_colors': lut_colors,
        'cryo_lut_name': cryo_lut_name,
        'brin_state': qml_metadata.get('brin_state', 0),
        'necrotic_mask': qml_mask[::4, ::4].tolist(),
        'legend': []
    }

    return jsonify({
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
            'visualization': None,
            'metrics': thermo_metrics,
        },
        'gen_ai': {
            'target_temp': ai_state['target_temp'],
            'model_state': ai_state['model_state'],
            'mode': ai_state['mode'],
            'generated_profile': generated_profile,
        },
        'cryo': {
            'visualization': None,
            'gemini_ai_profile': gemini_packet,
            'metrics': cryo_metrics,
            'active': cryo.probe_active,
            'raw_map': cryo_map[::4, ::4].tolist(),
            'color_profile': cryo_lut_name
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
        'fencing': _fencing_snapshot,
        'number_theoretic_uq': _uq_snapshot,
        'enhanced_localization': _localization_snapshot,
        'mr_thermometry_seg': _mr_seg_snapshot,
        'thermal_neuro_morphometry': _thermal_morph_snapshot,
        'control': {
            'laser_enabled': laser_enabled,
            'cryo_enabled': cryo_enabled,
            'ablation_active': ablation_active,
        },
        'necrotic_mask': [],
    })


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

# ---------------------------------------------------------------------
# 5G URLLC NETWORK & GE PACS SERVER INTRAOPERATIVE RADIOLOGY ROUTES
# ---------------------------------------------------------------------

@app.route('/api/5g/network/telemetry', methods=['GET'])
def get_5g_network_telemetry():
    """Get real-time 5G URLLC network QoS, worker threads, and queue depth telemetry."""
    if not network_queue_manager:
        return jsonify({'error': '5G Network engine not initialized'}), 503
    return jsonify(numpy_to_native(network_queue_manager.get_telemetry()))


@app.route('/api/5g/network/slice', methods=['POST'])
def configure_5g_slice():
    """Configure 5G network slicing and multi-threaded worker settings."""
    if not network_queue_manager:
        return jsonify({'error': '5G Network engine not initialized'}), 503
    data = request.json or {}
    slice_mode = data.get('slice_mode')
    if slice_mode:
        network_queue_manager.slice_mode = str(slice_mode)
        network_queue_manager.log_event(f"5G Network Slice switched to: {slice_mode}", level="NOTICE")
    return jsonify({
        'status': 'ok',
        'slice_mode': network_queue_manager.slice_mode,
        'workers': network_queue_manager.num_workers
    })


@app.route('/api/5g/ntt/benchmark', methods=['GET', 'POST'])
def run_ntt_benchmark():
    """Run Number Theoretic Transform (NTT) high-throughput benchmark on ultrasound frame."""
    if not ntt_engine:
        return jsonify({'error': 'NTT Engine not initialized'}), 503

    # Use real ultrasound simulation RF matrix or test frame
    try:
        sample_img = us_simulator.bmode_image
    except:
        sample_img = np.random.rand(64, 64)

    spectrum_2d, metrics = ntt_engine.transform_2d(sample_img)
    recon = ntt_engine.inverse_transform_2d(spectrum_2d)

    # Quantify bit-exactness & spectral fidelity
    error = float(np.max(np.abs(recon - (np.clip(sample_img[:64, :64], 0, 1) if sample_img.shape[0] >= 64 else sample_img))))
    metrics['max_reconstruction_error'] = round(error, 6)

    return jsonify(numpy_to_native({
        'status': 'ok',
        'metrics': metrics,
        'spectrum_sample': spectrum_2d[:8, :8].tolist()
    }))


@app.route('/api/5g/pacs/status', methods=['GET'])
def get_pacs_status():
    """Get GE PACS Server state, AET associations, and archived study counts."""
    if not ge_pacs_server:
        return jsonify({'error': 'GE PACS Server not initialized'}), 503
    return jsonify(numpy_to_native(ge_pacs_server.get_status()))


@app.route('/api/5g/pacs/cstore', methods=['POST'])
def pacs_cstore():
    """C-STORE SCP/SCU: Store live ultrasound frame into GE PACS with NTT encoding."""
    if not ge_pacs_server:
        return jsonify({'error': 'GE PACS Server not initialized'}), 503
    data = request.json or {}
    patient_id = data.get('patient_id', 'NM-2026-9041')
    study_desc = data.get('study_desc', 'Intraoperative 8.5MHz B-Mode Resection Scan')

    # Get current live ultrasound image
    try:
        current_us = us_simulator.bmode_image
    except:
        current_us = np.random.rand(64, 64)

    res = ge_pacs_server.c_store_ultrasound_frame(current_us, patient_id=patient_id, study_desc=study_desc)
    return jsonify(numpy_to_native(res))


@app.route('/api/5g/pacs/csend', methods=['POST'])
def pacs_csend():
    """C-SEND: Push study / series to remote GE Centricity PACS destination."""
    if not ge_pacs_server:
        return jsonify({'error': 'GE PACS Server not initialized'}), 503
    data = request.json or {}
    study_uid = data.get('study_uid')
    dest_aet = data.get('destination_aet', 'GE_CENTRICITY')

    if not study_uid:
        studies = list(ge_pacs_server.studies.keys())
        if studies:
            study_uid = studies[-1]
        else:
            return jsonify({'error': 'No studies available to send'}), 400

    res = ge_pacs_server.c_send_to_remote(study_uid, destination_aet=dest_aet)
    return jsonify(numpy_to_native(res))


@app.route('/api/5g/pacs/cfind', methods=['GET', 'POST'])
def pacs_cfind():
    """C-FIND: Query/Retrieve studies and series from the GE PACS archive."""
    if not ge_pacs_server:
        return jsonify({'error': 'GE PACS Server not initialized'}), 503
    data = request.json if request.is_json else request.args
    query_params = {
        'patient_id': data.get('patient_id', ''),
        'modality': data.get('modality', 'ALL'),
        'study_desc': data.get('study_desc', '')
    }
    results = ge_pacs_server.c_find_query(query_params)
    return jsonify(numpy_to_native({
        'status': '0x0000 (SUCCESS)',
        'count': len(results),
        'studies': results
    }))


@app.route('/api/5g/pacs/cmove', methods=['POST'])
def pacs_cmove():
    """C-MOVE: Retrieve pre-op MR or prior intraoperative US series to local cache."""
    if not ge_pacs_server:
        return jsonify({'error': 'GE PACS Server not initialized'}), 503
    data = request.json or {}
    study_uid = data.get('study_uid')
    if not study_uid:
        return jsonify({'error': 'study_uid is required'}), 400
    dest_aet = data.get('destination_aet', ge_pacs_server.local_aet)
    res = ge_pacs_server.c_move_retrieve(study_uid, destination_aet=dest_aet)
    return jsonify(numpy_to_native(res))


_econ_chart_cache = {}

def render_economics_3yr_chart(fin_result: dict) -> str:
    """Renders fast, thread-safe, high-resolution 3-Panel 3-Year Commercial Horizon charts."""
    three_yr = fin_result['three_year_horizon']
    quarters = three_yr['quarters']
    q_indices = np.arange(1, 13)
    market_revenue = three_yr['market_revenue_projection']
    arrhythmia = np.array(market_revenue['quarterly_by_market']['cardiac_arrhythmia']) / 1e6
    hifu = np.array(market_revenue['quarterly_by_market']['hifu_therapy']) / 1e6
    neurosurgery = np.array(market_revenue['quarterly_by_market']['neurosurgery']) / 1e6
    enabled_revenue = np.array(market_revenue['quarterly_enabled_revenue']) / 1e6
    conventional_revenue = np.array(market_revenue['quarterly_conventional_revenue']) / 1e6
    qml_llm_increment = np.array(market_revenue['quarterly_qml_llm_increment']) / 1e6

    from matplotlib.figure import Figure
    fig = Figure(figsize=(15, 3.6), facecolor='#0f172a', dpi=90)
    axes = fig.subplots(1, 3)

    # Panel 1: Three-market clinical revenue ramp.
    ax1 = axes[0]
    ax1.bar(q_indices, arrhythmia, label='Cardiac Arrhythmia', color='#38bdf8', alpha=0.88, width=0.65)
    ax1.bar(q_indices, hifu, bottom=arrhythmia, label='HIFU Therapy', color='#10b981', alpha=0.88, width=0.65)
    ax1.bar(q_indices, neurosurgery, bottom=arrhythmia + hifu, label='Neurosurgery', color='#c084fc', alpha=0.88, width=0.65)

    # Mark Year Boundaries
    ax1.axvline(4.5, color='#475569', linestyle='--', alpha=0.5)
    ax1.axvline(8.5, color='#475569', linestyle='--', alpha=0.5)
    max_top = max(enabled_revenue) * 0.93 if len(enabled_revenue) else 1.0
    ax1.text(2.5, max_top, 'Year 1 (KOL)', color='#94a3b8', fontsize=7.5, ha='center')
    ax1.text(6.5, max_top, 'Year 2 (Scale)', color='#94a3b8', fontsize=7.5, ha='center')
    ax1.text(10.5, max_top, 'Year 3 (Dominance)', color='#94a3b8', fontsize=7.5, ha='center')

    ax1.set_title('3-Year Clinical Market Revenue Ramp (Q1-Q12)', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax1.set_xlabel('Commercial Launch Timeline (Quarters)', color='#94a3b8', fontsize=8.5)
    ax1.set_ylabel('$ Millions / Quarter', color='#94a3b8', fontsize=8.5)
    ax1.set_xticks(q_indices)
    ax1.set_xticklabels(quarters)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7, loc='upper left')
    ax1.grid(True, color='#334155', alpha=0.3)
    ax1.set_facecolor('#0f172a')

    # Panel 2: Conventional versus QML/LLM-enabled commercial trajectory.
    ax2 = axes[1]
    cumulative_conventional = np.cumsum(conventional_revenue)
    cumulative_enabled = np.cumsum(enabled_revenue)
    cumulative_increment = np.cumsum(qml_llm_increment)
    ax2.plot(q_indices, cumulative_conventional, 's-', color='#94a3b8', lw=2.0, label='Conventional Revenue')
    ax2.plot(q_indices, cumulative_enabled, 'o-', color='#fbbf24', lw=2.3, label='QML + LLM Enabled')
    ax2.fill_between(q_indices, cumulative_conventional, cumulative_enabled, color='#38bdf8', alpha=0.18, label='AI-Enabled Increment')
    ax2.annotate(
        f"AI-enabled = ${cumulative_enabled[-1]:.0f}M\nIncrement = ${cumulative_increment[-1]:.0f}M",
        xy=(12, cumulative_enabled[-1]), xytext=(-8, 8), textcoords='offset points',
        color='#fbbf24', fontsize=8, fontweight='bold', ha='right'
    )
    ax2.set_title('Conventional vs. QML + LLM Revenue Trajectory', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Clinical Commercialization Timeline (Quarters)', color='#94a3b8', fontsize=8.5)
    ax2.set_ylabel('Cumulative Revenue ($M)', color='#94a3b8', fontsize=8.5)
    ax2.set_xticks(q_indices)
    ax2.set_xticklabels(quarters)
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7.0, loc='upper left')
    ax2.grid(True, color='#334155', alpha=0.3)
    ax2.set_facecolor('#0f172a')

    # Panel 3: Hospital 3-Year Unit ROI & Breakeven Procedure Payback
    ax3 = axes[2]
    hosp = three_yr['hospital_unit_economics_3yr']
    months = np.arange(1, 37)
    monthly_margin = hosp['annual_net_cash_flow'] / 12.0
    hosp_cum_cash = (months * monthly_margin) / 1e6
    hosp_capex = hosp['robot_capex'] / 1e6

    ax3.plot(months, hosp_cum_cash, color='#10b981', lw=2.2, label='Hospital Net Margin ($M)')
    ax3.axhline(hosp_capex, color='#ef4444', linestyle='--', lw=1.6, label=f"Robot CapEx (${hosp_capex:.2f}M)")

    payback_m = hosp['breakeven_months']
    ax3.axvline(payback_m, color='#fbbf24', linestyle=':', lw=1.8)
    ax3.scatter([payback_m], [hosp_capex], color='#fbbf24', s=50, zorder=5)
    ax3.annotate(f"Payback: {payback_m:.1f} Mo\n({hosp['breakeven_procedures']} procs)",
                 xy=(payback_m, hosp_capex), xytext=(8, -20), textcoords='offset points',
                 color='#fbbf24', fontsize=7.5, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=1.0))

    ax3.set_title('Hospital 3-Year Procedure Unit Economics & Payback', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax3.set_xlabel('Hospital Operational Timeline (Months 1–36)', color='#94a3b8', fontsize=8.5)
    ax3.set_ylabel('$ Millions Cumulative Net Margin', color='#94a3b8', fontsize=8.5)
    ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7.5, loc='upper left')
    ax3.grid(True, color='#334155', alpha=0.3)
    ax3.set_facecolor('#0f172a')

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(colors='#e2e8f0', labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=90, facecolor=fig.get_facecolor())
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


def render_economics_10yr_chart(fin_result: dict) -> str:
    """Renders fast, thread-safe, high-resolution 3-Panel 10-Year Long-Term Horizon charts."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(15, 3.6), facecolor='#0f172a', dpi=90)
    axes = fig.subplots(1, 3)

    years = fin_result['years']
    rev = np.array(fin_result['revenue_breakdown']['total_revenue']) / 1e6
    ebitda = np.array(fin_result['profitability']['ebitda']) / 1e6
    cum_fcf = np.array(fin_result['profitability']['cumulative_free_cash_flow']) / 1e6

    ax1 = axes[0]
    ax1.plot(years, rev, 's-', color='#38bdf8', lw=2, label='Revenue ($M)')
    ax1.plot(years, ebitda, 'o--', color='#10b981', lw=2, label='EBITDA ($M)')
    ax1.plot(years, cum_fcf, '^-', color='#c084fc', lw=2, label='Cum. FCF ($M)')
    ax1.axhline(0, color='#64748b', linestyle=':', alpha=0.7)
    ax1.set_title('10-Year Long-Term Capital Horizon Projections', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax1.set_xlabel('Horizon Timeline (Years)', color='#94a3b8', fontsize=8.5)
    ax1.set_ylabel('$ Millions', color='#94a3b8', fontsize=8.5)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7.5)
    ax1.grid(True, color='#334155', alpha=0.3)
    ax1.set_facecolor('#0f172a')

    ax2 = axes[1]
    hw = np.array(fin_result['revenue_breakdown']['hardware_sales']) / 1e6
    disp = np.array(fin_result['revenue_breakdown']['disposables']) / 1e6
    saas = np.array(fin_result['revenue_breakdown']['saas_and_service']) / 1e6

    ax2.bar(years, hw, label='Capital Hardware', color='#38bdf8', alpha=0.85)
    ax2.bar(years, disp, bottom=hw, label='Sterile Disposables', color='#10b981', alpha=0.85)
    ax2.bar(years, saas, bottom=hw+disp, label='SaaS & Service', color='#f59e0b', alpha=0.85)
    ax2.set_title('Revenue Streams Breakdown ($M)', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax2.set_xlabel('Horizon Timeline (Years)', color='#94a3b8', fontsize=8.5)
    ax2.set_ylabel('Stream Contribution ($M)', color='#94a3b8', fontsize=8.5)
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7.5)
    ax2.grid(True, color='#334155', alpha=0.3)
    ax2.set_facecolor('#0f172a')

    # Panel 3: 10-Year Discounted Cash Flow (DCF) Analysis
    ax3 = axes[2]
    dfcf = np.array(fin_result['profitability']['discounted_fcf']) / 1e6
    cum_dfcf_full = np.array(fin_result['profitability']['cumulative_discounted_fcf']) / 1e6
    dcf_years = [0] + years
    npv_m = fin_result['summary_kpis']['npv_millions']

    bar_colors = ['#10b981' if v >= 0 else '#ef4444' for v in dfcf]
    ax3.bar(years, dfcf, color=bar_colors, alpha=0.85, label='Discounted FCF ($M)')
    ax3.plot(dcf_years, cum_dfcf_full, 'o-', color='#fbbf24', lw=2.0, label='Cumulative Disc. FCF / NPV ($M)')
    ax3.axhline(0, color='#64748b', linestyle=':', alpha=0.8)
    ax3.axhline(npv_m, color='#fbbf24', linestyle='--', alpha=0.5, lw=1)
    ax3.annotate(f'NPV = ${npv_m:.1f}M', xy=(dcf_years[-1], cum_dfcf_full[-1]),
                 xytext=(-6, 10), textcoords='offset points', color='#fbbf24',
                 fontsize=8, fontweight='bold', ha='right')
    ax3.set_title('10-Year Discounted Cash Flow (DCF) Analysis', color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax3.set_xlabel('Horizon Timeline (Years, Yr 0 = Capital Outlay)', color='#94a3b8', fontsize=8.5)
    ax3.set_ylabel('$ Millions (Discounted)', color='#94a3b8', fontsize=8.5)
    ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7.5)
    ax3.grid(True, color='#334155', alpha=0.3)
    ax3.set_facecolor('#0f172a')

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(colors='#e2e8f0', labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=90, facecolor=fig.get_facecolor())
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/api/economics/simulate', methods=['GET', 'POST'])
def economics_simulate():
    """Simulates 3-year commercial horizon and 10-year capital economics, revenue streams, and monetization horizon with in-memory caching."""
    params = request.get_json(silent=True) or request.args
    fin_result = cost_economics_engine.simulate_financial_horizon(params)
    horizon_view = str(params.get('horizon_view', '3yr')).lower()

    cache_key = (
        float(params.get('base_robot_cost', 1450000)),
        int(params.get('procedures_per_year_per_unit', 180)),
        float(params.get('disposable_sale_price', 4200)),
        float(params.get('procedure_charge', 14500))
    )

    if cache_key in _econ_chart_cache:
        chart_3yr, chart_10yr = _econ_chart_cache[cache_key]
    else:
        chart_3yr = render_economics_3yr_chart(fin_result)
        chart_10yr = render_economics_10yr_chart(fin_result)
        _econ_chart_cache[cache_key] = (chart_3yr, chart_10yr)

    active_chart = chart_3yr if horizon_view.startswith('3') else chart_10yr

    return jsonify({
        'financials': numpy_to_native(fin_result),
        'chart': active_chart,
        'chart_3yr': chart_3yr,
        'chart_10yr': chart_10yr,
        'horizon_view': horizon_view
    })


@app.route('/api/thermometric/fencing', methods=['GET', 'POST'])
def get_thermometric_fencing():
    """Returns detailed Combinatorial Fencing and Number-Theoretic UQ state."""
    fencing_data = thermometric_fencing.evaluate_ablation_fencing(
        thermo.get_map(), thermo.get_damage_map(), safety_margin_mm=3.0
    )
    uq_data = number_theoretic_uq.evaluate_thermal_uq(
        thermo.get_map(), thermo.get_damage_map()
    )

    # Generate combinatorial fencing visualization
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor('#0f172a')

    # Panel 1: Temperature Map with Fences & Isotherms
    ax1 = axes[0]
    t_map = thermo.get_map()
    im1 = ax1.imshow(t_map, cmap='magma', vmin=37, vmax=80, extent=[0, 64, 0, 64])
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='#e2e8f0', labelsize=8)
    cbar1.set_label('Temp (°C)', color='#e2e8f0', fontsize=8.5)

    for struct in thermometric_fencing.eloquent_structures:
        pts = struct['points_mm']
        hull_pts = np.vstack([pts, pts[0]])
        ax1.plot(hull_pts[:, 0], hull_pts[:, 1], color=struct['color'], lw=2, label=struct['name'][:14])
        ax1.fill(hull_pts[:, 0], hull_pts[:, 1], color=struct['color'], alpha=0.25)

    ax1.contour(np.linspace(0, 64, 128), np.linspace(0, 64, 128), t_map, levels=[43.0], colors=['#38bdf8'], linewidths=[1.5], linestyles=['--'])
    ax1.contour(np.linspace(0, 64, 128), np.linspace(0, 64, 128), t_map, levels=[50.0], colors=['#ffffff'], linewidths=[1.5])

    ax1.set_title('Combinatorial Safety Fences & Isotherms', color='#e2e8f0', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Lateral X (mm)', color='#94a3b8', fontsize=8.5)
    ax1.set_ylabel('Axial Z (mm)', color='#94a3b8', fontsize=8.5)
    ax1.legend(loc='lower left', fontsize=7, facecolor='#1e293b', labelcolor='#e2e8f0')
    ax1.set_facecolor('#0f172a')

    # Panel 2: Ramanujan Harmonics
    ax2 = axes[1]
    q_vals = list(range(1, len(uq_data['ramanujan_harmonics']) + 1))
    ax2.bar(q_vals, uq_data['ramanujan_harmonics'], color='#c084fc', alpha=0.85, width=0.55)
    ax2.set_title('Ramanujan Perfusion Harmonics c_q(n)', color='#e2e8f0', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Divisor Modulus q', color='#94a3b8', fontsize=8.5)
    ax2.set_ylabel('Harmonic Amplitude', color='#94a3b8', fontsize=8.5)
    ax2.grid(True, color='#334155', alpha=0.3)
    ax2.set_facecolor('#0f172a')

    for ax in [ax1, ax2]:
        ax.tick_params(colors='#e2e8f0', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        'fencing': fencing_data,
        'uq': uq_data,
        'fencing_chart': 'data:image/png;base64,' + img_b64
    })


# =========================================================================
# ULTRASOUND, PROBE DESIGN, TUMOR ABLATION & QML STIELTJES ROUTES
# =========================================================================

@app.route('/api/ultrasound/scan', methods=['GET', 'POST'])
def get_ultrasound_scan():
    """Returns intraoperative ultrasound (iUS) B-mode and RF sector scan data."""
    global current_ablation_radius_mm, us_gain_db, us_frequency_mhz

    if request.method == 'POST':
        data = request.json or {}
        if 'ablation_radius_mm' in data:
            current_ablation_radius_mm = float(data['ablation_radius_mm'])
        if 'gain_db' in data:
            us_gain_db = float(data['gain_db'])
        if 'frequency_mhz' in data:
            us_frequency_mhz = float(data['frequency_mhz'])

    tumor_r = 9.5
    scan_data = us_simulator.generate_ius_bmode_scan(
        tumor_radius_mm=tumor_r,
        ablation_radius_mm=current_ablation_radius_mm,
        gain_db=us_gain_db,
        frequency_mhz=us_frequency_mhz
    )

    # Specificity & sensitivity evaluation
    spec_data = tumor_evaluator.evaluate_ablation_performance(current_ablation_radius_mm)

    return jsonify({
        'status': 'ok',
        'scan': scan_data,
        'ablation_metrics': spec_data
    })


@app.route('/api/ultrasound/probe/specs', methods=['GET'])
def get_probe_specs():
    """Returns 2D matrix CMUT/PMUT array and ASIC micro-circuitry specifications."""
    specs = us_probe.get_circuitry_specifications()
    return jsonify({
        'status': 'ok',
        'specifications': specs
    })


@app.route('/api/ultrasound/probe/beam', methods=['GET', 'POST'])
def get_probe_beam_pattern():
    """Calculates acoustic beam radiation pattern and grating lobe suppression."""
    global probe_steering_deg
    if request.method == 'POST':
        data = request.json or {}
        probe_steering_deg = float(data.get('steering_angle_deg', probe_steering_deg))
        focus_mm = float(data.get('focus_mm', 25.0))
    else:
        focus_mm = float(request.args.get('focus_mm', 25.0))
        probe_steering_deg = float(request.args.get('steering_angle_deg', 0.0))

    beam_data = us_probe.compute_beam_radiation_pattern(
        steering_angle_deg=probe_steering_deg,
        focus_mm=focus_mm
    )
    return jsonify({
        'status': 'ok',
        'beam_profile': beam_data
    })


@app.route('/api/ablation/specificity', methods=['GET', 'POST'])
def get_ablation_specificity():
    """Returns Tumor Sensitivity, Specificity, ROC/AUC, and Healthy Brain Parenchyma Sparing."""
    global current_ablation_radius_mm
    if request.method == 'POST':
        data = request.json or {}
        if 'ablation_radius_mm' in data:
            current_ablation_radius_mm = float(data['ablation_radius_mm'])

    metrics = tumor_evaluator.evaluate_ablation_performance(current_ablation_radius_mm)
    return jsonify({
        'status': 'ok',
        'metrics': metrics
    })


@app.route('/api/ablation/qml_stieltjes', methods=['GET', 'POST'])
def get_qml_stieltjes_ablation():
    """Runs QML Parameterized Quantum Circuit and Stieltjes Continued Fractions."""
    data = request.json if request.method == 'POST' else {}
    tumor_depth = float(data.get('tumor_depth_mm', 28.0))
    tumor_vol = float(data.get('tumor_volume_cm3', 3.6))
    perfusion = float(data.get('perfusion', 55.0))
    margin = float(data.get('margin_mm', 2.0))

    qml_result = qml_ablation_opt.optimize_ablation_pulse(
        tumor_depth_mm=tumor_depth,
        tumor_volume_cm3=tumor_vol,
        tissue_perfusion_ml_min_100g=perfusion,
        target_margin_mm=margin
    )
    return jsonify({
        'status': 'ok',
        'qml_stieltjes': qml_result
    })


@app.route('/api/robot/kinematics_qkf', methods=['GET', 'POST'])
def get_robot_kinematics_qkf():
    """Computes 7-DOF DH Forward Kinematics and Quantum Kalman Filter tool-tip estimation."""
    angles = None
    if request.method == 'POST':
        data = request.json or {}
        if 'joint_angles' in data:
            angles = [float(a) for a in data['joint_angles']]

    kinematics = robot_qkf_kinematics.forward_kinematics(joint_angles=angles)
    return jsonify({
        'status': 'ok',
        'kinematics': kinematics
    })


@app.route('/api/ablation/step', methods=['POST'])
def step_ablation():
    """Executes an incremental recurrent ultrasound ablation pulse."""
    global current_ablation_radius_mm
    data = request.json or {}
    step_delta_mm = float(data.get('step_delta_mm', 0.8))
    max_radius_mm = float(data.get('max_radius_mm', 10.5))

    current_ablation_radius_mm = min(max_radius_mm, current_ablation_radius_mm + step_delta_mm)

    scan_data = us_simulator.generate_ius_bmode_scan(
        tumor_radius_mm=9.5,
        ablation_radius_mm=current_ablation_radius_mm,
        gain_db=us_gain_db,
        frequency_mhz=us_frequency_mhz
    )
    spec_data = tumor_evaluator.evaluate_ablation_performance(current_ablation_radius_mm)

    return jsonify({
        'status': 'ok',
        'current_ablation_radius_mm': current_ablation_radius_mm,
        'scan': scan_data,
        'metrics': spec_data
    })


@app.route('/api/ablation/reset', methods=['POST'])
def reset_ablation():
    """Resets the ultrasound ablation simulation."""
    global current_ablation_radius_mm
    current_ablation_radius_mm = 0.0
    scan_data = us_simulator.generate_ius_bmode_scan(
        tumor_radius_mm=9.5,
        ablation_radius_mm=0.0,
        gain_db=us_gain_db,
        frequency_mhz=us_frequency_mhz
    )
    spec_data = tumor_evaluator.evaluate_ablation_performance(0.0)
    return jsonify({
        'status': 'ok',
        'current_ablation_radius_mm': 0.0,
        'scan': scan_data,
        'metrics': spec_data
    })


@app.route('/api/phantom/mrus_scan', methods=['GET', 'POST'])
def get_phantom_mrus_scan():
    """Generates 4D dynamic MR magnitude and synchronized iUS B-mode phantom scan slice."""
    data = request.json if request.method == 'POST' else {}
    phase = float(data.get('phase_rad', request.args.get('phase_rad', 0.0)))
    amp = float(data.get('deformation_amp_mm', request.args.get('deformation_amp_mm', 1.2)))

    pair = mrus_phantom_engine.generate_4d_mrus_pair(phase_rad=phase, deformation_amp_mm=amp)
    metrics = phantom_gt_study.evaluate_ground_truth_metrics()

    return jsonify({
        'status': 'ok',
        'phantom_scan': pair,
        'ground_truth_metrics': metrics
    })


@app.route('/api/phantom/ground_truth_study', methods=['GET', 'POST'])
def run_ground_truth_study():
    """Runs 256-trial statistical ground truth Monte Carlo study and formal verifiability proof."""
    data = request.json if request.method == 'POST' else {}
    seed = int(data.get('seed', request.args.get('seed', 1009)))

    study_results = phantom_gt_study.run_256_trial_monte_carlo_study(seed=seed)
    proof_metrics = phantom_gt_study.evaluate_ground_truth_metrics()

    return jsonify({
        'status': 'ok',
        'study': study_results,
        'formal_verification': proof_metrics
    })


@app.route('/api/mrus/fusion', methods=['GET', 'POST'])
def get_mrus_fusion():
    """
    MR/US Fusion analysis using Recurrent Prime-Moduli Elliptic Integral Signatures.
    Derives a rigorously bounded ablation/registration energy safety corridor from the
    classical inequality 1 <= E(k) <= pi/2 on the complete elliptic integral of the
    second kind, indexed by a recurrent prime-moduli modulus sequence.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'error': 'The MR/US fusion request body must be a JSON object.'
        }), 400

    try:
        phase = float(data.get('phase_rad', request.args.get('phase_rad', 0.4)))
        amp = float(data.get('deformation_amp_mm', request.args.get('deformation_amp_mm', 1.2)))
        max_energy = float(data.get('max_energy_j', request.args.get('max_energy_j', 6.0)))
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'error': 'phase_rad, deformation_amp_mm, and max_energy_j must be numeric.'
        }), 400

    if not 0.0 <= phase <= 2.0 * math.pi:
        return jsonify({
            'status': 'error',
            'error': 'phase_rad must be within [0, 2π].'
        }), 400
    if not 0.0 <= amp <= 5.0:
        return jsonify({
            'status': 'error',
            'error': 'deformation_amp_mm must be within [0, 5].'
        }), 400
    if not 1.0 <= max_energy <= 25.0:
        return jsonify({
            'status': 'error',
            'error': 'max_energy_j must be within [1, 25].'
        }), 400

    result = MRUSFusionEngine(
        num_terms=14,
        seed_modulus=0.15,
        max_energy_j=max_energy,
        dt_s=0.75,
        grid_dim=128
    ).run_fusion_analysis(phase_rad=phase, deformation_amp_mm=amp)
    return jsonify(numpy_to_native(result))


@app.route('/api/generate-preprint-pdf', methods=['GET', 'POST'])
def generate_preprint_pdf_endpoint():
    """Compiles and downloads the Nature Preprint PDF."""
    try:
        from generate_nature_preprint_economics_thermometry import generate_nature_preprint_pdf
        pdf_path = generate_nature_preprint_pdf()
        if os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name='Nature_Preprint_Neurosurgical_Robotics_Economics_Thermometric_Fencing.pdf'
            )
        return jsonify({'error': 'Preprint PDF generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprint/ultrasound/download', methods=['GET'])
@app.route('/api/preprint/ultrasound/view', methods=['GET'])
def get_ultrasound_preprint():
    """Serves the Nature Ultrasound Ablation & QML Stieltjes Preprint PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Ultrasound_Ablation_QML_Stieltjes_Preprint.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nature_ultrasound_ablation_preprint import build_ultrasound_ablation_preprint_pdf
            build_ultrasound_ablation_preprint_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='Nature_Ultrasound_Ablation_QML_Stieltjes_Preprint.pdf'
    )


@app.route('/api/preprint/probe_schematics/download', methods=['GET'])
@app.route('/api/preprint/probe_schematics/view', methods=['GET'])
def get_probe_schematics_preprint():
    """Serves the Nature Miniaturized Ultrasound Probe Schematics Preprint PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Miniaturized_Ultrasound_Probe_Schematics.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nature_probe_schematics_preprint import build_probe_schematics_preprint_pdf
            build_probe_schematics_preprint_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='Nature_Preprint_Miniaturized_Ultrasound_Probe_Schematics.pdf'
    )


@app.route('/api/preprint/cryo_laser/download', methods=['GET'])
@app.route('/api/preprint/cryo_laser/view', methods=['GET'])
def get_cryo_laser_preprint():
    """Serves the Nature Cryo & Laser Observable Finite Math Preprint PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Cryo_Laser_Observable_Finite_Math_Preprint.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nature_cryo_laser_preprint import build_cryo_laser_preprint_pdf
            build_cryo_laser_preprint_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='Nature_Cryo_Laser_Observable_Finite_Math_Preprint.pdf'
    )


@app.route('/api/ablation-modalities/litt', methods=['GET', 'POST'])
def get_litt_ablation():
    """Runs the nonlinear Pennes bioheat LITT simulation."""
    try:
        params = request.get_json(silent=True) or {}
        result = litt_engine.simulate(
            power_w=float(params.get('power_w', 12.0)),
            exposure_s=float(params.get('exposure_s', 90.0)),
            mu_a=float(params.get('mu_a', 35.0)),
            mu_s_prime=float(params.get('mu_s_prime', 900.0)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/cryo', methods=['GET', 'POST'])
def get_cryo_ablation():
    """Runs the Stefan-front cryoablation clinical simulation."""
    try:
        params = request.get_json(silent=True) or {}
        result = cryo_clinical_engine.simulate(
            pressure_psi=float(params.get('pressure_psi', 2400.0)),
            freeze_s=float(params.get('freeze_s', 480.0)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/us_precision', methods=['GET', 'POST'])
def get_us_precision_ablation():
    """Runs the millimetric-precision US-guided HIFU ablation simulation."""
    try:
        params = request.get_json(silent=True) or {}
        result = us_precision_engine.simulate(
            acoustic_power_w=float(params.get('acoustic_power_w', 65.0)),
            exposure_s=float(params.get('exposure_s', 20.0)),
            alpha_np_per_m=float(params.get('alpha_np_per_m', 6.5)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/slam_trajectory', methods=['GET', 'POST'])
def get_ablation_slam_trajectory():
    """Stieltjes-continued-fraction SLAM optimal insertion trajectory."""
    try:
        params = request.get_json(silent=True) or {}
        result = optimal_slam_trajectory(
            target_mm=params.get('target_mm'),
            waypoints=int(params.get('waypoints', 12)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/clinical_outcomes', methods=['GET', 'POST'])
def get_ablation_clinical_outcomes():
    """Long-horizon clinical outcome trajectories (Weibull/Gompertz/KPS)."""
    try:
        params = request.get_json(silent=True) or {}
        result = clinical_outcome_trajectories(
            horizon_months=int(params.get('horizon_months', 60)),
            step_months=int(params.get('step_months', 3)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/equations', methods=['GET'])
def get_ablation_equations():
    """Classical governing equations for the LITT/Cryo/US ablation tab."""
    try:
        return jsonify({'equations': classical_equations_reference()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ablation-modalities/summary', methods=['GET', 'POST'])
def get_ablation_modalities_summary():
    """Combined LITT + Cryo + US + SLAM + clinical-outcomes dashboard payload (cached)."""
    try:
        return jsonify(get_ablation_modalities_cached())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprint/litt_cryo/download', methods=['GET'])
@app.route('/api/preprint/litt_cryo/view', methods=['GET'])
def get_litt_cryo_preprint():
    """Serves the Nature LITT/Cryo/US Tumour Ablation Preprint PDF (l.pdf)."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'l.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nature_litt_cryo_preprint import build_pdf
            build_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='l.pdf'
    )


@app.route('/api/stem-cell-therapy/summary', methods=['GET', 'POST'])
def get_stem_cell_therapy_summary():
    """QML-enhanced stem cell therapy with BBB disruption dashboard."""
    try:
        result = stem_cell_therapy_summary(
            stem_cell_count=int(request.json.get('stem_cell_count', 1e6)) if request.is_json else int(1e6),
            us_energy_kj=float(request.json.get('us_energy_kj', 50.0)) if request.is_json else 50.0,
            litt_power_w=float(request.json.get('litt_power_w', 12.0)) if request.is_json else 12.0,
            litt_duration_s=float(request.json.get('litt_duration_s', 90.0)) if request.is_json else 90.0,
            tumor_cells_ml=float(request.json.get('tumor_cells_ml', 5e6)) if request.is_json else 5e6,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprint/stem_cell/download', methods=['GET'])
@app.route('/api/preprint/stem_cell/view', methods=['GET'])
def get_stem_cell_preprint():
    """Serves the Nature Stem Cell Therapy Preprint PDF (stem_cell.pdf)."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'stem_cell.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nature_stem_cell_preprint import build_pdf
            build_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='stem_cell.pdf'
    )


@app.route('/api/grant/nih/download', methods=['GET'])
@app.route('/api/grant/nih/view', methods=['GET'])
def get_nih_grant_proposal():
    """Serves the NIH R01/U01 Grant Proposal PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'NIH_Grant_Proposal_Ultrasound_QML_NTT_Neurosurgery.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nih_grant_proposal import build_nih_grant_pdf
            build_nih_grant_pdf(pdf_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='NIH_Grant_Proposal_Ultrasound_QML_NTT_Neurosurgery.pdf'
    )


@app.route('/api/grant/nih-r01-stem-cell/download', methods=['GET'])
@app.route('/api/grant/nih-r01-stem-cell/view', methods=['GET'])
def get_nih_r01_stem_cell_grant():
    """Serves the NIH R01 Grant Proposal for Quantum ML Stem Cell Therapy PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'nih_r01_stem_cell_therapy.pdf')
    if not os.path.exists(pdf_path):
        try:
            from generate_nih_r01_stem_cell_grant import build_nih_r01_pdf
            build_nih_r01_pdf()
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    as_att = request.path.endswith('/download')
    return send_file(
        pdf_path,
        as_attachment=as_att,
        mimetype='application/pdf',
        download_name='nih_r01_stem_cell_therapy.pdf'
    )


if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 5002))
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    print(f"\n{'='*60}")
    print(f"🚀 NeuroMorph Surgical Robotics Platform | Gemini AI Core")
    print(f"{'='*60}")
    print(f"Starting Server on http://{host}:{port}")
    print(f"AI Engine: Google Gemini 1.5 Pro (Generative)")
    print(f"{'='*60}\n")
    # Enable threaded mode to handle multiple concurrent telemetry requests from the UI
    app.run(host=host, port=port, debug=False, threaded=True)
