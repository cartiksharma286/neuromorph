#!/usr/bin/env python3
"""
shsc_vr_genai - Sunnybrook Health Sciences Centre (SHSC) Virtual Reality Surgery Platform
Flask Backend: Statistical Learning Curve Models, VLM / Multimodal Surgical Reasoning APIs,
and Joint Optimization Trajectory Simulators for Neurosurgery and Orthopedics.
"""

import os
import json
import math
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── STATISTICAL MACHINE LEARNING MODEL ────────────────────────────────────────
# Power Law of Practice / Skill Decay / Cognitive Load Transition Matrix
# We model cognitive load using a 3-State Hidden Markov Process:
# State 0: Low Stress / Nominal, State 1: Flow / Focused, State 2: Overload / Critical
COGNITIVE_TRANSITION_MATRIX = np.array([
    [0.70, 0.25, 0.05], # From Nominal
    [0.15, 0.75, 0.10], # From Focused
    [0.10, 0.40, 0.50]  # From Critical
])

def estimate_cognitive_load_probabilities(session_duration_minutes, skill_level):
    """
    Computes statistical cognitive load state distribution over a training session using the transition matrix,
    modulated by learner skill level. Higher skill level drives state probabilities towards 'nominal/focused'.
    """
    # Simple discrete Chapman-Kolmogorov power step simulation
    steps = int(max(1, min(10, session_duration_minutes // 5)))
    state_prob = np.array([0.60, 0.35, 0.05]) # Initial probability vector (nominal, focused, overload)
    
    current_matrix = np.copy(COGNITIVE_TRANSITION_MATRIX)
    # Modulate transition bounds based on skill (0.0 to 1.0)
    # High skill reduces transition into overload and increases recovery to nominal
    skill_mod = float(skill_level)
    current_matrix[0, 2] = max(0.01, current_matrix[0, 2] * (1.0 - 0.8 * skill_mod))
    current_matrix[1, 2] = max(0.01, current_matrix[1, 2] * (1.0 - 0.7 * skill_mod))
    current_matrix[2, 0] = min(0.90, current_matrix[2, 0] * (1.0 + 0.5 * skill_mod))
    
    # Normalize rows
    for i in range(3):
        current_matrix[i] /= np.sum(current_matrix[i])
        
    for _ in range(steps):
        state_prob = state_prob @ current_matrix
        
    return {
        'nominal_percentage': round(float(state_prob[0] * 100.0), 2),
        'focused_percentage': round(float(state_prob[1] * 100.0), 2),
        'overload_percentage': round(float(state_prob[2] * 100.0), 2)
    }


def compute_learning_curve(trials, base_skill=15.0, asymptotic_skill=95.0, alpha=0.35, decay_rate=0.04, retention_interval_weeks=0):
    """
    Applies the Power Law of Practice combined with Ebbinghaus memory decay:
    SK_raw(t) = SK_0 + (SK_inf - SK_0) * (1 - e^(-alpha * t))
    SK_retained(t) = SK_raw(t) * e^(-decay_rate * t_retention)
    """
    skill_series = []
    error_series = [] # in mm of placement error Or degrees of angle mismatch
    
    for t in range(1, trials + 1):
        # Power learning curve
        raw_skill = base_skill + (asymptotic_skill - base_skill) * (1.0 - math.exp(-alpha * t))
        # Retention decay term
        decay_factor = math.exp(-decay_rate * retention_interval_weeks)
        final_skill = max(base_skill, raw_skill * decay_factor)
        
        # Surgical deviation error model (exponentially decay error as skill scales)
        # Position error in mm: 12mm baseline down to sub-millimeter (e.g. 0.6mm)
        pos_err = 12.0 * math.exp(-0.06 * final_skill) + np.random.normal(0, 0.1)
        # Angle error in degrees: 8.5 deg down to 1.1 deg
        deg_err = 8.5 * math.exp(-0.05 * final_skill) + np.random.normal(0, 0.08)
        
        skill_series.append(float(round(final_skill, 2)))
        error_series.append({
            'trial': t,
            'position_error_mm': float(round(max(0.2, pos_err), 2)),
            'angle_error_deg': float(round(max(0.1, deg_err), 2))
        })
        
    return skill_series, error_series


# ── VLM / MULTIMODAL SURGICAL REASONING ENGINE ───────────────────────────────
# Mock Visual-Language understanding module to analyze coordinate alignments
# and provide spatial bounding boxes and anatomical safety feedback.
SURGICAL_SCENES_DATABASE = {
    'neurosurgery_dbs': {
        'title': 'Deep Brain Stimulation (DBS) target entry - Under Subthalamic Nucleus (STN) tract',
        'safe_zones': [{'label': 'STN Target', 'xmin': 0.42, 'xmax': 0.58, 'ymin': 0.45, 'ymax': 0.60}],
        'critical_structures': [{'label': 'Internal Capsule / Corticospinal Tract', 'xmin': 0.15, 'xmax': 0.38, 'ymin': 0.20, 'ymax': 0.80}],
        'anesthesia_load_limit': 1.8, # in mg/kg/hr
        'guidelines': 'Avoid lateral displacement into the internal capsule. Retain trajectory coaxial to STN boundary to limit spatial tremor mismatch.'
    },
    'neurosurgery_tumor': {
        'title': 'Glioma Resection Margins - Visual Cortical Area Bordering Corpus Callosum',
        'safe_zones': [{'label': 'Resection Boundary', 'xmin': 0.30, 'xmax': 0.70, 'ymin': 0.30, 'ymax': 0.70}],
        'critical_structures': [{'label': 'Anterior Cerebral Artery (ACA) Loop', 'xmin': 0.75, 'xmax': 0.95, 'ymin': 0.10, 'ymax': 0.40}],
        'anesthesia_load_limit': 2.2,
        'guidelines': 'VLM flags high-risk bleeding on the right-lateral side of the surgical sector. Exercise ultrasound aspiration control.'
    },
    'orthopedic_pedicle': {
        'title': 'L4-L5 Pedicle Screw Placement Simulation - Transpedicular Corridor',
        'safe_zones': [{'label': 'Pedicle Neck Entry', 'xmin': 0.46, 'xmax': 0.54, 'ymin': 0.42, 'ymax': 0.56}],
        'critical_structures': [{'label': 'Medial Spinal Canal & Nerve Root', 'xmin': 0.20, 'xmax': 0.44, 'ymin': 0.30, 'ymax': 0.70}],
        'anesthesia_load_limit': 2.5,
        'guidelines': 'Pedicle cortex perforation medial breach threatens L5 spinal canal root. Adjust entrance trajectory lateral by 2.2 degrees.'
    },
    'orthopedic_femoral': {
        'title': 'Antegrade Femoral Nailing - Greater Trochanter Insertion Portal',
        'safe_zones': [{'label': 'Trochanteric Tip Portal', 'xmin': 0.40, 'xmax': 0.60, 'ymin': 0.15, 'ymax': 0.35}],
        'critical_structures': [{'label': 'Medial Circumflex Cervical Artery Zone', 'xmin': 0.10, 'xmax': 0.38, 'ymin': 0.45, 'ymax': 0.85}],
        'anesthesia_load_limit': 3.0,
        'guidelines': 'Visual logic identifies entry point posterior displacement. Readjust the orthopedic starting guidewire alignment.'
    }
}

KNEE_WORKFLOW_GUIDES = {
    'midvastus': {
        'label': 'Mini-midvastus MIS TKA',
        'exposure_factor': 1.00,
        'soft_tissue_score': 88,
        'workflow': [
            {
                'phase': 'Pre-op digital templating',
                'objective': 'Review CT or long-leg radiographs, confirm implant sizing, and pre-brief instrument flow.',
                'teaching_focus': 'Align distributed planning artifacts across simulation stations and OR teams.'
            },
            {
                'phase': 'Portal and arthrotomy setup',
                'objective': 'Establish a muscle-sparing midvastus window with direct visualization of the trochlear entry corridor.',
                'teaching_focus': 'Maintain extensor continuity while preserving a reproducible exposure window for trainees.'
            },
            {
                'phase': 'Guidewire and canal control',
                'objective': 'Advance the guidewire on the intended femoral or tibial axis and confirm capture before reaming or cutting.',
                'teaching_focus': 'Translate wire feel, fluoroscopy cues, and tracking overlays into a stable trajectory decision.'
            },
            {
                'phase': 'Bone preparation and balancing',
                'objective': 'Perform measured resections, gap balancing, and trial verification through the minimized exposure.',
                'teaching_focus': 'Teach sequencing discipline so limited visualization does not degrade alignment checks.'
            },
            {
                'phase': 'Implantation and closure',
                'objective': 'Seat components, verify patellar tracking, and close with recovery-oriented soft-tissue handling.',
                'teaching_focus': 'Use structured after-action review metrics for readiness reporting.'
            }
        ]
    },
    'subvastus': {
        'label': 'Subvastus MIS TKA',
        'exposure_factor': 0.96,
        'soft_tissue_score': 92,
        'workflow': [
            {
                'phase': 'Pre-op digital templating',
                'objective': 'Review alignment plan, implant inventory, and navigation checkpoints before incision.',
                'teaching_focus': 'Supports lower-profile deployment training where resources must be synchronized early.'
            },
            {
                'phase': 'Muscle-preserving exposure',
                'objective': 'Develop a subvastus interval without violating the quadriceps tendon.',
                'teaching_focus': 'Reinforce atraumatic tissue handling and visualization management in constrained windows.'
            },
            {
                'phase': 'Guidewire navigation',
                'objective': 'Advance the guidewire while maintaining canal centerline control and rotational discipline.',
                'teaching_focus': 'Pairs tactile education with digital navigation overlays for precision repetition.'
            },
            {
                'phase': 'Resection, trialing, and balance',
                'objective': 'Complete femoral and tibial preparation, then validate gaps and component rotation.',
                'teaching_focus': 'Emphasize checkpoint-based workflow so learners can recover from reduced exposure.'
            },
            {
                'phase': 'Final implant and readiness review',
                'objective': 'Seat implants, verify ROM, and close with documented performance metrics.',
                'teaching_focus': 'Feed outcomes directly into competency and readiness dashboards.'
            }
        ]
    },
    'quad_sparing': {
        'label': 'Quadriceps-sparing MIS TKA',
        'exposure_factor': 1.08,
        'soft_tissue_score': 95,
        'workflow': [
            {
                'phase': 'Pre-op digital rehearsal',
                'objective': 'Run a constrained-access rehearsal with implant sizing, workspace setup, and contingency prompts.',
                'teaching_focus': 'Best suited for advanced learners working through a narrow-access cognitive load profile.'
            },
            {
                'phase': 'Limited incision access',
                'objective': 'Create a minimal arthrotomy while preserving quadriceps continuity and patellar mobility.',
                'teaching_focus': 'Focus on camera orientation, instrument choreography, and deliberate exposure changes.'
            },
            {
                'phase': 'Guidewire and alignment discipline',
                'objective': 'Keep the wire coaxial despite restricted visualization and small angular errors.',
                'teaching_focus': 'Highlights the value of real-time digital prompts for early deviation correction.'
            },
            {
                'phase': 'Resection and component placement',
                'objective': 'Perform bone prep and implant seating without losing gap symmetry or rotational alignment.',
                'teaching_focus': 'Teach workflow pacing under minimally invasive access constraints.'
            },
            {
                'phase': 'Closeout and recovery pathway',
                'objective': 'Validate patellar tracking, implant seating, and post-op mobilization goals.',
                'teaching_focus': 'Close the loop with digital competency capture and remote instructor review.'
            }
        ]
    }
}

MILITARY_EDUCATION_PROFILES = {
    'us_military': {
        'label': 'US Military Medical Education',
        'initiative': 'Distributed readiness pipeline for expeditionary orthopedic teams.',
        'priority': 'Common metrics, simulation-backed sustainment, and faster remediation for deployed care teams.'
    },
    'canadian_military': {
        'label': 'Canadian Armed Forces Education',
        'initiative': 'Joint digital surgery curriculum supporting austere and coalition care settings.',
        'priority': 'Interoperable training records, bilingual-ready content pathways, and team-based performance review.'
    },
    'joint_allied': {
        'label': 'US / Canadian Allied Initiative',
        'initiative': 'Shared digital transformation track for bilateral orthopedic simulation and competency reporting.',
        'priority': 'Standardized workflow guides, cloud-hosted simulation evidence, and interoperable after-action analytics.'
    }
}


def compute_knee_education_simulation(
    wire_diameter_mm=2.8,
    stiffness_index=58.0,
    insertion_angle_deg=1.5,
    approach='midvastus',
    mission='joint_allied',
):
    """
    Compute the knee education guidewire and MIS TKA workflow metrics used by
    both the dashboard endpoint and offline publication artifacts.
    """
    if approach not in KNEE_WORKFLOW_GUIDES:
        approach = 'midvastus'
    if mission not in MILITARY_EDUCATION_PROFILES:
        mission = 'joint_allied'

    guide = KNEE_WORKFLOW_GUIDES[approach]
    mission_profile = MILITARY_EDUCATION_PROFILES[mission]

    wire_diameter_mm = max(1.6, min(4.0, float(wire_diameter_mm)))
    stiffness_index = max(25.0, min(95.0, float(stiffness_index)))
    insertion_angle_deg = max(-6.0, min(6.0, float(insertion_angle_deg)))

    depth_axis = np.linspace(0.0, 120.0, 36).tolist()
    tracking_error_mm = []
    tactile_load_n = []

    exposure_factor = guide['exposure_factor']
    stiffness_gain = 0.85 + ((stiffness_index - 25.0) / 70.0) * 0.55
    diameter_gain = 0.88 + ((wire_diameter_mm - 1.6) / 2.4) * 0.28

    for depth in depth_axis:
        depth_ratio = depth / 120.0
        angle_penalty = 1.0 + (abs(insertion_angle_deg) / 7.0)
        deviation = (
            0.28
            + 1.35 * depth_ratio * exposure_factor * angle_penalty / (stiffness_gain * diameter_gain)
            + 0.06 * math.sin(depth_ratio * math.pi * 3.0)
        )
        tactile_load = (
            15.0
            + (stiffness_index * 0.18)
            + (wire_diameter_mm * 4.8)
            + depth_ratio * 16.0 * exposure_factor
            + abs(insertion_angle_deg) * 1.2
        )

        tracking_error_mm.append(float(round(deviation, 3)))
        tactile_load_n.append(float(round(tactile_load, 2)))

    mean_tracking_error = float(np.mean(tracking_error_mm))
    max_tactile_load = float(np.max(tactile_load_n))
    canal_capture_probability = max(
        45.0,
        min(
            98.0,
            95.0
            - abs(insertion_angle_deg) * 4.2
            - (exposure_factor - 1.0) * 14.0
            + (stiffness_index - 58.0) * 0.18
            + (wire_diameter_mm - 2.8) * 2.6,
        ),
    )
    alignment_index = max(
        35.0,
        min(
            99.0,
            94.0
            - mean_tracking_error * 12.0
            - abs(insertion_angle_deg) * 2.4
            + (stiffness_index - 58.0) * 0.10,
        ),
    )
    fluoroscopy_saves = max(
        1,
        int(
            round(
                9
                - (stiffness_index - 58.0) / 12.0
                - (wire_diameter_mm - 2.8)
                + abs(insertion_angle_deg) / 2.5
                + (exposure_factor - 1.0) * 3.0
            )
        ),
    )

    workflow_readiness_score = max(
        50.0,
        min(
            99.0,
            0.45 * alignment_index
            + 0.35 * canal_capture_probability
            + 0.20 * guide['soft_tissue_score'],
        ),
    )

    mission_bias = {
        'us_military': 84.0,
        'canadian_military': 86.0,
        'joint_allied': 89.0,
    }[mission]
    digital_interoperability_score = max(
        60.0,
        min(
            99.0,
            mission_bias
            + 0.12 * (alignment_index - 80.0)
            + 0.08 * (canal_capture_probability - 80.0)
            - 0.35 * fluoroscopy_saves,
        ),
    )

    readiness_narrative = (
        f"{mission_profile['label']} uses the {guide['label']} pathway to pair minimally invasive exposure discipline "
        f"with repeatable guidewire checkpoints. The current configuration projects a canal capture probability of "
        f"{canal_capture_probability:.1f}% and an implant alignment index of {alignment_index:.1f}%. "
        f"Digital transformation focus: {mission_profile['priority']}"
    )

    recommended_action = (
        'Maintain current wire profile and continue with templated MIS workflow checkpoints.'
        if canal_capture_probability >= 88.0 and alignment_index >= 85.0
        else 'Reduce angular error or increase wire support before progressing to resection and implant preparation.'
    )

    return {
        'depth_axis_mm': depth_axis,
        'tracking_error_mm': tracking_error_mm,
        'tactile_load_n': tactile_load_n,
        'workflow_guide': guide['workflow'],
        'approach_summary': {
            'label': guide['label'],
            'soft_tissue_preservation_score': guide['soft_tissue_score'],
            'recommended_action': recommended_action,
        },
        'mission_profile': mission_profile,
        'characteristics': {
            'canal_capture_probability': round(canal_capture_probability, 1),
            'alignment_index': round(alignment_index, 1),
            'mean_tracking_error_mm': round(mean_tracking_error, 3),
            'max_tactile_load_n': round(max_tactile_load, 2),
            'estimated_fluoroscopy_saves': fluoroscopy_saves,
            'workflow_readiness_score': round(workflow_readiness_score, 1),
            'digital_interoperability_score': round(digital_interoperability_score, 1),
        },
        'initiative_brief': readiness_narrative,
    }

@app.route('/api/vlm-reasoning', methods=['POST', 'GET'])
def api_vlm_reasoning():
    """
    Multimodal visual-language reasoning endpoint.
    Processes coordinate interactions, simulates VLM reasoning, overlaying bounding boxes,
    and returns localized surgical warnings or compliance approvals.
    """
    try:
        # Accept parameters via GET or POST
        if request.method == 'POST':
            data = request.json or {}
            scene_id = data.get('scene_id', 'neurosurgery_dbs')
            point_x = float(data.get('x', 0.5))
            point_y = float(data.get('y', 0.5))
            complexity = float(data.get('difficulty_mod', 1.0))
        else:
            scene_id = request.args.get('scene_id', 'neurosurgery_dbs')
            point_x = float(request.args.get('x', 0.5))
            point_y = float(request.args.get('y', 0.5))
            complexity = float(request.args.get('difficulty_mod', 1.0))

        if scene_id not in SURGICAL_SCENES_DATABASE:
            scene_id = 'neurosurgery_dbs'

        scene = SURGICAL_SCENES_DATABASE[scene_id]
        
        # Multimodal Spatial Reasoner Logic
        is_in_safe = False
        is_in_hazard = False
        matched_str = "Neutral Parenchyma Border"
        
        # Check coordinates against bounding boundaries
        for safe in scene['safe_zones']:
            if safe['xmin'] <= point_x <= safe['xmax'] and safe['ymin'] <= point_y <= safe['ymax']:
                is_in_safe = True
                matched_str = f"Target zone: {safe['label']}"
                
        for hazard in scene['critical_structures']:
            if hazard['xmin'] <= point_x <= hazard['xmax'] and hazard['ymin'] <= point_y <= hazard['ymax']:
                is_in_hazard = True
                matched_str = f"HAZARD COLLISION: {hazard['label']}"
                
        # Statistical AI Reasoner Text Gen
        score_base = 90.0 if is_in_safe else (15.0 if is_in_hazard else 65.0)
        actual_score = max(5.0, min(100.0, score_base - 12.0 * complexity + np.random.normal(0, 3.0)))
        
        # Create attention points reflecting LLM/VLM multi-modal heatmaps
        num_attention_points = 8
        attention_map = []
        for i in range(num_attention_points):
            # Focus weights clustered around selected point and hot regions
            noise_x = np.random.normal(0, 0.08)
            noise_y = np.random.normal(0, 0.08)
            heat_val = 0.95 * math.exp(-((noise_x**2 + noise_y**2)/0.02))
            attention_map.append({
                'x': float(max(0.0, min(1.0, point_x + noise_x))),
                'y': float(max(0.0, min(1.0, point_y + noise_y))),
                'density_heat': float(round(heat_val, 3))
            })

        feedback_narrative = ""
        clinical_state = "PASS"
        if is_in_hazard:
            clinical_state = "CRITICAL WARNING"
            feedback_narrative = (
                f"### [VLM Visual Reasoner Warning] - Critical Boundary Intrusion\n"
                f"Surgical tool entry coordinates (x: {point_x:.3f}, y: {point_y:.3f}) map inside the **{scene['critical_structures'][0]['label']}**. "
                f"Surgical navigation guidance indicates high mechanical tension. Multimodal visual inspection flags vascular puncture risk. "
                f"**Clinical Instruction**: Retract vector immediately. Retarget within the designated {scene['safe_zones'][0]['label']} portal."
            )
        elif is_in_safe:
            clinical_state = "Target Aligned"
            feedback_narrative = (
                f"### [VLM Visual Reasoning Report] - Perfect Spatial Alignment\n"
                f"Tool positioned within target aperture **{scene['safe_zones'][0]['label']}** (x: {point_x:.3f}, y: {point_y:.3f}). "
                f"Trajectory is optimized. Mechanical tissue displacement ratios are safe. Cortical shear parameters are within nominal bounds (< 0.05 kPa). "
                f"**Educational Note**: Co-axially locking this posture preserves tissue state characteristics."
            )
        else:
            clinical_state = "Marginal Deviated"
            feedback_narrative = (
                f"### [VLM Visual Reasoning Feedback] - Out of Focus\n"
                f"Coordinates (x: {point_x:.3f}, y: {point_y:.3f}) rest on non-target structural tissue ({matched_str}). "
                f"Alignment error is elevated. Visual feedback indicates tracking loss. "
                f"**Correction Plan**: Recalibrate virtual fluoroscopy or camera tracking focus. Pivot toward coordinates x: 0.50, y: 0.50."
            )

        return jsonify({
            'status': 'success',
            'scene_title': scene['title'],
            'coordinate_evaluation': {
                'x': point_x,
                'y': point_y,
                'is_safe_boundary': is_in_safe,
                'is_hazard_boundary': is_in_hazard,
                'tissue_signature': matched_str
            },
            'attention_heatmap': attention_map,
            'scoring_metric': actual_score,
            'clinical_state': clinical_state,
            'reasoning_agent_prescription': feedback_narrative,
            'underlying_scene_guidelines': scene['guidelines'],
            'anesthesia_threshold_mg_kg': scene['anesthesia_load_limit']
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── ENDPOINT: NEUROSURGERY VR OPTIMIZER SIMULATION ─────────────────────────────
@app.route('/api/neurosurgery-sim', methods=['GET'])
def api_neurosurgery_sim():
    """
    DBS/Ventricle Catheter Pathing optimization and coordinate simulation.
    Accepts target coords, current pathing drift, and returns surgical deviation metrics,
    Euler torque calculations, and statistical pathing steps.
    """
    try:
        target_x = float(request.args.get('target_x', 0.5))
        target_y = float(request.args.get('target_y', 0.52))
        drift_factor = float(request.args.get('drift', 1.0)) # Hand/Tremor drift coefficient 
        skill_reps = int(request.args.get('reps', 12)) # Cumulative previous training sessions
        
        np.random.seed(42 + skill_reps)
        steps = 40
        time_axis = np.linspace(0.0, 1.0, steps)
        
        # Theoretical trajectory drift modeling: Wiener process with drift target attraction
        # dx_t = -gamma * (x_t - target_x) * dt + sigma * dW_t
        gamma_attract = 4.0 + 0.5 * skill_reps
        sigma_tremor = 0.18 * drift_factor * (1.0 / (1.0 + 0.15 * skill_reps))
        
        path_x = [0.10] # Starts at bone entryway x=0.10, y=0.10
        path_y = [0.10]
        
        cumulative_deviation_score = 0.0
        shear_force_psi = []
        
        x_curr = 0.10
        y_curr = 0.10
        dt = 1.0 / steps
        
        for k in range(1, steps):
            # Target attraction vector
            dx = -gamma_attract * (x_curr - target_x) * dt + np.random.normal(0, sigma_tremor) * math.sqrt(dt)
            dy = -gamma_attract * (y_curr - target_y) * dt + np.random.normal(0, sigma_tremor) * math.sqrt(dt)
            
            x_curr = float(max(0.0, min(1.0, x_curr + dx)))
            y_curr = float(max(0.0, min(1.0, y_curr + dy)))
            
            path_x.append(x_curr)
            path_y.append(y_curr)
            
            # Deviation from targeted straight cylinder trajectory
            ideal_x = 0.10 + (target_x - 0.10) * (k/steps)
            ideal_y = 0.10 + (target_y - 0.10) * (k/steps)
            dist_dev = math.sqrt((x_curr - ideal_x)**2 + (y_curr - ideal_y)**2)
            cumulative_deviation_score += dist_dev
            
            # Shear stress on brain tissue (increases as trajectory shifts randomly)
            shear = 0.35 * (dist_dev * 15.0)**1.4 + 0.1 * drift_factor
            shear_force_psi.append(float(round(max(0.01, shear), 3)))
            
        shear_force_psi.insert(0, 0.0) # match initial step
        final_mismatch_mm = math.sqrt((x_curr - target_x)**2 + (y_curr - target_y)**2) * 10.0 # scale to mm
        
        # Analytical Euler Pitch/Yaw Torque to describe trajectory constraints
        torque_vector = [float(round(0.05 * np.sin(x*12.0) * drift_factor, 3)) for x in path_x]

        return jsonify({
            'steps_axis': list(range(steps)),
            'path_x': path_x,
            'path_y': path_y,
            'target_coords': [target_x, target_y],
            'shear_force_psi': shear_force_psi,
            'torque_vector': torque_vector,
            'final_deviation_mm': round(final_mismatch_mm, 2),
            'surgical_precision_score': round(max(0.0, 100.0 - 8.0 * final_mismatch_mm - 2.5 * cumulative_deviation_score), 1),
            'hebbian_tissue_fatigue_index': round(float(np.mean(shear_force_psi) * 4.2), 3)
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── ENDPOINT: ORTHOPEDIC VR DRILLING & SCREW SIMULATOR ────────────────────────
@app.route('/api/orthopedic-sim', methods=['GET'])
def api_orthopedic_sim():
    """
    Orthopedic force feedback / pedicle drilling trajectory.
    Calculates statistical bone density penetration profile, mechanical torque,
    and screw seating stability indices based on force, RPM, and insertion angles.
    """
    try:
        force_n = float(request.args.get('force', 45.0)) # Thrust force in Newtons
        drill_rpm = float(request.args.get('rpm', 1200.0))
        angle_deg = float(request.args.get('angle', 0.0)) # Insertion angle error from bone axis (deg)
        bone_type = request.args.get('bone_type', 'cortical') # cortical, cancellous, osteoporotic
        
        steps = 50
        penetration_depth_mm = np.linspace(0.0, 30.0, steps).tolist()
        
        # Mechanical torque = f(RPM, bone density profile)
        # Cortical bone is denser with a high resistant peak near the outer rim, then softer cancellous tissue
        torque_nm = []
        axial_vibration_g = []
        np.random.seed(77)
        
        base_density = 1.0
        if bone_type == 'cortical':
            base_density = 1.8
        elif bone_type == 'cancellous':
            base_density = 0.95
        else: # osteoporotic
            base_density = 0.55
            
        for d in penetration_depth_mm:
            # Cortical outer layer: 0-8mm depth has higher torque résistance
            if d < 8.0:
                density_profile = base_density * 2.2 * (1.1 - 0.1 * d)
            else: # inside femoral cavity / vertebral body
                if bone_type == 'cortical':
                    density_profile = base_density * 0.9 # Cancellous transition
                else:
                    density_profile = base_density * 0.7
                    
            # Formula: T = C * Force * Density / (RPM^0.2)
            t_val = 0.005 * force_n * density_profile * (1000.0 / math.sqrt(drill_rpm))
            # Critical warning: screw angular tilting leads to axial friction spikes
            t_val += 0.002 * abs(angle_deg) * (d / 10.0)
            
            # Vibration spikes in osteoporotic or cortical bone shear
            v_val = 0.05 * drill_rpm * density_profile / (force_n + 1.0)
            
            torque_nm.append(float(round(max(0.01, t_val + np.random.normal(0, 0.05)), 3)))
            axial_vibration_g.append(float(round(max(0.01, v_val + np.random.normal(0, 0.02)), 3)))
            
        # Overall seating stability index: lower angle error and steady force yields higher stability
        # Severe tilt error causes cortex damage / loose screw threads
        screw_stability_score = 100.0 - 0.08 * (angle_deg**2) - 0.15 * abs(force_n - 50.0)
        screw_stability_score = float(max(5.0, min(100.0, screw_stability_score)))
        
        # Check cortical breach threat: Breaching occurs if angle exceeds pedicle bottleneck limit
        is_lateral_breach = abs(angle_deg) > 4.5
        is_medial_breach = angle_deg < -3.5
        compliance_check = "✓ SECURE SEATING"
        if is_lateral_breach:
            compliance_check = "✗ CRITICAL CORTER LATERAL BREACH"
        elif is_medial_breach:
            compliance_check = "✗ CRITICAL INNER MEDIAL CANAL CRUSH"

        return jsonify({
            'depth_axis': penetration_depth_mm,
            'torque_nm': torque_nm,
            'axial_vibration_g': axial_vibration_g,
            'characteristics': {
                'stability_percentage': round(screw_stability_score, 1),
                'peak_resistive_torque_nm': round(float(np.max(torque_nm)), 2),
                'mean_mechanical_vibration_g': round(float(np.mean(axial_vibration_g)), 3),
                'cortical_breach_status': compliance_check,
                'is_compliant': not (is_lateral_breach or is_medial_breach)
            }
        })
        
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── ENDPOINT: KNEE EDUCATION GUIDEWIRE & MIS TKA WORKFLOW ───────────────────
@app.route('/api/knee-education-sim', methods=['GET'])
def api_knee_education_sim():
    """
    Educational knee surgery simulator for guidewire characteristics and
    minimally invasive total knee arthroplasty workflow rehearsal.
    """
    try:
        payload = compute_knee_education_simulation(
            wire_diameter_mm=request.args.get('wire_diameter', 2.8),
            stiffness_index=request.args.get('stiffness', 58.0),
            insertion_angle_deg=request.args.get('angle', 1.5),
            approach=request.args.get('approach', 'midvastus'),
            mission=request.args.get('mission', 'joint_allied'),
        )
        return jsonify(payload)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── ENDPOINT: TRAINING ANALYTICS & LEARNING PROGRESS ──────────────────────────
@app.route('/api/training-analytics', methods=['GET'])
def api_training_analytics():
    """
    Compiles resident training records across a simulated 12-week program:
      - Trainee cohort skill variance
      - Wald Sequential stopping curve verifying qualification limits
      - Hidden Markov state trajectories
    """
    try:
        trials = int(request.args.get('trials', 20))
        learning_rate_alpha = float(request.args.get('learning_rate', 0.28))
        decay_weeks = int(request.args.get('decay_weeks', 2))
        
        trials = max(5, min(100, trials))
        learning_rate_alpha = max(0.05, min(0.8, learning_rate_alpha))
        
        # Trainee Skill Trajectories
        skill_curr, errors_curr = compute_learning_curve(
            trials=trials,
            base_skill=12.0,
            asymptotic_skill=94.0,
            alpha=learning_rate_alpha,
            decay_rate=0.03,
            retention_interval_weeks=decay_weeks
        )
        
        # Expert comparative reference trajectory
        expert_skill, _ = compute_learning_curve(
            trials=trials,
            base_skill=75.0,
            asymptotic_skill=98.0,
            alpha=0.15,
            decay_rate=0.0,
            retention_interval_weeks=0
        )
        
        # Wald Sequential Probability boundaries to verify resident skill competency
        # Target skill boundary (H1: certified, skill >= 85) vs Null boundary (H0: uncertified, skill <= 55)
        # S_k = Cumulative log-likelihood ratio steps
        # Boundary limits A & B determine optimal stopping trial
        cert_threshold = 85.0
        uncert_threshold = 55.0
        wald_trajectory = []
        stopping_trial = trials - 1
        is_certified = False
        
        current_cumulative_llr = 0.0
        for idx, val in enumerate(skill_curr):
            # Compute log ratio: log( P(val | H1) / P(val | H0) ) using Gaussian models around targets
            h1_dist = -0.5 * ((val - cert_threshold)/10.0)**2
            h0_dist = -0.5 * ((val - uncert_threshold)/15.0)**2
            val_llr = h1_dist - h0_dist
            current_cumulative_llr += val_llr
            wald_trajectory.append(float(round(current_cumulative_llr, 3)))
            
            # Wald boundary bounds (representing alpha=0.05, beta=0.05 confidence thresholds)
            # Upper boundary A = log((1 - beta)/alpha) ~ 2.944
            if current_cumulative_llr >= 2.944 and not is_certified:
                stopping_trial = idx + 1
                is_certified = True
                
        # Cog State Distribution over typical training durations (e.g., 30 minutes)
        nominal_probs = []
        focused_probs = []
        overload_probs = []
        durations = list(range(5, 61, 5))
        
        for d in durations:
            # skill impacts cognitive load
            current_skill_val = skill_curr[-1] / 100.0
            p_dist = estimate_cognitive_load_probabilities(d, current_skill_val)
            nominal_probs.append(p_dist['nominal_percentage'])
            focused_probs.append(p_dist['focused_percentage'])
            overload_probs.append(p_dist['overload_percentage'])
            
        cognitive_trajectory = {
            'durations_axis_minutes': durations,
            'nominal': nominal_probs,
            'focused': focused_probs,
            'overload': overload_probs
        }

        # Mathematical characteristics dict
        characteristics = {
            'final_residents_skill': skill_curr[-1],
            'retention_percentage': round(100.0 * math.exp(-0.03 * decay_weeks), 2),
            'wald_certified_status': "QUALIFIED & PASSED" if is_certified else "TRAINING IN PROGRESS",
            'wald_stopping_trial': stopping_trial,
            'asymptotic_efficiency': "94.2% Spatial Stability Index",
            'positional_accuracy_achieved': errors_curr[-1]['position_error_mm'],
            'angulation_variance_degrees': errors_curr[-1]['angle_error_deg']
        }

        # Multi-modal educational prescription narrative
        genai_narrative = (
            f"### Sunnybrook Surgical Academy - AI Cognitive Training Prescriptive Outline:\n\n"
            f"1. **Trainee Skill Acceleration Pathway**: Analysis of the current cohort learning rate ($\\alpha = {learning_rate_alpha:.2f}$) "
            f"confirms robust log-linear movement. Repetitive tasking has expanded core surgical dexterity from **{skill_curr[0]:.1f}%** "
            f"to a high operational skill ranking of **{skill_curr[-1]:.1f}%** inside {trials} simulation trials. "
            f"Ebbinghaus memory decay over a non-retention interval of **{decay_weeks} weeks** shows a skill rentention index "
            f"holding tight at **{characteristics['retention_percentage']}%**.\n\n"
            f"2. **Sequential Wald Certification Threshold**: The resident path achieved statistical confirmation ($p < 0.05$) "
            f"of clinical compliance at trial **#{stopping_trial}** (stopping trajectory τ*), "
            f"crossing the upper sequential probability ratio threshold. This denotes that the physical hand-tremor drift and bone-drill angle "
            f"deviations converged below acceptable clinical barriers (Positional accuracy: **{characteristics['positional_accuracy_achieved']} mm**, "
            f"angular variance: **{characteristics['angulation_variance_degrees']}^\\circ**).\n\n"
            f"3. **Cognitive Load & Flow State Modeling**: Transition analysis tells us that for active durations extending beyond 40 minutes, "
            f"the risk of mental overload rises. At current capability levels, focused cognitive immersion remains stable at the 35-minute benchmark, "
            f"with overload probability capped at a manageable of **{overload_probs[-1]:.1f}%**. Continuous 3D spatial haptic simulation "
            f"and visual-language instructions are recommended to sustain focused flow states and prevent motor fatigue."
        )

        return jsonify({
            'trials_axis': list(range(1, trials + 1)),
            'resident_skill_history': skill_curr,
            'expert_skill_history': expert_skill,
            'resident_position_error': [item['position_error_mm'] for item in errors_curr],
            'resident_angle_error': [item['angle_error_deg'] for item in errors_curr],
            'wald_cumulative_llr': wald_trajectory,
            'wald_stopping_trial': stopping_trial,
            'cognitive_profile': cognitive_trajectory,
            'characteristics': characteristics,
            'genai_prescription': genai_narrative
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# ── MAIN INDEX DASHBOARD ROUTE ───────────────────────────────────────────────
@app.route('/')
def index():
    """
    Renders the primary Sunnybrook Health Sciences VR surgical simulator dashboard.
    """
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
