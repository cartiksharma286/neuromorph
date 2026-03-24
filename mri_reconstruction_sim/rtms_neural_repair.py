#!/usr/bin/env python3
"""
rTMS Neural Repair Engine for Improved Cognition & Smart Aging
==============================================================

Optimal treatment paradigms for repetitive Transcranial Magnetic Stimulation
targeting neural repair, cognitive enhancement, and age-related decline reversal.

Models:
  - LTP/LTD synaptic plasticity dynamics
  - BDNF-mediated neurogenesis
  - Cortical excitability mapping (E-field FEA)
  - Optimal protocol search (frequency, intensity, duration, target)
  - Cognitive domain scoring (memory, attention, executive function, processing speed)
  - Smart aging biomarker trajectories
"""

import math
import os
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats, ndimage, signal as sp_signal
from scipy.optimize import minimize

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA_TMS = 42.577e6       # MHz/T for reference
MU_0 = 4 * math.pi * 1e-7  # vacuum permeability

# ── Cortical Targets (MNI coordinates) ────────────────────────────────────────
CORTICAL_TARGETS = {
    "L-DLPFC":   {"mni": [-46, 45, 38],  "role": "Executive function, working memory",
                  "brodmann": 46, "depth_mm": 15},
    "R-DLPFC":   {"mni": [46, 45, 38],   "role": "Inhibition, decision making",
                  "brodmann": 46, "depth_mm": 15},
    "L-M1":      {"mni": [-38, -22, 54], "role": "Motor cortex, plasticity marker",
                  "brodmann": 4,  "depth_mm": 12},
    "L-IPL":     {"mni": [-40, -60, 45], "role": "Attention, spatial cognition",
                  "brodmann": 40, "depth_mm": 18},
    "L-STS":     {"mni": [-52, -40, 8],  "role": "Language, social cognition",
                  "brodmann": 22, "depth_mm": 20},
    "mPFC":      {"mni": [0, 52, 28],    "role": "Default mode, self-referential",
                  "brodmann": 10, "depth_mm": 14},
    "PCC":       {"mni": [0, -52, 26],   "role": "Memory consolidation, DMN hub",
                  "brodmann": 31, "depth_mm": 22},
    "Hippocampus-indirect": {"mni": [-26, -18, -18], "role": "Memory encoding (via parietal)",
                             "brodmann": 0, "depth_mm": 45},
    "Cerebellum": {"mni": [0, -76, -32], "role": "Motor learning, timing",
                   "brodmann": 0, "depth_mm": 10},
}

# ── Treatment Paradigms ──────────────────────────────────────────────────────
TREATMENT_PARADIGMS = {
    "cognitive_enhancement": {
        "label": "Cognitive Enhancement",
        "targets": ["L-DLPFC", "L-IPL"],
        "frequency_hz": 10.0,
        "intensity_pct": 80,
        "pulses_per_session": 2000,
        "sessions": 20,
        "schedule": "5x/week × 4 weeks",
        "protocol": "iTBS",  # intermittent theta burst
        "description": "High-frequency excitatory rTMS over L-DLPFC for working memory and executive function enhancement"
    },
    "memory_repair": {
        "label": "Memory Repair (MCI/Early AD)",
        "targets": ["L-DLPFC", "L-IPL", "PCC"],
        "frequency_hz": 20.0,
        "intensity_pct": 100,
        "pulses_per_session": 1600,
        "sessions": 30,
        "schedule": "5x/week × 6 weeks",
        "protocol": "rTMS-20Hz",
        "description": "Multi-target high-frequency rTMS for hippocampal-cortical network strengthening"
    },
    "smart_aging": {
        "label": "Smart Aging / Neuroprotection",
        "targets": ["L-DLPFC", "mPFC", "L-M1"],
        "frequency_hz": 10.0,
        "intensity_pct": 90,
        "pulses_per_session": 3000,
        "sessions": 40,
        "schedule": "3x/week × 13 weeks",
        "protocol": "iTBS + 10Hz",
        "description": "Combined protocol targeting age-related cortical thinning and BDNF upregulation"
    },
    "stroke_motor": {
        "label": "Post-Stroke Motor Recovery",
        "targets": ["L-M1", "Cerebellum"],
        "frequency_hz": 1.0,
        "intensity_pct": 110,
        "pulses_per_session": 1200,
        "sessions": 15,
        "schedule": "5x/week × 3 weeks",
        "protocol": "LF-contralesional + HF-ipsilesional",
        "description": "Inhibitory 1Hz contralateral + excitatory ipsilateral for interhemispheric rebalancing"
    },
    "depression_repair": {
        "label": "Depression Neural Repair",
        "targets": ["L-DLPFC", "R-DLPFC"],
        "frequency_hz": 10.0,
        "intensity_pct": 120,
        "pulses_per_session": 3000,
        "sessions": 30,
        "schedule": "5x/week × 6 weeks",
        "protocol": "Deep TMS H-coil",
        "description": "FDA-approved excitatory L-DLPFC protocol with deep H-coil for treatment-resistant depression"
    },
    "neuroplasticity_boost": {
        "label": "Neuroplasticity Boost",
        "targets": ["L-M1", "L-DLPFC", "Cerebellum"],
        "frequency_hz": 50.0,
        "intensity_pct": 80,
        "pulses_per_session": 600,
        "sessions": 10,
        "schedule": "daily × 10 days",
        "protocol": "cTBS",
        "description": "Continuous theta burst stimulation for rapid LTP-like plasticity induction"
    },
}

# ── Cognitive Domains ─────────────────────────────────────────────────────────
COGNITIVE_DOMAINS = [
    "working_memory", "episodic_memory", "attention",
    "processing_speed", "executive_function", "language",
    "visuospatial", "social_cognition"
]

# ── Aging Biomarkers ──────────────────────────────────────────────────────────
AGING_BIOMARKERS = {
    "cortical_thickness_mm": {"baseline_60y": 2.42, "decline_per_year": 0.015, "unit": "mm"},
    "hippocampal_volume_mm3": {"baseline_60y": 3400, "decline_per_year": 35, "unit": "mm³"},
    "white_matter_integrity": {"baseline_60y": 0.42, "decline_per_year": 0.005, "unit": "FA"},
    "bdnf_pg_ml": {"baseline_60y": 18.0, "decline_per_year": 0.4, "unit": "pg/mL"},
    "cerebral_blood_flow": {"baseline_60y": 48.0, "decline_per_year": 0.5, "unit": "mL/100g/min"},
    "synaptic_density": {"baseline_60y": 0.72, "decline_per_year": 0.008, "unit": "normalised"},
    "myelination_index": {"baseline_60y": 0.65, "decline_per_year": 0.006, "unit": "normalised"},
    "neuroinflammation_marker": {"baseline_60y": 1.2, "decline_per_year": -0.08, "unit": "TSPO-PET SUVr"},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Synaptic Plasticity Model (LTP / LTD)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_plasticity_dynamics(
    frequency_hz: float,
    intensity_pct: float,
    n_pulses: int,
    sessions: int,
    target_depth_mm: float = 15.0,
) -> dict:
    """
    BCM (Bienenstock-Cooper-Munro) plasticity model for rTMS.
    
    High-freq (≥5 Hz) → LTP (long-term potentiation) → excitatory
    Low-freq (≤1 Hz)  → LTD (long-term depression) → inhibitory
    
    Returns time-course of synaptic weight, BDNF, and repair index.
    """
    # E-field attenuation with depth (figure-8 coil model)
    e_field_surface = 150.0 * (intensity_pct / 100.0)  # V/m at cortex
    e_field = e_field_surface * np.exp(-target_depth_mm / 28.0)

    # BCM threshold (sliding)
    theta_m = 100.0  # modification threshold (V/m)

    # Plasticity direction
    if frequency_hz >= 5.0:
        # LTP regime
        eta = 0.02 * (frequency_hz / 10.0)  # learning rate scales with freq
        direction = 1.0
    else:
        # LTD regime
        eta = 0.015 * (1.0 / max(frequency_hz, 0.1))
        direction = -1.0

    # Session-by-session plasticity accumulation
    n_time = sessions * 10  # 10 time points per session
    t = np.linspace(0, sessions, n_time)
    synaptic_weight = np.ones(n_time)
    bdnf_level = np.ones(n_time) * 18.0  # baseline pg/mL
    repair_index = np.zeros(n_time)
    ltp_accumulator = 0.0

    for i in range(1, n_time):
        session_idx = int(t[i])
        pulse_factor = n_pulses / 1000.0

        # BCM rule: Δw = η · φ(w) · (E - θ_m) · direction
        phi_w = synaptic_weight[i-1] * (synaptic_weight[i-1] - 0.5)
        delta_w = eta * phi_w * (e_field - theta_m) / theta_m * direction * pulse_factor
        
        # Homeostatic metaplasticity (prevents runaway)
        homeostatic = -0.005 * (synaptic_weight[i-1] - 1.0)
        
        synaptic_weight[i] = np.clip(
            synaptic_weight[i-1] + delta_w + homeostatic + np.random.normal(0, 0.003),
            0.3, 2.5
        )

        # BDNF upregulation (activity-dependent)
        bdnf_rate = 0.4 * abs(delta_w) * frequency_hz * (intensity_pct / 100.0)
        bdnf_decay = 0.02 * (bdnf_level[i-1] - 18.0)
        bdnf_level[i] = max(5.0, bdnf_level[i-1] + bdnf_rate - bdnf_decay
                            + np.random.normal(0, 0.1))

        # Repair index = combination of LTP and BDNF
        ltp_accumulator += max(0, delta_w) * 0.1
        repair_index[i] = np.tanh(ltp_accumulator) * (bdnf_level[i] / 25.0)

    return {
        "time_sessions": t.tolist(),
        "synaptic_weight": synaptic_weight.tolist(),
        "bdnf_pg_ml": bdnf_level.tolist(),
        "repair_index": repair_index.tolist(),
        "final_synaptic_weight": round(float(synaptic_weight[-1]), 4),
        "final_bdnf": round(float(bdnf_level[-1]), 2),
        "final_repair_index": round(float(repair_index[-1]), 4),
        "plasticity_type": "LTP" if direction > 0 else "LTD",
        "e_field_at_target": round(float(e_field), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Cortical Excitability (E-field FEA)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_efield_map(
    target: str = "L-DLPFC",
    intensity_pct: float = 80.0,
    coil_type: str = "figure8",
    resolution: int = 64,
) -> dict:
    """
    Finite-element E-field simulation over cortical surface.
    Models figure-8 and H-coil geometries with tissue conductivity layers.
    """
    x = np.linspace(-80, 80, resolution)  # mm
    y = np.linspace(-80, 80, resolution)
    X, Y = np.meshgrid(x, y)

    target_info = CORTICAL_TARGETS.get(target, CORTICAL_TARGETS["L-DLPFC"])
    tx, ty = target_info["mni"][0], target_info["mni"][1]

    # Coil geometry
    if coil_type == "figure8":
        # Two opposing loops
        d = 35  # mm half-separation
        E = (intensity_pct / 100.0) * 150 * (
            np.exp(-((X - tx + d)**2 + (Y - ty)**2) / 800) -
            np.exp(-((X - tx - d)**2 + (Y - ty)**2) / 800)
        )
        E = np.abs(E)
        focality_mm2 = float(np.sum(E > 0.5 * E.max()) * (160/resolution)**2)
    elif coil_type == "hcoil":
        # Deep H-coil: broader, deeper field
        E = (intensity_pct / 100.0) * 120 * np.exp(
            -((X - tx)**2 + (Y - ty)**2) / 2500
        )
        focality_mm2 = float(np.sum(E > 0.5 * E.max()) * (160/resolution)**2)
    else:
        # Circular coil
        r = np.sqrt((X - tx)**2 + (Y - ty)**2)
        E = (intensity_pct / 100.0) * 130 * np.exp(-r**2 / 1200) * (1 + 0.3 * np.cos(r / 20))
        focality_mm2 = float(np.sum(E > 0.5 * E.max()) * (160/resolution)**2)

    # Add sulcal geometry effects
    sulci = 0.15 * np.sin(X / 8) * np.cos(Y / 6)
    E = np.clip(E * (1 + sulci), 0, None)

    # Tissue layers attenuation
    depth_map = target_info["depth_mm"] + 2 * np.random.randn(resolution, resolution)
    depth_map = np.clip(depth_map, 5, 50)
    E_cortex = E * np.exp(-depth_map / 28.0)

    peak_efield = float(E_cortex.max())
    mean_efield = float(E_cortex[E_cortex > 0.1 * peak_efield].mean())

    return {
        "efield_map": E_cortex.tolist(),
        "peak_efield_vm": round(peak_efield, 2),
        "mean_efield_vm": round(mean_efield, 2),
        "focality_mm2": round(focality_mm2, 1),
        "target": target,
        "target_mni": target_info["mni"],
        "coil_type": coil_type,
        "resolution": resolution,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Cognitive Domain Assessment
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cognitive_trajectory(
    paradigm: str = "cognitive_enhancement",
    age: int = 65,
    baseline_mmse: float = 26.0,
) -> dict:
    """
    Model cognitive domain improvement trajectories under rTMS treatment.
    Incorporates age-dependent response curves and ceiling effects.
    """
    protocol = TREATMENT_PARADIGMS.get(paradigm, TREATMENT_PARADIGMS["cognitive_enhancement"])
    sessions = protocol["sessions"]
    freq = protocol["frequency_hz"]

    # Age-dependent response factor (younger responds better)
    age_factor = np.clip(1.0 - (age - 50) * 0.008, 0.4, 1.0)

    # MMSE-dependent room for improvement
    mmse_room = (30.0 - baseline_mmse) / 30.0

    # Baseline cognitive scores (z-scores, age-normed)
    baseline = {}
    for domain in COGNITIVE_DOMAINS:
        # Mild deficit baseline scaled by MMSE
        baseline[domain] = -0.5 - mmse_room * 1.5 + np.random.normal(0, 0.2)

    # Treatment response per domain
    # Domains most affected by target sites
    domain_sensitivity = {
        "L-DLPFC": {"working_memory": 0.8, "executive_function": 0.9, "attention": 0.5,
                    "processing_speed": 0.4},
        "L-IPL": {"attention": 0.7, "visuospatial": 0.6, "working_memory": 0.3},
        "mPFC": {"social_cognition": 0.6, "executive_function": 0.3},
        "PCC": {"episodic_memory": 0.8, "working_memory": 0.4},
        "L-M1": {"processing_speed": 0.5},
        "Cerebellum": {"processing_speed": 0.4, "visuospatial": 0.3},
        "L-STS": {"language": 0.7, "social_cognition": 0.5},
        "R-DLPFC": {"executive_function": 0.4, "attention": 0.4},
    }

    # Aggregate sensitivity from all targets
    agg_sens = {d: 0.0 for d in COGNITIVE_DOMAINS}
    for tgt in protocol["targets"]:
        for domain, sens in domain_sensitivity.get(tgt, {}).items():
            agg_sens[domain] = max(agg_sens[domain], sens)

    # Session-by-session trajectories
    trajectories = {d: [] for d in COGNITIVE_DOMAINS}
    for d in COGNITIVE_DOMAINS:
        score = baseline[d]
        max_gain = agg_sens[d] * age_factor * (freq / 10.0) ** 0.3
        for s in range(sessions + 1):
            trajectories[d].append(round(score, 3))
            # Saturating exponential improvement
            gain = max_gain * (1 - np.exp(-s / (sessions * 0.3)))
            noise = np.random.normal(0, 0.02)
            score = baseline[d] + gain + noise

    # Post-treatment MMSE estimate
    mean_improvement = np.mean([trajectories[d][-1] - trajectories[d][0]
                                for d in COGNITIVE_DOMAINS])
    mmse_post = min(30.0, baseline_mmse + mean_improvement * 3.0)

    return {
        "paradigm": paradigm,
        "protocol_label": protocol["label"],
        "age": age,
        "baseline_mmse": baseline_mmse,
        "post_treatment_mmse": round(float(mmse_post), 1),
        "mmse_change": round(float(mmse_post - baseline_mmse), 1),
        "sessions": sessions,
        "baseline_scores": {d: round(baseline[d], 3) for d in COGNITIVE_DOMAINS},
        "final_scores": {d: round(trajectories[d][-1], 3) for d in COGNITIVE_DOMAINS},
        "trajectories": trajectories,
        "domain_sensitivity": {d: round(agg_sens[d], 2) for d in COGNITIVE_DOMAINS},
        "age_response_factor": round(age_factor, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Smart Aging Biomarker Model
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aging_biomarkers(
    age: int = 65,
    paradigm: str = "smart_aging",
    follow_up_years: int = 5,
) -> dict:
    """
    Model the effect of rTMS treatment on aging biomarkers over a follow-up period.
    Compares treated vs untreated trajectories.
    """
    protocol = TREATMENT_PARADIGMS.get(paradigm, TREATMENT_PARADIGMS["smart_aging"])
    freq = protocol["frequency_hz"]
    intensity = protocol["intensity_pct"]
    sessions = protocol["sessions"]

    years = np.arange(0, follow_up_years + 0.25, 0.25)
    results = {}

    for biomarker, params in AGING_BIOMARKERS.items():
        baseline = params["baseline_60y"] + params["decline_per_year"] * (age - 60)
        decline = params["decline_per_year"]

        # Untreated trajectory (natural aging)
        untreated = [baseline + decline * y + np.random.normal(0, abs(decline) * 0.1)
                     for y in years]

        # Treatment effect: rTMS slows decline + partial reversal
        # Effect magnitude depends on frequency, intensity, and biomarker type
        if biomarker in ("bdnf_pg_ml", "synaptic_density"):
            # BDNF and synaptic density are most responsive
            treatment_boost = 0.15 * (freq / 10.0) * (intensity / 100.0) * (sessions / 20.0)
            maintenance = 0.6  # fraction of effect maintained per year without booster
        elif biomarker in ("cortical_thickness_mm", "hippocampal_volume_mm3"):
            treatment_boost = 0.05 * (freq / 10.0) * (sessions / 20.0)
            maintenance = 0.8
        elif biomarker == "neuroinflammation_marker":
            treatment_boost = -0.08 * (freq / 10.0)  # decrease is good
            maintenance = 0.5
        else:
            treatment_boost = 0.08 * (freq / 10.0) * (intensity / 100.0)
            maintenance = 0.7

        treated = []
        for y in years:
            natural = baseline + decline * y
            # Treatment effect with exponential decay after protocol ends
            tx_duration_y = protocol["sessions"] / (3 * 52)  # approximate treatment duration in years
            if y <= tx_duration_y:
                effect = treatment_boost * (y / tx_duration_y) * abs(baseline)
            else:
                peak_effect = treatment_boost * abs(baseline)
                years_post = y - tx_duration_y
                effect = peak_effect * (maintenance ** years_post)
            treated.append(natural + effect + np.random.normal(0, abs(decline) * 0.05))

        # Biological age calculation
        bio_age_diff = (np.array(treated) - np.array(untreated)) / abs(decline) if decline != 0 else np.zeros_like(years)

        results[biomarker] = {
            "unit": params["unit"],
            "baseline": round(baseline, 3),
            "years": years.tolist(),
            "untreated": [round(v, 3) for v in untreated],
            "treated": [round(v, 3) for v in treated],
            "final_untreated": round(float(untreated[-1]), 3),
            "final_treated": round(float(treated[-1]), 3),
            "preservation_pct": round(
                float((treated[-1] - untreated[-1]) / (abs(baseline) + 1e-12) * 100), 2),
        }

    # Composite "brain age gap"
    age_gaps = [results[b]["preservation_pct"] for b in AGING_BIOMARKERS]
    brain_age_benefit = np.mean(age_gaps) * 0.1  # years younger

    return {
        "biomarkers": results,
        "brain_age_benefit_years": round(float(brain_age_benefit), 2),
        "paradigm": paradigm,
        "age": age,
        "follow_up_years": follow_up_years,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Optimal Protocol Search
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_protocol(
    goal: str = "cognitive_enhancement",
    age: int = 65,
    baseline_mmse: float = 26.0,
    constraints: dict = None,
) -> dict:
    """
    Bayesian-inspired search over rTMS parameter space to find
    optimal treatment paradigm for a given goal, age, and baseline.
    """
    if constraints is None:
        constraints = {}

    # Parameter ranges
    freq_range = constraints.get("freq_range", [1, 50])
    intensity_range = constraints.get("intensity_range", [60, 120])
    session_range = constraints.get("session_range", [10, 40])

    best_score = -np.inf
    best_params = {}
    search_history = []

    # Latin hypercube sampling of parameter space
    n_samples = 40
    np.random.seed(42)
    freqs = np.random.uniform(freq_range[0], freq_range[1], n_samples)
    intensities = np.random.uniform(intensity_range[0], intensity_range[1], n_samples)
    sess = np.random.randint(session_range[0], session_range[1] + 1, n_samples)

    target_options = list(CORTICAL_TARGETS.keys())[:6]

    for i in range(n_samples):
        f = freqs[i]
        inten = intensities[i]
        s = int(sess[i])
        targets = np.random.choice(target_options, size=min(3, len(target_options)),
                                   replace=False).tolist()

        # Quick plasticity eval
        plast = compute_plasticity_dynamics(f, inten, 2000, s, 15.0)

        # Score: repair index + BDNF boost - safety penalty
        score = plast["final_repair_index"] * 2.0
        score += (plast["final_bdnf"] - 18.0) / 10.0
        score += plast["final_synaptic_weight"] * 0.5

        # Safety: penalise extreme intensities
        if inten > 110:
            score -= (inten - 110) * 0.02
        if f > 25 and inten > 100:
            score -= 0.1  # SAR concern

        # Age penalty
        age_factor = np.clip(1.0 - (age - 50) * 0.005, 0.5, 1.0)
        score *= age_factor

        search_history.append({
            "iteration": i + 1,
            "frequency_hz": round(float(f), 2),
            "intensity_pct": round(float(inten), 1),
            "sessions": s,
            "targets": targets,
            "score": round(float(score), 4),
        })

        if score > best_score:
            best_score = score
            best_params = {
                "frequency_hz": round(float(f), 2),
                "intensity_pct": round(float(inten), 1),
                "sessions": s,
                "targets": targets,
                "repair_index": plast["final_repair_index"],
                "bdnf_final": plast["final_bdnf"],
                "synaptic_weight": plast["final_synaptic_weight"],
                "plasticity_type": plast["plasticity_type"],
                "e_field_vm": plast["e_field_at_target"],
            }

    return {
        "goal": goal,
        "age": age,
        "baseline_mmse": baseline_mmse,
        "optimal_parameters": best_params,
        "best_score": round(float(best_score), 4),
        "search_history": search_history,
        "n_evaluated": n_samples,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Visualization: Generate Plots (base64)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rtms_plots(
    plasticity: dict,
    cognitive: dict,
    aging: dict,
    efield: dict,
    optimal: dict,
) -> dict:
    """Generate publication-quality plots for all rTMS results. Returns base64 PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plots = {}

    # ── Plot 1: Plasticity Dynamics ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor('#0a0e1a')

    t = plasticity["time_sessions"]
    ax = axes[0]
    ax.plot(t, plasticity["synaptic_weight"], color='#00e5ff', linewidth=2)
    ax.axhline(1.0, color='#555', linestyle='--', linewidth=0.8)
    ax.set_title("Synaptic Weight (BCM)", color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel("Session", color='white')
    ax.set_ylabel("w / w₀", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')

    ax = axes[1]
    ax.plot(t, plasticity["bdnf_pg_ml"], color='#76ff03', linewidth=2)
    ax.axhline(18.0, color='#555', linestyle='--', linewidth=0.8)
    ax.fill_between(t, 18.0, plasticity["bdnf_pg_ml"], alpha=0.15, color='#76ff03')
    ax.set_title("BDNF Level", color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel("Session", color='white')
    ax.set_ylabel("pg/mL", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')

    ax = axes[2]
    ax.plot(t, plasticity["repair_index"], color='#ff9100', linewidth=2)
    ax.set_title("Neural Repair Index", color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel("Session", color='white')
    ax.set_ylabel("Repair Index", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')

    fig.suptitle(f"rTMS Plasticity Dynamics — {plasticity['plasticity_type']} | "
                 f"E={plasticity['e_field_at_target']} V/m",
                 color='#38bdf8', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    plots["plasticity"] = _fig_to_b64(fig)

    # ── Plot 2: Cognitive Trajectories ────────────────────────────────────────
    n_domains = len(COGNITIVE_DOMAINS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor('#0a0e1a')
    domain_colors = ['#00e5ff', '#76ff03', '#ff9100', '#e040fb',
                     '#ff5252', '#ffff00', '#18ffff', '#ea80fc']

    for idx, domain in enumerate(COGNITIVE_DOMAINS):
        ax = axes[idx // 4, idx % 4]
        traj = cognitive["trajectories"][domain]
        sessions_x = list(range(len(traj)))
        ax.plot(sessions_x, traj, color=domain_colors[idx], linewidth=2)
        ax.axhline(0, color='#555', linestyle='--', linewidth=0.5)
        change = traj[-1] - traj[0]
        arrow = "↑" if change > 0 else "↓"
        ax.set_title(f"{domain.replace('_',' ').title()}\n{arrow}{abs(change):.2f} SD",
                     color='white', fontsize=9, fontweight='bold')
        ax.set_xlabel("Session", color='white', fontsize=7)
        ax.set_ylabel("z-score", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=7)
        ax.set_facecolor('#0a0e1a')

    fig.suptitle(f"Cognitive Domain Trajectories — {cognitive['protocol_label']} | "
                 f"MMSE {cognitive['baseline_mmse']}→{cognitive['post_treatment_mmse']}",
                 color='#38bdf8', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    plots["cognitive"] = _fig_to_b64(fig)

    # ── Plot 3: E-field Map ───────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.patch.set_facecolor('#0a0e1a')
    efield_data = np.array(efield["efield_map"])
    cmap = LinearSegmentedColormap.from_list('efield', [
        (0, '#000033'), (0.25, '#0033cc'), (0.5, '#00cc66'),
        (0.75, '#ffcc00'), (1.0, '#ff3300')
    ], N=256)
    im = ax.imshow(efield_data, cmap=cmap, extent=[-80, 80, -80, 80])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('E-field (V/m)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    # Mark target
    tgt = efield["target_mni"]
    ax.plot(tgt[0], tgt[1], 'x', color='#ff0000', markersize=15, markeredgewidth=3)
    ax.annotate(efield["target"], xy=(tgt[0], tgt[1]),
                xytext=(tgt[0]+10, tgt[1]+10), color='white', fontsize=10,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white'))
    ax.set_title(f"Cortical E-field | Peak={efield['peak_efield_vm']} V/m | "
                 f"Focality={efield['focality_mm2']:.0f} mm²",
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("MNI x (mm)", color='white')
    ax.set_ylabel("MNI y (mm)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')
    fig.tight_layout()
    plots["efield"] = _fig_to_b64(fig)

    # ── Plot 4: Aging Biomarkers ──────────────────────────────────────────────
    bio_keys = list(aging["biomarkers"].keys())
    n_bio = len(bio_keys)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.patch.set_facecolor('#0a0e1a')

    for idx, bk in enumerate(bio_keys):
        ax = axes[idx // 4, idx % 4]
        bdata = aging["biomarkers"][bk]
        years = bdata["years"]
        ax.plot(years, bdata["untreated"], color='#ff5252', linewidth=2,
                linestyle='--', label='Untreated', alpha=0.8)
        ax.plot(years, bdata["treated"], color='#00e5ff', linewidth=2,
                label='rTMS-treated')
        ax.fill_between(years, bdata["untreated"], bdata["treated"],
                        alpha=0.1, color='#00e5ff')
        ax.set_title(f"{bk.replace('_',' ').title()}\n({bdata['unit']})",
                     color='white', fontsize=8, fontweight='bold')
        ax.set_xlabel("Years", color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_facecolor('#0a0e1a')
        if idx == 0:
            ax.legend(fontsize=6, facecolor='#1a1a2e', edgecolor='#333',
                      labelcolor='white')

    fig.suptitle(f"Smart Aging Biomarkers — Brain Age Benefit: "
                 f"{aging['brain_age_benefit_years']:.1f} years younger",
                 color='#38bdf8', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    plots["aging"] = _fig_to_b64(fig)

    # ── Plot 5: Optimization Landscape ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0e1a')

    history = optimal["search_history"]
    freqs = [h["frequency_hz"] for h in history]
    intens = [h["intensity_pct"] for h in history]
    scores = [h["score"] for h in history]

    ax = axes[0]
    sc = ax.scatter(freqs, intens, c=scores, cmap='viridis', s=60, edgecolors='white',
                    linewidth=0.5, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Score', color='white')
    cbar.ax.tick_params(colors='white')
    best = optimal["optimal_parameters"]
    ax.scatter([best["frequency_hz"]], [best["intensity_pct"]],
               marker='*', s=300, c='#ff0000', edgecolors='white', linewidth=2, zorder=5)
    ax.set_title("Protocol Optimisation Space", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", color='white')
    ax.set_ylabel("Intensity (% MSO)", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')

    ax = axes[1]
    iters = [h["iteration"] for h in history]
    cummax = np.maximum.accumulate(scores)
    ax.plot(iters, scores, 'o', color='#555', markersize=4, alpha=0.5)
    ax.plot(iters, cummax, color='#00e5ff', linewidth=2, label='Best so far')
    ax.axhline(optimal["best_score"], color='#ff0000', linestyle='--', linewidth=1)
    ax.set_title("Convergence", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Iteration", color='white')
    ax.set_ylabel("Score", color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0e1a')
    ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    fig.suptitle(f"Optimal: {best['frequency_hz']} Hz @ {best['intensity_pct']}% MSO | "
                 f"Score={optimal['best_score']:.3f}",
                 color='#38bdf8', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    plots["optimization"] = _fig_to_b64(fig)

    return plots


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_rtms_neural_repair_pipeline(
    paradigm: str = "cognitive_enhancement",
    age: int = 65,
    baseline_mmse: float = 26.0,
    target: str = "L-DLPFC",
    coil_type: str = "figure8",
    follow_up_years: int = 5,
) -> dict:
    """
    Complete rTMS neural repair pipeline:
    1. Plasticity dynamics (BCM model)
    2. E-field FEA simulation
    3. Cognitive domain trajectories
    4. Smart aging biomarker modelling
    5. Optimal protocol search
    6. Visualisation generation
    """
    protocol = TREATMENT_PARADIGMS.get(paradigm, TREATMENT_PARADIGMS["cognitive_enhancement"])

    # 1. Plasticity
    target_info = CORTICAL_TARGETS.get(target, CORTICAL_TARGETS["L-DLPFC"])
    plasticity = compute_plasticity_dynamics(
        protocol["frequency_hz"], protocol["intensity_pct"],
        protocol["pulses_per_session"], protocol["sessions"],
        target_info["depth_mm"],
    )

    # 2. E-field
    efield = simulate_efield_map(target, protocol["intensity_pct"], coil_type)

    # 3. Cognitive trajectory
    cognitive = compute_cognitive_trajectory(paradigm, age, baseline_mmse)

    # 4. Aging biomarkers
    aging = compute_aging_biomarkers(age, paradigm, follow_up_years)

    # 5. Optimal protocol
    optimal = find_optimal_protocol(paradigm, age, baseline_mmse)

    # 6. Plots
    plots = generate_rtms_plots(plasticity, cognitive, aging, efield, optimal)

    return {
        "paradigm": paradigm,
        "protocol": protocol,
        "target": target,
        "coil_type": coil_type,
        "plasticity": plasticity,
        "efield": {
            "peak_efield_vm": efield["peak_efield_vm"],
            "mean_efield_vm": efield["mean_efield_vm"],
            "focality_mm2": efield["focality_mm2"],
            "target_mni": efield["target_mni"],
        },
        "cognitive": {
            "baseline_mmse": cognitive["baseline_mmse"],
            "post_treatment_mmse": cognitive["post_treatment_mmse"],
            "mmse_change": cognitive["mmse_change"],
            "baseline_scores": cognitive["baseline_scores"],
            "final_scores": cognitive["final_scores"],
            "domain_sensitivity": cognitive["domain_sensitivity"],
            "age_response_factor": cognitive["age_response_factor"],
        },
        "aging": {
            "brain_age_benefit_years": aging["brain_age_benefit_years"],
            "biomarker_summary": {
                bk: {
                    "baseline": v["baseline"],
                    "final_treated": v["final_treated"],
                    "final_untreated": v["final_untreated"],
                    "preservation_pct": v["preservation_pct"],
                    "unit": v["unit"],
                }
                for bk, v in aging["biomarkers"].items()
            },
        },
        "optimal": {
            "best_score": optimal["best_score"],
            "optimal_parameters": optimal["optimal_parameters"],
        },
        "plots": plots,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running rTMS Neural Repair Pipeline …")
    result = run_rtms_neural_repair_pipeline(
        paradigm="smart_aging", age=65, baseline_mmse=26.0,
        target="L-DLPFC", coil_type="figure8", follow_up_years=5,
    )
    print(f"  Plasticity: {result['plasticity']['plasticity_type']} | "
          f"Repair index = {result['plasticity']['final_repair_index']:.4f}")
    print(f"  BDNF: {result['plasticity']['final_bdnf']:.1f} pg/mL")
    print(f"  E-field peak: {result['efield']['peak_efield_vm']} V/m")
    print(f"  MMSE: {result['cognitive']['baseline_mmse']} → "
          f"{result['cognitive']['post_treatment_mmse']} (Δ{result['cognitive']['mmse_change']})")
    print(f"  Brain age benefit: {result['aging']['brain_age_benefit_years']:.1f} years")
    print(f"  Optimal: {result['optimal']['optimal_parameters']['frequency_hz']} Hz @ "
          f"{result['optimal']['optimal_parameters']['intensity_pct']}% MSO")
    print("Done.")
