"""Statistical rTMS stroke-repair staging and long-horizon trajectory model."""

import math
from typing import Dict, List

import numpy as np


def continued_fraction(value: float, depth: int = 6) -> Dict[str, object]:
    coefficients: List[int] = []
    remainder = float(value)
    for _ in range(max(1, int(depth))):
        coefficient = math.floor(remainder)
        coefficients.append(int(coefficient))
        fractional = remainder - coefficient
        if abs(fractional) < 1e-9:
            break
        remainder = 1.0 / fractional

    convergents = []
    p_previous, p_current = 0, 1
    q_previous, q_current = 1, 0
    for coefficient in coefficients:
        numerator = coefficient * p_current + p_previous
        denominator = coefficient * q_current + q_previous
        convergents.append({
            "p": int(numerator),
            "q": int(denominator),
            "fraction": f"{int(numerator)}/{int(denominator)}",
            "value": float(numerator / denominator) if denominator else float(numerator),
        })
        p_previous, p_current = p_current, numerator
        q_previous, q_current = q_current, denominator

    return {"coefficients": coefficients, "convergents": convergents}


def elliptic_moduli_gate(modulus: float) -> Dict[str, float]:
    """Compute a bounded elliptic-moduli gate using the AGM approximation to K(m)."""
    m = float(np.clip(modulus, 0.05, 0.95))
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(12):
        a, b = (a + b) / 2.0, math.sqrt(a * b)
    k_m = math.pi / (2.0 * a)
    complement = 1.0 - m
    ac, bc = 1.0, math.sqrt(1.0 - complement)
    for _ in range(12):
        ac, bc = (ac + bc) / 2.0, math.sqrt(ac * bc)
    k_complement = math.pi / (2.0 * ac)
    nome = math.exp(-math.pi * k_complement / k_m)
    return {"m": round(m, 5), "K_m": round(k_m, 5), "K_1_minus_m": round(k_complement, 5), "nome_q": round(nome, 6), "gate_score": round(1.0 - nome, 5)}


def simulate_stroke_repair_rtms(
    baseline_fugl_meyer: float = 32.0,
    horizon_weeks: int = 52,
    galois_modulus: int = 13,
    cf_depth: int = 6,
    target_sync_ratio: float = 1.6180339887,
    closed_loop_gain: float = 1.35,
) -> Dict[str, object]:
    """Return a deterministic research simulation, not a clinical prescription."""
    baseline = float(np.clip(baseline_fugl_meyer, 10.0, 60.0))
    weeks = list(range(0, int(np.clip(horizon_weeks, 12, 104)) + 1))
    modulus = max(3, int(galois_modulus))
    gain = float(np.clip(closed_loop_gain, 0.4, 2.5))
    cf = continued_fraction(target_sync_ratio, cf_depth)
    moduli = elliptic_moduli_gate(0.5 + 0.35 * (gain - 0.4) / 2.1)
    rng = np.random.default_rng(20260924)

    stage_names = ["stabilize", "prime", "potentiate", "consolidate", "maintain"]
    stage_lengths = [2, 4, 6, 12, max(1, len(weeks) - 24)]
    stages = []
    stage_index = 0
    for week in weeks:
        while stage_index < len(stage_lengths) - 1 and week >= sum(stage_lengths[:stage_index + 1]):
            stage_index += 1
        stages.append(stage_names[stage_index])

    protocols = {
        "stabilize": {"frequency_hz": 1.0, "intensity_pct_mso": 80, "sessions_per_week": 3},
        "prime": {"frequency_hz": 10.0, "intensity_pct_mso": 90, "sessions_per_week": 5},
        "potentiate": {"frequency_hz": 20.0, "intensity_pct_mso": 90, "sessions_per_week": 5},
        "consolidate": {"frequency_hz": 10.0, "intensity_pct_mso": 85, "sessions_per_week": 3},
        "maintain": {"frequency_hz": 10.0, "intensity_pct_mso": 80, "sessions_per_week": 1},
    }

    neurovascular_signatures = {
        "stabilize": {"regions": "ipsilesional M1 hand knob + premotor cortex", "signature": "perilesional CBF asymmetry; preserved low-frequency sensorimotor coupling", "vascular_marker": "ASL-CBF / MRA patency", "gate_threshold": 0.42},
        "prime": {"regions": "ipsilesional M1-S1 border + dorsal premotor cortex", "signature": "motor-network hypoperfusion with preserved corticospinal reserve", "vascular_marker": "ASL-CBF recovery slope", "gate_threshold": 0.48},
        "potentiate": {"regions": "contralesional M1 inhibitory network + ipsilesional SMA", "signature": "interhemispheric imbalance with increasing peri-infarct perfusion", "vascular_marker": "BOLD cerebrovascular reactivity", "gate_threshold": 0.55},
        "consolidate": {"regions": "ipsilesional M1, S1 and supplementary motor area", "signature": "stable perfusion and strengthening sensorimotor connectivity", "vascular_marker": "CBF/connectivity concordance", "gate_threshold": 0.62},
        "maintain": {"regions": "individualized ipsilesional M1 hotspot", "signature": "plateaued recovery with stable vascular reserve", "vascular_marker": "longitudinal ASL-CBF stability", "gate_threshold": 0.70},
    }

    trajectories = {"usual_care": [], "fixed_rtms": [], "adaptive_rtms": []}
    distributions = {"week": [], "adaptive_mean": [], "adaptive_lower": [], "adaptive_upper": []}
    states = []
    for week, stage in zip(weeks, stages):
        stage_gain = {"stabilize": 0.10, "prime": 0.42, "potentiate": 0.55, "consolidate": 0.24, "maintain": 0.09}[stage]
        fixed_mean = baseline + 24.0 * (1.0 - math.exp(-0.045 * week))
        adaptive_mean = baseline + (29.0 * gain / 1.35) * (1.0 - math.exp(-0.075 * week))
        adaptive_mean += 1.8 * stage_gain * math.sin(math.pi * week / 6.0) * math.exp(-0.025 * week)
        usual = baseline + 0.05 * week
        fixed = fixed_mean + rng.normal(0.0, 1.7 * math.exp(-0.018 * week))
        adaptive = adaptive_mean + rng.normal(0.0, 1.0 * math.exp(-0.022 * week))
        trajectories["usual_care"].append(round(float(np.clip(usual, 0, 66)), 2))
        trajectories["fixed_rtms"].append(round(float(np.clip(fixed, 0, 66)), 2))
        trajectories["adaptive_rtms"].append(round(float(np.clip(adaptive, 0, 66)), 2))

        spread = 2.2 * math.exp(-0.018 * week) + 0.6
        distributions["week"].append(week)
        distributions["adaptive_mean"].append(round(float(adaptive_mean), 2))
        distributions["adaptive_lower"].append(round(float(adaptive_mean - 1.96 * spread), 2))
        distributions["adaptive_upper"].append(round(float(adaptive_mean + 1.96 * spread), 2))
        states.append(int(round((adaptive_mean / 66.0) * (modulus - 1))) % modulus)

    stage_protocol = []
    for name in stage_names:
        protocol = protocols[name].copy()
        protocol["stage"] = name
        protocol["duration_weeks"] = stage_lengths[stage_names.index(name)]
        protocol["sync_fraction"] = cf["convergents"][min(stage_names.index(name), len(cf["convergents"]) - 1)]["fraction"]
        protocol.update(neurovascular_signatures[name])
        protocol["elliptic_modulus"] = round(moduli["m"] + 0.04 * stage_names.index(name), 5)
        protocol["elliptic_gate"] = round(min(0.98, moduli["gate_score"] + 0.05 * stage_names.index(name)), 5)
        stage_protocol.append(protocol)

    trajectory_checkpoints = [
        {"week": week, "stage": stages[week], "region": next(row["regions"] for row in stage_protocol if row["stage"] == stages[week]), "vascular_gate": next(row["elliptic_gate"] for row in stage_protocol if row["stage"] == stages[week]), "adaptive_score": trajectories["adaptive_rtms"][week]}
        for week in [0, min(2, len(weeks) - 1), min(6, len(weeks) - 1), min(12, len(weeks) - 1), len(weeks) - 1]
    ]

    return {
        "status": "research_simulation",
        "disclaimer": "Model outputs are hypothesis-generating and not a clinical treatment plan.",
        "weeks": weeks,
        "trajectories": trajectories,
        "distribution": distributions,
        "stages": stages,
        "stage_protocol": stage_protocol,
        "neurovascular_signatures": neurovascular_signatures,
        "elliptic_moduli": moduli,
        "trajectory_checkpoints": trajectory_checkpoints,
        "finite_states": {"galois_modulus": modulus, "stage_state": states},
        "continued_fraction": cf,
        "kpis": {
            "baseline_fugl_meyer": baseline,
            "final_adaptive_fugl_meyer": trajectories["adaptive_rtms"][-1],
            "gain_vs_usual_care": round(trajectories["adaptive_rtms"][-1] - trajectories["usual_care"][-1], 2),
            "cf_error": round(abs(float(target_sync_ratio) - cf["convergents"][-1]["value"]), 8),
            "stages": len(stage_names),
            "elliptic_gate_score": moduli["gate_score"],
        },
        "math_equations": [
            {"title": "Distributional trajectory", "latex": "Y_t \\sim \\mathcal{N}(\\mu_t,\\sigma_t^2),\\quad \\mu_t=B+(A-B)(1-e^{-kt})"},
            {"title": "Finite stage transition", "latex": "s_{t+1}=(s_t+\\lfloor \\gamma r_t\\rfloor)\\bmod p,\\quad s_t\\in\\mathbb{F}_p"},
            {"title": "Continued-fraction synchronization", "latex": "\\rho=[a_0;a_1,\\ldots,a_n]\\approx p_n/q_n,\\quad p_nq_{n-1}-p_{n-1}q_n=(-1)^{n-1}"},
        ],
    }