"""Deterministic computational model for the depression rTMS research tab."""

import math

import numpy as np


def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for value in range(2, int(math.sqrt(limit)) + 1):
        if sieve[value]:
            sieve[value * value:limit + 1:value] = False
    return np.flatnonzero(sieve).tolist()


def _continued_fraction(value, depth=7):
    coefficients = []
    convergents = []
    current = float(value)
    p_prev2, p_prev1 = 0, 1
    q_prev2, q_prev1 = 1, 0
    for _ in range(depth):
        coefficient = math.floor(current)
        coefficients.append(int(coefficient))
        numerator = coefficient * p_prev1 + p_prev2
        denominator = coefficient * q_prev1 + q_prev2
        convergents.append({
            'fraction': f'{numerator}/{denominator}',
            'value': float(numerator / denominator),
            'error': float(abs(value - numerator / denominator)),
        })
        remainder = current - coefficient
        if abs(remainder) < 1e-12:
            break
        current = 1.0 / remainder
        p_prev2, p_prev1 = p_prev1, numerator
        q_prev2, q_prev1 = q_prev1, denominator
    return coefficients, convergents


def _optimal_stage_gates(symptoms, control, cognitive_state):
    """Exhaustively rank finite induction/consolidation gate candidates."""
    horizon = len(symptoms) - 1
    candidates = []
    induction_start = max(4, int(math.ceil(0.25 * horizon)))
    induction_stop = max(induction_start + 1, int(math.floor(0.60 * horizon)))
    for induction_end in range(induction_start, induction_stop + 1):
        consolidation_stop = min(horizon - 2, int(math.floor(0.90 * horizon)))
        for consolidation_end in range(induction_end + 3, consolidation_stop + 1):
            consolidation_slice = np.asarray(symptoms[induction_end:consolidation_end + 1])
            maintenance_control = np.asarray(control[consolidation_end:horizon + 1])
            gate_cost = (
                (symptoms[induction_end] - 10.0) ** 2
                + 0.65 * (symptoms[consolidation_end] - 7.0) ** 2
                + 2.5 * float(np.mean(np.abs(np.diff(consolidation_slice))))
                + 1.8 * float(np.mean(maintenance_control ** 2))
                + 0.8 * (symptoms[-1] - 5.0) ** 2
                + 0.05 * (induction_end - 0.45 * horizon) ** 2
            )
            candidates.append({
                'induction_end': induction_end,
                'consolidation_end': consolidation_end,
                'cost': float(gate_cost),
            })

    candidates.sort(key=lambda item: item['cost'])
    optimal = candidates[0]
    induction_end = optimal['induction_end']
    consolidation_end = optimal['consolidation_end']
    stage_index = []
    for session in range(horizon + 1):
        if session <= induction_end:
            stage_index.append(0)
        elif session <= consolidation_end:
            stage_index.append(1)
        else:
            stage_index.append(2)

    stages = [
        {
            'name': 'Induction', 'start': 0, 'end': induction_end,
            'mean_control': float(np.mean(control[1:induction_end + 1])),
            'end_phq9': float(symptoms[induction_end]),
            'research_goal': 'initial modeled symptom reduction under bounded control',
        },
        {
            'name': 'Consolidation', 'start': induction_end + 1, 'end': consolidation_end,
            'mean_control': float(np.mean(control[induction_end + 1:consolidation_end + 1])),
            'end_phq9': float(symptoms[consolidation_end]),
            'research_goal': 'stabilize trajectory and cognitive-state change',
        },
        {
            'name': 'Maintenance', 'start': consolidation_end + 1, 'end': horizon,
            'mean_control': float(np.mean(control[consolidation_end + 1:horizon + 1])),
            'end_phq9': float(symptoms[-1]),
            'research_goal': 'minimize modeled burden while monitoring terminal error',
        },
    ]
    return {
        'optimal_induction_end': induction_end,
        'optimal_consolidation_end': consolidation_end,
        'optimal_cost': optimal['cost'],
        'stage_index': stage_index,
        'stages': stages,
        'candidate_rank': list(range(1, min(30, len(candidates)) + 1)),
        'candidate_cost': [item['cost'] for item in candidates[:30]],
        'candidate_gates': [f"{item['induction_end']}/{item['consolidation_end']}" for item in candidates[:30]],
        'candidate_count': len(candidates),
    }


def simulate_depression_rtms(
    baseline_phq9=19.0,
    sessions=30,
    rtms_frequency_hz=10.0,
    cbt_weight=0.65,
    control_gain=0.85,
    signature_ratio=1.61803398875,
):
    """Return a seeded in-silico depression trajectory and finite-control telemetry."""
    baseline_phq9 = float(np.clip(baseline_phq9, 5.0, 27.0))
    sessions = int(np.clip(sessions, 10, 40))
    rtms_frequency_hz = float(np.clip(rtms_frequency_hz, 1.0, 20.0))
    cbt_weight = float(np.clip(cbt_weight, 0.0, 1.0))
    control_gain = float(np.clip(control_gain, 0.1, 1.5))
    signature_ratio = float(np.clip(signature_ratio, 0.5, 3.0))

    rng = np.random.RandomState(286)
    session_axis = np.arange(0, sessions + 1, dtype=float)
    target_phq9 = 5.0
    symptom = baseline_phq9
    cognitive_distortion = min(1.0, 0.35 + baseline_phq9 / 36.0)
    adaptive = [symptom]
    distortion = [cognitive_distortion]
    control = [0.0]
    objective = [(symptom - target_phq9) ** 2]
    prime_sessions = set(_primes(sessions))
    signature = [0]

    for session in range(1, sessions + 1):
        error = max(0.0, symptom - target_phq9)
        bounded_control = float(np.clip(control_gain * error / max(baseline_phq9, 1.0), 0.0, 1.0))
        prime_gain = 1.04 if session in prime_sessions else 1.0
        cbt_update = cbt_weight * 0.055 * cognitive_distortion
        cognitive_distortion = max(0.05, cognitive_distortion - cbt_update)
        rtms_rate = 0.052 * (rtms_frequency_hz / 10.0) * (0.45 + 0.55 * bounded_control) * prime_gain
        cbt_rate = 0.032 * cbt_weight * (1.0 - cognitive_distortion)
        symptom = max(2.0, target_phq9 + (symptom - target_phq9) * math.exp(-(rtms_rate + cbt_rate)) + rng.normal(0.0, 0.16))
        adaptive.append(float(symptom))
        distortion.append(float(cognitive_distortion))
        control.append(bounded_control)
        objective.append(float((symptom - target_phq9) ** 2 + 0.25 * bounded_control ** 2))
        signature.append(int((session * session + 3 * session + 7) % 17))

    usual_care = baseline_phq9 - 0.05 * session_axis + rng.normal(0, 0.28, sessions + 1)
    cbt_only = target_phq9 + (baseline_phq9 - target_phq9) * np.exp(-0.038 * cbt_weight * session_axis) + rng.normal(0, 0.20, sessions + 1)
    rtms_only = target_phq9 + (baseline_phq9 - target_phq9) * np.exp(-0.047 * (rtms_frequency_hz / 10.0) * session_axis) + rng.normal(0, 0.18, sessions + 1)
    usual_care = np.maximum(2.0, usual_care)
    cbt_only = np.maximum(2.0, cbt_only)
    rtms_only = np.maximum(2.0, rtms_only)

    cohort = np.clip(rng.normal(baseline_phq9, 4.1, 600), 5.0, 27.0)
    post_cohort = np.clip(cohort - rng.normal(baseline_phq9 - adaptive[-1], 2.2, 600), 0.0, 27.0)
    baseline_counts, bin_edges = np.histogram(cohort, bins=np.linspace(0, 27, 19))
    post_counts, _ = np.histogram(post_cohort, bins=bin_edges)
    coefficients, convergents = _continued_fraction(signature_ratio)

    response_pct = 100.0 * (baseline_phq9 - adaptive[-1]) / baseline_phq9
    staging = _optimal_stage_gates(adaptive, control, distortion)
    paradigm = {
        'status': 'in-silico research scenario; not a clinical prescription',
        'target': 'left dorsolateral prefrontal cortex research model',
        'frequency_hz': rtms_frequency_hz,
        'sessions': sessions,
        'cbt_component': 'behavioral activation, cognitive reappraisal, and relapse-planning state updates',
        'control_rule': 'bounded symptom-error feedback with prime-session modulation',
        'safety': 'requires clinician screening, device labeling, motor-threshold procedures, and adverse-event monitoring',
    }

    return {
        'sessions': session_axis.astype(int).tolist(),
        'phq9_usual_care': usual_care.tolist(),
        'phq9_cbt_only': cbt_only.tolist(),
        'phq9_rtms_only': rtms_only.tolist(),
        'phq9_adaptive_combined': adaptive,
        'cognitive_distortion_state': distortion,
        'control_effort': control,
        'objective': objective,
        'number_signature': signature,
        'prime_sessions': sorted(prime_sessions),
        'continued_fraction': {'coefficients': coefficients, 'convergents': convergents},
        'staging': staging,
        'distribution': {
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist(),
            'baseline_counts': baseline_counts.tolist(),
            'post_counts': post_counts.tolist(),
            'sample_size': int(len(cohort)),
        },
        'metrics': {
            'baseline_phq9': baseline_phq9,
            'final_phq9': float(adaptive[-1]),
            'modeled_response_pct': float(response_pct),
            'final_distortion_state': float(distortion[-1]),
            'mean_control_effort': float(np.mean(control[1:])),
            'remission_threshold_crossed': bool(adaptive[-1] < target_phq9),
        },
        'params': {
            'baseline_phq9': baseline_phq9,
            'sessions': sessions,
            'rtms_frequency_hz': rtms_frequency_hz,
            'cbt_weight': cbt_weight,
            'control_gain': control_gain,
            'signature_ratio': signature_ratio,
        },
        'paradigm': paradigm,
        'limitations': [
            'Synthetic cohort and seeded noise; no participant data are used.',
            'PHQ-9 trajectories are assumed model outputs, not efficacy estimates.',
            'Number-theoretic signatures are scheduling features without established biological meaning.',
            'The model omits anatomy-specific electric fields, medication effects, comorbidity, and adverse events.',
        ],
    }
