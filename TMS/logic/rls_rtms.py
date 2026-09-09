"""Research-only finite simulation for an RLS rTMS study dashboard."""

import math
import numpy as np


LOBE_PROTOCOL = [
    ("Supplementary motor area", "Medial frontal", 1.0, 0.78, "Figure-8 70 mm"),
    ("Dorsolateral prefrontal", "Frontal", 10.0, 0.66, "Figure-8 70 mm"),
    ("Primary sensorimotor", "Parietal", 1.0, 0.72, "Figure-8 70 mm"),
    ("Temporoparietal junction", "Temporal", 1.0, 0.58, "Cooled figure-8"),
    ("Occipital control region", "Occipital", 1.0, 0.34, "Figure-8 70 mm"),
]


def simulate_rls_rtms(baseline_irls=28.0, treatment_weeks=8, coil_towers=2, session_price=275.0):
    """Return deterministic, synthetic design outputs; this is not a clinical protocol."""
    baseline_irls = float(np.clip(baseline_irls, 10.0, 40.0))
    treatment_weeks = int(np.clip(treatment_weeks, 4, 16))
    coil_towers = int(np.clip(coil_towers, 1, 8))
    session_price = float(np.clip(session_price, 150.0, 650.0))
    rng = np.random.default_rng(20260902)
    weeks = np.arange(0, treatment_weeks + 1)

    # A finite recurrent state update couples each lobe's carry-over to its design weight.
    lobe_rows = []
    recurrent_curves = []
    for index, (target, lobe, frequency, weight, coil) in enumerate(LOBE_PROTOCOL):
        state = 0.0
        curve = []
        for week in weeks:
            if week == 0:
                curve.append(0.0)
                continue
            state = 0.63 * state + weight * (1.0 - math.exp(-0.42 * week))
            curve.append(round(min(1.0, state), 4))
        recurrent_curves.append({"name": target, "values": curve})
        lobe_rows.append({
            "target": target, "lobe": lobe, "frequency_hz": frequency,
            "design_weight": round(weight, 2), "coil": coil,
            "allocation_pct": round(100 * weight / sum(item[3] for item in LOBE_PROTOCOL), 1),
        })

    combined_signal = np.mean([item["values"] for item in recurrent_curves], axis=0)
    symptom_index = baseline_irls * (1.0 - 0.54 * combined_signal)
    symptom_index += rng.normal(0.0, 0.32, len(weeks))
    symptom_index = np.maximum(4.0, symptom_index).round(2)
    recovery_pct = (100.0 * (baseline_irls - symptom_index) / baseline_irls).round(1)

    # Finite candidate set and D-optimal information scores, deliberately not patient probabilities.
    candidates = []
    for frequency in (1, 5, 10, 20):
        for intensity in (80, 90, 100, 110):
            information = math.exp(-((frequency - 10) / 7.0) ** 2 - ((intensity - 100) / 18.0) ** 2)
            candidates.append({"frequency_hz": frequency, "intensity_pct": intensity, "information_score": round(information, 4)})
    best_design = max(candidates, key=lambda item: item["information_score"])

    samples = np.clip(rng.beta(7.5, 4.3, 900) * 100, 0, 100)
    response_likelihood = float(np.mean(samples >= 50) * 100)
    ci_low, ci_high = np.quantile(samples, [0.1, 0.9])

    bem_axis = np.linspace(-1.0, 1.0, 42)
    grid_x, grid_y = np.meshgrid(bem_axis, bem_axis)
    field = np.zeros_like(grid_x)
    centers = [(-0.35, 0.35), (0.28, 0.24), (0.0, -0.1), (0.5, -0.32), (-0.5, -0.32)]
    for (_, _, _, weight, _), (center_x, center_y) in zip(LOBE_PROTOCOL, centers):
        field += weight * np.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / 0.13)
    field = (field / field.max() * 100).round(2)

    courses_per_tower = int(7 * 245 / 24)
    annual_courses = coil_towers * courses_per_tower
    course_price = 24 * session_price
    year_one = annual_courses * course_price
    growth = [1.0, 1.12, 1.25, 1.37, 1.48]
    revenue = [round(year_one * rate, 2) for rate in growth]
    capex = coil_towers * 145000.0
    opex = year_one * 0.38
    discount = 0.08
    npv = -capex + sum((value - opex) / ((1 + discount) ** year) for year, value in enumerate(revenue, 1))
    payback = capex / max((year_one - opex) / 12, 1)
    modeled_savings = round(2650 * response_likelihood / 100, 0)

    equipment = [
        {"item": "Figure-8 TMS system", "role": "superficial motor-network field model", "quantity": coil_towers},
        {"item": "Neuronavigation workstation", "role": "research coordinate reproducibility", "quantity": 1},
        {"item": "EMG / accelerometry kit", "role": "research outcome capture", "quantity": coil_towers},
        {"item": "Cooling and safety monitor", "role": "device workflow assumption", "quantity": coil_towers},
    ]
    return {
        "weeks": weeks.tolist(), "symptom_index": symptom_index.tolist(), "recovery_pct": recovery_pct.tolist(),
        "recurrent_curves": recurrent_curves, "lobe_protocol": lobe_rows,
        "experimental_design": {"candidates": candidates, "best": best_design, "response_samples": samples.round(2).tolist(), "response_likelihood_pct": round(response_likelihood, 1), "interval_10_90": [round(float(ci_low), 1), round(float(ci_high), 1)]},
        "bem_field": {"x": bem_axis.round(3).tolist(), "y": bem_axis.round(3).tolist(), "z": field.tolist(), "peak": float(field.max())},
        "equipment": equipment,
        "economics": {"annual_courses": annual_courses, "course_price": course_price, "modeled_savings_per_course": modeled_savings, "revenue_5yr": revenue, "total_revenue_5yr": round(sum(revenue), 2), "capex": capex, "annual_opex": round(opex, 2), "npv": round(npv, 2), "payback_months": round(payback, 1)},
        "disclaimer": "Research-only synthetic simulation. It is not a clinical protocol, does not estimate cure probability, and must not guide treatment.",
    }
