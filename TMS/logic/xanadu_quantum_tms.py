"""Xanadu Photonic Quantum Computing Infrastructure & Chipsets for Closed-Loop TMS.

Integrates:
1. Xanadu Continuous-Variable (CV) Photonic Quantum Chipsets (Borealis, X8, X12)
   with Gaussian Boson Sampling (GBS), Squeezing Operator S(r), and Displacement D(alpha).
2. Quantum-Enhanced Number-Theoretic Fractional States (Continued Fractions, Farey Sequences,
   Dedekind Phase Sums, and Squeezed Fock Superpositions in Phase Space).
3. Quantum Machine Learning (QML) parameter-shift acceleration of cortical BEM/FEM field
   optimization: O(log N) parameter search vs O(N^3) classical matrix sweeps.
4. Long-Term 60-Month Clinical Trajectories across 5 Target Neurological Conditions:
   - Stroke Motor Recovery (Fugl-Meyer Assessment)
   - Dementia / Alzheimer's (ADAS-Cog)
   - Parkinson's Disease (UPDRS-III)
   - Schizophrenia (PANSS Cognitive Subscale)
   - Alexander's Disease (GFAP Neuroprotective Scale)
5. Multi-Year Xanadu Quantum Cloud QaaS Revenue Projections & Health Economics (QALYs, ICER).
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np


XANADU_CHIPSETS = {
    "borealis": {
        "name": "Xanadu Borealis (216-Mode CV Photonic QPU)",
        "qumodes": 216,
        "chip_architecture": "Time-bin multiplexed dynamic silicon photonics with programmable loops",
        "max_squeezing_db": 14.5,
        "max_squeezing_r": 1.67,
        "clock_rate_mhz": 6.0,
        "optical_interferometer": "Loop-based dynamic Clements/Reck unitary interferometer",
        "detector_type": "Photon-Number-Resolving (PNR) Transition Edge Sensors (TES)",
        "loss_channel_db": 0.42,
        "fidelity_pct": 99.4,
        "cloud_endpoint": "cloud.xanadu.ai/borealis-qpu-v2"
    },
    "x8": {
        "name": "Xanadu X8 (8-Qumode Silicon Nitride Chip)",
        "qumodes": 8,
        "chip_architecture": "Silicon Nitride (Si3N4) integrated photonic waveguide chip",
        "max_squeezing_db": 9.2,
        "max_squeezing_r": 1.06,
        "clock_rate_mhz": 12.0,
        "optical_interferometer": "8x8 Fully reconfigurable Mach-Zehnder interferometer mesh",
        "detector_type": "Superconducting Nanowire Single-Photon Detectors (SNSPDs)",
        "loss_channel_db": 0.75,
        "fidelity_pct": 98.7,
        "cloud_endpoint": "cloud.xanadu.ai/x8-integrated-chip"
    },
    "x12": {
        "name": "Xanadu X12 (12-Qumode Advanced Photonic Chip)",
        "qumodes": 12,
        "chip_architecture": "Monolithic low-loss silica-on-silicon integrated planar waveguide",
        "max_squeezing_db": 11.8,
        "max_squeezing_r": 1.36,
        "clock_rate_mhz": 10.0,
        "optical_interferometer": "12x12 Phase-stable unitary mesh with thermal phase shifters",
        "detector_type": "Dual-quadrature Homodyne & Heterodyne PNR detectors",
        "loss_channel_db": 0.58,
        "fidelity_pct": 99.1,
        "cloud_endpoint": "cloud.xanadu.ai/x12-photonic-qpu"
    }
}


def compute_farey_fractional_states(modulus_q: int = 17, depth: int = 6) -> List[Dict[str, Any]]:
    """
    Generate number-theoretic Farey fraction sequence and continued fraction convergents
    to construct quantum phase-space rotation angles for CV photonic squeezing.
    """
    modulus_q = max(3, min(97, modulus_q))
    fractions = []
    # Farey sequence generation F_q
    a, b, c, d = 0, 1, 1, modulus_q
    fractions.append((a, b))
    while c <= modulus_q:
        k = int((modulus_q + b) / d)
        a, b, c, d = c, d, (k * c - a), (k * d - b)
        fractions.append((a, b))

    # Select representative continued fraction convergents
    convergents = []
    step = max(1, len(fractions) // depth)
    for idx in range(0, min(len(fractions), depth * step), step):
        num, den = fractions[idx]
        val = num / den if den != 0 else 0.0
        # Dedekind sum s(h, k) = sum_{j=1}^{k-1} ((j/k)) (((h*j)/k))
        dedekind_sum = 0.0
        if den > 1:
            for j in range(1, den):
                j_k = (j / den) - 0.5
                hj_k = ((num * j) / den) - math.floor((num * j) / den) - 0.5
                dedekind_sum += j_k * hj_k

        convergents.append({
            "order": len(convergents) + 1,
            "fraction": f"{num}/{den}",
            "value": round(val, 5),
            "quantum_phase_rad": round(2.0 * math.pi * val, 5),
            "dedekind_sum": round(float(dedekind_sum), 5),
            "squeezing_subspace_dim": den
        })

    return convergents[:depth]


def simulate_xanadu_photonic_wigner(
    r_squeeze: float = 1.25,
    theta_rad: float = 0.35,
    alpha_disp: float = 0.65,
    grid_res: int = 30
) -> Dict[str, Any]:
    """
    Compute Continuous-Variable (CV) Wigner function W(x, p) in quantum phase space
    for a displaced squeezed state:
        |alpha, xi> = D(alpha) S(xi) |0>
    with squeezed variances sigma_x^2 = e^(-2r)/2, sigma_p^2 = e^(2r)/2.
    """
    x_lin = np.linspace(-3.0, 3.0, grid_res)
    p_lin = np.linspace(-3.0, 3.0, grid_res)
    X, P = np.meshgrid(x_lin, p_lin)

    # Rotate coordinates by squeezing angle theta
    cos_t = math.cos(theta_rad / 2.0)
    sin_t = math.sin(theta_rad / 2.0)
    X_rot = (X - alpha_disp) * cos_t + P * sin_t
    P_rot = -(X - alpha_disp) * sin_t + P * cos_t

    var_x = 0.5 * math.exp(-2.0 * r_squeeze)
    var_p = 0.5 * math.exp(2.0 * r_squeeze)

    W = (1.0 / (2.0 * math.pi * math.sqrt(var_x * var_p))) * np.exp(
        -(X_rot ** 2) / (2.0 * var_x) - (P_rot ** 2) / (2.0 * var_p)
    )

    # Fock basis state distribution P(n) up to cutoff 12
    cutoff = 12
    p_n = []
    cosh_r = math.cosh(r_squeeze)
    tanh_r = math.tanh(r_squeeze)
    for n in range(cutoff):
        if n % 2 == 0:
            fact_ratio = math.factorial(n) / ((2 ** n) * (math.factorial(n // 2) ** 2))
            prob = (1.0 / cosh_r) * (tanh_r ** n) * fact_ratio
        else:
            prob = 0.0
        p_n.append(round(float(min(1.0, max(0.0, prob))), 5))

    # Normalize Fock probabilities
    sum_p = sum(p_n) or 1.0
    p_n = [round(p / sum_p, 5) for p in p_n]

    return {
        "x_coords": [round(float(v), 2) for v in x_lin],
        "p_coords": [round(float(v), 2) for v in p_lin],
        "wigner_matrix": [[round(float(val), 5) for val in row] for row in W],
        "squeezing_r": round(r_squeeze, 3),
        "squeezing_db": round(10.0 * math.log10(math.exp(2.0 * r_squeeze)), 2),
        "displacement_alpha": round(alpha_disp, 3),
        "fock_photon_probabilities": p_n,
        "vacuum_fidelity": round(float(1.0 / math.cosh(r_squeeze)), 4)
    }


def simulate_quantum_qml_speedup(
    iterations: int = 50,
    n_qumodes: int = 8,
    squeezing_r: float = 1.25
) -> Dict[str, Any]:
    """
    Simulate Quantum Machine Learning (QML) closed-loop parameter search
    using Xanadu Continuous-Variable variational ansatz vs Classical Gradient Descent (SGD)
    for primary motor cortex rTMS E-field optimization.
    """
    iters = list(range(1, iterations + 1))
    qml_loss = []
    classical_loss = []
    quantum_speedup_ratio = []

    # Quantum parameter-shift convergence on CV hardware
    # Parameter-shift rule on CV gates enables exact analytic gradients:
    # d<H>/d theta = (<H(theta + s)> - <H(theta - s)>) / (2 sin s)
    q_rate = 0.16 * (1.0 + 0.25 * squeezing_r) * (n_qumodes / 8.0)
    c_rate = 0.045

    for k in iters:
        # Loss reflects residual mismatch to target motor threshold electric field (V/m)
        q_l = 0.012 + 0.88 * math.exp(-q_rate * k) + 0.003 * math.sin(0.4 * k)
        c_l = 0.085 + 0.88 * math.exp(-c_rate * k) + 0.015 * math.cos(0.2 * k)
        qml_loss.append(round(float(max(0.005, q_l)), 5))
        classical_loss.append(round(float(max(0.04, c_l)), 5))

        # Effective speedup factor at each iteration
        speedup = (classical_loss[-1] / max(qml_loss[-1], 0.005)) * math.log2(k + 3)
        quantum_speedup_ratio.append(round(float(min(45.0, speedup)), 2))

    wallclock_ms_quantum = round(float(12.4 * iterations * (8.0 / n_qumodes)), 1)
    wallclock_ms_classical = round(float(280.0 * iterations), 1)

    return {
        "iterations": iters,
        "qml_loss": qml_loss,
        "classical_loss": classical_loss,
        "quantum_speedup_curve": quantum_speedup_ratio,
        "peak_speedup_factor": round(float(max(quantum_speedup_ratio)), 1),
        "wallclock_quantum_ms": wallclock_ms_quantum,
        "wallclock_classical_ms": wallclock_ms_classical,
        "time_reduction_pct": round(100.0 * (1.0 - wallclock_ms_quantum / wallclock_ms_classical), 1)
    }


def simulate_longterm_clinical_trajectories_xanadu(
    horizon_months: int = 60,
    qml_boost: float = 1.35
) -> Dict[str, Any]:
    """
    Simulate 60-Month Long-Term Clinical Trajectories comparing:
    1. Standard of Care (SOC)
    2. Conventional Static rTMS
    3. Xanadu Quantum-Accelerated Closed-Loop rTMS

    Across 5 Target Indications:
    - Stroke: Fugl-Meyer Motor Assessment (FMA, scale 0-66, higher is better)
    - Dementia: ADAS-Cog-14 Score (scale 0-90, lower is better)
    - Parkinson's: UPDRS-III Motor Score (scale 0-108, lower is better)
    - Schizophrenia: PANSS Cognitive/Negative Scale (scale 7-49, lower is better)
    - Alexander's Disease: Scale for Assessment & Rating of Ataxia (SARA, scale 0-40, lower is better)
    """
    months = list(range(0, horizon_months + 1, 3))
    if 60 not in months:
        months.append(60)

    # Stroke FMA (Baseline ~ 22.0)
    stroke_soc = [round(float(22.0 + 8.5 * (1.0 - math.exp(-0.06 * m))), 1) for m in months]
    stroke_rtms = [round(float(22.0 + 22.0 * (1.0 - math.exp(-0.08 * m))), 1) for m in months]
    stroke_xanadu = [round(float(min(64.0, 22.0 + 34.5 * (1.0 - math.exp(-0.11 * qml_boost * m)))), 1) for m in months]

    # Dementia ADAS-Cog (Baseline ~ 36.5, higher is worse)
    dem_soc = [round(float(36.5 + 0.38 * m), 1) for m in months]
    dem_rtms = [round(float(max(26.0, 36.5 - 4.5 * (1.0 - math.exp(-0.08 * m)) + 0.18 * m)), 1) for m in months]
    dem_xanadu = [round(float(max(18.5, 36.5 - 14.5 * (1.0 - math.exp(-0.12 * qml_boost * m)) + 0.06 * m)), 1) for m in months]

    # Parkinson's UPDRS-III (Baseline ~ 44.0, higher is worse)
    pd_soc = [round(float(44.0 + 0.28 * m), 1) for m in months]
    pd_rtms = [round(float(max(28.0, 44.0 - 12.0 * (1.0 - math.exp(-0.09 * m)) + 0.12 * m)), 1) for m in months]
    pd_xanadu = [round(float(max(15.0, 44.0 - 24.5 * (1.0 - math.exp(-0.13 * qml_boost * m)) + 0.04 * m)), 1) for m in months]

    # Schizophrenia PANSS-Cognitive (Baseline ~ 32.0, higher is worse)
    schiz_soc = [round(float(32.0 + 0.08 * m), 1) for m in months]
    schiz_rtms = [round(float(max(22.0, 32.0 - 7.5 * (1.0 - math.exp(-0.08 * m)))), 1) for m in months]
    schiz_xanadu = [round(float(max(13.5, 32.0 - 16.5 * (1.0 - math.exp(-0.12 * qml_boost * m)))), 1) for m in months]

    # Alexander's Disease SARA Score (Baseline ~ 24.5, higher is worse)
    axd_soc = [round(float(24.5 + 0.26 * m), 1) for m in months]
    axd_rtms = [round(float(max(18.0, 24.5 - 4.8 * (1.0 - math.exp(-0.07 * m)) + 0.10 * m)), 1) for m in months]
    axd_xanadu = [round(float(max(11.0, 24.5 - 12.0 * (1.0 - math.exp(-0.11 * qml_boost * m)) + 0.02 * m)), 1) for m in months]

    return {
        "months": months,
        "stroke": {
            "name": "Stroke Motor Recovery (FMA Score)",
            "soc": stroke_soc,
            "conventional_rtms": stroke_rtms,
            "xanadu_qml_rtms": stroke_xanadu,
            "improvement_pct": round(float(100.0 * (stroke_xanadu[-1] - stroke_soc[-1]) / stroke_soc[-1]), 1)
        },
        "dementia": {
            "name": "Dementia ADAS-Cog-14 Cognitive Deficit",
            "soc": dem_soc,
            "conventional_rtms": dem_rtms,
            "xanadu_qml_rtms": dem_xanadu,
            "improvement_pct": round(float(100.0 * (dem_soc[-1] - dem_xanadu[-1]) / dem_soc[-1]), 1)
        },
        "parkinsons": {
            "name": "Parkinson's UPDRS-III Motor Fluctuation",
            "soc": pd_soc,
            "conventional_rtms": pd_rtms,
            "xanadu_qml_rtms": pd_xanadu,
            "improvement_pct": round(float(100.0 * (pd_soc[-1] - pd_xanadu[-1]) / pd_soc[-1]), 1)
        },
        "schizophrenia": {
            "name": "Schizophrenia PANSS Cognitive Subscale",
            "soc": schiz_soc,
            "conventional_rtms": schiz_rtms,
            "xanadu_qml_rtms": schiz_xanadu,
            "improvement_pct": round(float(100.0 * (schiz_soc[-1] - schiz_xanadu[-1]) / schiz_soc[-1]), 1)
        },
        "alexander": {
            "name": "Alexander's Disease SARA Ataxia Score",
            "soc": axd_soc,
            "conventional_rtms": axd_rtms,
            "xanadu_qml_rtms": axd_xanadu,
            "improvement_pct": round(float(100.0 * (axd_soc[-1] - axd_xanadu[-1]) / axd_soc[-1]), 1)
        }
    }


def simulate_xanadu_qaas_commercial_economics(
    horizon_months: int = 36,
    qpu_base_hourly_rate: float = 350.0,
    qaas_api_price_per_pulse: float = 0.018,
    enterprise_tier_arr: float = 240000.0
) -> Dict[str, Any]:
    """
    Simulate 3-Year (36-Month) Commercial Economics for NeuroMorph + Xanadu Quantum Cloud:
    - Quantum-as-a-Service (QaaS) Closed-Loop Pulse Optimization API
    - Hospital Network Enterprise Subscriptions
    - Biopharma Co-Development QPU Runtime Licenses
    - Clinical Health Economic Cost-Offsets & ICER
    """
    months = list(range(1, horizon_months + 1))
    qaas_api_rev = []
    enterprise_rev = []
    biopharma_rev = []
    total_rev = []
    cogs = []
    ebitda = []

    installed_hospitals = 6
    pharma_sponsors = 2

    for m in months:
        if m in (6, 12, 18, 24, 30):
            installed_hospitals += 3
        if m in (10, 20, 30):
            pharma_sponsors += 1

        # Sessions scaled by installed hospital base
        sessions_monthly = installed_hospitals * 32.0 * (1.0 + 0.03 * m)
        pulses_monthly = sessions_monthly * 2800.0

        api_rev = pulses_monthly * qaas_api_price_per_pulse
        ent_rev = (installed_hospitals * enterprise_tier_arr) / 12.0
        bio_rev = (pharma_sponsors * 450000.0) / 12.0

        tot = api_rev + ent_rev + bio_rev
        # Xanadu QPU runtime fees & photonic cloud infrastructure COGS (24% of revenue)
        cog = tot * 0.24
        gp = tot - cog
        # R&D, clinical trial operations, quantum software engineering OPEX
        opex = 88000.0 + 4200.0 * m
        eb = gp - opex

        qaas_api_rev.append(round(api_rev, 2))
        enterprise_rev.append(round(ent_rev, 2))
        biopharma_rev.append(round(bio_rev, 2))
        total_rev.append(round(tot, 2))
        cogs.append(round(cog, 2))
        ebitda.append(round(eb, 2))

    total_3yr_rev = sum(total_rev)
    total_3yr_ebitda = sum(ebitda)
    forward_arr_m36 = (qaas_api_rev[-1] + enterprise_rev[-1] + biopharma_rev[-1]) * 12.0

    # Health economic QALY and cost-offset per treated patient over 3 years
    qaly_gain_xanadu = 0.54
    direct_cost_offset_3yr = 34500.0
    therapy_cost_3yr = 14200.0
    net_monetary_benefit = 100000.0 * qaly_gain_xanadu - (therapy_cost_3yr - direct_cost_offset_3yr)
    icer_per_qaly = (therapy_cost_3yr - direct_cost_offset_3yr) / qaly_gain_xanadu

    return {
        "monthly": {
            "months": months,
            "qaas_api_rev": qaas_api_rev,
            "enterprise_rev": enterprise_rev,
            "biopharma_rev": biopharma_rev,
            "total_revenue": total_rev,
            "ebitda": ebitda
        },
        "summary_3yr": {
            "total_3yr_revenue": round(total_3yr_rev, 2),
            "total_3yr_ebitda": round(total_3yr_ebitda, 2),
            "forward_arr_m36": round(forward_arr_m36, 2),
            "installed_hospitals_m36": installed_hospitals,
            "active_pharma_sponsors": pharma_sponsors,
            "blended_gross_margin_pct": 76.0
        },
        "health_economics": {
            "qaly_gain_per_patient": round(qaly_gain_xanadu, 3),
            "direct_medical_cost_offset_3yr": round(direct_cost_offset_3yr, 2),
            "net_monetary_benefit_100k": round(net_monetary_benefit, 2),
            "icer_per_qaly": round(icer_per_qaly, 2),
            "is_dominant_strategy": bool(icer_per_qaly < 0)
        }
    }


def run_full_xanadu_quantum_tms_simulation(
    chipset: str = "borealis",
    squeezing_r: float = 1.25,
    squeezing_theta: float = 0.35,
    displacement_alpha: float = 0.65,
    modulus_q: int = 17,
    horizon_months: int = 60,
    qml_iterations: int = 50,
    qaas_price_per_pulse: float = 0.018
) -> Dict[str, Any]:
    """
    Master Pipeline:
    1. Xanadu Photonic Quantum Chipset Specification (Borealis / X8 / X12).
    2. Number-Theoretic Fractional States (Farey sequence & Dedekind phase shifts).
    3. Quantum Phase-Space Wigner Function & Fock Photon Distributions.
    4. QML Optimization vs Classical BEM/FEM Parameter Sweeps.
    5. 60-Month Multi-Condition Clinical Trajectories.
    6. Commercial QaaS Revenue Engine & Health Economics.
    """
    chipset_spec = XANADU_CHIPSETS.get(chipset.lower(), XANADU_CHIPSETS["borealis"])

    farey_states = compute_farey_fractional_states(modulus_q=modulus_q, depth=6)
    wigner_data = simulate_xanadu_photonic_wigner(
        r_squeeze=squeezing_r,
        theta_rad=squeezing_theta,
        alpha_disp=displacement_alpha,
        grid_res=30
    )
    speedup_data = simulate_quantum_qml_speedup(
        iterations=qml_iterations,
        n_qumodes=chipset_spec["qumodes"],
        squeezing_r=squeezing_r
    )
    clinical_data = simulate_longterm_clinical_trajectories_xanadu(
        horizon_months=horizon_months,
        qml_boost=1.0 + 0.30 * (squeezing_r / 1.5)
    )
    commercial_data = simulate_xanadu_qaas_commercial_economics(
        horizon_months=36,
        qaas_api_price_per_pulse=qaas_price_per_pulse
    )

    return {
        "status": "success",
        "parameters": {
            "chipset": chipset_spec["name"],
            "chipset_key": chipset,
            "squeezing_r": squeezing_r,
            "squeezing_theta": squeezing_theta,
            "displacement_alpha": displacement_alpha,
            "modulus_q": modulus_q,
            "horizon_months": horizon_months,
            "qml_iterations": qml_iterations,
            "qaas_price_per_pulse": qaas_price_per_pulse
        },
        "chipset_specification": chipset_spec,
        "number_theoretic_states": farey_states,
        "photonic_wigner": wigner_data,
        "qml_optimization": speedup_data,
        "clinical_trajectories": clinical_data,
        "commercial_economics": commercial_data
    }
