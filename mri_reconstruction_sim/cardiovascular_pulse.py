"""
cardiovascular_pulse.py
Cardiovascular MR Pulse Sequence Engine
────────────────────────────────────────
Implements:
  1. Pulse sequence generation  — bSSFP, FLASH, SSFP, SE, IR-bSSFP, VENC phase-contrast
  2. Statistical optimization   — Bayesian / SGD convergence of TR/TE/FA
  3. Continued fractions        — rational approximation of optimal FA/TR ratios
  4. Classical optimization     — Gradient Descent, Nelder-Mead, BFGS
  5. SNR / contrast models      — Ernst angle, T1/T2 steady-state signal equations
  6. Cardiac-gating simulation  — heart-rate variability effect on k-space
"""

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────
# 1. Tissue parameters (T1, T2, PD) at 3T for cardiac structures
# ─────────────────────────────────────────────────────────────────
CARDIAC_TISSUES = {
    "Myocardium":       {"T1": 1471, "T2":  47, "PD": 0.80},
    "Blood (LV)":       {"T1": 1932, "T2": 275, "PD": 1.00},
    "Pericardial Fat":  {"T1":  370, "T2":  90, "PD": 0.90},
    "Aortic Wall":      {"T1": 1200, "T2":  42, "PD": 0.75},
    "Coronary Vessel":  {"T1": 1850, "T2": 250, "PD": 0.98},
    "Liver (ref)":      {"T1":  812, "T2":  42, "PD": 0.71},
}

# ─────────────────────────────────────────────────────────────────
# 2. Signal equations
# ─────────────────────────────────────────────────────────────────

def ernst_angle(T1, TR):
    """Optimal flip angle for spoiled GRE: cos(α*) = exp(-TR/T1)."""
    ratio = np.exp(-TR / T1)
    return float(np.degrees(np.arccos(np.clip(ratio, -1, 1))))


def signal_spgr(FA_deg, TR, TE, T1, T2s, PD=1.0):
    """Spoiled GRE (FLASH/SPGR) signal."""
    a  = np.radians(FA_deg)
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TE / T2s)
    S  = PD * np.sin(a) * (1 - E1) / (1 - np.cos(a) * E1) * E2
    return float(np.clip(S, 0, 1))


def signal_bssfp(FA_deg, TR, TE, T1, T2, PD=1.0):
    """
    Balanced SSFP signal (on-resonance).
    S = PD · sin(α) · (1 - E1) / (1 - (E1-E2)·cos(α) - E1·E2)  · E2^(TE/TR)
    """
    a  = np.radians(FA_deg)
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TR / T2)
    Et = np.exp(-TE / T2)
    denom = 1 - (E1 - E2) * np.cos(a) - E1 * E2
    if abs(denom) < 1e-12:
        return 0.0
    S = PD * np.sin(a) * (1 - E1) / denom * Et
    return float(np.clip(S, 0, 1))


def signal_ir_bssfp(FA_deg, TR, TI, T1, T2, PD=1.0):
    """Inversion-Recovery bSSFP — T1 weighting for myocardium nulling."""
    a  = np.radians(FA_deg)
    E1 = np.exp(-TR / T1)
    Ei = np.exp(-TI / T1)
    E2 = np.exp(-TR / T2)
    denom = 1 - (E1 - E2) * np.cos(a) - E1 * E2
    if abs(denom) < 1e-12:
        return 0.0
    Mz_post_inv = PD * (1 - 2 * Ei)
    S = abs(Mz_post_inv * np.sin(a) * (1 - E1) / denom)
    return float(np.clip(S, 0, 1))


def contrast_ratio(S_tissue, S_blood):
    """CNR-proxy: |S_tissue - S_blood| / (S_tissue + S_blood + 1e-9)."""
    return float(abs(S_tissue - S_blood) / (S_tissue + S_blood + 1e-9))


# ─────────────────────────────────────────────────────────────────
# 3. Pulse sequence catalogue
# ─────────────────────────────────────────────────────────────────

SEQUENCES = {
    "bSSFP": {
        "full_name": "Balanced Steady-State Free Precession",
        "use":       "Cardiac cine, bright-blood morphology",
        "default":   {"FA": 60, "TR": 3.5,  "TE": 1.75, "TI": None, "VENC": None},
        "fa_range":  [30, 90],
        "tr_range":  [2.5, 8.0],
    },
    "SPGR": {
        "full_name": "Spoiled Gradient Echo (FLASH)",
        "use":       "Perfusion, first-pass contrast",
        "default":   {"FA": 15, "TR": 5.0,  "TE": 2.5,  "TI": None, "VENC": None},
        "fa_range":  [5, 40],
        "tr_range":  [3.0, 12.0],
    },
    "IR-bSSFP": {
        "full_name": "Inversion-Recovery bSSFP (MOLLI / ShMOLLI)",
        "use":       "Myocardial T1 mapping, LGE",
        "default":   {"FA": 35, "TR": 3.5,  "TE": 1.75, "TI": 280, "VENC": None},
        "fa_range":  [20, 50],
        "tr_range":  [2.5, 6.0],
    },
    "PC-MRA": {
        "full_name": "Phase-Contrast MR Angiography",
        "use":       "Flow quantification, VENC",
        "default":   {"FA": 20, "TR": 6.0,  "TE": 3.0,  "TI": None, "VENC": 150},
        "fa_range":  [10, 35],
        "tr_range":  [4.0, 10.0],
    },
    "T2-STAR": {
        "full_name": "T2* GRE (Multi-Echo)",
        "use":       "Iron overload, haemorrhage, myocardial iron",
        "default":   {"FA": 20, "TR": 200,   "TE": 2.5,  "TI": None, "VENC": None},
        "fa_range":  [10, 30],
        "tr_range":  [100, 500],
    },
}


# ─────────────────────────────────────────────────────────────────
# 4. Continued fraction optimizer — FA and TR rational approximants
# ─────────────────────────────────────────────────────────────────

def continued_fraction_approx(target, depth=10):
    """
    Expand `target` as a continued fraction:
      target = a0 + 1/(a1 + 1/(a2 + …))
    Return the convergents p_k/q_k as rational candidates.
    """
    x = float(target)
    p, q = [1, 0], [0, 1]
    convergents = []
    for k in range(depth):
        a = int(x)
        p_k = a * p[-1] + p[-2]
        q_k = a * q[-1] + q[-2]
        approx = p_k / q_k if q_k != 0 else float('inf')
        err    = abs(approx - target) / (abs(target) + 1e-12) * 100
        convergents.append({
            "k": k, "a_k": a,
            "numerator": int(p_k), "denominator": int(q_k),
            "approx": round(float(approx), 6),
            "error_pct": round(float(err), 6)
        })
        p.append(p_k); q.append(q_k)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac
    return convergents


# ─────────────────────────────────────────────────────────────────
# 5. Statistical optimization  — Bayesian-SGD parameter refinement
# ─────────────────────────────────────────────────────────────────

def statistical_optimize(seq_name, tissue_a="Myocardium", tissue_b="Blood (LV)",
                         n_iter=30, lr=0.12, noise_std=0.03):
    """
    Stochastic gradient ascent on CNR for the chosen sequence type.
    Optimises FA and TR jointly.

    Update rule:
      θ_{k+1} = θ_k + α · ∇_θ CNR(θ_k) + η_k,   η_k ~ N(0, σ²I)
    """
    seq   = SEQUENCES[seq_name]
    fa    = float(seq["default"]["FA"])
    TR    = float(seq["default"]["TR"])
    TE    = float(seq["default"]["TE"])
    TI    = seq["default"]["TI"]
    ta    = CARDIAC_TISSUES[tissue_a]
    tb    = CARDIAC_TISSUES[tissue_b]

    fa_lo, fa_hi = seq["fa_range"]
    tr_lo, tr_hi = seq["tr_range"]

    trajectory = []

    for i in range(n_iter):
        # Perturb and evaluate numerical gradient
        dfa, dtr = 0.5, 0.05
        if seq_name == "IR-bSSFP":
            Sa  = signal_ir_bssfp(fa + dfa, TR, TI, ta["T1"], ta["T2"], ta["PD"])
            Sa_ = signal_ir_bssfp(fa - dfa, TR, TI, ta["T1"], ta["T2"], ta["PD"])
            Sb  = signal_ir_bssfp(fa,        TR, TI, tb["T1"], tb["T2"], tb["PD"])
            Sb2 = signal_ir_bssfp(fa,     TR+dtr,TI, tb["T1"], tb["T2"], tb["PD"])
            Sb3 = signal_ir_bssfp(fa,     TR-dtr,TI, tb["T1"], tb["T2"], tb["PD"])
        elif seq_name in ("SPGR", "T2-STAR"):
            Sa  = signal_spgr(fa + dfa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
            Sa_ = signal_spgr(fa - dfa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
            Sb  = signal_spgr(fa, TR, TE, tb["T1"], tb["T2"], tb["PD"])
            Sb2 = signal_spgr(fa, TR+dtr, TE, tb["T1"], tb["T2"], tb["PD"])
            Sb3 = signal_spgr(fa, TR-dtr, TE, tb["T1"], tb["T2"], tb["PD"])
        else:  # bSSFP, PC-MRA
            Sa  = signal_bssfp(fa + dfa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
            Sa_ = signal_bssfp(fa - dfa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
            Sb  = signal_bssfp(fa, TR, TE, tb["T1"], tb["T2"], tb["PD"])
            Sb2 = signal_bssfp(fa, TR+dtr, TE, tb["T1"], tb["T2"], tb["PD"])
            Sb3 = signal_bssfp(fa, TR-dtr, TE, tb["T1"], tb["T2"], tb["PD"])

        cnr_now = contrast_ratio(
            signal_bssfp(fa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
            if seq_name in ("bSSFP","PC-MRA","IR-bSSFP")
            else signal_spgr(fa, TR, TE, ta["T1"], ta["T2"], ta["PD"]),
            Sb
        )

        grad_fa = (contrast_ratio(Sa, Sb) - contrast_ratio(Sa_, Sb)) / (2 * dfa)
        grad_tr = (contrast_ratio(Sa, Sb2) - contrast_ratio(Sa, Sb3)) / (2 * dtr)

        # SGD step + noise
        fa += lr * grad_fa + np.random.normal(0, noise_std)
        TR += lr * grad_tr + np.random.normal(0, noise_std * 0.1)

        fa = float(np.clip(fa, fa_lo, fa_hi))
        TR = float(np.clip(TR, tr_lo, tr_hi))

        trajectory.append({
            "iteration": i + 1,
            "FA":  round(fa, 3),
            "TR":  round(TR, 4),
            "CNR": round(cnr_now, 5)
        })

    return {
        "final_FA":  round(fa, 2),
        "final_TR":  round(TR, 3),
        "ernst_angle": round(ernst_angle(ta["T1"], TR), 2),
        "trajectory": trajectory
    }


# ─────────────────────────────────────────────────────────────────
# 6. Classical optimizers — Gradient Descent, Nelder-Mead, BFGS
# ─────────────────────────────────────────────────────────────────

def _cnr_objective(params, seq_name, tissue_a, tissue_b):
    """Negative CNR for minimisation (we want to maximise CNR)."""
    fa, TR = params
    seq  = SEQUENCES[seq_name]
    TE   = seq["default"]["TE"]
    TI   = seq["default"]["TI"]
    ta   = CARDIAC_TISSUES[tissue_a]
    tb   = CARDIAC_TISSUES[tissue_b]

    if fa <= 0 or TR <= 0:
        return 1.0

    if seq_name == "IR-bSSFP":
        Sa = signal_ir_bssfp(fa, TR, TI, ta["T1"], ta["T2"], ta["PD"])
        Sb = signal_ir_bssfp(fa, TR, TI, tb["T1"], tb["T2"], tb["PD"])
    elif seq_name in ("SPGR", "T2-STAR"):
        Sa = signal_spgr(fa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
        Sb = signal_spgr(fa, TR, TE, tb["T1"], tb["T2"], tb["PD"])
    else:
        Sa = signal_bssfp(fa, TR, TE, ta["T1"], ta["T2"], ta["PD"])
        Sb = signal_bssfp(fa, TR, TE, tb["T1"], tb["T2"], tb["PD"])

    return -contrast_ratio(Sa, Sb)


def classical_optimize(seq_name, tissue_a="Myocardium", tissue_b="Blood (LV)"):
    """
    Run three classical optimizers and return their results + convergence.

    Methods:
      1. Gradient Descent  (manual finite-difference)
      2. Nelder-Mead       (scipy.optimize.minimize)
      3. L-BFGS-B          (scipy.optimize.minimize with bounds)
    """
    seq  = SEQUENCES[seq_name]
    x0   = [float(seq["default"]["FA"]), float(seq["default"]["TR"])]
    fa_lo, fa_hi = seq["fa_range"]
    tr_lo, tr_hi = seq["tr_range"]
    bounds = [(fa_lo, fa_hi), (tr_lo, tr_hi)]

    results = {}

    # ── 1. Manual Gradient Descent ───────────────────────────────
    fa, TR   = x0
    lr       = 0.5
    gd_traj  = []
    for step in range(40):
        dfa, dtr = 0.5, 0.05
        g_fa = (_cnr_objective([fa - dfa, TR], seq_name, tissue_a, tissue_b) -
                _cnr_objective([fa + dfa, TR], seq_name, tissue_a, tissue_b)) / (2 * dfa)
        g_tr = (_cnr_objective([fa, TR - dtr], seq_name, tissue_a, tissue_b) -
                _cnr_objective([fa, TR + dtr], seq_name, tissue_a, tissue_b)) / (2 * dtr)
        fa   = float(np.clip(fa + lr * g_fa, fa_lo, fa_hi))
        TR   = float(np.clip(TR + lr * g_tr, tr_lo, tr_hi))
        cnr  = -_cnr_objective([fa, TR], seq_name, tissue_a, tissue_b)
        gd_traj.append({"step": step + 1, "FA": round(fa, 3), "TR": round(TR, 4), "CNR": round(cnr, 5)})
        lr  *= 0.97  # decay

    results["gradient_descent"] = {
        "final_FA":  round(fa, 2),
        "final_TR":  round(TR, 3),
        "final_CNR": round(-_cnr_objective([fa, TR], seq_name, tissue_a, tissue_b), 5),
        "trajectory": gd_traj
    }

    # ── 2. Nelder-Mead ───────────────────────────────────────────
    nm_traj = []
    def nm_callback(xk):
        cnr = -_cnr_objective(xk, seq_name, tissue_a, tissue_b)
        nm_traj.append({"step": len(nm_traj) + 1, "FA": round(float(xk[0]), 3),
                        "TR": round(float(xk[1]), 4), "CNR": round(cnr, 5)})

    res_nm = minimize(_cnr_objective, x0, args=(seq_name, tissue_a, tissue_b),
                      method='Nelder-Mead', callback=nm_callback,
                      options={'maxiter': 60, 'xatol': 1e-4, 'fatol': 1e-5})
    results["nelder_mead"] = {
        "final_FA":  round(float(res_nm.x[0]), 2),
        "final_TR":  round(float(res_nm.x[1]), 3),
        "final_CNR": round(float(-res_nm.fun), 5),
        "n_iters":   int(res_nm.nit),
        "trajectory": nm_traj
    }

    # ── 3. L-BFGS-B ─────────────────────────────────────────────
    bfgs_traj = []
    def bfgs_callback(xk):
        cnr = -_cnr_objective(xk, seq_name, tissue_a, tissue_b)
        bfgs_traj.append({"step": len(bfgs_traj) + 1, "FA": round(float(xk[0]), 3),
                          "TR": round(float(xk[1]), 4), "CNR": round(cnr, 5)})

    res_bfgs = minimize(_cnr_objective, x0, args=(seq_name, tissue_a, tissue_b),
                        method='L-BFGS-B', bounds=bounds, callback=bfgs_callback,
                        options={'maxiter': 60, 'ftol': 1e-9})
    results["lbfgsb"] = {
        "final_FA":  round(float(res_bfgs.x[0]), 2),
        "final_TR":  round(float(res_bfgs.x[1]), 3),
        "final_CNR": round(float(-res_bfgs.fun), 5),
        "n_iters":   int(res_bfgs.nit),
        "trajectory": bfgs_traj
    }

    return results


# ─────────────────────────────────────────────────────────────────
# 7. Cardiac gating simulation
# ─────────────────────────────────────────────────────────────────

def cardiac_gating_sim(HR_bpm=70, scan_duration_s=15, TR_ms=3.5, n_phases=20):
    """
    Simulate k-space line acquisition gated to cardiac phase.
    Returns per-beat phase timing and k-space fill fraction.
    """
    RR_ms   = 60000.0 / HR_bpm
    n_beats = int(scan_duration_s * 1000 / RR_ms)
    lines_per_beat = int(RR_ms / TR_ms)

    beats = []
    total_lines = 0
    for b in range(n_beats):
        # Simulate mild HRV
        rr_actual = RR_ms * (1 + np.random.normal(0, 0.02))
        lines      = int(rr_actual / TR_ms)
        total_lines += lines
        # Assign each line to a cardiac phase bin
        phases = np.linspace(0, 100, lines)
        beats.append({
            "beat":       b + 1,
            "RR_ms":      round(float(rr_actual), 1),
            "lines_acq":  lines,
            "phases_pct": [round(float(p), 1) for p in phases]
        })

    required_lines = n_phases * 96  # 96 phase-encodes per phase (typical)
    fill_fraction  = min(1.0, total_lines / required_lines)

    return {
        "HR_bpm":          HR_bpm,
        "n_beats":         n_beats,
        "total_lines":     total_lines,
        "k_fill_fraction": round(float(fill_fraction), 4),
        "beats":           beats[:12],  # return first 12 beats only (payload limit)
        "phase_summary": {
            "n_phases":       n_phases,
            "lines_per_beat": lines_per_beat,
            "TR_ms":          TR_ms
        }
    }


# ─────────────────────────────────────────────────────────────────
# 8. Main API call — full cardiovascular pulse sequence analysis
# ─────────────────────────────────────────────────────────────────

def run_cardiovascular_analysis(seq_name="bSSFP",
                                tissue_a="Myocardium",
                                tissue_b="Blood (LV)",
                                HR_bpm=70):
    """
    Master function called by the Flask route.
    Returns all analysis data in a single JSON-serialisable dict.
    """
    seq = SEQUENCES.get(seq_name, SEQUENCES["bSSFP"])

    # Compute signal curves over FA sweep
    fa_sweep = np.linspace(5, 90, 80)
    TE       = seq["default"]["TE"]
    TR_def   = seq["default"]["TR"]
    TI       = seq["default"]["TI"]
    ta       = CARDIAC_TISSUES[tissue_a]
    tb       = CARDIAC_TISSUES[tissue_b]

    sig_a, sig_b, cnr_curve = [], [], []
    for fa in fa_sweep:
        if seq_name == "IR-bSSFP":
            Sa = signal_ir_bssfp(fa, TR_def, TI, ta["T1"], ta["T2"], ta["PD"])
            Sb = signal_ir_bssfp(fa, TR_def, TI, tb["T1"], tb["T2"], tb["PD"])
        elif seq_name in ("SPGR", "T2-STAR"):
            Sa = signal_spgr(fa, TR_def, TE, ta["T1"], ta["T2"], ta["PD"])
            Sb = signal_spgr(fa, TR_def, TE, tb["T1"], tb["T2"], tb["PD"])
        else:
            Sa = signal_bssfp(fa, TR_def, TE, ta["T1"], ta["T2"], ta["PD"])
            Sb = signal_bssfp(fa, TR_def, TE, tb["T1"], tb["T2"], tb["PD"])
        sig_a.append(round(float(Sa), 5))
        sig_b.append(round(float(Sb), 5))
        cnr_curve.append(round(float(contrast_ratio(Sa, Sb)), 5))

    # Continued fraction approx of optimal FA
    optimal_fa = fa_sweep[int(np.argmax(cnr_curve))]
    cf_fa      = continued_fraction_approx(optimal_fa, depth=8)

    # Continued fraction approx of TR/TE ratio
    cf_tr_te   = continued_fraction_approx(TR_def / TE, depth=8)

    # Statistical optimization
    stat_opt = statistical_optimize(seq_name, tissue_a, tissue_b)

    # Classical optimizers
    classical = classical_optimize(seq_name, tissue_a, tissue_b)

    # Cardiac gating
    gating = cardiac_gating_sim(HR_bpm=HR_bpm, TR_ms=TR_def)

    # Ernst angle for tissue A
    ernst = ernst_angle(ta["T1"], TR_def)

    return {
        "sequence":     seq_name,
        "sequence_info":seq,
        "tissues":      {tissue_a: ta, tissue_b: tb},
        "fa_sweep": {
            "angles":   [round(float(f), 2) for f in fa_sweep],
            "signal_a": sig_a,
            "signal_b": sig_b,
            "cnr":      cnr_curve,
            "tissue_a": tissue_a,
            "tissue_b": tissue_b
        },
        "optimal_fa":       round(float(optimal_fa), 2),
        "ernst_angle":      round(ernst, 2),
        "cf_fa":            cf_fa,
        "cf_tr_te":         cf_tr_te,
        "statistical_opt":  stat_opt,
        "classical_opt":    classical,
        "cardiac_gating":   gating,
    }
