import numpy as np


def generate_nash_geodesic_registration():
    """
    Generates data for Multimodal Laser-Scan / MRI / CT registration analysis:
      1. Nash Equilibrium convergence distributions (per-modality utility trajectories)
      2. Laplace-Beltrami eigen spectra (coordinate-invariant shape descriptors)
      3. Cauchy-Schwarz inequality bound tracking during registration optimization
      4. Geodesic mapping point-cloud (Riemannian manifold path between modalities)
    """
    rng = np.random.default_rng(42)

    # ── 1. Nash Equilibrium Distributions ──────────────────────────────
    # Three "players": Laser Scan, MRI, CT transformation utilities converging
    # toward a shared equilibrium utility value under repeated best-response updates.
    iterations = np.arange(1, 61)
    eq_utility = 0.97
    u_laser = eq_utility - 0.55 * np.exp(-0.09 * iterations) + rng.normal(0, 0.008, len(iterations))
    u_mri   = eq_utility - 0.62 * np.exp(-0.085 * iterations) + rng.normal(0, 0.008, len(iterations))
    u_ct    = eq_utility - 0.48 * np.exp(-0.095 * iterations) + rng.normal(0, 0.008, len(iterations))

    # Nash gap: max utility improvement any single player could unilaterally achieve
    nash_gap = 0.6 * np.exp(-0.11 * iterations) + np.abs(rng.normal(0, 0.004, len(iterations)))

    # Equilibrium probability distribution (mixed strategy) over discretized
    # transform-perturbation bins at final converged iteration
    bins = np.linspace(-3, 3, 41)
    nash_density = np.exp(-0.5 * (bins / 0.55) ** 2)
    nash_density /= nash_density.sum()

    nash_data = {
        "iterations": iterations.tolist(),
        "utility_laser": u_laser.tolist(),
        "utility_mri": u_mri.tolist(),
        "utility_ct": u_ct.tolist(),
        "nash_gap": nash_gap.tolist(),
        "equilibrium_bins": bins.tolist(),
        "equilibrium_density": nash_density.tolist(),
    }

    # ── 2. Laplace-Beltrami Eigen Spectra ───────────────────────────────
    # Coordinate-invariant shape descriptors for each modality's surface mesh
    n_modes = 30
    modes = np.arange(1, n_modes + 1)

    def spectrum(scale, jitter_seed):
        rng_local = np.random.default_rng(jitter_seed)
        base = scale * modes ** 1.15
        return (base + rng_local.normal(0, base * 0.03)).tolist()

    eigen_data = {
        "modes": modes.tolist(),
        "eigenvalues_laser": spectrum(0.014, 1),
        "eigenvalues_mri": spectrum(0.0138, 2),
        "eigenvalues_ct": spectrum(0.0142, 3),
    }

    # ── 3. Cauchy-Schwarz Bound Tracking ────────────────────────────────
    # Ratio r = |<f,g>|^2 / (<f,f><g,g>) must remain in [0, 1] by Cauchy-Schwarz.
    # As registration converges, r -> 1 (perfect co-linear alignment in Hilbert space).
    cs_ratio_lasermri = 1.0 - 0.62 * np.exp(-0.10 * iterations) - np.abs(rng.normal(0, 0.003, len(iterations)))
    cs_ratio_mrict    = 1.0 - 0.71 * np.exp(-0.095 * iterations) - np.abs(rng.normal(0, 0.003, len(iterations)))
    cs_ratio_laserct  = 1.0 - 0.66 * np.exp(-0.09 * iterations) - np.abs(rng.normal(0, 0.003, len(iterations)))
    cs_ratio_lasermri = np.clip(cs_ratio_lasermri, 0.0, 1.0)
    cs_ratio_mrict    = np.clip(cs_ratio_mrict, 0.0, 1.0)
    cs_ratio_laserct  = np.clip(cs_ratio_laserct, 0.0, 1.0)

    cauchy_schwarz_data = {
        "iterations": iterations.tolist(),
        "ratio_laser_mri": cs_ratio_lasermri.tolist(),
        "ratio_mri_ct": cs_ratio_mrict.tolist(),
        "ratio_laser_ct": cs_ratio_laserct.tolist(),
        "upper_bound": [1.0] * len(iterations),
    }

    # ── 4. Geodesic Mapping Point Cloud ─────────────────────────────────
    # Geodesic path on a synthetic Riemannian cortical-surface manifold connecting
    # aligned landmark points across Laser / MRI / CT coordinate frames.
    t = np.linspace(0, 1, 60)
    # Parametrize a smooth curved geodesic path with 3 twisting components
    gx = 3.0 * np.sin(t * np.pi) * np.cos(2.4 * np.pi * t)
    gy = 3.0 * np.sin(t * np.pi) * np.sin(2.4 * np.pi * t)
    gz = 4.0 * t

    landmark_idx = np.linspace(0, len(t) - 1, 8).astype(int)
    geodesic_data = {
        "x": gx.tolist(),
        "y": gy.tolist(),
        "z": gz.tolist(),
        "landmark_x": gx[landmark_idx].tolist(),
        "landmark_y": gy[landmark_idx].tolist(),
        "landmark_z": gz[landmark_idx].tolist(),
        "max_geodesic_error_mm": 0.22,
        "mean_geodesic_error_mm": 0.14,
    }

    # ── 5. Longitudinal Registration Tracking ───────────────────────────
    # Repeated clinical registration sessions over a 12-month follow-up
    # pathway, tracking drift/re-registration accuracy across visits.
    n_sessions = 12
    session_idx = np.arange(1, n_sessions + 1)
    rng_lt = np.random.default_rng(7)

    # Registration error tends to stay low with periodic re-calibration drift
    recalibration_bump = np.where(session_idx % 4 == 0, 0.10, 0.0)
    mean_error_mm = 0.14 + 0.03 * np.sin(session_idx * 0.5) + recalibration_bump + np.abs(rng_lt.normal(0, 0.015, n_sessions))
    max_error_mm  = mean_error_mm + 0.10 + np.abs(rng_lt.normal(0, 0.02, n_sessions))

    # Final Nash gap and Cauchy-Schwarz ratio achieved at convergence, per session
    nash_gap_final = 0.02 + 0.015 * np.sin(session_idx * 0.4) + np.abs(rng_lt.normal(0, 0.004, n_sessions))
    cs_ratio_final = 1.0 - (0.03 + 0.01 * np.cos(session_idx * 0.3) + np.abs(rng_lt.normal(0, 0.003, n_sessions)))
    cs_ratio_final = np.clip(cs_ratio_final, 0.0, 1.0)

    longitudinal_data = {
        "sessions": session_idx.tolist(),
        "mean_error_mm": mean_error_mm.tolist(),
        "max_error_mm": max_error_mm.tolist(),
        "clinical_tolerance_mm": [1.0] * n_sessions,
        "nash_gap_final": nash_gap_final.tolist(),
        "cauchy_schwarz_final": cs_ratio_final.tolist(),
    }

    return {
        "nash": nash_data,
        "eigen_spectra": eigen_data,
        "cauchy_schwarz": cauchy_schwarz_data,
        "geodesic": geodesic_data,
        "longitudinal": longitudinal_data,
    }
