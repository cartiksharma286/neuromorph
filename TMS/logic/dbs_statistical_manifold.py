import numpy as np

def generate_dbs_treatment_protocol():
    """
    Generates data for:
    1. Statistical Manifold Distribution (Dementia Imaging)
    2. Optimal Stage Gating Trajectory
    3. DBS Stimulation Timeline
    """
    # 1. Statistical Manifold Distribution (Imaging)
    # Using a 2D grid to represent the manifold space M with probability densities
    res = 50
    x = np.linspace(-2, 2, res)
    y = np.linspace(-2, 2, res)
    X, Y = np.meshgrid(x, y)
    
    # Statistical manifold model: Radon-Nikodym density proxy
    # Mixture of Gaussian-like distributions for Tau and Amyloid aggregations
    Z_tau = np.exp(-(X - 1.0)**2 - (Y - 0.5)**2) * 2.5
    Z_amyloid = np.exp(-(X + 0.8)**2 - (Y + 1.2)**2) * 1.8
    Z_background = np.sin(X * 4) * np.cos(Y * 4) * 0.3
    
    Z_density = Z_tau + Z_amyloid + Z_background + 0.5
    Z_density = np.maximum(Z_density, 0)

    manifold_data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": Z_density.tolist()
    }

    # 2. Optimal Stage Gating Trajectory
    # Simulating progression of treatment effect across multiple DBS sessions
    sessions = np.arange(1, 101)
    
    # Stage gating thresholds: T1, T2, T3
    # Patient state evolves dynamically through stages via treatment interventions
    base_decline = -0.05 * sessions # natural dementia decline
    dbs_effect = 0.8 * (1 - np.exp(-0.03 * sessions)) # DBS neuromodulation recovery
    hebbian_boost = 0.2 * np.sin(sessions * 0.2) # Synaptic plasticity waves

    patient_state = 10.0 + base_decline + dbs_effect * 8.0 + hebbian_boost
    
    stage_lines = {
        "Stage 1 (Neuro-protective)": [9.0] * len(sessions),
        "Stage 2 (Neuro-restorative)": [11.0] * len(sessions),
        "Stage 3 (Synaptic Amplification)": [13.0] * len(sessions)
    }

    gating_data = {
        "sessions": sessions.tolist(),
        "patient_state": patient_state.tolist(),
        "stages": stage_lines
    }

    # 3. DBS Treatment Protocol Timeline & Parameters
    # Frequencies (Hz), Intensity (% MSO), Burst Patterns
    target_freqs = 130 + 10 * np.cos(sessions * 0.1) # Variable High Frequency DBS
    intensity = 60 + 20 * (1 - np.exp(-0.05 * sessions)) # Gradual titration
    
    dbs_timeline = {
        "sessions": sessions.tolist(),
        "frequency": target_freqs.tolist(),
        "intensity": intensity.tolist()
    }

    return {
        "manifold": manifold_data,
        "gating": gating_data,
        "dbs_timeline": dbs_timeline
    }
