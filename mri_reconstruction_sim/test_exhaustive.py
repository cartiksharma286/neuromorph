import numpy as np
from simulator_core import MRIReconstructionSimulator

# Define pulse sequences and coil modes to test
pulse_sequences = ['SE', 'GRE', 'SSFP', 'SpinEcho', 'InversionRecovery']
coil_modes = ['standard', 'quantum_vascular', 'head_coil_50_turn']

# Threshold for white pixel detection (intensity close to 1.0)
WHITE_PIXEL_THRESHOLD = 0.95

failures = []

for seq in pulse_sequences:
    for coil in coil_modes:
        try:
            sim = MRIReconstructionSimulator(resolution=128)
            # Setup phantom (real data if available)
            sim.setup_phantom(use_real_data=True, phantom_type='brain')
            # Generate coil sensitivities for the given coil type
            sim.generate_coil_sensitivities(num_coils=8, coil_type=coil, optimal_shimming=False)
            # Acquire signal
            kspace, M_ref = sim.acquire_signal(sequence_type=seq, TR=2000, TE=100, TI=500, flip_angle=30, noise_level=0.01)
            # Reconstruct image (includes boosting and artifact removal)
            recon_img, _ = sim.reconstruct_image(kspace, method='SoS')
            # Check for white pixels
            max_val = np.max(recon_img)
            if max_val > WHITE_PIXEL_THRESHOLD:
                failures.append({
                    'sequence': seq,
                    'coil': coil,
                    'max_intensity': float(max_val)
                })
                print(f"[FAIL] {seq} with {coil}: max intensity {max_val:.4f} exceeds threshold")
            else:
                print(f"[PASS] {seq} with {coil}: max intensity {max_val:.4f}")
        except Exception as e:
            print(f"[ERROR] {seq} with {coil}: {e}")
            failures.append({'sequence': seq, 'coil': coil, 'error': str(e)})

print('\n=== Summary ===')
if failures:
    print(f"Found {len(failures)} cases with potential white pixel artifacts or errors:")
    for f in failures:
        print(f)
else:
    print('All tested combinations produced noise‑free images with no white pixel artifacts.')
