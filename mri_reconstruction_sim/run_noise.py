import json
import base64
import os
from simulator_core import MRIReconstructionSimulator

sim = MRIReconstructionSimulator(resolution=128)
sim.setup_phantom(use_real_data=True, phantom_type='brain')
sim.generate_coil_sensitivities(num_coils=8, coil_type='quantum_vascular', optimal_shimming=False)
kspace, M_ref = sim.acquire_signal(sequence_type='QuantumGenerativeRecon', TR=2000, TE=100, TI=500, flip_angle=30, noise_level=0.05)
recon_img, coil_imgs = sim.reconstruct_image(kspace, method='SoS', noise_filter='Median', morphological_cleanup=True)

# Generate plots
plots = sim.generate_plots(kspace, recon_img, M_ref)

img_dir = os.path.join(os.getcwd(), 'static', 'report_images')
os.makedirs(img_dir, exist_ok=True)
for key, b64_str in plots.items():
    if b64_str:
        with open(os.path.join(img_dir, f"{key}_noise_filtered.png"), "wb") as f:
            f.write(base64.b64decode(b64_str))

print('SUCCESS')
