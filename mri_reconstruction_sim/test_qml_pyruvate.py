import sys
sys.path.append('.')
from statistical_adaptive_pulse import create_adaptive_sequence

seq = create_adaptive_sequence('qml_pyruvate', nvqlink_enabled=False)
tissue_stats = {'mean_intensity': 0.8, 'std_intensity': 0.05}
params = seq.generate_sequence(tissue_stats)
recon = seq.simulate_signal_reconstruction(noise_level=0.03)

print("Sequence:")
print(params)
print("\nRecon:")
print(recon)
