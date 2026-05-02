import sys
import numpy as np

# Add the current directory so that absolute/relative imports will resolve if needed
sys.path.append('.')

from statistical_adaptive_pulse import create_adaptive_sequence

def run_simulation():
    # 1. Create the sequence
    print("Initializing Hyperpolarized Pulse Sequence...")
    hyper_seq = create_adaptive_sequence('hyperpolarized', nvqlink_enabled=False)
    
    # 2. Mock statistical optimization data (tissue baseline)
    tissue_stats = {
        'mean_intensity': 0.8,
        'std_intensity': 0.05
    }
    
    print("Generating statistical optimal parameters...")
    seq_params = hyper_seq.generate_sequence(tissue_stats)
    print("Optimized Sequence Parameters:")
    for key, value in seq_params.items():
        print(f"  {key}: {value}")
        
    print("\nSimulating signal reconstruction and calculating SNR...")
    
    # 3. Simulate Reconstruction and SNR
    recon_results = hyper_seq.simulate_signal_reconstruction(noise_level=0.03)
    
    print("\nReconstruction Results:")
    print(f"  Reconstructed Signal Amplitude: {recon_results['reconstructed_amplitude']:.2f}")
    print(f"  Estimated SNR: {recon_results['snr_estimate']:.2f}")
    print(f"  Noise Level Used: {recon_results['noise_level']:.2f}")

if __name__ == '__main__':
    run_simulation()
