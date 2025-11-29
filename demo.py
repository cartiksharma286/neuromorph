"""
Demo script for Statistical Learning with SNR Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from snr_optimizer import SNROptimizer, AdaptiveSNRLearner


def generate_test_signal(n_samples: int = 1000) -> tuple:
    """Generate a test signal with noise."""
    t = np.linspace(0, 4 * np.pi, n_samples)
    
    # Create clean signal: combination of sinusoids
    signal = (np.sin(t) + 
             0.5 * np.sin(3 * t) + 
             0.3 * np.sin(7 * t))
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.5, n_samples)
    
    return signal, noise, t


def demo_single_distribution():
    """Demonstrate SNR optimization with a single distribution."""
    print("=" * 60)
    print("Demo 1: Single Distribution SNR Optimization (Gaussian)")
    print("=" * 60)
    
    # Generate test data
    signal, noise, t = generate_test_signal()
    noisy_signal = signal + noise
    
    # Compute initial SNR
    optimizer = SNROptimizer(distribution_type='gaussian')
    initial_snr = optimizer.compute_snr(signal, noise)
    print(f"\nInitial SNR: {initial_snr:.2f} dB")
    
    # Learn optimal distribution
    print("\nLearning optimal distribution parameters...")
    params = optimizer.learn_optimal_distribution(signal, noise, iterations=10)
    
    print(f"\nLearned parameters:")
    print(f"  μ = {params['mu']:.4f}")
    print(f"  σ = {params['sigma']:.4f}")
    
    # Denoise and compute final SNR
    denoised = optimizer._denoise_signal(noisy_signal, params)
    estimated_noise = noisy_signal - denoised
    final_snr = optimizer.compute_snr(signal, estimated_noise)
    
    print(f"\nFinal SNR after optimization: {final_snr:.2f} dB")
    print(f"SNR improvement: {final_snr - initial_snr:.2f} dB")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(t, signal, 'g-', label='Clean Signal', linewidth=2)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Clean Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, noisy_signal, 'r-', alpha=0.7, label='Noisy Signal')
    axes[1].plot(t, signal, 'g--', alpha=0.5, label='Clean Signal')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Noisy Signal (SNR: {initial_snr:.2f} dB)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, denoised, 'b-', label='Denoised Signal', linewidth=2)
    axes[2].plot(t, signal, 'g--', alpha=0.5, label='Clean Signal')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Denoised Signal (SNR: {final_snr:.2f} dB)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/.gemini/antigravity/scratch/snr_learning/single_distribution_demo.png', 
                dpi=150, bbox_inches='tight')
    print("\nPlot saved to: single_distribution_demo.png")
    
    # Plot SNR history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(optimizer.snr_history) + 1), 
            optimizer.snr_history, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=initial_snr, color='r', linestyle='--', label='Initial SNR')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Optimization Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('C:/Users/User/.gemini/antigravity/scratch/snr_learning/snr_history.png',
                dpi=150, bbox_inches='tight')
    print("SNR history plot saved to: snr_history.png")


def demo_adaptive_learning():
    """Demonstrate adaptive distribution selection for SNR optimization."""
    print("\n" + "=" * 60)
    print("Demo 2: Adaptive Distribution Selection")
    print("=" * 60)
    
    # Generate test data with non-Gaussian noise
    signal, _, t = generate_test_signal()
    
    # Use Laplace (heavy-tailed) noise
    noise = np.random.laplace(0, 0.3, len(signal))
    noisy_signal = signal + noise
    
    # Compute initial SNR
    initial_snr = 10 * np.log10(np.var(signal) / np.var(noise))
    print(f"\nInitial SNR: {initial_snr:.2f} dB")
    print(f"True noise distribution: Laplace")
    
    # Use adaptive learner
    print("\nTesting multiple distributions...")
    learner = AdaptiveSNRLearner()
    results = learner.fit(signal, noise)
    
    print(f"\nResults by distribution:")
    for dist_type, res in results['all_results'].items():
        print(f"  {dist_type:12s}: SNR = {res['snr']:.2f} dB")
    
    print(f"\nBest distribution: {results['best_distribution']}")
    print(f"Best SNR: {results['best_snr']:.2f} dB")
    print(f"SNR improvement: {results['best_snr'] - initial_snr:.2f} dB")
    
    # Denoise using best model
    denoised = learner.denoise(noisy_signal)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Time domain plots
    axes[0, 0].plot(t[:200], signal[:200], 'g-', label='Clean', linewidth=2)
    axes[0, 0].plot(t[:200], noisy_signal[:200], 'r-', alpha=0.6, label='Noisy')
    axes[0, 0].set_title('Input Signals (zoomed)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t[:200], signal[:200], 'g-', label='Clean', linewidth=2)
    axes[0, 1].plot(t[:200], denoised[:200], 'b-', alpha=0.8, label='Denoised')
    axes[0, 1].set_title('Denoised vs Clean (zoomed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Noise distribution histograms
    axes[1, 0].hist(noise, bins=50, density=True, alpha=0.7, label='True Noise')
    axes[1, 0].set_title('True Noise Distribution (Laplace)')
    axes[1, 0].set_xlabel('Noise Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # SNR comparison bar chart
    dist_names = list(results['all_results'].keys())
    snr_values = [results['all_results'][d]['snr'] for d in dist_names]
    colors = ['green' if d == results['best_distribution'] else 'skyblue' 
              for d in dist_names]
    
    axes[1, 1].bar(dist_names, snr_values, color=colors, alpha=0.7)
    axes[1, 1].axhline(y=initial_snr, color='r', linestyle='--', 
                       linewidth=2, label='Initial SNR')
    axes[1, 1].set_ylabel('SNR (dB)')
    axes[1, 1].set_title('SNR by Distribution Type')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('C:/Users/User/.gemini/antigravity/scratch/snr_learning/adaptive_learning_demo.png',
                dpi=150, bbox_inches='tight')
    print("\nPlot saved to: adaptive_learning_demo.png")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Statistical Learning through Distributions - SNR Optimization")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demos
    demo_single_distribution()
    demo_adaptive_learning()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
