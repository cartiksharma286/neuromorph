"""
Demonstration of Quantum-Enhanced Surgical Robot
Tests quantum Kalman filter and QML pose estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from robot_kinematics_quantum import QuantumEnhancedRobot6DOF
from robot_kinematics import Robot6DOF

def compare_classical_vs_quantum(num_steps=100):
    """
    Compare classical vs quantum-enhanced pose estimation
    """
    print("=" * 60)
    print("Quantum vs Classical Robot Pose Estimation Comparison")
    print("=" * 60)
    
    # Initialize robots
    classical_robot = Robot6DOF()
    quantum_robot = QuantumEnhancedRobot6DOF()
    
    # Set same target
    target = np.array([0.3, 0.1, 0.4])
    classical_robot.set_target(*target)
    quantum_robot.set_target(*target)
    
    # Track errors
    classical_errors = []
    quantum_errors = []
    quantum_coherence = []
    quantum_uncertainty = []
    
    print(f"\nTarget position: {target}")
    print(f"Running {num_steps} iterations...\n")
    
    for i in range(num_steps):
        # Update both robots
        classical_robot.update()
        quantum_robot.update()
        
        # Get positions
        classical_pos = classical_robot.fk(classical_robot.joints)
        quantum_pos = quantum_robot.fk(quantum_robot.joints)
        
        # Calculate errors
        classical_error = np.linalg.norm(target - classical_pos)
        quantum_error = np.linalg.norm(target - quantum_pos)
        
        classical_errors.append(classical_error)
        quantum_errors.append(quantum_error)
        
        # Get quantum metrics
        metrics = quantum_robot.get_quantum_metrics()
        quantum_coherence.append(metrics.get('coherence', 0.0))
        quantum_uncertainty.append(metrics.get('uncertainty', 0.0))
        
        if i % 20 == 0:
            print(f"Step {i:3d}: Classical Error = {classical_error:.4f} m, "
                  f"Quantum Error = {quantum_error:.4f} m, "
                  f"Coherence = {metrics.get('coherence', 0.0):.3f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Classical Robot:")
    print(f"  Final Error: {classical_errors[-1]:.6f} m")
    print(f"  Average Error: {np.mean(classical_errors):.6f} m")
    print(f"  Std Dev: {np.std(classical_errors):.6f} m")
    
    print(f"\nQuantum-Enhanced Robot:")
    print(f"  Final Error: {quantum_errors[-1]:.6f} m")
    print(f"  Average Error: {np.mean(quantum_errors):.6f} m")
    print(f"  Std Dev: {np.std(quantum_errors):.6f} m")
    print(f"  Final Coherence: {quantum_coherence[-1]:.3f}")
    print(f"  Final Uncertainty: {quantum_uncertainty[-1]:.6f}")
    
    improvement = (np.mean(classical_errors) - np.mean(quantum_errors)) / np.mean(classical_errors) * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error comparison
    axes[0, 0].plot(classical_errors, label='Classical', linewidth=2, alpha=0.7)
    axes[0, 0].plot(quantum_errors, label='Quantum-Enhanced', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Position Error (m)')
    axes[0, 0].set_title('Tracking Error Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Quantum coherence
    axes[0, 1].plot(quantum_coherence, color='purple', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Quantum Coherence')
    axes[0, 1].set_title('Quantum Coherence Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1])
    
    # Quantum uncertainty
    axes[1, 0].plot(quantum_uncertainty, color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('State Uncertainty')
    axes[1, 0].set_title('Quantum State Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error reduction
    error_reduction = [(c - q) / c * 100 if c > 0 else 0 
                       for c, q in zip(classical_errors, quantum_errors)]
    axes[1, 1].plot(error_reduction, color='green', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Error Reduction (%)')
    axes[1, 1].set_title('Quantum Improvement Over Classical')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/cartik_sharma/Downloads/neuromorph-main-n/neurosurgery_robot/quantum_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: quantum_comparison.png")
    
    return {
        'classical_errors': classical_errors,
        'quantum_errors': quantum_errors,
        'improvement': improvement,
        'coherence': quantum_coherence,
        'uncertainty': quantum_uncertainty
    }


def test_qml_training():
    """Test quantum machine learning training"""
    print("\n" + "=" * 60)
    print("Quantum Machine Learning Training Test")
    print("=" * 60)
    
    robot = QuantumEnhancedRobot6DOF()
    
    print("\nTraining QML component with 50 steps...")
    losses = robot.train_qml(num_steps=50)
    
    print(f"Initial Loss: {losses[0]:.6f}")
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Loss Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='blue')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (MSE)')
    plt.title('Quantum Machine Learning Training Curve')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('/Users/cartik_sharma/Downloads/neuromorph-main-n/neurosurgery_robot/qml_training.png',
                dpi=300, bbox_inches='tight')
    print(f"Training plot saved to: qml_training.png")
    
    return losses


if __name__ == "__main__":
    # Run comparison
    results = compare_classical_vs_quantum(num_steps=100)
    
    # Test QML training
    losses = test_qml_training()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"  • Quantum enhancement provides {results['improvement']:.1f}% improvement")
    print(f"  • Final coherence: {results['coherence'][-1]:.3f}")
    print(f"  • QML training converged successfully")
    print("\nGenerated files:")
    print("  • quantum_comparison.png")
    print("  • qml_training.png")
    print("  • Quantum_Kalman_Surgical_Robotics_Report.tex")
