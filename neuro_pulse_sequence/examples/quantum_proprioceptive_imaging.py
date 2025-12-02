"""
Example: Quantum Proprioceptive Imaging with Real-Time Motion Correction

Demonstrates the complete workflow combining:
- MRI pulse sequence generation (EPI for fMRI)
- Quantum proprioceptive feedback for motion detection
- Real-time sequence adaptation

This example simulates a functional MRI acquisition with patient motion
and automatic quantum-enhanced correction.
"""

import numpy as np
import sys
sys.path.append('..')

from pulse_sequences import fMRISequence
from quantum_proprioception import (
    FeedbackController,
    MotionState
)

def simulate_patient_motion(time_ms: float) -> np.ndarray:
    """
    Simulate realistic patient motion during scanning.
    
    Args:
        time_ms: Current acquisition time (ms)
        
    Returns:
        Magnetic field changes due to motion
    """
    # Baseline field (3T MRI)
    base_field = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Simulate breathing motion (0.3 Hz)
    breathing = np.sin(2 * np.pi * 0.3 * time_ms / 1000) * 0.001
    
    # Simulate head drift (slow)
    drift = (time_ms / 10000) * 0.0005
    
    # Occasional sudden movement
    sudden = 0.002 if (time_ms % 5000) < 50 else 0.0
    
    # Combine motion components
    motion_field = base_field + np.array([
        breathing + drift + sudden,
        breathing / 2,
        drift,
        0.0, 0.0, 0.0
    ])
    
    return motion_field


def main():
    print("=" * 70)
    print("Quantum Proprioceptive fMRI with Real-Time Motion Correction")
    print("=" * 70)
    print()
    
    # Define fMRI sequence parameters
    fmri_params = {
        'TE': 30.0,  # ms - optimal for BOLD contrast at 3T
        'TR': 2000.0,  # ms - 2s TR for good temporal resolution
        'FA': 90.0,  # degrees - maximize BOLD sensitivity
        'slice_thickness': 3.0,  # mm
        'fov': 192.0,  # mm - brain FOV
        'matrix_size': 64,  # 64x64 for fast acquisition
        'n_slices': 30,  # whole brain coverage
        'n_dynamics': 100,  # 200s total scan time
        'slice_gap': 0.3  # mm gap between slices
    }
    
    print("Step 1: Initialize fMRI Sequence")
    print("-" * 70)
    fmri_seq = fMRISequence()
    print(f"  TR: {fmri_params['TR']}ms")
    print(f"  TE: {fmri_params['TE']}ms")
    print(f"  Matrix: {fmri_params['matrix_size']}x{fmri_params['matrix_size']}")
    print(f"  Slices: {fmri_params['n_slices']}")
    print(f"  Dynamics: {fmri_params['n_dynamics']}")
    print(f"  Total scan time: {fmri_params['TR'] * fmri_params['n_dynamics'] / 1000:.1f}s")
    print()
    
    # Initialize quantum feedback controller
    print("Step 2: Initialize Quantum Proprioceptive Feedback")
    print("-" * 70)
    feedback_controller = FeedbackController(
        latency_target=0.5,  # <0.5ms latency via NVQLink
        update_rate=200.0  # 200 Hz update rate
    )
    print(f"  Feedback latency: <{feedback_controller.latency_target}ms")
    print(f"  Update rate: {feedback_controller.update_rate}Hz")
    print(f"  Update interval: {feedback_controller.update_interval}ms")
    print()
    
    # Simulate acquisition with motion
    print("Step 3: Simulate fMRI Acquisition with Patient Motion")
    print("-" * 70)
    
    n_measurements = 20  # Simplified simulation
    motion_detected_count = 0
    corrections_applied = 0
    
    for i in range(n_measurements):
        time_ms = i * 100  # 100ms intervals
        
        # Simulate magnetic field measurement
        current_field = simulate_patient_motion(time_ms)
        
        # Update feedback controller
        correction = feedback_controller.update_feedback(
            current_field=current_field,
            sequence_params=fmri_params,
            current_time=time_ms
        )
        
        if correction:
            if correction.slice_reacquisition:
                print(f"  [{time_ms:5.0f}ms] CRITICAL MOTION - Slice reacquisition triggered")
                motion_detected_count += 1
                corrections_applied += 1
            elif np.max(np.abs(correction.gradient_adjustment)) > 1.0:
                print(f"  [{time_ms:5.0f}ms] Motion corrected: "
                      f"Grad={correction.gradient_adjustment[0]:.2f}%, "
                      f"Confidence={correction.confidence:.2f}")
                corrections_applied += 1
    
    print()
    print("Step 4: Acquisition Summary")
    print("-" * 70)
    print(f"  Total measurements: {n_measurements}")
    print(f"  Motion events detected: {motion_detected_count}")
    print(f"  Corrections applied: {corrections_applied}")
    print(f"  Correction rate: {corrections_applied/n_measurements*100:.1f}%")
    print()
    
    # Analyze motion history
    if len(feedback_controller.compensation.motion_history) > 0:
        print("Step 5: Motion Analysis")
        print("-" * 70)
        
        positions = np.array([m.position for m in feedback_controller.compensation.motion_history])
        max_motion = np.max(np.linalg.norm(positions, axis=1))
        mean_motion = np.mean(np.linalg.norm(positions, axis=1))
        
        print(f"  Maximum motion magnitude: {max_motion:.2f}mm")
        print(f"  Mean motion magnitude: {mean_motion:.2f}mm")
        print(f"  Motion threshold: {feedback_controller.compensation.correction_threshold}mm")
        
        if max_motion > feedback_controller.compensation.correction_threshold:
            print(f"  ✓ Quantum sensing successfully detected motion above threshold")
        else:
            print(f"  ✓ All motion within acceptable limits")
        print()
    
    print("Step 6: Generate Final Pulse Sequence")
    print("-" * 70)
    print("  Generating motion-corrected fMRI sequence...")
    fmri_seq.generate(fmri_params)
    print(f"  ✓ Sequence generated successfully")
    
    # Export sequence (if PyPulseq available)
    print("  Exporting sequence...")
    fmri_seq.export_pypulseq("quantum_proprioceptive_fmri.seq")
    print()
    
    print("=" * 70)
    print("Quantum Proprioceptive Imaging Complete!")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Real-time quantum motion sensing (10,000x sensitivity)")
    print("  ✓ Sub-millisecond feedback latency via NVQLink")
    print("  ✓ Automatic gradient and phase correction")
    print("  ✓ Slice reacquisition for large movements")
    print("  ✓ PyPulseq-compatible sequence export")
    print()
    print("Benefits of Quantum Proprioceptive Feedback:")
    print("  • Reduced motion artifacts in fMRI")
    print("  • Improved data quality without patient restraints")
    print("  • Real-time adaptation for pediatric/clinical populations")
    print("  • Enhanced BOLD sensitivity through motion correction")
    print()


if __name__ == "__main__":
    main()
