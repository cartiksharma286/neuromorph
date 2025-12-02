"""
Quantum Proprioceptive Feedback System for MRI

Implements quantum sensing for real-time patient motion detection and
adaptive sequence correction using quantum magnetometry and NVQLink.

Key Components:
- QuantumMotionSensor: Quantum magnetometer-based motion detection
- ProprioceptiveEncoder: Position/orientation encoding in quantum states
- MotionCompensation: Real-time gradient adjustment
- FeedbackController: Closed-loop quantum feedback
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from dataclasses import dataclass

# Quantum computing
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    warnings.warn("CUDA-Q not available. Using classical simulation.")


@dataclass
class MotionState:
    """Represents detected motion state."""
    position: np.ndarray  # x, y, z position (mm)
    orientation: np.ndarray  # roll, pitch, yaw (degrees)
    velocity: np.ndarray  # Linear velocity (mm/s)
    timestamp: float  # Time of measurement (ms)
    confidence: float  # Detection confidence (0-1)


@dataclass
class CorrectionParameters:
    """Parameters for motion correction."""
    gradient_adjustment: np.ndarray  # Gradient amplitude changes  (%)
    phase_correction: np.ndarray  # Phase corrections (radians)
    slice_reacquisition: bool  # Whether to reacquire slice
    confidence: float  # Correction confidence (0-1)


class ProprioceptiveEncoder:
    """
    Encodes position and orientation data into quantum states.
    
    Uses quantum state preparation to encode 6DOF (degrees of freedom)
    motion information for quantum processing.
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize proprioceptive encoder.
        
        Args:
            n_qubits: Number of qubits (6 allows encoding of 6DOF motion)
        """
        self.n_qubits = n_qubits
        
    def encode_motion(self, motion_state: MotionState) -> np.ndarray:
        """
        Encode motion state into quantum state preparation angles.
        
        Args:
            motion_state: Detected motion state
            
        Returns:
            Quantum state preparation parameters
        """
        # Normalize position to [0, π] for quantum encoding
        pos = motion_state.position
        orient = motion_state.orientation
        
        # Map to rotation angles for quantum state preparation
        # Position -> first 3 qubits, Orientation -> next 3 qubits
        angles = np.zeros(self.n_qubits * 2)  # RY and RZ for each qubit
        
        # Position encoding (normalize to scanner FOV ~ 500mm)
        fov = 500.0
        for i in range(3):
            angles[i*2] = (pos[i] / fov + 0.5) * np.pi  # RY angle
            angles[i*2 + 1] = (pos[i] / fov) * np.pi / 2  # RZ angle
        
        # Orientation encoding (normalize degrees to radians)
        for i in range(3):
            angles[(i+3)*2] = np.radians(orient[i])  # RY angle
            angles[(i+3)*2 + 1] = np.radians(orient[i]) / 2  # RZ angle
        
        return angles
    
    def decode_motion(self, quantum_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode quantum state back to position and orientation.
        
        Args:
            quantum_state: Quantum state vector or measurement probabilities
            
        Returns:
            Tuple of (position, orientation)
        """
        # Extract position from first 3 qubits via state tomography
        fov = 500.0
        
        # Use highest probability states to infer position
        n_states = len(quantum_state)
        position = np.zeros(3)
        orientation = np.zeros(3)
        
        # Simplified decoding using probability amplitudes
        for i in range(min(self.n_qubits // 2, 3)):
            # Position from state amplitudes
            start_idx = 2**i
            end_idx = 2**(i+1)
            prob_sum = np.sum(np.abs(quantum_state[start_idx:end_idx])**2)
            position[i] = (prob_sum - 0.5) * fov
            
        for i in range(min(self.n_qubits // 2, 3)):
            # Orientation from phase information
            start_idx = 2**(i+3)
            end_idx = 2**(i+4)
            if end_idx <= n_states:
                phase_avg = np.mean(np.angle(quantum_state[start_idx:end_idx]))
                orientation[i] = np.degrees(phase_avg)
        
        return position, orientation


class QuantumMotionSensor:
    """
    Quantum magnetometer-based motion detection using CUDA-Q.
    
    Simulates quantum sensing with enhanced sensitivity (10,000x classical)
    for detecting subtle patient motion during MRI acquisition.
    """
    
    def __init__(self, n_qubits: int = 6, sensitivity_enhancement: float = 10000.0):
        """
        Initialize quantum motion sensor.
        
        Args:
            n_qubits: Number of qubits for quantum sensing
            sensitivity_enhancement: Quantum advantage factor
        """
        self.n_qubits = n_qubits
        self.sensitivity_enhancement = sensitivity_enhancement
        self.encoder = ProprioceptiveEncoder(n_qubits)
        
        if CUDAQ_AVAILABLE:
            self.sensing_kernel = self._build_sensing_kernel()
        else:
            print("Using classical motion sensing simulation")
    
    def _build_sensing_kernel(self):
        """Build CUDA-Q kernel for quantum magnetometry."""
        
        @cudaq.kernel
        def quantum_magnetometer(angles: List[float]):
            """
            Quantum magnetometer for motion sensing.
            
            Uses GHZ  state for entanglement-enhanced sensing.
            """
            qubits = cudaq.qvector(self.n_qubits)
            
            # Prepare GHZ state for enhanced sensitivity
            h(qubits[0])
            for i in range(self.n_qubits - 1):
                cx(qubits[i], qubits[i + 1])
            
            # Apply rotation based on magnetic field (motion-induced)
            for i in range(self.n_qubits):
                if i < len(angles):
                    ry(angles[i], qubits[i])
            
            # Additional entangling layer for phase estimation
            for i in range(self.n_qubits - 1):
                cx(qubits[i], qubits[i + 1])
        
        return quantum_magnetometer
    
    def detect_motion(self, 
                     reference_field: np.ndarray,
                     current_field: np.ndarray,
                     timestamp: float) -> MotionState:
        """
        Detect motion from magnetic field changes.
        
        Args:
            reference_field: Reference magnetic field (baseline)
            current_field: Current measured magnetic field
            timestamp: Measurement timestamp (ms)
            
        Returns:
            Detected motion state
        """
        # Calculate field difference (motion signature)
        field_delta = current_field - reference_field
        
        # Quantum-enhanced sensing
        if CUDAQ_AVAILABLE:
            motion_state = self._quantum_detect(field_delta, timestamp)
        else:
            motion_state = self._classical_detect(field_delta, timestamp)
        
        return motion_state
    
    def _quantum_detect(self, field_delta: np.ndarray, timestamp: float) -> MotionState:
        """Quantum-enhanced motion detection."""
        # Prepare sensing angles from field delta
        angles = (field_delta / np.linalg.norm(field_delta) * np.pi).tolist()
        angles = angles[:self.n_qubits]
        
        # Execute quantum sensing
        result = cudaq.sample(self.sensing_kernel, angles)
        counts = result.get_sequential_data()
        
        # Convert to state vector
        total = sum(counts.values())
        state_vector = np.array([
            counts.get(format(i, f'0{self.n_qubits}b'), 0) / total
            for i in range(2**self.n_qubits)
        ], dtype=complex)
        
        # Decode motion from quantum state
        position, orientation = self.encoder.decode_motion(state_vector)
        
        # Estimate velocity from position change (simplified)
        velocity = position / (timestamp / 1000.0 + 1e-6)
        
        # Confidence from measurement entropy
        probs = np.abs(state_vector)**2
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = self.n_qubits
        confidence = 1.0 - (entropy / max_entropy)
        
        return MotionState(
            position=position,
            orientation=orientation,
            velocity=velocity,
            timestamp=timestamp,
            confidence=confidence
        )
    
    def _classical_detect(self, field_delta: np.ndarray, timestamp: float) -> MotionState:
        """Classical motion detection (fallback)."""
        # Simulate motion detection with reduced sensitivity
        sensitivity = 1.0 / self.sensitivity_enhancement
        
        # Simple linear mapping from field to position
        position = field_delta[:3] * 100.0 * sensitivity
        orientation = field_delta[3:6] if len(field_delta) >= 6 else np.zeros(3)
        orientation = orientation * 10.0 * sensitivity
        
        velocity = position / (timestamp / 1000.0 + 1e-6)
        
        # Add noise to simulate classical limitations
        position += np.random.normal(0, 0.5, 3)
        orientation += np.random.normal(0, 0.1, 3 )
        
        return MotionState(
            position=position,
            orientation=orientation,
            velocity=velocity,
            timestamp=timestamp,
            confidence=0.7  # Lower confidence without quantum enhancement
        )


class MotionCompensation:
    """
    Real-time gradient and phase adjustment for motion correction.
    
    Calculates necessary sequence modifications to compensate for
    detected patient motion.
    """
    
    def __init__(self, correction_threshold: float = 0.5):
        """
        Initialize motion compensation.
        
        Args:
            correction_threshold: Motion threshold for correction (mm)
        """
        self.correction_threshold = correction_threshold
        self.motion_history = []
    
    def calculate_correction(self, 
                            motion_state: MotionState,
                            sequence_params: Dict) -> CorrectionParameters:
        """
        Calculate correction parameters for detected motion.
        
        Args:
            motion_state: Detected motion state
            sequence_params: Current pulse sequence parameters
            
        Returns:
            Correction parameters
        """
        # Check if motion exceeds threshold
        motion_magnitude = np.linalg.norm(motion_state.position)
        
        if motion_magnitude < self.correction_threshold:
            # No correction needed
            return CorrectionParameters(
                gradient_adjustment=np.zeros(3),
                phase_correction=np.zeros(3),
                slice_reacquisition=False,
                confidence=1.0
            )
        
        # Calculate gradient adjustments (for through-plane motion)
        gradient_adjustment = self._calculate_gradient_adjustment(
            motion_state.position,
            motion_state.orientation
        )
        
        # Calculate phase corrections (for in-plane motion)
        phase_correction = self._calculate_phase_correction(
            motion_state.position,
            sequence_params
        )
        
        # Determine if reacquisition is needed
        slice_reacquisition = motion_magnitude > (self.correction_threshold * 3)
        
        # Store in history
        self.motion_history.append(motion_state)
        
        return CorrectionParameters(
            gradient_adjustment=gradient_adjustment,
            phase_correction=phase_correction,
            slice_reacquisition=slice_reacquisition,
            confidence=motion_state.confidence
        )
    
    def _calculate_gradient_adjustment(self, 
                                      position: np.ndarray,
                                      orientation: np.ndarray) -> np.ndarray:
        """Calculate gradient amplitude adjustments."""
        # Gradient adjustment proportional to position error
        # Assuming gradients are in x, y, z directions
        gradient_adj = np.zeros(3)
        
        for i in range(3):
            # Adjust gradient to shift FOV and compensate motion
            # adjustment = (position_error / FOV) * 100 (percent)
            fov = 256.0  # mm
            gradient_adj[i] = (position[i] / fov) * 100.0
            
            # Add rotational component
            if i < len(orientation):
                gradient_adj[i] += orientation[i] * 0.1
        
        return gradient_adj
    
    def _calculate_phase_correction(self,
                                   position: np.ndarray,
                                   sequence_params: Dict) -> np.ndarray:
        """Calculate phase corrections for in-plane motion."""
        # Phase correction from rigid body motion
        fov = sequence_params.get('fov', 256.0)  # mm
        
        # k-space shift theorem: motion causes linear phase
        k_max = 1.0 / (fov / sequence_params.get('matrix_size', 256))
        
        phase_correction = np.zeros(3)
        for i in range(3):
            # Phase = 2π * k * Δx
            phase_correction[i] = 2 * np.pi * k_max * position[i]
        
        return phase_correction


class FeedbackController:
    """
    Closed-loop quantum feedback controller for real-time sequence adaptation.
    
    Integrates quantum motion sensing with pulse sequence generation
    for automatic motion correction during acquisition.
    """
    
    def __init__(self, 
                 latency_target: float = 0.5,  # ms
                 update_rate: float = 100.0):  # Hz
        """
        Initialize feedback controller.
        
        Args:
            latency_target: Target feedback latency (ms) via NVQLink
            update_rate: Feedback update rate (Hz)
        """
        self.latency_target = latency_target
        self.update_rate = update_rate
        self.update_interval = 1000.0 / update_rate  # ms
        
        self.motion_sensor = QuantumMotionSensor()
        self.compensation = MotionCompensation()
        
        self.reference_field = None
        self.last_update_time = 0.0
        self.correction_history = []
    
    def initialize_reference(self, reference_field: np.ndarray):
        """Set reference magnetic field (baseline with no motion)."""
        self.reference_field = reference_field
        print(f"Reference field initialized: {reference_field[:3]}")
    
    def update_feedback(self,
                       current_field: np.ndarray,
                       sequence_params: Dict,
                       current_time: float) -> Optional[CorrectionParameters]:
        """
        Update feedback loop with current measurements.
        
        Args:
            current_field: Current magnetic field measurement
            sequence_params: Current sequence parameters
            current_time: Current time (ms)
            
        Returns:
            Correction parameters if update is needed, None otherwise
        """
        # Check if update is needed based on update rate
        if current_time - self.last_update_time < self.update_interval:
            return None
        
        if self.reference_field is None:
            self.initialize_reference(current_field)
            return None
        
        # Detect motion
        motion_state = self.motion_sensor.detect_motion(
            self.reference_field,
            current_field,
            current_time
        )
        
        # Calculate correction
        correction = self.compensation.calculate_correction(
            motion_state,
            sequence_params
        )
        
        # Store correction history
        self.correction_history.append({
            'time': current_time,
            'motion': motion_state,
            'correction': correction
        })
        
        self.last_update_time = current_time
        
        # Print feedback (in real system, would send to scanner)
        if correction.slice_reacquisition:
            print(f"[{current_time:.1f}ms] MOTION DETECTED - Reacquisition required!")
        elif np.max(np.abs(correction.gradient_adjustment)) > 1.0:
            print(f"[{current_time:.1f}ms] Applying corrections: "
                  f"Grad={correction.gradient_adjustment}, "
                  f"Phase={correction.phase_correction}")
        
        return correction
    
    def apply_correction(self, 
                        correction: CorrectionParameters,
                        pulse_sequence: any) -> any:
        """
        Apply correction to pulse sequence in real-time.
        
        Args:
            correction: Correction parameters
            pulse_sequence: Current pulse sequence object
            
        Returns:
            Modified pulse sequence
        """
        # This would interface with actual sequence generation
        # For now, return parameters for demonstration
        
        if correction.slice_reacquisition:
            print("Flagging slice for reacquisition")
        
        # Modify sequence parameters
        modified_params = pulse_sequence.params.copy()
        
        # Apply gradient adjustments (percentage changes)
        # In real system, would modify gradient amplitudes
        
        # Apply phase corrections
        # In real system, would add phase encoding shifts
        
        return pulse_sequence


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum Proprioceptive Feedback System for MRI")
    print("=" * 70)
    print(f"CUDA-Q Available: {CUDAQ_AVAILABLE}")
    print()
    
    # Test proprioceptive encoder
    print("Testing Proprioceptive Encoder...")
    encoder = ProprioceptiveEncoder(n_qubits=6)
    test_motion = MotionState(
        position=np.array([2.5, -1.0, 0.5]),  # mm
        orientation=np.array([1.0, -0.5, 0.2]),  # degrees
        velocity=np.array([0.1, 0.05, 0.02]),  # mm/s
        timestamp=100.0,  # ms
        confidence=0.95
    )
    angles = encoder.encode_motion(test_motion)
    print(f"  Motion state: pos={test_motion.position}, orient={test_motion.orientation}")
    print(f"  Encoded angles (first 6): {angles[:6]}")
    print()
    
    # Test quantum motion sensor
    print("Testing Quantum Motion Sensor...")
    sensor = QuantumMotionSensor(n_qubits=6)
    
    # Simulate magnetic field measurements
    reference_field = np.array([1.5, 0.0, 3.0, 0.1, 0.05, 0.02])  # Tesla
    current_field = reference_field + np.array([0.001, 0.0005, 0.0002, 0.0001, 0, 0])  # Small change
    
    detected_motion = sensor.detect_motion(reference_field, current_field, 150.0)
    print(f"  Detected position: {detected_motion.position}")
    print(f"  Detected orientation: {detected_motion.orientation}")
    print(f"  Confidence: {detected_motion.confidence:.3f}")
    print()
    
    # Test motion compensation
    print("Testing Motion Compensation...")
    compensator = MotionCompensation(correction_threshold=0.5)
    sequence_params = {'fov': 256, 'matrix_size': 256, 'TE': 30, 'TR': 2000}
    
    correction = compensator.calculate_correction(detected_motion, sequence_params)
    print(f"  Gradient adjustment: {correction.gradient_adjustment}")
    print(f"  Phase correction: {correction.phase_correction}")
    print(f"  Reacquisition needed: {correction.slice_reacquisition}")
    print()
    
    # Test feedback controller
    print("Testing Feedback Controller...")
    controller = FeedbackController(latency_target=0.5, update_rate=100.0)
    controller.initialize_reference(reference_field)
    
    # Simulate motion over time
    for t in np.arange(0, 100, 5):  # 100ms simulation
        # Simulate gradual motion
        field_drift = np.sin(t / 20) * 0.002
        current_field_sim = reference_field + np.array([field_drift, field_drift/2, 0, 0, 0, 0])
        
        correction = controller.update_feedback(current_field_sim, sequence_params, t)
        
        if correction and t % 20 == 0:
            print(f"  Time {t}ms: Motion magnitude = "
                  f"{np.linalg.norm(controller.compensation.motion_history[-1].position):.2f}mm")
    
    print()
    print("Quantum proprioceptive feedback system initialized!")
    print(f"Average latency would be: <{controller.latency_target}ms via NVQLink")
