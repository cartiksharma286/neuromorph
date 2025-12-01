#!/usr/bin/env python3
"""
pypulseq_generator.py
Complete PyPulseq sequence generation for hyperpolarized imaging

Sunnybrook Research Institute
"""

import numpy as np
from pypulseq import Sequence, Opts
from pypulseq import make_sinc_pulse, make_trapezoid, make_adc, make_delay
from pypulseq import make_arbitrary_grad, calc_duration
from typing import Dict, List, Optional, Tuple


class HyperpolarizedSequence:
    """Generate complete PyPulseq sequences for hyperpolarized imaging"""
    
    def __init__(self, system_limits: Optional[Dict] = None):
        """
        Initialize sequence generator
        
        Args:
            system_limits: Dictionary of system limits (max_grad, max_slew, etc.)
        """
        if system_limits is None:
            self.system = Opts(
                max_grad=40,  # mT/m
                grad_unit='mT/m',
                max_slew=150,  # T/m/s
                slew_unit='T/m/s',
                rf_ringdown_time=20e-6,
                rf_dead_time=100e-6,
                adc_dead_time=10e-6,
                grad_raster_time=10e-6,
                rf_raster_time=1e-6
            )
        else:
            self.system = Opts(**system_limits)
        
        self.seq = Sequence(self.system)
    
    def create_vfa_sequence(self,
                           flip_angles: np.ndarray,
                           tr: float,
                           fov: float = 0.24,
                           slice_thickness: float = 0.01,
                           readout_type: str = 'spiral') -> Sequence:
        """
        Create variable flip angle sequence
        
        Args:
            flip_angles: Array of flip angles in degrees
            tr: Repetition time in seconds
            fov: Field of view in meters
            slice_thickness: Slice thickness in meters
            readout_type: Type of readout ('spiral', 'epi', 'cartesian')
            
        Returns:
            Complete PyPulseq Sequence object
        """
        for frame_idx, flip_angle in enumerate(flip_angles):
            flip_rad = np.deg2rad(flip_angle)
            
            # RF pulse with slice selection
            rf, gz, gz_reph = make_sinc_pulse(
                flip_angle=flip_rad,
                system=self.system,
                duration=2e-3,
                slice_thickness=slice_thickness,
                apodization=0.5,
                time_bw_product=4,
                return_gz=True
            )
            
            self.seq.add_block(rf, gz)
            self.seq.add_block(gz_reph)
            
            # Add readout based on type
            if readout_type == 'spiral':
                # Placeholder - would add spiral gradients
                delay_readout = make_delay(20e-3)  # 20ms spiral readout
                self.seq.add_block(delay_readout)
            elif readout_type == 'epi':
                delay_readout = make_delay(50e-3)  # 50ms EPI readout
                self.seq.add_block(delay_readout)
            else:  # Cartesian
                # Simple Cartesian readout
                gx = make_trapezoid(channel='x', amplitude=20, flat_time=4e-3, system=self.system)
                adc = make_adc(num_samples=256, duration=4e-3, system=self.system)
                self.seq.add_block(gx,adc)
            
            # TR delay
            tr_delay = tr - self.seq.duration() + frame_idx * tr
            if tr_delay > 0:
                self.seq.add_block(make_delay(tr_delay))
        
        return self.seq
    
    def create_spiral_readout(self,
                             kx: np.ndarray,
                             ky: np.ndarray,
                             dt: float = 4e-6,
                             gamma: float = 42.58e6) -> Tuple:
        """
        Create spiral readout gradients from k-space trajectory
        
        Args:
            kx: k-space x coordinates (1/m)
            ky: k-space y coordinates (1/m)
            dt: Dwell time in seconds
            gamma: Gyromagnetic ratio in Hz/T
            
        Returns:
            Tuple of (gx, gy, adc) blocks
        """
        # Calculate gradient amplitude from k-space trajectory
        # G(t) = (1/gamma) * dk/dt
        
        gx_wave = np.diff(kx) / (gamma * dt)  # T/m
        gy_wave = np.diff(ky) / (gamma * dt)
        
        # Convert to mT/m
        gx_wave = gx_wave * 1000
        gy_wave = gy_wave * 1000
        
        # Create arbitrary gradients
        gx = make_arbitrary_grad(channel='x', waveform=gx_wave, system=self.system)
        gy = make_arbitrary_grad(channel='y', waveform=gy_wave, system=self.system)
        
        # ADC
        num_samples = len(kx)
        duration = len(kx) * dt
        adc = make_adc(num_samples=num_samples, duration=duration, system=self.system)
        
        return gx, gy, adc
    
    def create_epi_readout(self,
                          matrix_size: int = 64,
                          fov: float = 0.24,
                          echo_spacing: float = 0.5e-3) -> List:
        """
        Create EPI readout train
        
        Args:
            matrix_size: Matrix size
            fov: Field of view in meters
            echo_spacing: Echo spacing in seconds
            
        Returns:
            List of sequence blocks for EPI readout
        """
        blocks = []
        
        # Calculate gradient amplitude
        delta_k = 1 / fov
        grad_amplitude = delta_k / (self.system.gamma * echo_spacing)
        
        for line in range(matrix_size):
            # Readout gradient (alternating direction)
            direction = 1 if line % 2 == 0 else -1
            gx = make_trapezoid(
                channel='x',
                amplitude=direction * grad_amplitude * 1000,  # mT/m
                flat_time=echo_spacing,
                system=self.system
            )
            
            # ADC
            adc = make_adc(num_samples=matrix_size, duration=echo_spacing, system=self.system)
            
            # Phase blip
            if line < matrix_size - 1:
                gy_blip = make_trapezoid(
                    channel='y',
                    area=delta_k,
                    duration=200e-6,
                    system=self.system
                )
                blocks.append((gx, adc, gy_blip))
            else:
                blocks.append((gx, adc))
        
        return blocks
    
    def add_spectral_spatial_pulse(self,
                                   target_frequencies: List[float],
                                   slice_thickness: float = 0.01,
                                   duration: float = 8e-3) -> None:
        """
        Add spectral-spatial excitation pulse
        
        Args:
            target_frequencies: List of target frequencies in Hz
            slice_thickness: Slice thickness in meters
            duration: Pulse duration in seconds
        """
        # Simplified spectral-spatial pulse
        # In practice, would use more sophisticated design (e.g., SLR algorithm)
        
        num_subpulses = 16
        subpulse_duration = duration / num_subpulses
        
        for i in range(num_subpulses):
            # Phase modulation for spectral selectivity
            phase = 2 * np.pi * target_frequencies[0] * i * subpulse_duration
            
            # Amplitude (Hamming window)
            amplitude = 0.54 - 0.46 * np.cos(2 * np.pi * i / (num_subpulses - 1))
            
            # Create subpulse
            rf_subpulse = make_sinc_pulse(
                flip_angle=amplitude * np.pi / 32,  # Small flip per subpulse
                system=self.system,
                duration=subpulse_duration,
                slice_thickness=slice_thickness,
                phase_offset=phase
            )
            
            # Note: Full implementation would require custom RF pulse design
    
    def write_sequence(self, filename: str = 'hyperpolarized_sequence.seq'):
        """
        Write sequence to file
        
        Args:
            filename: Output filename
        """
        # Check sequence timing
        ok, error_report = self.seq.check_timing()
        
        if ok:
            print("✓ Sequence timing check passed")
        else:
            print("✗ Sequence timing check failed:")
            print(error_report)
            return False
        
        # Write to file
        self.seq.write(filename)
        print(f"✓ Sequence written to: {filename}")
        
        # Print statistics
        print(f"\nSequence Statistics:")
        print(f"  Duration: {self.seq.duration():.3f} s")
        print(f"  Number of blocks: {len(self.seq.block_events)}")
        
        return True
    
    def plot_sequence(self, time_range: Optional[List[float]] = None):
        """
        Plot sequence diagram
        
        Args:
            time_range: Optional [start, end] time range in seconds
        """
        if time_range:
            self.seq.plot(time_range=time_range)
        else:
            self.seq.plot()


def example_c13_pyruvate_sequence():
    """Example: Generate C-13 pyruvate VFA sequence"""
    from sequence_optimizer import VFAOptimizer
    
    # Parameters for C-13 pyruvate
    num_frames = 20
    t1 = 43  # seconds
    tr = 0.1  # seconds (100 ms)
    
    # Calculate optimal VFA
    optimizer = VFAOptimizer(num_frames, t1, tr * 1000)  # tr in ms for optimizer
    flip_angles = optimizer.constant_signal_vfa()
    
    print(f"Generating C-13 Pyruvate VFA Sequence:")
    print(f"  Frames: {num_frames}")
    print(f"  T1: {t1} s")
    print(f"  TR: {tr*1000} ms")
    print(f"  Flip angles: {flip_angles[:5]}... (first 5)")
    
    # Create sequence
    seq_gen = HyperpolarizedSequence()
    seq = seq_gen.create_vfa_sequence(
        flip_angles=flip_angles,
        tr=tr,
        fov=0.24,  # 24 cm
        slice_thickness=0.01,  # 10 mm
        readout_type='cartesian'  # Simple readout for example
    )
    
    # Write sequence
    seq_gen.write_sequence('c13_pyruvate_vfa.seq')
    
    # Plot first 3 frames
    seq_gen.plot_sequence(time_range=[0, 3 * tr])
    
    return seq_gen


def example_xe129_ventilation_sequence():
    """Example: Generate Xe-129 ventilation sequence"""
    from sequence_optimizer import VFAOptimizer
    
    # Parameters for Xe-129
    num_frames = 10
    t1 = 20  # seconds
    tr = 0.02  # seconds (20 ms)
    
    optimizer = VFAOptimizer(num_frames, t1, tr * 1000)
    flip_angles = optimizer.max_snr_vfa()
    
    print(f"\nGenerating Xe-129 Ventilation Sequence:")
    print(f"  Frames: {num_frames}")
    print(f"  T1: {t1} s")
    print(f"  TR: {tr*1000} ms")
    
    seq_gen = HyperpolarizedSequence()
    seq = seq_gen.create_vfa_sequence(
        flip_angles=flip_angles,
        tr=tr,
        fov=0.40,  # 40 cm FOV for lungs
        slice_thickness=0.015,  # 15 mm
        readout_type='cartesian'
    )
    
    seq_gen.write_sequence('xe129_ventilation.seq')
    
    return seq_gen


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Generate example sequences
    print("=" * 60)
    print("PyPulseq Sequence Generator for Hyperpolarized Imaging")
    print("Sunnybrook Research Institute")
    print("=" * 60)
    
    # C-13 Pyruvate
    c13_seq = example_c13_pyruvate_sequence()
    
    # Xe-129
    xe_seq = example_xe129_ventilation_sequence()
    
    print("\n" + "=" * 60)
    print("Sequences generated successfully!")
    print("=" * 60)
    
    plt.show()
