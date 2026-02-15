"""
MRI Pulse Sequence Generation Module

Implements various MRI pulse sequences with quantum ML optimization support
and PyPulseq export capability.

Sequences implemented:
- GradientEcho (GRE): T1-weighted, T2*-weighted imaging
- SpinEcho (SE): T2-weighted imaging
- EchoPlanarImaging (EPI): Rapid imaging for fMRI
- fMRISequence: BOLD-sensitive functional MRI
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings

# PyPulseq for MRI sequence generation
try:
    import pypulseq as pp
    from pypulseq.Sequence.sequence import Sequence
    PYPULSEQ_AVAILABLE = True
except ImportError:
    PYPULSEQ_AVAILABLE = False
    warnings.warn("PyPulseq not available. Export functionality limited.")


class PulseSequenceBase(ABC):
    """
    Abstract base class for all MRI pulse sequences.
    
    Provides common functionality for sequence generation, parameter validation,
    k-space trajectory calculation, and PyPulseq export.
    """
    
    def __init__(self, system_specs: Optional[Dict] = None):
        """
        Initialize pulse sequence.
        
        Args:
            system_specs: MRI system specifications (gradient limits, RF limits, etc.)
        """
        self.system_specs = system_specs or self._default_system_specs()
        
        if PYPULSEQ_available:
            self.seq = Sequence(system=pp.Opts(
                max_grad=self.system_specs['max_grad'],
                grad_unit='mT/m',
                max_slew=self.system_specs['max_slew'],
                slew_unit='T/m/s',
                rf_ringdown_time=20e-6,
                rf_dead_time=100e-6,
                adc_dead_time=10e-6
            ))
        else:
            self.seq = None
            
        self.params = {}
        self.k_space_trajectory = None
        
    def _default_system_specs(self) -> Dict:
        """Default MRI system specifications."""
        return {
            'max_grad': 40,  # mT/m
            'max_slew': 170,  # T/m/s
            'rf_dead_time': 100e-6,  # s
            'rf_ringdown_time': 20e-6,  # s
            'adc_dead_time': 10e-6,  # s
            'grad_raster_time': 10e-6,  # s
            'rf_raster_time': 1e-6,  # s
        }
    
    @abstractmethod
    def generate(self, params: Dict) -> 'PulseSequenceBase':
        """
        Generate the pulse sequence with given parameters.
        
        Args:
            params: Sequence parameters (TE, TR, FA, etc.)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def calculate_k_space(self) -> np.ndarray:
        """
        Calculate k-space trajectory for this sequence.
        
        Returns:
            K-space trajectory array (n_points x 3) for kx, ky, kz
        """
        pass
    
    def export_pypulseq(self, filename: str):
        """
        Export sequence to PyPulseq .seq file.
        
        Args:
            filename: Output filename
        """
        if not PYPULSEQ_AVAILABLE or self.seq is None:
            print(f"Warning: PyPulseq not available, saving parameters only")
            self._export_params_json(filename.replace('.seq', '.json'))
            return
        
        self.seq.write(filename)
        print(f"Sequence exported to {filename}")
    
    def _export_params_json(self, filename: str):
        """Export parameters as JSON (fallback)."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.params, f, indent=2)
        print(f"Parameters exported to {filename}")
    
    def validate_params(self, params: Dict) -> bool:
        """Validate sequence parameters."""
        required_keys = self.get_required_params()
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        return True


class GradientEcho(PulseSequenceBase):
    """
    Gradient Recalled Echo (GRE) sequence.
    
    A fundamental MRI sequence using gradient  reversal for echo formation.
    Provides T1 or T2* weighting depending on parameters.
    """
    
    def get_required_params(self) -> List[str]:
        return ['TE', 'TR', 'FA', 'slice_thickness', 'fov', 'matrix_size']
    
    def generate(self, params: Dict) -> 'GradientEcho':
        """
        Generate GRE sequence.
        
        Args:
            params: Dictionary with keys:
                - TE: Echo time (ms)
                - TR: Repetition time (ms)
                - FA: Flip angle (degrees)
                - slice_thickness: Slice thickness (mm)
                - fov: Field of view (mm)
                - matrix_size: Matrix size (e.g., 256)
                
        Returns:
            Self for method chaining
        """
        self.validate_params(params)
        self.params = params
        
        # Convert units
        TE = params['TE'] * 1e-3  # ms to s
        TR = params['TR'] * 1e-3
        FA = params['FA']  # degrees
        slice_thickness = params['slice_thickness'] * 1e-3  # mm to m
        fov = params['fov'] * 1e-3  # mm to m
        Nx = params['matrix_size']
        Ny = params.get('matrix_size_y', Nx)
        
        if not PYPULSEQ_AVAILABLE:
            print(f"GRE sequence configured: TE={params['TE']}ms, TR={params['TR']}ms, FA={FA}°")
            return self
        
        # Create RF pulse (slice-selective excitation)
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=FA * np.pi / 180,
            duration=3e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.seq.system,
            return_gz=True
        )
        
        # Calculate readout gradient and ADC
        delta_k = 1 / fov
        k_width = Nx * delta_k
        readout_time = 6.4e-3  # 6.4 ms
        
        gx = pp.make_trapezoid(
            channel='x',
            flat_area=k_width,
            flat_time=readout_time,
            system=self.seq.system
        )
        
        adc = pp.make_adc(
            num_samples=Nx,
            duration=readout_time,
            delay=gx.rise_time,
            system=self.seq.system
        )
        
        # Phase encoding gradient
        phase_areas = (np.arange(Ny) - Ny / 2) * delta_k
        
        # Prephasing gradients
        gx_pre = pp.make_trapezoid(
            channel='x',
            area=-gx.area / 2,
            system=self.seq.system
        )
        
        # Timing calculations
        TE_delay = TE - pp.calc_duration(gzr) - pp.calc_duration(gx) / 2
        TR_delay = TR - pp.calc_duration(gz) - pp.calc_duration(gzr) - \
                   pp.calc_duration(gx) - TE_delay
        
        # Build sequence
        for i in range(Ny):
            # RF excitation
            self.seq.add_block(rf, gz)
            
            # Slice rephasing
            self.seq.add_block(gzr)
            
            # Phase encoding
            gy_pre = pp.make_trapezoid(
                channel='y',
                area=phase_areas[i],
                system=self.seq.system
            )
            self.seq.add_block(gx_pre, gy_pre)
            
            # TE delay
            if TE_delay > 0:
                self.seq.add_block(pp.make_delay(TE_delay))
            
            # Readout
            self.seq.add_block(gx, adc)
            
            # TR delay
            if TR_delay > 0:
                self.seq.add_block(pp.make_delay(TR_delay))
        
        print(f"GRE sequence generated: {Ny} phase encoding steps")
        return self
    
    def calculate_k_space(self) -> np.ndarray:
        """Calculate k-space trajectory for GRE (Cartesian)."""
        Nx = self.params['matrix_size']
        Ny = self.params.get('matrix_size_y', Nx)
        fov = self.params['fov'] * 1e-3
        
        # Cartesian grid
        kx = np.linspace(-0.5/fov, 0.5/fov, Nx)
        ky = np.linspace(-0.5/fov, 0.5/fov, Ny)
        
        # Create trajectory (raster scan)
        k_traj = []
        for ky_val in ky:
            for kx_val in kx:
                k_traj.append([kx_val, ky_val, 0])
        
        self.k_space_trajectory = np.array(k_traj)
        return self.k_space_trajectory


class SpinEcho(PulseSequenceBase):
    """
    Spin Echo (SE) sequence.
    
    Uses 90° excitation followed by 180° refocusing pulse.
    Provides true T2 weighting with reduced susceptibility artifacts.
    """
    
    def get_required_params(self) -> List[str]:
        return ['TE', 'TR', 'slice_thickness', 'fov', 'matrix_size']
    
    def generate(self, params: Dict) -> 'SpinEcho':
        """Generate spin echo sequence."""
        self.validate_params(params)
        self.params = params
        
        TE = params['TE'] * 1e-3
        TR = params['TR'] * 1e-3
        slice_thickness = params['slice_thickness'] * 1e-3
        fov = params['fov'] * 1e-3
        Nx = params['matrix_size']
        Ny = params.get('matrix_size_y', Nx)
        
        if not PYPULSEQ_AVAILABLE:
            print(f"SE sequence configured: TE={params['TE']}ms, TR={params['TR']}ms")
            return self
        
        # 90° excitation pulse
        rf90, gz90, gzr90 = pp.make_sinc_pulse(
            flip_angle=90 * np.pi / 180,
            duration=3e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.seq.system,
            return_gz=True
        )
        
        # 180° refocusing pulse
        rf180, gz180, gzr180 = pp.make_sinc_pulse(
            flip_angle=180 * np.pi / 180,
            duration=5e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.seq.system,
            return_gz=True,
            use='refocusing'
        )
        
        # Readout gradient and ADC
        delta_k = 1 / fov
        k_width = Nx * delta_k
        readout_time = 6.4e-3
        
        gx = pp.make_trapezoid(
            channel='x',
            flat_area=k_width,
            flat_time=readout_time,
            system=self.seq.system
        )
        
        adc = pp.make_adc(
            num_samples=Nx,
            duration=readout_time,
            delay=gx.rise_time,
            system=self.seq.system
        )
        
        # Phase encoding
        phase_areas = (np.arange(Ny) - Ny / 2) * delta_k
        
        # Build sequence for each phase encode
        for i in range(Ny):
            # 90° excitation
            self.seq.add_block(rf90, gz90)
            self.seq.add_block(gzr90)
            
            # Phase encoding
            gy = pp.make_trapezoid(
                channel='y',
                area=phase_areas[i],
                system=self.seq.system
            )
            self.seq.add_block(gy)
            
            # Delay to TE/2
            delay1 = TE/2 - pp.calc_duration(gz90) - pp.calc_duration(gzr90) - \
                    pp.calc_duration(gy) - pp.calc_duration(gz180)/2
            if delay1 > 0:
                self.seq.add_block(pp.make_delay(delay1))
            
            # 180° refocusing
            self.seq.add_block(rf180, gz180)
            self.seq.add_block(gzr180)
            
            # Delay to TE
            delay2 = TE/2 - pp.calc_duration(gz180)/2 - pp.calc_duration(gzr180) - \
                    pp.calc_duration(gx)/2
            if delay2 > 0:
                self.seq.add_block(pp.make_delay(delay2))
            
            # Readout
            self.seq.add_block(gx, adc)
            
            # TR delay
            TR_delay = TR - TE - pp.calc_duration(gx)/2
            if TR_delay > 0:
                self.seq.add_block(pp.make_delay(TR_delay))
        
        print(f"SE sequence generated: {Ny} phase encoding steps")
        return self
    
    def calculate_k_space(self) -> np.ndarray:
        """Calculate k-space trajectory for SE (Cartesian)."""
        return GradientEcho.calculate_k_space(self)


class EchoPlanarImaging(PulseSequenceBase):
    """
    Echo Planar Imaging (EPI) sequence.
    
    Rapid single-shot imaging sequence that acquires all k-space data
    after a single excitation. Used for fMRI and diffusion imaging.
    """
    
    def get_required_params(self) -> List[str]:
        return ['TE', 'TR', 'FA', 'slice_thickness', 'fov', 'matrix_size']
    
    def generate(self, params: Dict) -> 'EchoPlanarImaging':
        """Generate EPI sequence."""
        self.validate_params(params)
        self.params = params
        
        TE = params['TE'] * 1e-3
        TR = params['TR'] * 1e-3
        FA = params['FA']
        slice_thickness = params['slice_thickness'] * 1e-3
        fov = params['fov'] * 1e-3
        Nx = params['matrix_size']
        Ny = params.get('matrix_size_y', Nx)
        
        if not PYPULSEQ_AVAILABLE:
            print(f"EPI sequence configured: TE={params['TE']}ms, TR={params['TR']}ms, FA={FA}°")
            return self
        
        # RF excitation
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=FA * np.pi / 180,
            duration=3e-3,
            slice_thickness=slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.seq.system,
            return_gz=True
        )
        
        # EPI readout train
        delta_k = 1 / fov
        k_width = Nx * delta_k
        dwell_time = 4e-6  # 4 µs
        readout_time = Nx * dwell_time
        
        # Readout gradient (bipolar for EPI)
        gx = pp.make_trapezoid(
            channel='x',
            flat_area=k_width,
            flat_time=readout_time,
            system=self.seq.system
        )
        
        # Phase encoding blip
        gy_blip = pp.make_trapezoid(
            channel='y',
            area=delta_k,
            system=self.seq.system
        )
        
        # Prephasing gradients
        gx_pre = pp.make_trapezoid(
            channel='x',
            area=-k_width/2,
            system=self.seq.system
        )
        
        gy_pre = pp.make_trapezoid(
            channel='y',
            area=-Ny/2 * delta_k,
            system=self.seq.system
        )
        
        # Build EPI readout
        self.seq.add_block(rf, gz)
        self.seq.add_block(gzr)
        self.seq.add_block(gx_pre, gy_pre)
        
        # EPI readout train
        for i in range(Ny):
            # Readout
            adc = pp.make_adc(
                num_samples=Nx,
                dwell=dwell_time,
                delay=gx.rise_time,
                system=self.seq.system
            )
            
            if i % 2 == 0:
                # Positive readout
                self.seq.add_block(gx, adc)
            else:
                # Negative readout (flyback)
                gx_neg = pp.scale_grad(gx, -1)
                self.seq.add_block(gx_neg, adc)
            
            # Phase encoding blip (except last line)
            if i < Ny - 1:
                self.seq.add_block(gy_blip)
        
        # TR delay
        seq_duration = pp.calc_duration(self.seq)
        TR_delay = TR - seq_duration
        if TR_delay > 0:
            self.seq.add_block(pp.make_delay(TR_delay))
        
        print(f"EPI sequence generated: {Nx}x{Ny} single-shot acquisition")
        return self
    
    def calculate_k_space(self) -> np.ndarray:
        """Calculate EPI k-space trajectory (zigzag pattern)."""
        Nx = self.params['matrix_size']
        Ny = self.params.get('matrix_size_y', Nx)
        fov = self.params['fov'] * 1e-3
        
        kx = np.linspace(-0.5/fov, 0.5/fov, Nx)
        ky = np.linspace(-0.5/fov, 0.5/fov, Ny)
        
        k_traj = []
        for i, ky_val in enumerate(ky):
            if i % 2 == 0:
                # Forward readout
                for kx_val in kx:
                    k_traj.append([kx_val, ky_val, 0])
            else:
                # Reverse readout
                for kx_val in reversed(kx):
                    k_traj.append([kx_val, ky_val, 0])
        
        self.k_space_trajectory = np.array(k_traj)
        return self.k_space_trajectory


class fMRISequence(EchoPlanarImaging):
    """
    Functional MRI sequence.
    
    Extended EPI sequence optimized for BOLD contrast and temporal resolution.
    Includes multi-slice acquisition and dynamic time series.
    """
    
    def get_required_params(self) -> List[str]:
        base_params = super().get_required_params()
        base_params.extend(['n_slices', 'n_dynamics', 'slice_gap'])
        return base_params
    
    def generate(self, params: Dict) -> 'fMRISequence':
        """Generate fMRI sequence with multi-slice and dynamics."""
        self.validate_params(params)
        self.params = params
        
        n_slices = params['n_slices']
        n_dynamics = params['n_dynamics']
        
        print(f"Generating fMRI sequence: {n_slices} slices, {n_dynamics} dynamics")
        
        # Generate base EPI for each slice
        for dynamic in range(n_dynamics):
            for slice_idx in range(n_slices):
                # Each slice is an EPI readout
                super().generate(params)
                
                if dynamic == 0 and slice_idx == 0:
                    print(f"  First dynamic, slice {slice_idx+1}/{n_slices} generated")
        
        total_time = params['TR'] * n_slices * n_dynamics / 1000  # in seconds
        print(f"fMRI sequence complete: Total scan time = {total_time:.1f}s")
        
        return self


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("MRI Pulse Sequence Generator")
    print("=" * 70)
    print(f"PyPulseq Available: {PYPULSEQ_AVAILABLE}")
    print()
    
    # Test GRE sequence
    print("Testing Gradient Echo (GRE) Sequence...")
    gre = GradientEcho()
    gre_params = {
        'TE': 10,  # ms
        'TR': 500,  # ms
        'FA': 30,  # degrees
        'slice_thickness': 3,  # mm
        'fov': 256,  # mm
        'matrix_size': 256
    }
    gre.generate(gre_params)
    k_space_gre = gre.calculate_k_space()
    print(f"  K-space points: {k_space_gre.shape[0]}")
    print()
    
    # Test Spin Echo sequence
    print("Testing Spin Echo (SE) Sequence...")
    se = SpinEcho()
    se_params = {
        'TE': 80,  # ms
        'TR': 2000,  # ms
        'slice_thickness': 3,  # mm
        'fov': 256,  # mm
        'matrix_size': 256
    }
    se.generate(se_params)
    print()
    
    # Test EPI sequence
    print("Testing Echo Planar Imaging (EPI) Sequence...")
    epi = EchoPlanarImaging()
    epi_params = {
        'TE': 30,  # ms
        'TR': 2000,  # ms
        'FA': 90,  # degrees
        'slice_thickness': 3,  # mm
        'fov': 192,  # mm
        'matrix_size': 64
    }
    epi.generate(epi_params)
    k_space_epi = epi.calculate_k_space()
    print(f"  K-space points: {k_space_epi.shape[0]}")
    print()
    
    # Test fMRI sequence
    print("Testing fMRI Sequence...")
    fmri = fMRISequence()
    fmri_params = {
        'TE': 30,  # ms
        'TR': 2000,  # ms
        'FA': 90,  # degrees
        'slice_thickness': 3,  # mm
        'fov': 192,  # mm
        'matrix_size': 64,
        'n_slices': 20,
        'n_dynamics': 100,
        'slice_gap': 0.3  # mm
    }
    fmri.generate(fmri_params)
    print()
    
    print("Pulse sequence generation complete!")
