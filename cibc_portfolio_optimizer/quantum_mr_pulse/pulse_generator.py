import numpy as np
from quantum_integrals import QuantumIntegrals

class PulseGenerator:
    def __init__(self):
        self.gamma = 42.58 * 1e6  # Hz/T for Hydrogen
        self.quantum_integrals = QuantumIntegrals()

    def sinc_pulse(self, duration_ms, flip_angle_deg, points=100):
        t = np.linspace(-duration_ms/2, duration_ms/2, points)
        # Simplified sinc shape
        # Area under B1 should correspond to flip angle
        # Flip Angle = gamma * integral(B1) dt
        # For simplicity, we create a shape and normalize it
        sinc = np.sinc(t)
        current_area = np.trapz(sinc, dx=duration_ms/points * 1e-3)
        target_area = (flip_angle_deg * np.pi / 180) / (2 * np.pi * self.gamma)
        
        amplitude = target_area / current_area if current_area != 0 else 0
        b1 = sinc * amplitude
        return t, b1

    def generate_gre(self, te_ms, tr_ms, flip_angle_deg, fov_mm, matrix_size, optimize=False):
        if optimize:
            return self.optimize_gre(te_ms, tr_ms, flip_angle_deg, fov_mm, matrix_size)

        dt = 0.01 # ms time step
        total_time = tr_ms
        time = np.arange(0, total_time, dt)
        
        # Initialize channels
        rf = np.zeros_like(time)
        gx = np.zeros_like(time)
        gy = np.zeros_like(time)
        gz = np.zeros_like(time)
        adc = np.zeros_like(time)
        
        # 1. RF Excitation
        rf_duration = 2.0 # ms
        rf_t, rf_shape = self.sinc_pulse(rf_duration, flip_angle_deg, int(rf_duration/dt))
        
        rf_start_idx = 10
        rf_end_idx = rf_start_idx + len(rf_shape)
        if rf_end_idx < len(rf):
            rf[rf_start_idx:rf_end_idx] = rf_shape
            
            # Slice Select Gradient (Gz) during RF
            gz[rf_start_idx:rf_end_idx] = 0.5 # mT/m
            
        # 2. Phase Encoding (Gy) & Readout Pre-phasing (Gx)
        enc_start = rf_end_idx + 10
        enc_dur = 2.0 # ms
        enc_end = enc_start + int(enc_dur/dt)
        
        if enc_end < len(gy):
            gy[enc_start:enc_end] = 0.2 # Simulated phase step
            gx[enc_start:enc_end] = -0.5 # Pre-phasing
            
        # 3. Readout (Gx) and ADC
        ro_start = int(te_ms / dt) - int(2.0/dt) # Center around TE
        ro_dur = 4.0 # ms
        ro_end = ro_start + int(ro_dur/dt)
        
        if ro_start > enc_end and ro_end < len(gx):
            gx[ro_start:ro_end] = 0.5 # Readout gradient
            adc[ro_start:ro_end] = 1.0 # ADC ON
            
        # 4. Spoiler
        sp_start = ro_end + 10
        sp_dur = 1.0
        sp_end = sp_start + int(sp_dur/dt)
        if sp_end < len(gx):
            gx[sp_start:sp_end] = 0.8
            gy[sp_start:sp_end] = 0.8
            gz[sp_start:sp_end] = 0.8

        return {
            "time": time.tolist(),
            "rf": rf.tolist(),
            "gx": gx.tolist(),
            "gy": gy.tolist(),
            "gz": gz.tolist(),
            "adc": adc.tolist(),
            "metadata": {
                "te": te_ms,
                "tr": tr_ms,
                "type": "GRE"
            }
        }

    def optimize_gre(self, te_ms, tr_ms, flip_angle_deg, fov_mm, matrix_size):
        """
        Optimizes GRE parameters using Quantum Surface Integrals.
        Varies flip angle to maximize the surface integral metric.
        """
        best_metric = -np.inf
        best_params = None
        best_sequence = None
        
        # Grid search around the provided flip angle
        search_range = np.linspace(max(5, flip_angle_deg - 20), min(90, flip_angle_deg + 20), 10)
        
        for angle in search_range:
            # Generate temporary sequence
            seq = self.generate_gre(te_ms, tr_ms, angle, fov_mm, matrix_size, optimize=False)
            
            # Calculate Quantum Metric
            metrics = self.quantum_integrals.calculate_surface_integral(seq)
            metric_val = metrics['surface_integral']
            
            if metric_val > best_metric:
                best_metric = metric_val
                best_sequence = seq
                best_params = angle
                
        # Attach optimization metadata
        best_sequence['optimization_metadata'] = {
            'optimized_parameter': 'flip_angle',
            'optimal_value': best_params,
            'metric_value': best_metric,
            'method': 'Quantum Surface Integral Maximization'
        }
        return best_sequence

    def generate_se(self, te_ms, tr_ms, fov_mm, matrix_size, optimize=False):
        if optimize:
            return self.optimize_se(te_ms, tr_ms, fov_mm, matrix_size)

        # Spin Echo implementation
        dt = 0.01 # ms
        total_time = tr_ms
        time = np.arange(0, total_time, dt)
        
        rf = np.zeros_like(time)
        gx = np.zeros_like(time)
        gy = np.zeros_like(time)
        gz = np.zeros_like(time)
        adc = np.zeros_like(time)
        
        # 90 degree pulse
        rf_dur = 2.0
        rf90_t, rf90 = self.sinc_pulse(rf_dur, 90, int(rf_dur/dt))
        start90 = 10
        end90 = start90 + len(rf90)
        rf[start90:end90] = rf90
        gz[start90:end90] = 0.5
        
        # 180 degree pulse at TE/2
        te_2 = te_ms / 2
        start180 = int(te_2/dt) - int(rf_dur/2/dt)
        rf180_t, rf180 = self.sinc_pulse(rf_dur, 180, int(rf_dur/dt))
        end180 = start180 + len(rf180)
        
        if end180 < len(rf):
            rf[start180:end180] = rf180
            gz[start180:end180] = 0.5
            
        # Readout at TE
        ro_dur = 4.0
        start_ro = int(te_ms/dt) - int(ro_dur/2/dt)
        end_ro = start_ro + int(ro_dur/dt)
        
        if end_ro < len(adc):
            gx[start_ro:end_ro] = 0.5
            adc[start_ro:end_ro] = 1.0
            
        return {
            "time": time.tolist(),
            "rf": rf.tolist(),
            "gx": gx.tolist(),
            "gy": gy.tolist(),
            "gz": gz.tolist(),
            "adc": adc.tolist(),
            "metadata": {
                "te": te_ms,
                "tr": tr_ms,
                "type": "SE"
            }
        }

    def optimize_se(self, te_ms, tr_ms, fov_mm, matrix_size):
        """
        Optimizes SE parameters.
        Varies TE to maximize coherence metric.
        """
        best_metric = -np.inf
        best_sequence = None
        best_te = None

        # Search optimal TE around the target
        te_search = np.linspace(max(10, te_ms - 10), te_ms + 10, 5)

        for te in te_search:
            if te >= tr_ms: continue
            
            # Generate sequence
            seq = self.generate_se(te, tr_ms, fov_mm, matrix_size, optimize=False)
            
            # Calculate Metric
            metrics = self.quantum_integrals.calculate_surface_integral(seq)
            metric_val = metrics['coherence_metric'] # Maximize coherence for SE
            
            if metric_val > best_metric:
                best_metric = metric_val
                best_sequence = seq
                best_te = te
                
        if best_sequence is None:
             # Fallback if optimization fails (e.g. constraints)
             return self.generate_se(te_ms, tr_ms, fov_mm, matrix_size, optimize=False)

        best_sequence['optimization_metadata'] = {
            'optimized_parameter': 'te',
            'optimal_value': best_te,
            'metric_value': best_metric,
            'method': 'Quantum Coherence Maximization'
        }
        return best_sequence

    def generate_prime_weighted_sequence(self, duration_ms, scaling_factor, base_rf_amp=1.0):
        """
        Generates a pulse sequence where RF events are triggered based on Prime Number distributions.
        Events occur when int(t * scaling_factor) is a Prime Number.
        """
        dt = 0.01
        time = np.arange(0, duration_ms, dt)
        
        # Initialize
        rf = np.zeros_like(time)
        gx = np.zeros_like(time)
        gy = np.zeros_like(time)
        gz = np.zeros_like(time)
        adc = np.zeros_like(time)
        
        # Pre-compute primes for speed using a simple sieve for the range needed
        max_val = int(duration_ms * scaling_factor) + 100
        # Simple Sieve
        is_prime = np.ones(max_val, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(np.sqrt(max_val)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
                
        # Generate Pulse Train
        for i, t in enumerate(time):
            val = int(t * scaling_factor)
            if val < max_val and is_prime[val]:
                # Trigger RF pulse - modulated by the "primality" stability
                # Using a Gaussian blip for the pulse
                rf[i] = base_rf_amp
                
                # Associated Gradient blip for spatial encoding of this "prime event"
                gx[i] = 0.5 * (1 if val % 4 == 1 else -1) # Alternating gradient based on prime residue
                
        return {
            "time": time.tolist(),
            "rf": rf.tolist(),
            "gx": gx.tolist(),
            "gy": gy.tolist(),
            "gz": gz.tolist(),
            "adc": adc.tolist(),
            "metadata": {
                "type": "PRIME_WEIGHTED",
                "scaling_factor": scaling_factor,
                "duration": duration_ms
            }
        }

    def export_to_seq(self, pulse_data, filename="sequence.seq"):
        """
        Exports the pulse data to a Pulseq (.seq) compatible text file.
        This is a simplified writer that mimics the Pulseq file structure.
        """
        lines = []
        lines.append("# Pulseq Sequence File")
        lines.append(f"# Created by Quantum Pulse Architect")
        lines.append("# FormatVersion 1.2.0")
        lines.append("")
        
        lines.append("[DEFINITIONS]")
        lines.append("FOV 200 200 5")
        lines.append("Name Quantum_Sequence_001")
        lines.append("")
        
        # In a real Pulseq file, we define shape blocks. 
        # Here we will write a simplified block-event structure for demonstration.
        lines.append("[BLOCKS]")
        lines.append("# Format: ID RF GX GY GZ ADC")
        
        # We downsample to avoid creating a massive text file for the demo
        time = pulse_data['time']
        rf = pulse_data['rf']
        gx = pulse_data['gx']
        gy = pulse_data['gy']
        gz = pulse_data['gz']
        adc = pulse_data['adc']
        
        step = 10 # Downsample factor
        for i in range(0, len(time), step):
            if rf[i] != 0 or gx[i] != 0 or gy[i] != 0 or gz[i] != 0 or adc[i] != 0:
                # Mock event IDs (1 = ON, 0 = OFF)
                rf_id = 1 if abs(rf[i]) > 0 else 0
                gx_id = 1 if abs(gx[i]) > 0 else 0
                gy_id = 1 if abs(gy[i]) > 0 else 0
                gz_id = 1 if abs(gz[i]) > 0 else 0
                adc_id = 1 if adc[i] > 0 else 0
                
                # Write a line for this time block if anything is happening
                if rf_id or gx_id or gy_id or gz_id or adc_id:
                    lines.append(f"{i//step:4d} {rf_id:2d} {gx_id:2d} {gy_id:2d} {gz_id:2d} {adc_id:2d}")
                    
        content = "\n".join(lines)
        return content
