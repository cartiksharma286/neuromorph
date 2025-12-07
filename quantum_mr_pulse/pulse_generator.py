import numpy as np

class PulseGenerator:
    def __init__(self):
        self.gamma = 42.58 * 1e6  # Hz/T for Hydrogen

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

    def generate_gre(self, te_ms, tr_ms, flip_angle_deg, fov_mm, matrix_size):
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
            "adc": adc.tolist()
        }

    def generate_se(self, te_ms, tr_ms, fov_mm, matrix_size):
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
            "adc": adc.tolist()
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
