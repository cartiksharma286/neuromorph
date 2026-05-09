
import numpy as np
import os

try:
    import pypulseq as pp
    _PYPULSEQ_AVAILABLE = True
except Exception:
    pp = None
    _PYPULSEQ_AVAILABLE = False

def generate_seq_file(sequence_type, tr, te, flip_angle=90, matrix_size=128, fov=256e-3):
    """
    Generates a .seq file using pypulseq for the specified parameters.
    Returns the path to the generated file.
    """
    if not _PYPULSEQ_AVAILABLE:
        return None  # pypulseq unavailable due to dependency conflict
    system = pp.Opts(
        max_grad=32, 
        grad_unit='mT/m', 
        max_slew=130, 
        slew_unit='T/m/s', 
        rf_ringdown_time=100e-6, 
        rf_dead_time=100e-6, 
        adc_dead_time=10e-6
    )
    seq = pp.Sequence(system)

    if sequence_type == 'SE' or sequence_type == 'SpinEcho':
        # --- PREPARE SPIN ECHO ---
        rf_dur = 2e-3
        slice_thickness = 5e-3
        rf90, gz, _ = pp.make_sinc_pulse(flip_angle=90 * np.pi / 180, duration=rf_dur, 
                                        slice_thickness=slice_thickness, bandwidth=2000, 
                                        system=system, return_gz=True)
        
        rf180, gz180, _ = pp.make_sinc_pulse(flip_angle=180 * np.pi / 180, duration=rf_dur, 
                                          slice_thickness=slice_thickness, bandwidth=2000, 
                                          system=system, use='refocusing', return_gz=True)
        
        delta_k = 1 / fov
        gx = pp.make_trapezoid(channel='x', flat_area=matrix_size * delta_k, flat_time=6.4e-3, system=system)
        adc = pp.make_adc(num_samples=matrix_size, duration=gx.flat_time, delay=gx.rise_time, system=system)
        
        gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
        gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)
        
        # Simple timing check
        te_s = te / 1000.0
        tr_s = tr / 1000.0
        
        # Delay after 90 to 180
        delay_te1 = te_s / 2 - rf_dur / 2 - rf_dur / 2 - gz.fall_time - gz180.rise_time
        # Delay after 180 to readout
        delay_te2 = te_s / 2 - rf_dur / 2 - gx.rise_time - gx.flat_time / 2
        
        for i in range(matrix_size):
            seq.add_block(rf90, gz)
            seq.add_block(gz_reph, gx_pre)
            if delay_te1 > 0: seq.add_block(pp.make_delay(delay_te1))
            seq.add_block(rf180, gz180)
            if delay_te2 > 0: seq.add_block(pp.make_delay(delay_te2))
            
            phase_area = (i - matrix_size / 2) * delta_k
            gy_pre = pp.make_trapezoid(channel='y', area=phase_area, duration=2e-3, system=system)
            seq.add_block(gy_pre)
            seq.add_block(gx, adc)
            
            # TR delay
            if tr_s > te_s:
                seq.add_block(pp.make_delay(tr_s - te_s))

    elif sequence_type == 'GRE' or sequence_type == 'GradientEcho':
        # --- PREPARE GRADIENT ECHO ---
        rf_dur = 1e-3
        rf, gz, _ = pp.make_sinc_pulse(flip_angle=flip_angle * np.pi / 180, duration=rf_dur, 
                                      slice_thickness=5e-3, bandwidth=2000, 
                                      system=system, return_gz=True)
        
        delta_k = 1 / fov
        gx = pp.make_trapezoid(channel='x', flat_area=matrix_size * delta_k, flat_time=4e-3, system=system)
        adc = pp.make_adc(num_samples=matrix_size, duration=gx.flat_time, delay=gx.rise_time, system=system)
        
        gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
        gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)
        
        te_s = te / 1000.0
        tr_s = tr / 1000.0
        
        delay_te = te_s - rf_dur / 2 - gx.rise_time - gx.flat_time / 2
        
        for i in range(matrix_size):
            seq.add_block(rf, gz)
            seq.add_block(gx_pre, gz_reph)
            if delay_te > 0: seq.add_block(pp.make_delay(delay_te))
            
            phase_area = (i - matrix_size / 2) * delta_k
            gy_pre = pp.make_trapezoid(channel='y', area=phase_area, duration=2e-3, system=system)
            seq.add_block(gy_pre)
            seq.add_block(gx, adc)
            
            if tr_s > te_s:
                seq.add_block(pp.make_delay(tr_s - te_s))

    else:
        # Fallback for complex sequences - generate a placeholder
        seq.add_block(pp.make_delay(tr/1000.0))

    filename = f"{sequence_type}_{tr}_{te}.seq"
    output_path = os.path.join(os.getcwd(), filename)
    seq.write(output_path)
    return output_path
