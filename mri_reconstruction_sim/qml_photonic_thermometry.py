import numpy as np

def run_qml_thermometry_sim(sequence_name, coil_name, matrix_size=256):
    x = np.linspace(-1, 1, matrix_size)
    y = np.linspace(-1, 1, matrix_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    base_temp = 37.0 + 8.0 * np.exp(-R**2 / 0.1)
    
    if 'qubit' in coil_name.lower():
        base_temp += 2.0 * np.sin(5*R)
        
    if 'thermometry_v1' in sequence_name.lower():
        snr = 85.0
        contrast = base_temp * 1.2
    elif 'thermometry_v2' in sequence_name.lower():
        snr = 110.0
        contrast = base_temp * 1.5
    else:
        snr = 50.0
        contrast = base_temp

    return {
        'max_temp': float(np.max(contrast)),
        'mean_temp': float(np.mean(contrast)),
        'snr': float(snr),
        'status': 'Quantum Photonic Circuitry Active'
    }
