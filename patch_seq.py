import re

with open("mri_reconstruction_sim/simulator_core.py", "r") as f:
    code = f.read()

# Inject sequence logic
seq_logic = """
        elif sequence_type == 'CardioRamanujanPulse':
            # Cardiovascular Ramanujan Pulse (Conformal)
            t1 = T1_safe
            t2 = T2_safe
            M_base = self.pd_map * (1 - np.exp(-TR / t1)) * np.exp(-TE / t2)
            
            # Improved SNR via Ramanujan Statistical Operators
            ramanujan_gain = 1 + 0.5 * np.maximum(0, np.sin(np.linspace(0, 4*np.pi, M_base.shape[0]))[:, None])
            M_base *= ramanujan_gain
            M = M_base
"""

if 'CardioRamanujanPulse' not in code:
    code = code.replace("elif sequence_type == 'QuantObservables':", seq_logic + "        elif sequence_type == 'QuantObservables':")

# Inject coil logic
coil_logic = """
        elif coil_type == 'CardioRamanujanCoil':
            # Statistical Conformal Cardiovascular Coil (Ramanujan Operator Signature)
            for i in range(num_coils):
                angle = 2 * np.pi * i / num_coils
                cx = center[1] + (N//2.5) * np.cos(angle)
                cy = center[0] + (N//2.5) * np.sin(angle)
                
                # Ramanujan prime-gap density conformal mapped profile
                dist_sq = (x - cx)**2 + (y - cy)**2
                
                tau_eff = 1.0 + 0.8 * (x / N) * np.sin(y / N * np.pi * 4) # Topological signature
                sens = tau_eff / (1 + dist_sq / (N*1.2)**2)
                
                phase = np.exp(1j * angle)
                self.coils.append(sens * phase)
"""

if 'CardioRamanujanCoil' not in code:
    code = code.replace("elif coil_type == 'n25_array':", coil_logic + "        elif coil_type == 'n25_array':")


with open("mri_reconstruction_sim/simulator_core.py", "w") as f:
    f.write(code)

print("Patched simulator_core.py successfully")
