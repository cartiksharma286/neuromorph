import numpy as np
from scipy.integrate import odeint

# Differential equations model for Aducanumab taking action on amyloid beta plaques
# A_beta = Amyloid Beta
# Drug = Aducanumab
# C = Immune Complex
def amyloid_kinetics(y, t, k_prod, k_deg, k_bind, k_clear, k_phago, dose_rate):
    A_beta, Drug, C = y
    
    # Differential equations
    dA_dt = k_prod - k_deg * A_beta - k_bind * A_beta * Drug
    dD_dt = dose_rate - k_clear * Drug - k_bind * A_beta * Drug
    dC_dt = k_bind * A_beta * Drug - k_phago * C
    
    return [dA_dt, dD_dt, dC_dt]

def simulate_treatment(days, start_plaque, dose_mg, affinity, clearance):
    t = np.linspace(0, days, int(days)*10)
    
    # Assume constants based on rough kinetic values for monoclonal antibodies
    k_prod = 0.5   # Amyloid production rate (mg/day)
    k_deg = 0.05   # Natural amyloid degradation (1/day)
    
    k_bind = affinity      # Binding affinity mapping
    k_clear = clearance    # Drug clearance rate mapping
    k_phago = 0.8          # Macrophage clearance of immune complexes
    
    # Starting conditions
    y0 = [start_plaque, 0.0, 0.0]
    
    # Solve ODE
    solution = odeint(amyloid_kinetics, y0, t, args=(k_prod, k_deg, k_bind, k_clear, k_phago, dose_mg / 30.0))
    
    return {
        "time": t.tolist(),
        "amyloid_beta": solution[:, 0].tolist(),
        "aducanumab": solution[:, 1].tolist(),
        "immune_complex": solution[:, 2].tolist()
    }
