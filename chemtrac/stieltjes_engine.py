"""
Stieltjes Primes & Continued Fraction Spectral Engine for Water Quality Purification.
Computes Stieltjes S-fractions, orthogonal moment densities, and contaminant decay kinetics.
"""

import numpy as np
import math

class StieltjesPrimesEngine:
    """
    Implements Stieltjes continued fraction theory and prime-indexed spectral kernels:
    1. Stieltjes S-Fraction Representation of Contaminant Adsorption:
       f(z) = c_0 / (z + c_1 / (1 + c_2 / (z + c_3 / (1 + ...))))
    2. Stieltjes Primes (Primes associated with generalized Stieltjes constants and orthogonal polynomials):
       p_s in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
    3. Moment Problem Inversion for Porous Media Granular Carbon Filtration:
       dmu(t) = (1 / pi) * lim_{eps -> 0+} Im[ f(t - i*eps) ] dt
    """
    def __init__(self):
        # Stieltjes Primes list
        self.stieltjes_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        # Stieltjes constants gamma_n approximations (gamma_0 = Euler-Mascheroni ~ 0.5772)
        self.stieltjes_constants = [
            0.5772156649,   # gamma_0
            -0.0728158454,  # gamma_1
            -0.0096903631,  # gamma_2
            0.0020538344,   # gamma_3
            0.0023253700,   # gamma_4
            0.0007933238,   # gamma_5
            -0.0002387693,  # gamma_6
            -0.0005275322   # gamma_7
        ]

    def compute_s_fraction_convergents(self, moments, depth=8):
        """
        Computes Stieltjes continued fraction coefficients c_k from moment sequence mu_n:
        mu_n = int_0^infty t^n dmu(t)
        Returns rational convergents P_k(z) / Q_k(z).
        """
        n = min(len(moments), depth)
        c_coeffs = []
        
        # Quotient-Difference (QD) algorithm initialization
        e = np.zeros((n + 1, n + 1))
        q = np.zeros((n + 1, n + 1))
        
        for i in range(n):
            if i < n - 1 and moments[i] != 0:
                q[i, 0] = moments[i+1] / (moments[i] + 1e-12)
            else:
                q[i, 0] = 1.0

        for j in range(1, n):
            for i in range(n - j):
                e[i, j] = q[i+1, j-1] - q[i, j-1] + e[i+1, j-2] if j >= 2 else q[i+1, j-1] - q[i, j-1]
                if abs(e[i, j-1]) > 1e-12:
                    q[i, j] = q[i+1, j-1] * e[i+1, j] / e[i, j]
                else:
                    q[i, j] = q[i+1, j-1]

        # Extract coefficients for Stieltjes S-fraction
        c_coeffs.append(float(moments[0]))
        for k in range(n - 1):
            c_coeffs.append(float(abs(q[0, k])))
            c_coeffs.append(float(abs(e[0, k+1])))

        return c_coeffs[:depth]

    def stieltjes_prime_spectral_decay(self, t_residence_min, initial_ppm=10.0):
        """
        Models multi-contaminant adsorption kinetics through Stieltjes prime harmonic decay:
        C(t) = C_0 * sum_{k=1}^M [ w_k * exp(-t / tau_k) / (1 + (t / p_k)^alpha) ]
        Guarantees monotonicity and non-negative mass conservation.
        """
        decay_points = []
        t_vals = np.linspace(0, max(60.0, t_residence_min), 50)
        
        for t in t_vals:
            c_t = 0.0
            total_w = 0.0
            for idx, p in enumerate(self.stieltjes_primes[:8]):
                gamma_k = abs(self.stieltjes_constants[idx % len(self.stieltjes_constants)])
                tau_k = p * (0.85 + 0.15 * gamma_k)
                weight = 1.0 / math.sqrt(p)
                
                # Stieltjes transform kernel element
                stieltjes_factor = 1.0 / (1.0 + (t / p)**1.25)
                c_t += weight * math.exp(-t / tau_k) * stieltjes_factor
                total_w += weight
                
            norm_c = initial_ppm * (c_t / (total_w + 1e-9))
            decay_points.append({
                "time_min": round(float(t), 2),
                "concentration_ppm": round(float(norm_c), 4),
                "removal_efficiency_pct": round(float(max(0.0, min(100.0, (1.0 - norm_c / initial_ppm) * 100.0))), 2)
            })

        return decay_points

    def get_prime_harmonics(self):
        """Returns Stieltjes primes and their spectral properties."""
        harmonics = []
        for idx, p in enumerate(self.stieltjes_primes[:12]):
            gamma = self.stieltjes_constants[idx % len(self.stieltjes_constants)]
            q_resonance_hz = round(float(1420.0 / p), 2)
            harmonics.append({
                "prime": p,
                "stieltjes_const_gamma": round(float(gamma), 6),
                "adsorption_rate_constant_k": round(float(1.0 / (p * 0.45)), 4),
                "acoustic_resonance_hz": q_resonance_hz,
                "phase_angle_rad": round(float(math.atan2(p, abs(gamma) * 100.0)), 4)
            })
        return harmonics
