import numpy as np

class PrimeGeodesicSolver:
    def __init__(self):
        self.primes = self._sieve_of_eratosthenes(1000)
    
    def _sieve_of_eratosthenes(self, n):
        """Generate primes up to n."""
        primes = []
        is_prime = [True] * (n + 1)
        for p in range(2, n + 1):
            if is_prime[p]:
                primes.append(p)
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        return primes

    def calculate_geodesics(self, pulse_data):
        """
        Maps the pulse sequence onto a 'Prime Geodesic' manifold.
        The metric tensor depends on the local density of prime numbers relative to the
        sequence amplitude.
        """
        rf = np.array(pulse_data['rf'])
        time = np.array(pulse_data['time'])
        
        # 1. Prime Potential Field
        # We assume the 'energy' of the state is modulated by a potential V(x) 
        # driven by the distribution of primes (Riemann Zeta zeros proxy).
        # We map time steps to integers and check 'primality' or distance to nearest prime.
        
        potential = []
        geodesic_x = []
        geodesic_y = []
        geodesic_z = []
        
        # Initial Quantum State on Bloch Sphere
        state = np.array([0.0, 0.0, 1.0])
        
        for i, t_val in enumerate(time):
            # Scale time to integer space for prime check (mock)
            t_int = int(t_val * 100) 
            
            # Simple Prime Potential: 1/log(n) density proxy or distance to nearest prime
            # For visualization, we use distance to nearest prime as a perturbation
            dist = min([abs(t_int - p) for p in self.primes])
            prime_factor = np.exp(-dist / 5.0) # Stronger if close to a prime
            
            potential.append(prime_factor)
            
            # 2. Geodesic Equation (Simplified)
            # d2x/dt2 + Gamma * (dx/dt)^2 = 0
            # Here we simulate the trajectory being 'pulled' by prime-rich regions
            # Perturb the RF rotation axis based on prime_factor
            
            rf_amp = rf[i]
            
            # Standard rotation
            dt = 0.01 * 1e-3
            gamma = 42.58 * 1e6 * 2 * np.pi
            
            # Perturbation: localized phase shift due to "Prime Curvature"
            prime_phase = prime_factor * 0.1 * np.pi 
            
            # Rotation Matrix
            # Axis is slightly tilted by prime_phase
            alpha = gamma * rf_amp * dt
            
            # X-rotation with slight Z-tilt
            cx, sx = np.cos(alpha), np.sin(alpha)
            
            # New state update (Geodesic flow)
            # M = R_x(alpha) * M
            # + Prime Curvature drift
            
            # Rotate around X
            new_y = state[1] * cx - state[2] * sx
            new_z = state[1] * sx + state[2] * cx
            state[1] = new_y
            state[2] = new_z
            
            # Prime Drift (Projection State collapse probability)
            # If we are exactly on a prime, we have a higher chance of 'projection'
            # represented here as a slight decoherence or jump
            if dist == 0 and i % 5 == 0:
                 state[0] += 0.05 # Mock jump
            
            # Renormalize (Unitary evolution constraint, but geodesics might leave the sphere in some metrics)
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm
            
            geodesic_x.append(state[0])
            geodesic_y.append(state[1])
            geodesic_z.append(state[2])
            
        return {
            "prime_potential": potential,
            "geodesic_trajectory": {
                "x": geodesic_x[::10],
                "y": geodesic_y[::10],
                "z": geodesic_z[::10]
            },
            "projection_states": [
                {"time": t, "probability": p} 
                for t, p in zip(time[::50], potential[::50]) if p > 0.5
            ]
        }
