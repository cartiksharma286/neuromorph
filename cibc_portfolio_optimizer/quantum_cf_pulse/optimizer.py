import numpy as np
from scipy.optimize import minimize
from .solver import RiccatiSolver

class VariationalPulseOptimizer:
    def __init__(self, n_steps, dt, target_state):
        self.n_steps = n_steps
        self.dt = dt
        self.target_state = target_state
        self.solver = RiccatiSolver(dt)

    def cost_function(self, controls):
        """
        Cost function: 1 - Fidelity
        controls: flat array of [omega_x_0, ..., omega_x_N, omega_y_0, ..., omega_y_N]
        """
        omega_x = controls[:self.n_steps]
        omega_y = controls[self.n_steps:]
        
        # Initial state |0> (z-up)
        initial_state = np.array([1, 0], dtype=complex)
        
        # Evolve
        trajectory = self.solver.evolve(initial_state, omega_x, omega_y, delta=0.0)
        final_state = trajectory[-1]
        
        # Fidelity = |<target|final>|^2
        fidelity = np.abs(np.vdot(self.target_state, final_state))**2
        return 1.0 - fidelity

    def optimize(self, initial_guess=None):
        if initial_guess is None:
            initial_guess = np.zeros(2 * self.n_steps)
            
        result = minimize(self.cost_function, initial_guess, method='BFGS')
        
        optimized_controls = result.x
        omega_x = optimized_controls[:self.n_steps]
        omega_y = optimized_controls[self.n_steps:]
        
        return omega_x, omega_y, result.fun

class FermatPulseOptimizer:
    """
    Optimizes a pulse parameterized by Fermat Prime frequencies.
    Implements 'Ricci State' optimization by minimizing the geometric curvature of the trajectory.
    """
    def __init__(self, n_steps, dt, target_state):
        self.n_steps = n_steps
        self.dt = dt
        self.target_state = target_state
        self.solver = RiccatiSolver(dt)
        # Fermat Primes: 3, 5, 17, 257, 65537
        # We use the first 3 for practical modulation frequencies, scaled down.
        self.fermat_primes = [3, 5, 17, 257] 
        self.base_freq = 2 * np.pi * 0.1 # Base frequency scaling

    def generate_pulse_from_params(self, params):
        """
        Params: [Ax_0, ..., Ax_3, phix_0, ..., phix_3, Ay_0, ..., Ay_3, phiy_0, ..., phiy_3]
        """
        t = np.linspace(0, self.n_steps * self.dt, self.n_steps)
        n_freqs = len(self.fermat_primes)
        
        Ax = params[:n_freqs]
        phix = params[n_freqs:2*n_freqs]
        Ay = params[2*n_freqs:3*n_freqs]
        phiy = params[3*n_freqs:]
        
        omega_x = np.zeros_like(t)
        omega_y = np.zeros_like(t)
        
        for i, F in enumerate(self.fermat_primes):
            omega_x += Ax[i] * np.cos(F * self.base_freq * t + phix[i])
            omega_y += Ay[i] * np.cos(F * self.base_freq * t + phiy[i])
            
        return omega_x, omega_y

    def ricci_curvature_cost(self, trajectory):
        """
        Calculates a proxy for the Ricci curvature of the trajectory in Hilbert space.
        We approximate this by the variation in the 'speed' of the state evolution 
        (geodesic deviation).
        Ideally, a geodesic has constant speed. High curvature implies rapid changes in direction.
        """
        # Calculate discrete derivatives of the state
        states = trajectory
        velocities = []
        for i in range(len(states)-1):
            # Fubini-Study distance or simple Euclidean in CP1
            diff = states[i+1] - states[i]
            velocities.append(np.linalg.norm(diff))
            
        velocities = np.array(velocities)
        # Curvature proxy: Variance of the velocity (acceleration)
        # Minimizing this makes the trajectory smoother (geodesic-like)
        curvature = np.var(velocities)
        return curvature

    def cost_function(self, params):
        omega_x, omega_y = self.generate_pulse_from_params(params)
        
        initial_state = np.array([1, 0], dtype=complex)
        trajectory = self.solver.evolve(initial_state, omega_x, omega_y, delta=0.0)
        final_state = trajectory[-1]
        
        fidelity = np.abs(np.vdot(self.target_state, final_state))**2
        
        # Ricci Regularization
        ricci_cost = self.ricci_curvature_cost(trajectory)
        
        # Total Cost = Infidelity + lambda * Ricci_Curvature
        return (1.0 - fidelity) + 0.1 * ricci_cost

    def optimize(self):
        n_freqs = len(self.fermat_primes)
        # Initial guess: Small random amplitudes
        initial_guess = np.random.rand(4 * n_freqs) * 0.1
        
        result = minimize(self.cost_function, initial_guess, method='BFGS')
        
        omega_x, omega_y = self.generate_pulse_from_params(result.x)
        return omega_x, omega_y, result.fun
