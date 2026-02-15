import numpy as np
from scipy.linalg import expm

class RiccatiSolver:
    """
    Solves the dynamics of a quantum spin system using a Continued Fraction approach.
    Specifically, we use the Padé approximant (which is a continued fraction)
    to approximate the matrix exponential for time evolution.
    
    The Hamiltonian is assumed to be of the form:
    H(t) = (omega_x(t) * Sx + omega_y(t) * Sy + Delta * Sz)
    """
    
    def __init__(self, dt):
        self.dt = dt
        # Pauli matrices
        self.Sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        self.Sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        self.Sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        self.Id = np.eye(2, dtype=complex)

    def get_hamiltonian(self, omega_x, omega_y, delta):
        """Constructs the Hamiltonian matrix."""
        return omega_x * self.Sx + omega_y * self.Sy + delta * self.Sz

    def pade_approximant_step(self, H):
        """
        Computes the time evolution operator U = exp(-i * H * dt)
        using the [1/1] Padé approximant (Cayley transform), which is a simple continued fraction.
        
        exp(A) ~ (I + A/2) / (I - A/2)  (First order Padé)
        For higher accuracy, we could use [2/2] or higher.
        Here we use [2/2] Padé for better stability.
        """
        A = -1j * H * self.dt
        
        # [1/1] Padé (Cayley form):
        # R_11(A) = (I + A/2)(I - A/2)^-1
        # This is equivalent to the continued fraction:
        # exp(x) = 1 + 2x / (2 - x + ...)
        
        # Using numpy's solve for stability instead of explicit inverse
        # (I - A/2) U = (I + A/2)
        
        I = self.Id
        M_minus = I - A / 2
        M_plus = I + A / 2
        
        U = np.linalg.solve(M_minus, M_plus)
        return U

    def evolve(self, initial_state, omega_x_seq, omega_y_seq, delta):
        """
        Evolves the state through the pulse sequence.
        """
        state = np.array(initial_state, dtype=complex)
        trajectory = [state.copy()]
        
        for ox, oy in zip(omega_x_seq, omega_y_seq):
            H = self.get_hamiltonian(ox, oy, delta)
            U = self.pade_approximant_step(H)
            state = U @ state
            trajectory.append(state.copy())
            
        return np.array(trajectory)

    def state_to_bloch(self, state):
        """Converts a quantum state vector to a Bloch vector (x, y, z)."""
        rho = np.outer(state, state.conj())
        x = np.trace(rho @ self.Sx).real * 2
        y = np.trace(rho @ self.Sy).real * 2
        z = np.trace(rho @ self.Sz).real * 2
        return np.array([x, y, z])
