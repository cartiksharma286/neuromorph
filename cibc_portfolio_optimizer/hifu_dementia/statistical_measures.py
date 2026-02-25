
import numpy as np
from scipy.stats import norm
from scipy.linalg import logm

def calculate_var(data: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at a given confidence level alpha.
    Represents the maximum expected loss (or in this case, safety breach) 
    over a given time period.
    """
    if len(data) == 0:
        return 0.0
    # Parametric VaR assuming normal distribution for simulation speed
    mu = np.mean(data)
    sigma = np.std(data)
    return norm.ppf(1 - alpha, mu, sigma)

def calculate_cvar(data: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    Measures the average loss in the worst (1-alpha)% of cases.
    """
    if len(data) == 0:
        return 0.0
    var = calculate_var(data, alpha)
    return np.mean(data[data >= var]) if any(data >= var) else var

def quantum_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """
    Calculate the Fidelity between two pure quantum states (vectors).
    F = |<psi|phi>|^2
    """
    # Normalize states just in case
    state_a = state_a / np.linalg.norm(state_a)
    state_b = state_b / np.linalg.norm(state_b)
    overlap = np.dot(state_a.conj(), state_b)
    return np.abs(overlap) ** 2

def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Calculate Von Neumann Entropy S = -Tr(rho * ln(rho)).
    rho is the density matrix.
    """
    # Ensure rho is a valid density matrix (trace 1, positive semi-definite)
    # For simulation, we assume it is valid or close to it.
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zero or negative eigenvalues to avoid log error
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def kl_divergence_measure(p: np.ndarray, q: np.ndarray) -> float:
    """
    Kullback-Leibler divergence for variational inference comparison.
    """
    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))
