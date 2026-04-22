from flask import Flask, render_template, jsonify
import numpy as np
import scipy.special

app = Flask(__name__)

def generate_surfboard_points(n_points=200):
    """
    Generates a surfboard pattern (elliptical shape with tapered ends)
    used for anastomosis excisions.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = 10 * np.cos(t)
    y = 3 * np.sin(t) * (1 - 0.2 * np.cos(t))
    return np.column_stack((x, y))

def apply_continued_fraction_elliptic_deformation(points):
    """
    Applies the mathematical deformation combining:
    1. Continued fractions equivalence for elliptic integrals
    2. Finite properties of primes via residual differential equations
    """
    primes = [2, 3, 5, 7, 11, 13]
    deformed = np.copy(points)
    
    for i, pt in enumerate(deformed):
        x, y = pt
        r = np.sqrt(x**2 + y**2)
        
        # Continued fraction approximation related to elliptic properties
        cf_factor = np.sum([p / (r + p) if (r + p) != 0 else 0 for p in primes])
        
        # Residual Differential Equation effect (non-linear offset)
        deformed[i, 0] = x + 1.5 * np.sin(y * cf_factor) + 2
        deformed[i, 1] = y + 1.2 * np.cos(x * cf_factor) + 1
        
    return deformed

def apply_wishart_noise(points, scale=0.15):
    """
    Applies a covariance structure sampled from a Wishart distribution.
    """
    # Wishart distribution parameters (df=3, scale matrix)
    V = np.array([[1.0, 0.4], [0.4, 1.0]]) * scale
    covariance = np.random.randn(2, 2)
    covariance = np.dot(covariance, covariance.T) * V  # Covariance approximation matrix
    
    noise = np.random.multivariate_normal(mean=[0, 0], cov=covariance, size=len(points))
    return points + noise

def apply_pigeonhole_combinatorial_deformation(source, target):
    """
    Simulates a Pigeonhole Combinatorial Manifold Operator.
    Partitions the target manifold into 'holes' (regions) and forces the source 'pigeons' 
    (vertices) to map optimally into these regions reducing the geometric residuals.
    """
    n_points = len(source)
    steps = []
    current = np.copy(source)
    n_iterations = 30
    
    # Define "holes" as dominant attractor nodes on the target manifold
    n_holes = 10
    indices = np.linspace(0, n_points - 1, n_holes, dtype=int)
    attractors = target[indices]
    
    for i in range(n_iterations):
        # 1. Combinatorial Assignment (Pigeons to Holes)
        # For each point, find the nearest hole
        for j in range(n_points):
            distances = np.linalg.norm(attractors - current[j], axis=1)
            nearest_hole_idx = np.argmin(distances)
            nearest_hole = attractors[nearest_hole_idx]
            
            # 2. Manifold Operator (Move towards the hole with a non-linear combinatorial relaxation)
            # Dirichlet energy relaxation equivalent
            energy_gradient = (nearest_hole - current[j]) * 0.05
            
            # Add global target drift to prevent clustering at holes
            global_drift = (target[j] - current[j]) * 0.1
            
            current[j] += energy_gradient + global_drift
            
        steps.append(current.tolist())
        
    return steps

def apply_feynman_path_integral_registration(source, target):
    """
    Simulates a Feynman Path Integral approach mapping to equivalence properties
    of numbers for residual differential equations in registration.
    """
    n_points = len(source)
    steps = []
    current = np.copy(source)
    n_iterations = 25
    
    # Path integral simulation parameters (sum over histories)
    # We simulate multiple possible trajectories and average them,
    # weighted by an action related to residual differential equations.
    num_paths = 5
    
    for i in range(n_iterations):
        next_step_avg = np.zeros_like(current)
        total_weight = 0
        
        for p in range(num_paths):
            # Generate a random path variation (quantum fluctuation)
            fluctuation = np.random.randn(*current.shape) * 0.2 * (1.0 - i/n_iterations)
            candidate = current + fluctuation
            
            # Action: defined by properties of numbers and residual differential equation
            # Action evaluates how well the candidate aligns with properties of the target
            residuals = np.linalg.norm(target - candidate, axis=1)
            
            # Use numerical properties (e.g., mathematical constant e) for action coupling
            action = np.sum(residuals) + 0.1 * np.sum(np.sin(candidate * np.pi * 2.718))
            
            # Weight is exp(-Action)
            weight = np.exp(-action * 0.05)
            
            # Gradient step towards target from candidate
            diff = target - candidate
            # Use residual differential equation update
            candidate += diff * 0.15 * (1 + 0.05 * np.cos(residuals[:, None]))
            
            next_step_avg += candidate * weight
            total_weight += weight
            
        current = next_step_avg / total_weight
        steps.append(current.tolist())
        
    return steps

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cpd')
def cpd_simulation():
    # 1. Source Surfboard Pattern
    source = generate_surfboard_points()
    
    # 2. Target Pattern (Deformed by the elliptic/prime algorithm + Wishart variance)
    target = apply_continued_fraction_elliptic_deformation(source)
    target = apply_wishart_noise(target)
    
    # 3. Coherent Point Drift (CPD) Registration Mock Trajectory
    # Simulates the probabilistic alignment of the structures
    steps = []
    current = np.copy(source)
    n_iterations = 25
    
    for i in range(n_iterations):
        diff = target - current
        # Use prime-based learning rate decay "Residual differential gradient"
        prime_factor = 2.71828
        lr = 0.15 * (1 + 0.05 * np.sin(i * prime_factor))
        step = diff * lr
        current += step
        steps.append(current.tolist())
        
    return jsonify({
        'source': source.tolist(),
        'target': target.tolist(),
        'trajectory': steps
    })

@app.route('/api/pigeonhole')
def pigeonhole_simulation():
    # 1. Source Pattern
    source = generate_surfboard_points()
    
    # 2. Target Pattern 
    target = apply_continued_fraction_elliptic_deformation(source)
    target = apply_wishart_noise(target)
    
    # 3. Simulate combinatorial alignment operators (Pigeons & Holes)
    steps = apply_pigeonhole_combinatorial_deformation(source, target)
    
    return jsonify({
        'source': source.tolist(),
        'target': target.tolist(),
        'trajectory': steps
    })

@app.route('/api/feynman')
def feynman_simulation():
    # 1. Source Pattern
    source = generate_surfboard_points()
    
    # 2. Target Pattern 
    target = apply_continued_fraction_elliptic_deformation(source)
    target = apply_wishart_noise(target)
    
    # 3. Simulate Feynman path integrals
    steps = apply_feynman_path_integral_registration(source, target)
    
    return jsonify({
        'source': source.tolist(),
        'target': target.tolist(),
        'trajectory': steps
    })

if __name__ == '__main__':
    app.run(port=5004, debug=True)
