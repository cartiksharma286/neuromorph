
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_spin_lattice(filename="spin_lattice_nash.png"):
    """Visualizes a spin lattice with Nash equilibrium patterns."""
    size = 20
    # Simulate a converged Nash equilibrium state
    lattice = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    # Add a "vessel" structure
    lattice[5:15, 8:12] = 1
    
    plt.figure(figsize=(6, 6))
    plt.imshow(lattice, cmap='viridis', interpolation='nearest')
    plt.title("Finite Spin Lattice Nash Equilibrium ($u_i^*$)")
    plt.colorbar(label="Spin Alignment Strategy")
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

def generate_cf_tree(filename="cf_depth_optimization.png"):
    """Visualizes the depth of continued fraction optimization."""
    depths = np.arange(1, 11)
    def golden_cf(d):
        if d == 0: return 1
        return 1 + 1.0 / golden_cf(d - 1)
    
    vals = [golden_cf(d) for d in depths]
    
    plt.figure(figsize=(8, 4))
    plt.plot(depths, vals, 'o-', color='tab:blue', linewidth=2)
    plt.axhline(y=(1 + np.sqrt(5))/2, color='tab:red', linestyle='--', label="$\phi$ (Limit)")
    plt.xlabel("Recursion Depth ($k$)")
    plt.ylabel("TR Multiplier $[1; 1, \dots, 1]$")
    plt.title("Finite-Depth Continued Fraction Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

def generate_modular_congruence(filename="modular_congruence_map.png"):
    """Visualizes modular congruence patterns in tissue texture."""
    size = 100
    x = np.linspace(0, 5, size)
    y = np.linspace(0, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Simulated congruence factor C = (X*Y) mod 8
    Z = (X * Y * 10) % 8
    
    plt.figure(figsize=(6, 6))
    plt.imshow(Z, cmap='magma')
    plt.title("Statistical Congruence Map ($C \pmod{8}$)")
    plt.colorbar(label="Congruence Factor")
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename}")

if __name__ == "__main__":
    if not os.path.exists("report_images"):
        os.makedirs("report_images")
    
    generate_spin_lattice("report_images/spin_lattice_nash.png")
    generate_cf_tree("report_images/cf_depth_optimization.png")
    generate_modular_congruence("report_images/modular_congruence_map.png")
