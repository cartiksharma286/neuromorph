import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as mpatches

@dataclass
class RFCoilConfig:
    num_turns: int
    radius: float
    frequency: float
    
class StatisticalCoilEvolution:
    def __init__(self, config: RFCoilConfig, population_size: int = 10000):
        self.config = config
        self.population_size = population_size
        self.distributions = np.random.normal(0, 1, (population_size, 3))
        self.fitness_history = []
        
    def evaluate_fitness(self, params):
        """Calculate coil efficiency based on parameters"""
        impedance, q_factor, bandwidth = params
        return -(impedance**2 + q_factor**2 - bandwidth) + 1
    
    def evolve_generation(self, generations: int = 50):
        """Self-evolving distributions over generations"""
        for gen in range(generations):
            fitness = np.array([self.evaluate_fitness(p) for p in self.distributions])
            self.fitness_history.append(np.mean(fitness))
            
            top_idx = np.argsort(fitness)[-self.population_size//2:]
            self.distributions = self.distributions[top_idx]
            
            variance = 1.0 / (1.0 + gen * 0.1)
            mutations = np.random.normal(0, variance, self.distributions.shape)
            self.distributions = np.vstack([self.distributions, self.distributions + mutations])
    
    def plot_schematic(self):
        """Visualize coil schematic and evolution"""
        fig = plt.figure(figsize=(14, 5))
        
        # RF Coil schematic
        ax1 = plt.subplot(131)
        self._draw_coil(ax1)
        
        # Fitness evolution
        ax2 = plt.subplot(132)
        ax2.plot(self.fitness_history, linewidth=2, color='blue')
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Average Fitness")
        ax2.set_title("Optimization Progress")
        ax2.grid()
        
        # Parameter distribution
        ax3 = plt.subplot(133)
        scatter = ax3.scatter(self.distributions[:, 0], self.distributions[:, 1], 
                       c=self.distributions[:, 2], cmap='viridis', s=50, alpha=0.6)
        ax3.set_xlabel("Impedance")
        ax3.set_ylabel("Q-Factor")
        ax3.set_title("Final Parameter Distribution")
        ax3.grid()
        plt.colorbar(scatter, ax=ax3, label="Bandwidth")
        
        plt.tight_layout()
        plt.show()
    
    def _draw_coil(self, ax):
        """Draw RF coil schematic"""
        angles = np.linspace(0, 2*np.pi, self.config.num_turns, endpoint=False)
        
        for i, angle in enumerate(angles):
            x = self.config.radius * np.cos(angle)
            y = self.config.radius * np.sin(angle)
            circle = Circle((x, y), 0.01, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        # Draw outer loop
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.config.radius * np.cos(theta)
        y = self.config.radius * np.sin(theta)
        ax.plot(x, y, 'b-', linewidth=2, label='Coil Loop')
        
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_aspect('equal')
        ax.set_title(f"RF Coil Schematic\n({self.config.num_turns} turns @ {self.config.frequency/1e9:.1f} GHz)")
        ax.grid()
        ax.legend()

# Usage
config = RFCoilConfig(num_turns=10, radius=0.05, frequency=1e9)
coil = StatisticalCoilEvolution(config)
coil.evolve_generation(generations=50)
coil.plot_schematic()