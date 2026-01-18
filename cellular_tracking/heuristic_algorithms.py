"""
Cell Tracking and Differentiation System - Heuristic Algorithms Module
Implements genetic algorithms, PSO, simulated annealing, and ant colony optimization
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
from simanneal import Annealer


@dataclass
class TrackingAssignment:
    """Represents cell-to-track assignments"""
    assignments: np.ndarray  # Matrix of cell-track associations
    fitness: float = 0.0


class GeneticTrackingOptimizer:
    """Genetic algorithm for optimizing cell-track associations"""
    
    def __init__(self, n_cells: int, n_tracks: int, population_size: int = 100):
        self.n_cells = n_cells
        self.n_tracks = n_tracks
        self.population_size = population_size
        self.setup_ga()
        
    def setup_ga(self):
        """Initialize DEAP genetic algorithm components"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 0, self.n_tracks - 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_int, n=self.n_cells)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, 
                            up=self.n_tracks - 1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def fitness_function(self, individual: List[int], 
                        cell_features: np.ndarray,
                        track_features: np.ndarray,
                        motion_coherence: np.ndarray) -> Tuple[float]:
        """
        Calculate fitness based on:
        - Morphological similarity between cells and tracks
        - Motion coherence
        - Tracking consistency
        """
        fitness = 0.0
        
        for cell_idx, track_idx in enumerate(individual):
            # Morphological similarity
            morph_sim = 1.0 / (1.0 + np.linalg.norm(
                cell_features[cell_idx] - track_features[track_idx]))
            
            # Motion coherence
            motion_score = motion_coherence[cell_idx, track_idx]
            
            # Combined fitness
            fitness += 0.6 * morph_sim + 0.4 * motion_score
            
        return (fitness,)
    
    def optimize(self, cell_features: np.ndarray, 
                track_features: np.ndarray,
                motion_coherence: np.ndarray,
                n_generations: int = 100) -> Tuple[np.ndarray, List[float]]:
        """
        Run genetic algorithm optimization
        
        Returns:
            best_assignment: Optimal cell-track assignments
            fitness_history: Fitness values over generations
        """
        # Register fitness function with current data
        self.toolbox.register("evaluate", self.fitness_function,
                            cell_features=cell_features,
                            track_features=track_features,
                            motion_coherence=motion_coherence)
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # Run evolution
        fitness_history = []
        for gen in range(n_generations):
            # Evaluate fitness
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Record statistics
            fitness_history.append(max([ind.fitness.values[0] for ind in population]))
            
            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Re-evaluate modified individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
        
        # Get best individual
        best_ind = tools.selBest(population, 1)[0]
        best_assignment = np.array(best_ind)
        
        return best_assignment, fitness_history


class ParticleSwarmOptimizer:
    """PSO for parameter tuning"""
    
    def __init__(self, n_particles: int = 30, n_dimensions: int = 5):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        
    def optimize(self, objective_func: Callable, 
                bounds: List[Tuple[float, float]],
                max_iter: int = 100) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize parameters using PSO
        
        Args:
            objective_func: Function to minimize
            bounds: Parameter bounds [(min, max), ...]
            max_iter: Maximum iterations
            
        Returns:
            best_position: Optimal parameters
            cost_history: Cost values over iterations
        """
        # Initialize particles
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        positions = np.random.uniform(lb, ub, (self.n_particles, self.n_dimensions))
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_dimensions))
        
        # Personal best
        p_best_positions = positions.copy()
        p_best_costs = np.array([objective_func(p) for p in positions])
        
        # Global best
        g_best_idx = np.argmin(p_best_costs)
        g_best_position = p_best_positions[g_best_idx].copy()
        g_best_cost = p_best_costs[g_best_idx]
        
        cost_history = [g_best_cost]
        
        # PSO parameters
        w = 0.9  # Inertia weight
        c1 = 2.0  # Cognitive parameter
        c2 = 2.0  # Social parameter
        
        for iteration in range(max_iter):
            # Update inertia weight (adaptive)
            w = 0.9 - (0.9 - 0.4) * iteration / max_iter
            
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (p_best_positions[i] - positions[i])
                social = c2 * r2 * (g_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Update position
                positions[i] += velocities[i]
                
                # Enforce bounds
                positions[i] = np.clip(positions[i], lb, ub)
                
                # Evaluate
                cost = objective_func(positions[i])
                
                # Update personal best
                if cost < p_best_costs[i]:
                    p_best_costs[i] = cost
                    p_best_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if cost < g_best_cost:
                        g_best_cost = cost
                        g_best_position = positions[i].copy()
            
            cost_history.append(g_best_cost)
        
        return g_best_position, cost_history


class TrajectoryMatcher(Annealer):
    """Simulated annealing for trajectory matching"""
    
    def __init__(self, trajectory1: np.ndarray, trajectory2: np.ndarray):
        self.trajectory1 = trajectory1
        self.trajectory2 = trajectory2
        self.n_points = len(trajectory1)
        
        # Initial state: identity mapping
        initial_state = list(range(self.n_points))
        super(TrajectoryMatcher, self).__init__(initial_state)
        
    def move(self):
        """Swap two random points in the mapping"""
        i, j = random.sample(range(self.n_points), 2)
        self.state[i], self.state[j] = self.state[j], self.state[i]
        
    def energy(self):
        """Calculate trajectory distance with current mapping"""
        mapped_traj2 = self.trajectory2[self.state]
        return np.sum(np.linalg.norm(self.trajectory1 - mapped_traj2, axis=1))


class AntColonyLineage:
    """Ant colony optimization for lineage tree reconstruction"""
    
    def __init__(self, n_cells: int, n_ants: int = 20):
        self.n_cells = n_cells
        self.n_ants = n_ants
        self.pheromone = np.ones((n_cells, n_cells))
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        
    def heuristic_info(self, cell_features: np.ndarray) -> np.ndarray:
        """Calculate heuristic information based on morphological similarity"""
        distances = cdist(cell_features, cell_features, 'euclidean')
        # Convert to similarity (higher is better)
        heuristic = 1.0 / (1.0 + distances)
        np.fill_diagonal(heuristic, 0)
        return heuristic
        
    def construct_solution(self, heuristic: np.ndarray, 
                          division_events: List[int]) -> List[Tuple[int, int]]:
        """Construct lineage tree using ant colony"""
        lineage_edges = []
        
        for parent_cell in division_events:
            # Calculate probabilities for daughter cells
            probabilities = (self.pheromone[parent_cell] ** self.alpha * 
                           heuristic[parent_cell] ** self.beta)
            probabilities = probabilities / probabilities.sum()
            
            # Select two daughter cells
            daughters = np.random.choice(self.n_cells, size=2, 
                                        replace=False, p=probabilities)
            lineage_edges.append((parent_cell, daughters[0]))
            lineage_edges.append((parent_cell, daughters[1]))
            
        return lineage_edges
        
    def optimize(self, cell_features: np.ndarray,
                division_events: List[int],
                n_iterations: int = 50) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Optimize lineage tree reconstruction
        
        Returns:
            best_lineage: List of (parent, daughter) edges
            quality_history: Quality scores over iterations
        """
        heuristic = self.heuristic_info(cell_features)
        best_lineage = None
        best_quality = float('-inf')
        quality_history = []
        
        for iteration in range(n_iterations):
            # Construct solutions for all ants
            solutions = []
            qualities = []
            
            for ant in range(self.n_ants):
                lineage = self.construct_solution(heuristic, division_events)
                quality = self.evaluate_lineage(lineage, cell_features)
                solutions.append(lineage)
                qualities.append(quality)
                
                if quality > best_quality:
                    best_quality = quality
                    best_lineage = lineage
            
            quality_history.append(best_quality)
            
            # Update pheromones
            self.pheromone *= (1 - self.rho)  # Evaporation
            
            # Deposit pheromones
            for lineage, quality in zip(solutions, qualities):
                for parent, daughter in lineage:
                    self.pheromone[parent, daughter] += quality
        
        return best_lineage, quality_history
    
    def evaluate_lineage(self, lineage: List[Tuple[int, int]], 
                        cell_features: np.ndarray) -> float:
        """Evaluate lineage quality based on morphological coherence"""
        quality = 0.0
        for parent, daughter in lineage:
            similarity = 1.0 / (1.0 + np.linalg.norm(
                cell_features[parent] - cell_features[daughter]))
            quality += similarity
        return quality


class HybridOptimizer:
    """Combines multiple heuristics for robust solutions"""
    
    def __init__(self):
        self.ga_optimizer = None
        self.pso_optimizer = ParticleSwarmOptimizer()
        
    def optimize_tracking(self, cell_features: np.ndarray,
                         track_features: np.ndarray,
                         motion_coherence: np.ndarray) -> Dict:
        """
        Use hybrid approach: GA for assignment + PSO for parameter tuning
        """
        n_cells, n_tracks = cell_features.shape[0], track_features.shape[0]
        
        # Initialize GA
        self.ga_optimizer = GeneticTrackingOptimizer(n_cells, n_tracks)
        
        # First pass: GA optimization
        ga_assignment, ga_history = self.ga_optimizer.optimize(
            cell_features, track_features, motion_coherence, n_generations=50)
        
        # Second pass: PSO for fine-tuning detection parameters
        def objective(params):
            # Simulate tracking with different parameters
            # This is a placeholder - in practice, would re-run tracking
            # We add a penalty based on distance from "ideal" parameters (0.5) to simulate tuning
            base_score = -np.sum(motion_coherence[np.arange(n_cells), ga_assignment.astype(int)])
            penalty = np.sum((params - 0.5) ** 2)
            return base_score + penalty
        
        bounds = [(0.1, 0.9), (0.1, 0.9), (0.5, 2.0), (0.1, 0.5), (0.1, 0.9)]
        pso_params, pso_history = self.pso_optimizer.optimize(objective, bounds, max_iter=30)
        
        return {
            'assignment': ga_assignment,
            'parameters': pso_params,
            'ga_history': ga_history,
            'pso_history': pso_history
        }
