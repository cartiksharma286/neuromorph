"""
Parametric Learning Models for Cell Differentiation
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import GPy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d


@dataclass
class CellState:
    """Represents a cell state in latent space"""
    embedding: np.ndarray
    time: float
    cell_id: int
    features: np.ndarray


class CellStateModel:
    """Represent cell states in latent space using learned embeddings"""
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """Learn latent space from cell features"""
        self.pca.fit(features)
        self.is_fitted = True
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features to latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.pca.transform(features)
    
    def inverse_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform from latent space back to feature space"""
        return self.pca.inverse_transform(embeddings)
    
    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio"""
        return self.pca.explained_variance_ratio_


class DifferentiationModel:
    """Gaussian Process regression for modeling differentiation trajectories"""
    
    def __init__(self, n_dimensions: int = 3):
        self.n_dimensions = n_dimensions
        self.gp_models = []
        self.is_fitted = False
        
    def fit(self, times: np.ndarray, states: np.ndarray):
        """
        Fit GP models for each dimension
        
        Args:
            times: Time points (n_samples,)
            states: Cell states (n_samples, n_dimensions)
        """
        times = times.reshape(-1, 1)
        
        self.gp_models = []
        for dim in range(self.n_dimensions):
            # Define kernel: RBF + noise
            kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
            
            # Create GP model
            gp = GPy.models.GPRegression(times, states[:, dim:dim+1], kernel)
            
            # Optimize hyperparameters
            gp.optimize(messages=False)
            
            self.gp_models.append(gp)
        
        self.is_fitted = True
    
    def predict(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cell states at given times
        
        Returns:
            means: Predicted states (n_times, n_dimensions)
            variances: Prediction uncertainties (n_times, n_dimensions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        times = times.reshape(-1, 1)
        
        means = []
        variances = []
        
        for gp in self.gp_models:
            mean, var = gp.predict(times)
            means.append(mean)
            variances.append(var)
        
        means = np.hstack(means)
        variances = np.hstack(variances)
        
        return means, variances
    
    def sample_trajectories(self, times: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Sample trajectories from the GP posterior"""
        times = times.reshape(-1, 1)
        
        samples = []
        for gp in self.gp_models:
            # Sample from posterior
            dim_samples = gp.posterior_samples_f(times, size=n_samples)
            samples.append(dim_samples)
        
        # Combine dimensions: (n_times, n_samples, n_dimensions)
        trajectories = np.stack([s.squeeze() for s in samples], axis=-1)
        
        return trajectories


class ParameterEstimator:
    """Estimate differentiation rates, transition probabilities, and cell fate decisions"""
    
    def __init__(self):
        self.differentiation_rates = {}
        self.transition_matrix = None
        self.cell_types = []
        
    def estimate_differentiation_rate(self, trajectories: List[np.ndarray],
                                     times: List[np.ndarray]) -> Dict[str, float]:
        """
        Estimate differentiation rate from trajectory data
        
        Args:
            trajectories: List of cell trajectories in latent space
            times: Corresponding time points
            
        Returns:
            Dictionary with differentiation metrics
        """
        rates = []
        
        for traj, time in zip(trajectories, times):
            if len(traj) < 2:
                continue
            
            # Calculate velocity (rate of change in latent space)
            velocities = np.diff(traj, axis=0)
            time_diffs = np.diff(time)
            
            # Normalize by time
            rates_per_cell = np.linalg.norm(velocities, axis=1) / time_diffs
            rates.extend(rates_per_cell)
        
        rates = np.array(rates)
        
        return {
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'median_rate': np.median(rates),
            'max_rate': np.max(rates),
            'min_rate': np.min(rates)
        }
    
    def estimate_transition_probabilities(self, states: np.ndarray,
                                         n_cell_types: int = 3) -> np.ndarray:
        """
        Estimate transition probabilities between cell types
        
        Args:
            states: Cell states over time (n_samples, n_dimensions)
            n_cell_types: Number of cell types to identify
            
        Returns:
            Transition matrix (n_cell_types, n_cell_types)
        """
        # Cluster states into cell types
        kmeans = KMeans(n_clusters=n_cell_types, random_state=42)
        labels = kmeans.fit_predict(states)
        
        self.cell_types = [f"Type_{i}" for i in range(n_cell_types)]
        
        # Count transitions
        transition_counts = np.zeros((n_cell_types, n_cell_types))
        
        for i in range(len(labels) - 1):
            from_type = labels[i]
            to_type = labels[i + 1]
            transition_counts[from_type, to_type] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = transition_counts / row_sums
        
        return self.transition_matrix
    
    def predict_cell_fate(self, current_state: np.ndarray,
                         n_steps: int = 10) -> List[str]:
        """
        Predict cell fate trajectory using transition probabilities
        
        Args:
            current_state: Current cell state
            n_steps: Number of steps to predict
            
        Returns:
            List of predicted cell types
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition probabilities first")
        
        # Find closest cell type
        # This is simplified - in practice would use the clustering model
        current_type = np.random.randint(0, len(self.cell_types))
        
        trajectory = [self.cell_types[current_type]]
        
        for _ in range(n_steps - 1):
            # Sample next state based on transition probabilities
            next_type = np.random.choice(
                len(self.cell_types),
                p=self.transition_matrix[current_type]
            )
            trajectory.append(self.cell_types[next_type])
            current_type = next_type
        
        return trajectory


class TrajectoryAnalyzer:
    """Analyze and cluster differentiation paths"""
    
    def __init__(self):
        self.clusters = None
        self.cluster_labels = None
        
    def cluster_trajectories(self, trajectories: List[np.ndarray],
                            n_clusters: int = 3) -> np.ndarray:
        """
        Cluster trajectories based on similarity
        
        Args:
            trajectories: List of cell trajectories
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels for each trajectory
        """
        # Normalize trajectory lengths using interpolation
        max_len = max(len(traj) for traj in trajectories)
        
        normalized_trajs = []
        for traj in trajectories:
            if len(traj) < 2:
                continue
            
            # Interpolate to common length
            t_old = np.linspace(0, 1, len(traj))
            t_new = np.linspace(0, 1, max_len)
            
            normalized = np.zeros((max_len, traj.shape[1]))
            for dim in range(traj.shape[1]):
                interp_func = interp1d(t_old, traj[:, dim], kind='linear')
                normalized[:, dim] = interp_func(t_new)
            
            normalized_trajs.append(normalized.flatten())
        
        normalized_trajs = np.array(normalized_trajs)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(normalized_trajs)
        self.clusters = kmeans.cluster_centers_
        
        return self.cluster_labels
    
    def compute_trajectory_statistics(self, trajectories: List[np.ndarray]) -> Dict:
        """Compute statistics for trajectories"""
        lengths = [len(traj) for traj in trajectories]
        
        # Compute path lengths
        path_lengths = []
        for traj in trajectories:
            if len(traj) < 2:
                continue
            diffs = np.diff(traj, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            path_lengths.append(path_length)
        
        # Compute tortuosity (path length / straight line distance)
        tortuosities = []
        for traj in trajectories:
            if len(traj) < 2:
                continue
            straight_dist = np.linalg.norm(traj[-1] - traj[0])
            diffs = np.diff(traj, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            if straight_dist > 0:
                tortuosity = path_length / straight_dist
                tortuosities.append(tortuosity)
        
        return {
            'n_trajectories': len(trajectories),
            'mean_length': np.mean(lengths),
            'mean_path_length': np.mean(path_lengths) if path_lengths else 0,
            'mean_tortuosity': np.mean(tortuosities) if tortuosities else 0,
            'std_tortuosity': np.std(tortuosities) if tortuosities else 0
        }
