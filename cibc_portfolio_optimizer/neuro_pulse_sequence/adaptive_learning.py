"""
Adaptive Learning System for MRI Pulse Sequence Optimization

Implements reinforcement learning for real-time sequence parameter adaptation
based on image quality metrics and scan efficiency.

Components:
- SequenceEnvironment: Gymnasium-compatible RL environment
- AdaptiveAgent: PPO-based agent for parameter optimization
- RewardFunction: Custom reward based on sequence quality
- SequenceOptimizer: Combines quantum ML with adaptive learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import warnings

# RL imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    warnings.warn("Gymnasium/StableBaselines3 not available. Using simple optimization.")

# Import quantum optimizer
try:
    from quantum_optimizer import CUDAQOptimizer
    QUANTUM_OPT_AVAILABLE = True
except:
    QUANTUM_OPT_AVAILABLE = False


class SequenceEnvironment(gym.Env if RL_AVAILABLE else object):
    """
    Gymnasium environment for MRI sequence parameter optimization.
    
    State space: Current sequence parameters
    Action space: Parameter adjustments
    Reward: Image quality metrics (SNR, CNR, scan time)
    """
    
    def __init__(self, 
                 initial_params: Dict,
                 target_metrics: Dict,
                 parameter_ranges: Dict):
        """
        Initialize sequence optimization environment.
        
        Args:
            initial_params: Starting sequence parameters
            target_metrics: Target values for quality metrics
            parameter_ranges: Valid ranges for each parameter
        """
        super().__init__()
        
        self.initial_params = initial_params
        self.target_metrics = target_metrics
        self.parameter_ranges = parameter_ranges
        
        # Extract parameter names
        self.param_names = list(initial_params.keys())
        n_params = len(self.param_names)
        
        # Define action and observation spaces
        if RL_AVAILABLE:
            # Action space: continuous adjustments for each parameter
            self.action_space = spaces.Box(
                low=-0.2,  # Max 20% decrease
                high=0.2,  # Max 20% increase
                shape=(n_params,),
                dtype=np.float32
            )
            
            # Observation space: current parameter values (normalized)
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(n_params,),
                dtype=np.float32
            )
        
        self.current_params = initial_params.copy()
        self.episode_step = 0
        self.max_episode_steps = 50
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if RL_AVAILABLE and seed is not None:
            super().reset(seed=seed)
        
        self.current_params = self.initial_params.copy()
        self.episode_step = 0
        
        observation = self._get_observation()
        info = {}
        
        if RL_AVAILABLE:
            return observation, info
        return observation
    
    def step(self, action: np.ndarray):
        """
        Take a step in the environment.
        
        Args:
            action: Parameter adjustments
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply actions to parameters
        for i, param_name in enumerate(self.param_names):
            current_val = self.current_params[param_name]
            adjustment = action[i] * current_val  # Proportional adjustment
            new_val = current_val + adjustment
            
            # Clip to valid range
            if param_name in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param_name]
                new_val = np.clip(new_val, min_val, max_val)
            
            self.current_params[param_name] = new_val
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        self.episode_step += 1
        terminated = self.episode_step >= self.max_episode_steps
        truncated = False
        
        observation = self._get_observation()
        info = {'params': self.current_params.copy()}
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get normalized observation of current parameters."""
        obs = []
        for param_name in self.param_names:
            val = self.current_params[param_name]
            
            # Normalize to [0, 1]
            if param_name in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param_name]
                normalized = (val - min_val) / (max_val - min_val)
            else:
                normalized = val / (self.initial_params[param_name] * 2)
            
            obs.append(normalized)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on sequence quality metrics.
        
        Combines SNR, CNR, and scan time into single reward signal.
        """
        params = self.current_params
        
        # Simulate SNR (higher is better)
        te = params.get('TE', 30)
        tr = params.get('TR', 500)
        fa = params.get('FA', 90)
        
        # SNR model: inversely proportional to TE, proportional to sin(FA), TR
        snr = (1000 / te) * np.sin(np.radians(fa)) * (tr / 1000)
        
        # Simulate CNR (contrast-to-noise ratio)
        cnr = snr * 0.3  # Simplified
        
        # Scan time (lower is better)
        matrix_size = params.get('matrix_size', 256)
        scan_time = tr * matrix_size / 1000  # seconds
        
        # Combined reward
        reward = snr * 0.5 + cnr * 0.3 - scan_time * 0.1
        
        # Penalty for deviating too far from initial params
        param_deviation = 0
        for param_name in self.param_names:
            initial_val = self.initial_params[param_name]
            current_val = self.current_params[param_name]
            param_deviation += abs(current_val - initial_val) / initial_val
        
        reward -= param_deviation * 5
        
        return float(reward)


class AdaptiveAgent:
    """
    Adaptive agent using PPO for sequence optimization.
    
    Learns optimal parameter adjustment policy through interaction
    with the sequence environment.
    """
    
    def __init__(self, environment: SequenceEnvironment):
        """
        Initialize adaptive agent.
        
        Args:
            environment: Sequence optimization environment
        """
        self.env = environment
        
        if RL_AVAILABLE:
            # Create PPO agent
            self.model = PPO(
                'MlpPolicy',
                environment,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95
            )
        else:
            self.model = None
            print("Using simple gradient-based optimization")
    
    def train(self, total_timesteps: int = 10000):
        """
        Train the adaptive agent.
        
        Args:
            total_timesteps: Total training timesteps
        """
        if self.model is not None:
            print(f"Training adaptive agent for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            print("Training complete!")
        else:
            print("RL not available, skipping training")
    
    def optimize(self, initial_params: Dict) -> Dict:
        """
        Optimize parameters using trained policy.
        
        Args:
            initial_params: Starting parameters
            
        Returns:
            Optimized parameters
        """
        if self.model is None:
            # Simple optimization fallback
            return self._simple_optimize(initial_params)
        
        # Reset environment with new initial params
        self.env.initial_params = initial_params
        obs, _ = self.env.reset()
        
        # Run episode with trained policy
        for _ in range(self.env.max_episode_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        return self.env.current_params
    
    def _simple_optimize(self, initial_params: Dict) -> Dict:
        """Simple gradient-free optimization fallback."""
        # Just return slightly modified parameters
        optimized = initial_params.copy()
        optimized['TE'] = initial_params['TE'] * 0.9
        optimized['FA'] = min(initial_params['FA'] * 1.1, 90)
        return optimized
    
    def save(self, filepath: str):
        """Save trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model."""
        if RL_AVAILABLE:
            self.model = PPO.load(filepath)
            print(f"Model loaded from {filepath}")


class SequenceOptimizer:
    """
    Combines quantum ML and adaptive learning for sequence optimization.
    
    Hybrid approach leveraging both quantum computing and reinforcement
    learning for superior optimization performance.
    """
    
    def __init__(self,
                 use_quantum: bool = True,
                 use_adaptive: bool = True):
        """
        Initialize hybrid optimizer.
        
        Args:
            use_quantum: Whether to use quantum ML optimization
            use_adaptive: Whether to use adaptive learning
        """
        self.use_quantum = use_quantum and QUANTUM_OPT_AVAILABLE
        self.use_adaptive = use_adaptive and RL_AVAILABLE
        
        self.quantum_optimizer = None
        self.adaptive_agent = None
        
        print(f"Hybrid Optimizer: Quantum={self.use_quantum}, Adaptive={self.use_adaptive}")
    
    def optimize(self,
                initial_params: Dict,
                target_metrics: Dict,
                parameter_ranges: Dict,
                optimization_steps: int = 100) -> Dict:
        """
        Optimize sequence parameters using hybrid approach.
        
        Args:
            initial_params: Initial sequence parameters
            target_metrics: Target quality metrics
            parameter_ranges: Valid parameter ranges
            optimization_steps: Number of optimization iterations
            
        Returns:
            Optimized parameters
        """
        print(f"\nStarting hybrid optimization with {optimization_steps} steps...")
        current_params = initial_params.copy()
        
        # Phase 1: Quantum ML optimization (if enabled)
        if self.use_quantum:
            print("\nPhase 1: Quantum ML Optimization")
            print("-" * 50)
            
            if self.quantum_optimizer is None:
                from quantum_optimizer import CUDAQOptimizer
                self.quantum_optimizer = CUDAQOptimizer(n_qubits=6, n_layers=3)
            
            # Define target metric function
            def metric_function(params):
                te = params.get('TE', 30)
                tr = params.get('TR', 500)
                fa = params.get('FA', 90)
                snr = (1000 / te) * np.sin(np.radians(fa)) * (tr / 1000)
                return snr
            
            current_params = self.quantum_optimizer.optimize(
                initial_params=current_params,
                target_metric=metric_function,
                max_iterations=optimization_steps // 2
            )
            
            print(f"Quantum optimization complete: {current_params}")
        
        # Phase 2: Adaptive learning refinement (if enabled)
        if self.use_adaptive:
            print("\nPhase 2: Adaptive Learning")
            print("-" * 50)
            
            # Create environment
            env = SequenceEnvironment(
                initial_params=current_params,
                target_metrics=target_metrics,
                parameter_ranges=parameter_ranges
            )
            
            # Create and train agent
            self.adaptive_agent = AdaptiveAgent(env)
            self.adaptive_agent.train(total_timesteps=optimization_steps)
            
            # Use trained policy to optimize
            current_params = self.adaptive_agent.optimize(current_params)
            
            print(f"Adaptive optimization complete: {current_params}")
        
        # If neither method available, return simple optimization
        if not self.use_quantum and not self.use_adaptive:
            print("\nNo advanced optimization available, using simple heuristics")
            current_params['TE'] = initial_params['TE'] * 0.95
            current_params['FA'] = min(initial_params['FA'] * 1.05, 90)
        
        print("\n" + "=" * 50)
        print("Hybrid optimization complete!")
        print("=" * 50)
        print(f"Initial: {initial_params}")
        print(f"Optimized: {current_params}")
        
        return current_params


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Learning System for MRI Pulse Sequences")
    print("=" * 70)
    print(f"RL Available: {RL_AVAILABLE}")
    print(f"Quantum Optimizer Available: {QUANTUM_OPT_AVAILABLE}")
    print()
    
    # Define test parameters
    initial_params = {
        'TE': 30.0,  # ms
        'TR': 500.0,  # ms
        'FA': 70.0,  # degrees
        'matrix_size': 256
    }
    
    target_metrics = {
        'SNR': 100.0,
        'CNR': 30.0,
        'scan_time': 120.0  # seconds
    }
    
    parameter_ranges = {
        'TE': (5.0, 100.0),
        'TR': (100.0, 5000.0),
        'FA': (10.0, 90.0),
        'matrix_size': (64, 512)
    }
    
    # Test sequence environment
    if RL_AVAILABLE:
        print("Testing Sequence Environment...")
        env = SequenceEnvironment(initial_params, target_metrics, parameter_ranges)
        check_env(env)
        print("  Environment validation passed!")
        
        # Test episode
        obs, info = env.reset()
        print(f"  Initial observation: {obs}")
        
        action = np.array([0.05, -0.1, 0.02, 0.0])  # Sample action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  After action - Reward: {reward:.2f}, Params: {info['params']}")
        print()
    
    # Test hybrid optimizer
    print("Testing Hybrid Sequence Optimizer...")
    optimizer = SequenceOptimizer(use_quantum=True, use_adaptive=True)
    
    optimized_params = optimizer.optimize(
        initial_params=initial_params,
        target_metrics=target_metrics,
        parameter_ranges=parameter_ranges,
        optimization_steps=20  # Small for testing
    )
    
    print("\nAdaptive learning system ready!")
