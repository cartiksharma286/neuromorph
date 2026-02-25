import time
import random
import math
import asyncio
import numpy as np
from threading import Lock
from typing import Dict, List, Optional
from pydantic import BaseModel

class ANEStats(BaseModel):
    status: str
    active_cores: int
    tops_utilization: float
    temperature_c: float
    power_draw_w: float
    memory_bandwidth_gbps: float
    current_task: str
    core_map: List[float] # Utilization per core

class AppleNeuralEngineSim:
    """
    Simulates a Next-Gen 'M5' Ultra Neural Engine.
    OPTIMIZATION: Uses NumPy for Vectorized/SIMD operations, mimicking 
    native processor instructions and efficient memory layout.
    
    Architecture:
    - 32-core Neural Engine
    - 128 TOPS INT8 Performance
    - Unified Memory Architecture (UMA)
    """
    def __init__(self):
        self.name = "Apple ANE X-Series (Simulated)"
        self.num_cores = 32
        self.total_tops = 128.0
        
        # Vectorized State Arrays (SIMD Layout)
        # 0: Utilization (0.0 - 1.0)
        # 1: Temperature (C)
        # 2: Active State (0 or 1)
        self.state_matrix = np.zeros((3, self.num_cores), dtype=np.float32)
        
        # Initialize Temps
        self.state_matrix[1, :] = 35.0
        
        # Constants for vector math
        self.smoothing_factor = 0.8
        self.load_factor = 0.2
        self.heat_coeff = 1.5
        self.cool_coeff = 0.1
        self.ambient_temp = 30.0
        
        # System State
        self.memory_usage_gb = 0.0
        self.memory_bandwidth_used = 0.0
        self.power_w = 0.5 # Idle
        self.task_queue = []
        self.current_task_name = "IDLE"
        self.lock = Lock()
        
    def submit_job(self, name: str, complexity_flops: float):
        """Submit a compute job to the queue."""
        with self.lock:
            self.task_queue.append({
                "name": name,
                "flops": complexity_flops,
                "progress": 0.0
            })

    def update(self):
        """
        Main simulation loop tick.
        PERFORMANCE: fully vectorized matrix updates.
        """
        with self.lock:
            if not self.task_queue:
                self._update_idle_state()
                return

            # Active Job Processing
            job = self.task_queue[0]
            self.current_task_name = job["name"]
            
            # Massive Parallel Load (1.0 across all cores)
            target_load = np.ones(self.num_cores, dtype=np.float32)
            
            # Calculate Job Progress
            # Throughput per tick (0.1s)
            cpu_efficiency = 0.8
            total_throughput = self.total_tops * 1e12 * 0.1 * cpu_efficiency
            job["progress"] += total_throughput
            
            # Physics Update
            self._step_physics(target_load)
            
            # Power Calc (Vectorized sum implicitly handled by physics result)
            active_cores = np.sum(self.state_matrix[0, :] > 0.01)
            self.power_w = 5.0 + (active_cores * 0.5)
            self.memory_bandwidth_used = 200.0 + random.uniform(-10, 10)
            
            if job["progress"] >= job["flops"]:
                self.task_queue.pop(0)

    def _update_idle_state(self):
        """Handle background noise in idle state."""
        self.current_task_name = "IDLE (Background)"
        self.memory_bandwidth_used = max(5, self.memory_bandwidth_used * 0.95)
        self.power_w = 0.5 + random.uniform(0.0, 0.2)
        
        # Vectorized Random Noise Generation
        # Probabilistic load: 20% chance of 0.15 load
        random_mask = np.random.random(self.num_cores) < 0.2
        target_load = np.zeros(self.num_cores, dtype=np.float32)
        target_load[random_mask] = np.random.uniform(0.0, 0.15, size=np.sum(random_mask))
        
        self._step_physics(target_load)

    def _step_physics(self, target_load: np.ndarray):
        """
        Core physics simulation using optimized vector operations.
        Replaces 32 distinct object method calls with matrix math.
        """
        # 1. Update Utilization (Exponential Moving Average)
        # util = util * 0.8 + target * 0.2
        self.state_matrix[0, :] = (self.state_matrix[0, :] * self.smoothing_factor) + (target_load * self.load_factor)
        
        # 2. Update Temperature
        # heat = util * 1.5 - (temp - 30) * 0.1
        current_util = self.state_matrix[0, :]
        current_temp = self.state_matrix[1, :]
        
        heat_gen = current_util * self.heat_coeff
        passive_cooling = (current_temp - self.ambient_temp) * self.cool_coeff
        
        self.state_matrix[1, :] = current_temp + heat_gen - passive_cooling
        
        # 3. Update Active Flag
        self.state_matrix[2, :] = (current_util > 0.01).astype(np.float32)

    def get_stats(self) -> ANEStats:
        with self.lock:
            # Fast vectorized aggregation
            avg_temp = float(np.mean(self.state_matrix[1, :]))
            avg_util = float(np.mean(self.state_matrix[0, :]))
            active_count = int(np.sum(self.state_matrix[2, :]))
            
            # Extract core map as native Python list for JSON serialization
            core_map = self.state_matrix[0, :].tolist()
            
            return ANEStats(
                status="PROCESSING" if self.task_queue else "IDLE",
                active_cores=active_count,
                tops_utilization=avg_util * self.total_tops,
                temperature_c=float(f"{avg_temp:.1f}"),
                power_draw_w=float(f"{self.power_w:.1f}"),
                memory_bandwidth_gbps=float(f"{self.memory_bandwidth_used:.1f}"),
                current_task=self.current_task_name,
                core_map=core_map
            )

# Singleton Instance
ane_processor = AppleNeuralEngineSim()
