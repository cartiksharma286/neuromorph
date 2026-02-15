import time
import numpy as np
import matplotlib.pyplot as plt
from lbm_solver import LBMSolver, LBMParameters
from lbm_solver_gpu import LBMSolverGPU, LBMParametersGPU

def benchmark():
    print("Benchmarking CPU vs GPU LBM Solver...")
    print("=" * 60)
    
    resolutions = [32, 64] # Keep it small for quick test
    cpu_times = []
    gpu_times = []
    
    for res in resolutions:
        print(f"\nTesting Resolution: {res}x{res}x{res}")
        
        # Setup
        resolution = (res, res, res)
        geometry = np.zeros(resolution, dtype=bool)
        geometry[res//2-2:res//2+2, res//2-2:res//2+2, res//2-2:res//2+2] = True
        
        steps = 100
        
        # CPU Test
        print("  Running CPU Solver...")
        cpu_params = LBMParameters(
            resolution=resolution,
            tau=0.6,
            max_steps=steps
        )
        start = time.time()
        cpu_solver = LBMSolver(cpu_params, geometry)
        cpu_solver.solve()
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU Time: {cpu_time:.4f}s")
        
        # GPU Test
        print("  Running GPU Solver...")
        try:
            gpu_params = LBMParametersGPU(
                resolution=resolution,
                tau=0.6,
                max_steps=steps
            )
            start = time.time()
            gpu_solver = LBMSolverGPU(gpu_params, geometry)
            gpu_solver.solve()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)
            print(f"  GPU Time: {gpu_time:.4f}s")
            print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        except Exception as e:
            print(f"  GPU Failed: {e}")
            gpu_times.append(float('inf'))

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    benchmark()
