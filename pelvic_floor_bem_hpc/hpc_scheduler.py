"""
HPC Cluster Scheduler Simulation
Models distribution of the dense BEM panel system across an HPC cluster using
a SLURM-style job submission, MPI domain decomposition, and Amdahl's-law
runtime/efficiency projection.
"""

import numpy as np
from typing import Dict, Any
import uuid
import time


class HPCScheduler:
    """Simulates SLURM job submission and parallel scaling for BEM workloads"""

    def __init__(self, cluster_name: str = "neuromorph-hpc01", cores_per_node: int = 64):
        self.cluster_name = cluster_name
        self.cores_per_node = cores_per_node

    def estimate_serial_runtime_sec(self, n_panels: int, production_scale_factor: int = 40) -> float:
        """
        Dense BEM assembly + solve is O(n^3) for the direct solve, O(n^2) for assembly.
        The interactive studio uses a coarse panel count for responsiveness; HPC job
        sizing is projected against the full clinical-resolution mesh a production run
        would use (n_panels * production_scale_factor).
        """
        dof = 3 * n_panels * production_scale_factor
        assembly_flops = dof ** 2 * 20
        solve_flops = dof ** 3
        total_flops = assembly_flops + solve_flops
        gflops_per_core = 8.0
        return float(total_flops / (gflops_per_core * 1e9))

    def generate_sbatch_script(self, job_name: str, nodes: int, tasks_per_node: int,
                                wall_time_min: int = 30) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --cluster={self.cluster_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-hpc
#SBATCH --time=00:{wall_time_min:02d}:00
#SBATCH --output=bem_%j.out

module load openmpi/4.1 cuda/12.4 nvqlink/1.0
srun --mpi=pmix python3 bem_worker.py --job-name={job_name} \\
     --domain-decompose=metis --panels-per-rank=auto
"""

    def submit_job(self, n_panels: int, nodes: int = 4, tasks_per_node: int = 16,
                    parallel_fraction: float = 0.94) -> Dict[str, Any]:
        """
        Simulate submitting the BEM dense-solve workload to the HPC cluster.
        Applies Amdahl's law:  S(P) = 1 / ((1 - f) + f/P)
        plus a communication overhead term modeling MPI all-to-all exchange
        cost during domain-decomposed panel assembly.
        """
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        p_total = nodes * tasks_per_node

        t_serial = self.estimate_serial_runtime_sec(n_panels)

        f = parallel_fraction
        amdahl_speedup = 1.0 / ((1 - f) + f / p_total)

        # communication overhead grows with sqrt(P) for all-to-all boundary exchange
        comm_overhead_sec = 0.0004 * np.sqrt(p_total) * np.log1p(n_panels)

        t_parallel = (t_serial / amdahl_speedup) + comm_overhead_sec
        actual_speedup = t_serial / t_parallel
        efficiency = actual_speedup / p_total

        sbatch_script = self.generate_sbatch_script(job_id, nodes, tasks_per_node)

        return {
            'job_id': job_id,
            'cluster': self.cluster_name,
            'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'nodes': nodes,
            'tasks_per_node': tasks_per_node,
            'total_ranks': int(p_total),
            'parallel_fraction': f,
            'serial_runtime_sec': round(t_serial, 4),
            'amdahl_theoretical_speedup': round(float(amdahl_speedup), 3),
            'communication_overhead_sec': round(float(comm_overhead_sec), 5),
            'projected_wall_time_sec': round(float(t_parallel), 4),
            'actual_speedup': round(float(actual_speedup), 3),
            'parallel_efficiency_pct': round(float(efficiency) * 100, 2),
            'sbatch_script': sbatch_script,
            'status': 'COMPLETED (simulated)',
        }

    def scaling_curve(self, n_panels: int, parallel_fraction: float = 0.94,
                       max_ranks: int = 256) -> Dict[str, Any]:
        """Generate an Amdahl's-law strong-scaling curve for the dashboard chart"""
        t_serial = self.estimate_serial_runtime_sec(n_panels)
        rank_counts = [2 ** k for k in range(0, int(np.log2(max_ranks)) + 1)]
        speedups, efficiencies, wall_times = [], [], []
        for p in rank_counts:
            f = parallel_fraction
            s = 1.0 / ((1 - f) + f / p)
            comm = 0.0004 * np.sqrt(p) * np.log1p(n_panels)
            wt = (t_serial / s) + comm
            actual_s = t_serial / wt
            speedups.append(round(float(actual_s), 3))
            efficiencies.append(round(float(actual_s / p) * 100, 2))
            wall_times.append(round(float(wt), 4))
        return {
            'ranks': rank_counts,
            'speedup': speedups,
            'efficiency_pct': efficiencies,
            'wall_time_sec': wall_times,
            'serial_runtime_sec': round(t_serial, 4),
        }
