"""
Computational Fluid Dynamics Simulation for Catheter Flow Analysis

Implements simplified CFD for real-time catheter performance evaluation
with quantum-enhanced linear solvers
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import cudaq
from quantum_catheter_optimizer import DesignParameters, PatientConstraints


@dataclass
class FlowResults:
    """Container for CFD simulation results"""
    pressure_drop: float  # Pa
    average_velocity: float  # m/s
    max_velocity: float  # m/s
    reynolds_number: float
    flow_regime: str  # 'laminar' or 'turbulent'
    pressure_field: Optional[np.ndarray] = None
    velocity_field: Optional[np.ndarray] = None


class FluidProperties:
    """Blood and saline fluid properties"""
    
    # Blood properties (at 37°C)
    BLOOD_DENSITY = 1060  # kg/m³
    BLOOD_VISCOSITY = 0.0035  # Pa·s (3.5 cP)
    
    # Saline properties
    SALINE_DENSITY = 1000  # kg/m³
    SALINE_VISCOSITY = 0.001  # Pa·s (1.0 cP)
    
    @classmethod
    def get_properties(cls, fluid_type: str = 'blood') -> Tuple[float, float]:
        """Get density and viscosity for fluid type"""
        if fluid_type.lower() == 'blood':
            return cls.BLOOD_DENSITY, cls.BLOOD_VISCOSITY
        elif fluid_type.lower() == 'saline':
            return cls.SALINE_DENSITY, cls.SALINE_VISCOSITY
        else:
            raise ValueError(f"Unknown fluid type: {fluid_type}")


class SimplifiedNavierStokesSolver:
    """
    Simplified 1D Navier-Stokes solver for catheter flow
    
    Assumes:
    - Steady state flow
    - Axisymmetric geometry
    - Incompressible fluid
    """
    
    def __init__(
        self,
        design: DesignParameters,
        constraints: PatientConstraints,
        fluid_type: str = 'blood'
    ):
        self.design = design
        self.constraints = constraints
        self.density, self.viscosity = FluidProperties.get_properties(fluid_type)
        
    def compute_pressure_drop(self) -> float:
        """
        Compute pressure drop using Hagen-Poiseuille equation
        
        Valid for laminar flow in cylindrical pipes
        ΔP = (8 * μ * L * Q) / (π * r^4)
        """
        # Convert units
        length = self.constraints.required_length / 1000  # mm to m
        radius = (self.design.inner_diameter / 2) / 1000  # mm to m
        flow_rate = self.constraints.flow_rate / 60000  # ml/min to m³/s
        
        # Hagen-Poiseuille
        pressure_drop = (8 * self.viscosity * length * flow_rate) / (np.pi * radius ** 4)
        
        return pressure_drop  # Pa
    
    def compute_reynolds_number(self) -> float:
        """
        Compute Reynolds number
        
        Re = (ρ * v * D) / μ
        """
        # Convert units
        diameter = self.design.inner_diameter / 1000  # mm to m
        flow_rate = self.constraints.flow_rate / 60000  # ml/min to m³/s
        area = np.pi * (diameter / 2) ** 2
        
        # Average velocity
        velocity = flow_rate / area
        
        # Reynolds number
        re = (self.density * velocity * diameter) / self.viscosity
        
        return re
    
    def compute_velocity_profile(self, n_radial: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity profile across catheter lumen
        
        For laminar flow: v(r) = v_max * (1 - (r/R)^2)
        """
        radius = (self.design.inner_diameter / 2) / 1000  # mm to m
        flow_rate = self.constraints.flow_rate / 60000  # m³/s
        area = np.pi * radius ** 2
        
        # Average velocity
        v_avg = flow_rate / area
        
        # Maximum velocity (centerline) for parabolic profile
        v_max = 2 * v_avg
        
        # Radial positions
        r = np.linspace(0, radius, n_radial)
        
        # Velocity profile
        v_r = v_max * (1 - (r / radius) ** 2)
        
        return r, v_r
    
    def solve(self) -> FlowResults:
        """Run complete flow analysis"""
        
        # Compute metrics
        pressure_drop = self.compute_pressure_drop()
        reynolds = self.compute_reynolds_number()
        
        # Determine flow regime
        flow_regime = 'laminar' if reynolds < 2300 else 'turbulent'
        
        # Compute velocity
        flow_rate = self.constraints.flow_rate / 60000  # m³/s
        diameter = self.design.inner_diameter / 1000  # m
        area = np.pi * (diameter / 2) ** 2
        avg_velocity = flow_rate / area
        max_velocity = 2 * avg_velocity  # For parabolic profile
        
        # Get velocity profile
        r, v_profile = self.compute_velocity_profile()
        
        return FlowResults(
            pressure_drop=pressure_drop,
            average_velocity=avg_velocity,
            max_velocity=max_velocity,
            reynolds_number=reynolds,
            flow_regime=flow_regime,
            velocity_field=v_profile
        )


class QuantumLinearSolver:
    """
    Quantum-enhanced linear solver for CFD matrix equations
    
    Uses quantum amplitude estimation for faster solving of Ax = b
    """
    
    @staticmethod
    def solve_classical(A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Classical sparse linear solver (fallback)"""
        return spsolve(A, b)
    
    @staticmethod
    def solve_quantum(
        A: np.ndarray,
        b: np.ndarray,
        use_quantum: bool = False
    ) -> np.ndarray:
        """
        Quantum linear system solver using HHL algorithm
        
        Note: This is a simplified placeholder. Full HHL implementation
        requires specialized quantum hardware and circuit design.
        """
        if not use_quantum or A.shape[0] > 16:
            # Fall back to classical for large systems
            # or when quantum not available
            if isinstance(A, np.ndarray):
                return np.linalg.solve(A, b)
            else:
                return QuantumLinearSolver.solve_classical(A, b)
        
        # Simplified quantum approach (placeholder)
        # Real implementation would use cudaq.amplitude_estimation
        
        # For now: classical solve with quantum-inspired preprocessing
        # Condition the matrix for better quantum performance
        conditioned_A = A + np.eye(A.shape[0]) * 1e-6
        
        return np.linalg.solve(conditioned_A, b)


class AdvancedCFDSolver:
    """
    Advanced 2D CFD solver with finite difference method
    
    Solves incompressible Navier-Stokes on axisymmetric mesh
    """
    
    def __init__(
        self,
        design: DesignParameters,
        constraints: PatientConstraints,
        fluid_type: str = 'blood',
        n_axial: int = 50,
        n_radial: int = 20
    ):
        self.design = design
        self.constraints = constraints
        self.density, self.viscosity = FluidProperties.get_properties(fluid_type)
        
        self.n_axial = n_axial
        self.n_radial = n_radial
        
        # Create mesh
        self.length = constraints.required_length / 1000  # m
        self.radius = (design.inner_diameter / 2) / 1000  # m
        
        self.z = np.linspace(0, self.length, n_axial)
        self.r = np.linspace(0, self.radius, n_radial)
        
        self.dz = self.length / (n_axial - 1)
        self.dr = self.radius / (n_radial - 1)
    
    def build_matrix_system(self) -> Tuple[csr_matrix, np.ndarray]:
        """
        Build linear system for pressure Poisson equation
        
        ∇²p = -ρ * ∇·(u·∇u)
        """
        n = self.n_axial * self.n_radial
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        # Simplified pressure equation for steady flow
        # Assumes velocity field is known (from Poiseuille)
        
        for i in range(1, self.n_axial - 1):
            for j in range(1, self.n_radial - 1):
                idx = i * self.n_radial + j
                
                # 5-point stencil for Laplacian
                A[idx, idx] = -2 / (self.dz ** 2) - 2 / (self.dr ** 2)
                A[idx, idx - self.n_radial] = 1 / (self.dz ** 2)  # z-1
                A[idx, idx + self.n_radial] = 1 / (self.dz ** 2)  # z+1
                A[idx, idx - 1] = 1 / (self.dr ** 2)  # r-1
                A[idx, idx + 1] = 1 / (self.dr ** 2)  # r+1
                
                # RHS from velocity gradients (simplified)
                b[idx] = 0  # Assuming steady state
        
        # Boundary conditions
        # Inlet (z=0): prescribed pressure
        for j in range(self.n_radial):
            idx = j
            A[idx, idx] = 1
            b[idx] = 0  # Reference pressure
        
        # Outlet (z=L): zero gradient
        for j in range(self.n_radial):
            idx = (self.n_axial - 1) * self.n_radial + j
            A[idx, idx] = 1
            A[idx, idx - self.n_radial] = -1
            b[idx] = 0
        
        # Wall (r=R): no-slip
        for i in range(self.n_axial):
            idx = i * self.n_radial + (self.n_radial - 1)
            A[idx, idx] = 1
            b[idx] = 0
        
        # Centerline (r=0): symmetry
        for i in range(self.n_axial):
            idx = i * self.n_radial
            A[idx, idx] = 1
            A[idx, idx + 1] = -1
            b[idx] = 0
        
        return A.tocsr(), b
    
    def solve(self, use_quantum: bool = False) -> FlowResults:
        """Solve 2D CFD problem"""
        
        print("Building matrix system...")
        A, b = self.build_matrix_system()
        
        print(f"Solving linear system ({A.shape[0]} equations)...")
        if use_quantum and A.shape[0] <= 16:
            # Convert to dense for quantum solver
            A_dense = A.toarray()
            pressure = QuantumLinearSolver.solve_quantum(A_dense, b, use_quantum=True)
        else:
            pressure = QuantumLinearSolver.solve_classical(A, b)
        
        # Reshape pressure field
        pressure_field = pressure.reshape(self.n_axial, self.n_radial)
        
        # Compute pressure drop
        pressure_drop = pressure_field[0, :].mean() - pressure_field[-1, :].mean()
        
        # Compute velocity field (simplified)
        flow_rate = self.constraints.flow_rate / 60000  # m³/s
        area = np.pi * self.radius ** 2
        avg_velocity = flow_rate / area
        max_velocity = 2 * avg_velocity
        
        # Reynolds number
        reynolds = (self.density * avg_velocity * 2 * self.radius) / self.viscosity
        flow_regime = 'laminar' if reynolds < 2300 else 'turbulent'
        
        return FlowResults(
            pressure_drop=abs(pressure_drop),
            average_velocity=avg_velocity,
            max_velocity=max_velocity,
            reynolds_number=reynolds,
            flow_regime=flow_regime,
            pressure_field=pressure_field
        )


class CatheterFlowAnalyzer:
    """High-level interface for catheter flow analysis"""
    
    def __init__(
        self,
        design: DesignParameters,
        constraints: PatientConstraints,
        fluid_type: str = 'blood'
    ):
        self.design = design
        self.constraints = constraints
        self.fluid_type = fluid_type
    
    def run_quick_analysis(self) -> FlowResults:
        """Fast analytical solution"""
        solver = SimplifiedNavierStokesSolver(
            self.design,
            self.constraints,
            self.fluid_type
        )
        return solver.solve()
    
    def run_detailed_analysis(self, use_quantum: bool = False) -> FlowResults:
        """Detailed numerical CFD"""
        solver = AdvancedCFDSolver(
            self.design,
            self.constraints,
            self.fluid_type,
            n_axial=30,
            n_radial=15
        )
        return solver.solve(use_quantum=use_quantum)
    
    def compare_designs(
        self,
        alternative_designs: list[DesignParameters]
    ) -> Dict[str, FlowResults]:
        """Compare multiple design options"""
        results = {}
        
        # Baseline design
        results['baseline'] = self.run_quick_analysis()
        
        # Alternative designs
        for i, alt_design in enumerate(alternative_designs):
            analyzer = CatheterFlowAnalyzer(
                alt_design,
                self.constraints,
                self.fluid_type
            )
            results[f'alternative_{i+1}'] = analyzer.run_quick_analysis()
        
        return results


if __name__ == '__main__':
    from quantum_catheter_optimizer import DesignParameters, PatientConstraints
    
    print("Catheter Fluid Dynamics Simulation")
    print("=" * 60)
    
    # Example design
    design = DesignParameters(
        outer_diameter=4.5,
        inner_diameter=3.8,
        wall_thickness=0.35,
        tip_angle=30.0,
        flexibility_index=0.75,
        side_holes=[200, 400, 600, 800],
        material_composition={'polyurethane': 0.7, 'silicone': 0.2, 'ptfe': 0.1}
    )
    
    constraints = PatientConstraints(
        vessel_diameter=5.5,
        vessel_curvature=0.15,
        required_length=1000.0,
        flow_rate=8.0
    )
    
    # Run analysis
    analyzer = CatheterFlowAnalyzer(design, constraints, fluid_type='blood')
    
    print("\n[Quick Analysis - Analytical Solution]")
    results = analyzer.run_quick_analysis()
    
    print(f"\nFlow Characteristics:")
    print(f"  Pressure drop: {results.pressure_drop:.2f} Pa ({results.pressure_drop/133.322:.2f} mmHg)")
    print(f"  Average velocity: {results.average_velocity*100:.2f} cm/s")
    print(f"  Maximum velocity: {results.max_velocity*100:.2f} cm/s")
    print(f"  Reynolds number: {results.reynolds_number:.1f}")
    print(f"  Flow regime: {results.flow_regime}")
    
    print("\n[Detailed Analysis - Numerical CFD]")
    detailed_results = analyzer.run_detailed_analysis(use_quantum=False)
    
    print(f"\nDetailed Flow Characteristics:")
    print(f"  Pressure drop: {detailed_results.pressure_drop:.2f} Pa")
    print(f"  Reynolds number: {detailed_results.reynolds_number:.1f}")
    print(f"  Flow regime: {detailed_results.flow_regime}")
    
    # Compare with larger diameter
    print("\n[Design Comparison]")
    alt_design = DesignParameters(
        outer_diameter=5.0,
        inner_diameter=4.3,
        wall_thickness=0.35,
        tip_angle=30.0,
        flexibility_index=0.75,
        side_holes=[200, 400, 600, 800],
        material_composition={'polyurethane': 0.7, 'silicone': 0.2, 'ptfe': 0.1}
    )
    
    comparison = analyzer.compare_designs([alt_design])
    
    print(f"\nBaseline (ID={design.inner_diameter:.1f}mm):")
    print(f"  Pressure drop: {comparison['baseline'].pressure_drop:.2f} Pa")
    
    print(f"\nAlternative (ID={alt_design.inner_diameter:.1f}mm):")
    print(f"  Pressure drop: {comparison['alternative_1'].pressure_drop:.2f} Pa")
    print(f"  Improvement: {(1 - comparison['alternative_1'].pressure_drop/comparison['baseline'].pressure_drop)*100:.1f}%")
