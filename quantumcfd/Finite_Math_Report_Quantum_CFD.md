# Technical Report: Finite Math & Physics of Quantum CFD Solvers

**Date:** February 6, 2026
**Author:** Antigravity AI
**Subject:** Mathematical Formalism of the Quantum Hyper-Fluid Solver

## 1. Introduction

This report details the mathematical framework underpinning the Quantum CFD application. The solver implements a four-dimensional (4D) incompressible Navier-Stokes equation solver tailored for hyper-fluid dynamics, integrating quantum statistical mechanics for turbulence modeling and quantum interferometry for flow signature analysis.

## 2. Governing Equations

The core of the solver relies on the multidimensional Incompressible Navier-Stokes equations, generalized to $D=4$ dimensions (spatial coordinates $x, y, z, w$).

### 2.1. Momentum Equation

The conservation of momentum for an incompressible fluid in 4D is given by:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{F}_{quantum}
$$

Where:
*   $\mathbf{u} = (u, v, w_{vel}, a_{vel})$ is the 4D velocity vector field.
*   $\rho$ is the fluid density.
*   $p$ is the pressure scalar field.
*   $\nu$ is the kinematic viscosity.
*   $\mathbf{F}_{quantum}$ represents the stochastic quantum forcing term.
*   $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} + \frac{\partial^2}{\partial w^2}$ is the 4D Laplacian.

### 2.2. Continuity Equation

The incompressibility constraint mandates a divergence-free velocity field:

$$
\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w_{vel}}{\partial z} + \frac{\partial a_{vel}}{\partial w} = 0
$$

## 3. Finite Difference Discretization & Numerical Method

The solver employs the **Chorin Projection Method** (splitting method) to decouple the pressure and velocity updates. The problem is discretized on a 4D Cartesian grid using Finite Difference Methods (FDM).

### 3.1. The Projection Algorithm

1.  **Intermediate Velocity Step**: Compute a tentative velocity field $\mathbf{u}^*$ ignoring pressure (or using old pressure) but including advection, diffusion, and forcing terms.
    $$
    \mathbf{u}^* = \mathbf{u}^n + \Delta t \left( -(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n + \mathbf{F}_{quantum} \right)
    $$
    *Note: The current implementation simplifies advection, focusing primarily on the domination of the pressure/viscous terms for this specialized quantum fluid model.*

2.  **Pressure Poisson Equation (PPE)**: By enforcing $\nabla \cdot \mathbf{u}^{n+1} = 0$, we derive a Poisson equation for pressure:
    $$
    \nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
    $$

3.  **Velocity Correction**: Update the velocity field using the new pressure gradient to project it onto the divergence-free subspace:
    $$
    \mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}
    $$

### 3.2. Discrete Operators

The spatial derivatives are approximated using **Central Differences** of order $O(\Delta x^2)$.

**Gradient (e.g., $\frac{\partial p}{\partial x}$ at grid point $i,j,k,l$):**
$$
\left( \frac{\partial p}{\partial x} \right)_{i,j,k,l} \approx \frac{p_{i+1,j,k,l} - p_{i-1,j,k,l}}{2 \Delta x}
$$

**Laplacian (e.g., $\nabla^2 p$):**
In 4D, the 9-point stencil (center + 2 neighbors per axis) is used:
$$
(\nabla^2 p)_{i,j,k,l} \approx \sum_{d \in \{x,y,z,w\}} \frac{p_{next(d)} - 2p_{curr} + p_{prev(d)}}{\Delta d^2}
$$

## 4. Quantum Extensions

### 4.1. Stochastic Quantum Forcing

The term $\mathbf{F}_{quantum}$ models fluctuations derived from quantum statistical distributions. We utilize the **Fermi-Dirac** formulation to model non-classical turbulence injection:

$$
F_{q}(x) \propto \frac{1}{e^{(E(x) - \mu)/k_B T} + 1} \cdot \mathcal{N}(0, 1)
$$

This introduces coordinate-dependent stochasticity into the momentum equation, simulating quantum vacuum fluctuations interacting with the hyper-fluid.

### 4.2. Quantum Surface Integrals

Aerodynamic forces (Lift and Drag) are computed by integrating the stress tensor over the boundary $\partial \Omega$ of the obstacle.

$$
\mathbf{F}_{aero} = \oint_{\partial \Omega} \sigma \cdot \mathbf{n} \, dA
$$

Where $\sigma = -p \mathbf{I} + \tau$ is the stress tensor.
In the inviscid limit approximation often used for calculating form drag in this solver:
$$
\mathbf{F}_{drag} \approx \oint_{\partial \Omega} -p (\mathbf{n} \cdot \hat{x}) dA
$$

The solver numerically integrates this over the discrete voxelized surface of the high-dimensional airfoil.

### 4.3. Quantum Interferometry Signatures

Flow stability is analyzed via **Swap Test** fidelity metrics, measuring the overlap between velocity states at subsequent timesteps $|\psi(t)\rangle$ and $|\psi(t+\Delta t)\rangle$:

$$
\text{Fidelity} = |\langle \mathbf{u}(t) | \mathbf{u}(t+\Delta t) \rangle|^2
$$

This metric serves as a high-sensitivity detector for flow separation and turbulent onset.

---
*Generated by Antigravity AI Research Module*
