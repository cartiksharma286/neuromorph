# Quantum Phase MRI Simulation: Finite Mathematics & Circuit Derivations

**NeuroPulse Physics Engine v3.0**  
**Date:** January 10, 2026  
**Classification:** Advanced Quantum MRI Physics

---

## Executive Summary

This report presents the complete finite mathematical framework for quantum phase-based MRI reconstruction, including discrete Bloch equation solutions, RF coil circuit topology derivations, and Berry phase accumulation in topological pulse sequences. All derivations are presented in discrete form suitable for numerical implementation.

---

## 1. Discrete Bloch Equations for Quantum Phase Evolution

### 1.1 Finite Difference Formulation

The continuous Bloch equations are discretized using forward Euler method with time step $\Delta t$:

$$ M_x^{n+1} = M_x^n + \Delta t \left[ \gamma (M_y^n B_z - M_z^n B_y) - \frac{M_x^n}{T_2} \right] $$

$$ M_y^{n+1} = M_y^n + \Delta t \left[ \gamma (M_z^n B_x - M_x^n B_z) - \frac{M_y^n}{T_2} \right] $$

$$ M_z^{n+1} = M_z^n + \Delta t \left[ \gamma (M_x^n B_y - M_y^n B_x) - \frac{M_z^n - M_0}{T_1} \right] $$

where:
- $M^n = (M_x^n, M_y^n, M_z^n)$ is the magnetization vector at time step $n$
- $\gamma = 42.58 \text{ MHz/T}$ is the gyromagnetic ratio
- $B = (B_x, B_y, B_z)$ is the magnetic field vector
- $T_1, T_2$ are relaxation time constants

### 1.2 Stability Criterion

For numerical stability, the time step must satisfy:

$$ \Delta t < \frac{2}{\gamma B_{\max} + \max(1/T_1, 1/T_2)} $$

---

## 2. Quantum Berry Phase Accumulation

### 2.1 Geometric Phase in Parameter Space

The Berry phase $\gamma_B$ accumulated during adiabatic evolution along a closed path $C$ in parameter space is:

$$ \gamma_B(C) = i \oint_C \langle \psi_n(\mathbf{R}) | \nabla_{\mathbf{R}} | \psi_n(\mathbf{R}) \rangle \cdot d\mathbf{R} $$

For MRI applications with spatially varying gradients, this becomes:

$$ \gamma_B(\mathbf{r}) = \int_0^T \mathbf{A}(\mathbf{r}, t) \cdot \frac{d\mathbf{r}}{dt} dt $$

where $\mathbf{A}$ is the Berry connection (gauge potential).

### 2.2 Discrete Berry Phase Calculation

In discrete form with $N$ time steps:

$$ \gamma_B \approx \sum_{n=0}^{N-1} \mathbf{A}_n \cdot \Delta \mathbf{r}_n $$

where:

$$ \mathbf{A}_n = \text{Im} \left[ \langle \psi_n | \frac{\psi_{n+1} - \psi_n}{\Delta t} \rangle \right] $$

### 2.3 Berry Curvature and Topological Invariant

The Berry curvature tensor is:

$$ \Omega_{ij}(\mathbf{k}) = \partial_{k_i} A_j - \partial_{k_j} A_i $$

The Chern number (topological invariant) is:

$$ C = \frac{1}{2\pi} \int_{BZ} \Omega_{xy}(\mathbf{k}) d^2k $$

For discrete k-space sampling on an $N \times N$ grid:

$$ C \approx \frac{1}{2\pi} \sum_{i=1}^{N} \sum_{j=1}^{N} \Omega_{xy}(k_i, k_j) \Delta k_x \Delta k_y $$

---

## 3. RF Coil Circuit Topology

### 3.1 Birdcage Coil Impedance Matrix

For an $N$-rung birdcage coil, the impedance matrix $\mathbf{Z}$ is:

$$ Z_{ij} = \begin{cases}
R + j\omega L_{leg} + \frac{1}{j\omega C_{ring}} & i = j \\
\frac{1}{j\omega C_{ring}} & |i-j| = 1 \text{ or } N-1 \\
0 & \text{otherwise}
\end{cases} $$

The resonant frequencies are:

$$ \omega_m = \frac{1}{\sqrt{L_{leg} C_{ring}}} \left| 2\sin\left(\frac{m\pi}{N}\right) \right|^{-1}, \quad m = 1, 2, \ldots, N/2 $$

### 3.2 Mutual Inductance Calculation

For two circular loops with radii $a_1, a_2$ separated by distance $d$:

$$ M_{12} = \mu_0 \sqrt{a_1 a_2} \left[ \left(2 - k^2\right) K(k) - 2E(k) \right] $$

where:
- $k^2 = \frac{4a_1 a_2}{(a_1 + a_2)^2 + d^2}$
- $K(k), E(k)$ are complete elliptic integrals

Discrete approximation using mesh elements:

$$ M_{ij} \approx \sum_{p=1}^{N_1} \sum_{q=1}^{N_2} \frac{\mu_0}{4\pi} \frac{\mathbf{J}_i^p \cdot \mathbf{J}_j^q}{|\mathbf{r}_{ij}^{pq}|} \Delta A_p \Delta A_q $$

### 3.3 Quantum Surface Lattice Coupling

For a hexagonal lattice of quantum coil elements, the coupling Hamiltonian is:

$$ H_{coupling} = \sum_{\langle i,j \rangle} J_{ij} (\sigma_i^+ \sigma_j^- + \sigma_i^- \sigma_j^+) + \sum_i h_i \sigma_i^z $$

The flux quantization condition:

$$ \Phi = n \Phi_0, \quad \Phi_0 = \frac{h}{2e} = 2.067 \times 10^{-15} \text{ Wb} $$

---

## 4. K-Space Trajectory and Reconstruction

### 4.1 Discrete Fourier Transform

The k-space signal is related to the image by discrete Fourier transform:

$$ S(k_x, k_y) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} \rho(m,n) e^{-2\pi i (k_x m/M + k_y n/N)} $$

The inverse transform for reconstruction:

$$ \rho(m,n) = \frac{1}{MN} \sum_{k_x=0}^{M-1} \sum_{k_y=0}^{N-1} S(k_x, k_y) e^{2\pi i (k_x m/M + k_y n/N)} $$

### 4.2 Non-Uniform FFT (NUFFT)

For non-Cartesian trajectories, the NUFFT approximation:

$$ S(\mathbf{k}_j) \approx \sum_{m \in \mathcal{N}(\mathbf{k}_j)} c_m \rho_m e^{-2\pi i \mathbf{k}_j \cdot \mathbf{r}_m} $$

where $\mathcal{N}(\mathbf{k}_j)$ is the neighborhood of $\mathbf{k}_j$ and $c_m$ are Kaiser-Bessel interpolation coefficients.

### 4.3 Parallel Imaging with SENSE

For $N_c$ coils with sensitivity maps $C_j(\mathbf{r})$:

$$ S_j(\mathbf{k}) = \int C_j(\mathbf{r}) \rho(\mathbf{r}) e^{-2\pi i \mathbf{k} \cdot \mathbf{r}} d\mathbf{r} $$

The SENSE reconstruction solves:

$$ \hat{\rho} = \arg\min_\rho \sum_{j=1}^{N_c} \| \mathcal{F}\{C_j \rho\} - S_j \|_2^2 + \lambda R(\rho) $$

Discrete form:

$$ \hat{\rho}_n = \left( \sum_{j=1}^{N_c} C_j^* C_j + \lambda \mathbf{L}^T \mathbf{L} \right)^{-1} \sum_{j=1}^{N_c} C_j^* \mathcal{F}^{-1}\{S_j\} $$

---

## 5. Quantum Low Energy Beam Focusing

### 5.1 Energy Distribution Function

The beam intensity profile with quantum focusing:

$$ I(\mathbf{r}) = I_0 \left[ 1 + \eta \mathcal{H}(\chi(\mathbf{r})) \right] \exp\left(-\frac{|\mathbf{r} - \mathbf{r}_0|^2}{2\sigma^2}\right) $$

where:
- $\mathcal{H}(\chi)$ is the entropy-based attention function
- $\chi(\mathbf{r})$ is the local image complexity measure

### 5.2 Entropy-Driven Denoising

The local entropy at position $\mathbf{r}$ in a window $W$:

$$ \mathcal{H}(\mathbf{r}) = -\sum_{i=1}^{L} p_i(\mathbf{r}) \log_2 p_i(\mathbf{r}) $$

where $p_i(\mathbf{r})$ is the probability of intensity level $i$ in window $W$ centered at $\mathbf{r}$.

Discrete implementation:

$$ \mathcal{H}_n = -\sum_{i=0}^{255} \frac{h_i^n}{N_w} \log_2\left(\frac{h_i^n}{N_w}\right) $$

where $h_i^n$ is the histogram count for intensity $i$ in the $n$-th window.

---

## 6. Signal-to-Noise Ratio Analysis

### 6.1 SNR for Sum-of-Squares Reconstruction

For $N_c$ coils with uncorrelated noise:

$$ \text{SNR} = \frac{\sqrt{\sum_{j=1}^{N_c} |C_j \rho|^2}}{\sigma_n \sqrt{N_c}} $$

### 6.2 g-Factor for Parallel Imaging

The geometry factor quantifying SNR loss:

$$ g(\mathbf{r}) = \sqrt{[(\mathbf{C}^H \Psi^{-1} \mathbf{C})^{-1}]_{ii} \cdot [\mathbf{C}^H \Psi^{-1} \mathbf{C}]_{ii}} $$

where $\Psi$ is the noise covariance matrix.

For $R$-fold acceleration:

$$ \text{SNR}_{acc} = \frac{\text{SNR}_{full}}{g \sqrt{R}} $$

---

## 7. Shimming Optimization

### 7.1 B1+ Homogeneity Objective

The shimming problem minimizes field inhomogeneity:

$$ \mathbf{w}_{opt} = \arg\min_{\mathbf{w}} \left\| |\mathbf{A}\mathbf{w}| - \mathbf{T} \right\|_2^2 + \lambda \|\mathbf{w}\|_2^2 $$

where:
- $\mathbf{A}$ is the $N_{voxel} \times N_{coil}$ sensitivity matrix
- $\mathbf{T}$ is the target field
- $\lambda$ is the regularization parameter

### 7.2 Iterative Solution

Using gradient descent with step size $\alpha$:

$$ \mathbf{w}^{k+1} = \mathbf{w}^k - \alpha \nabla_{\mathbf{w}} \left[ \| |\mathbf{A}\mathbf{w}^k| - \mathbf{T} \|_2^2 + \lambda \|\mathbf{w}^k\|_2^2 \right] $$

Discrete gradient:

$$ \nabla_{\mathbf{w}} = 2\mathbf{A}^H \text{diag}\left(\frac{|\mathbf{A}\mathbf{w}| - \mathbf{T}}{|\mathbf{A}\mathbf{w}|}\right) \mathbf{A}\mathbf{w} + 2\lambda\mathbf{w} $$

---

## 8. Numerical Implementation Details

### 8.1 Grid Resolution

For a field of view (FOV) of $L \times L$ with $N \times N$ pixels:

$$ \Delta x = \Delta y = \frac{L}{N} $$

$$ \Delta k_x = \Delta k_y = \frac{1}{L} $$

### 8.2 Gradient Strength

The gradient required for k-space traversal:

$$ G = \frac{k_{max}}{\gamma t_{read}} = \frac{N/(2L)}{\gamma t_{read}} $$

For $N = 128$, $L = 25$ cm, $t_{read} = 10$ ms:

$$ G = \frac{128/(2 \times 0.25)}{42.58 \times 10^6 \times 0.01} = 6.0 \text{ mT/m} $$

### 8.3 Sampling Theorem

Nyquist criterion for artifact-free reconstruction:

$$ \Delta k \leq \frac{1}{2 \Delta x} $$

$$ k_{max} \geq \frac{N}{2L} $$

---

## 9. Quantum Entanglement Contrast

### 9.1 Entangled State Formulation

The two-contrast entangled state:

$$ |\Psi\rangle = \frac{1}{\sqrt{2}} (|T1\rangle \otimes |bright\rangle + |T2\rangle \otimes |dark\rangle) $$

Measurement yields superposition:

$$ M_{entangled} = \alpha M_{T1} + \beta M_{T2} + \gamma M_{T1} \odot M_{T2} $$

where $\odot$ denotes element-wise product (quantum correlation term).

### 9.2 Discrete Implementation

$$ M_{entangled}^{ij} = 0.5 M_{base}^{ij} + 0.25 M_{T1}^{ij} + 0.25 M_{T2}^{ij} $$

where:

$$ M_{T1}^{ij} = \rho^{ij} \left(1 - e^{-TR/T1^{ij}}\right) $$

$$ M_{T2}^{ij} = \rho^{ij} e^{-TE/T2^{ij}} $$

---

## 10. Computational Complexity

### 10.1 FFT Complexity

For $N \times N$ image:

$$ \mathcal{O}(N^2 \log N) $$

### 10.2 SENSE Reconstruction

Matrix inversion for each pixel:

$$ \mathcal{O}(N^2 N_c^3) $$

### 10.3 Berry Phase Calculation

For $N_t$ time steps and $N_p$ spatial points:

$$ \mathcal{O}(N_t N_p) $$

---

## Conclusion

This report provides the complete finite mathematical framework for quantum phase MRI simulation, including all discrete formulations necessary for numerical implementation. The derivations cover Bloch equation evolution, Berry phase accumulation, RF coil circuit analysis, k-space reconstruction, and advanced quantum-inspired contrast mechanisms.

All equations are presented in forms directly implementable in the NeuroPulse simulation engine, ensuring accurate modeling of quantum topological effects in magnetic resonance imaging.

---

**References:**
1. Berry, M. V. (1984). "Quantal Phase Factors Accompanying Adiabatic Changes"
2. Pruessmann, K. P. et al. (1999). "SENSE: Sensitivity Encoding for Fast MRI"
3. Hayes, C. E. et al. (1985). "An Efficient, Highly Homogeneous Radiofrequency Coil"
4. Sodickson, D. K. & Manning, W. J. (1997). "Simultaneous Acquisition of Spatial Harmonics"

**Document ID:** NP-QP-MRI-2026-001  
**Classification:** Advanced Physics Research  
**Generated by:** NeuroPulse Physics Engine v3.0
