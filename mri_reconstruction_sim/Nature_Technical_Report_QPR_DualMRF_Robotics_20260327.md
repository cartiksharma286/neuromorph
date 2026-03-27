# Nature-Style Technical Report
## Quantum Phase-Resolved Dual-MRF Pulse Sequences for Cardiac and Stroke Imaging with Interoperative MR-Guided Tumour Resection Robotics

**Authors:** NeuroMorph Imaging Systems Group  
**Date:** 2026-03-27

## Abstract
We present an integrated pulse-sequence framework combining cardiac cine imaging, acute stroke tissue characterization, and interoperative MR-guided tumour resection. The framework introduces elliptic modulo resonators for quantum phase-resolved RF control and dual MR fingerprinting (Dual-MRF) to jointly estimate relaxation and transport biomarkers. We derive finite reconstruction operators extending beyond classical partial Fourier methods by introducing fractional boundary gaps in k-space closure constraints. Simulation-grade sequence specification is released in a scanner-style .seq file for translational prototyping.

## 1. Clinical-Technical Motivation
A unified protocol is useful when a centre performs:
1. Cardiac viability and flow mapping.
2. Hyperacute stroke core-penumbra triage.
3. Intraoperative tumour resection with robotic guidance under MR thermometry.

Conventional workflows split these tasks across distinct pulse families, increasing latency and calibration overhead. We instead use one quantum phase-resolved scheduling engine with Dual-MRF readouts and robotics interoperability hooks.

## 2. Sequence Architecture
### 2.1 Elliptic Modulo Resonator
Resonator state is defined by the elliptic modular manifold

$$
\mathcal{E}_{q}: \frac{x^2}{a^2}+\frac{y^2}{b^2}=1 \pmod q,
$$

with $(a,b,q)=(1.0,0.64,17)$. Phase evolves by

$$
\theta_{n+1}=(3\theta_n+2)\bmod 2\pi,
$$

and the RF phase-resolved wave packet is

$$
\psi_n(t)=A_n\exp\left(i\left[\omega_n t+\theta_n+\frac{\pi n}{4}\right]\right), \quad n=0,\dots,7.
$$

### 2.2 Dual-MRF Streams
Two synchronous streams are played:
1. Stream A: variable TR/TE/FA schedule for T1-T2 tissue fingerprinting.
2. Stream B: diffusion-perfusion or flow-sensitive schedule for transport biomarkers.

Signal model:

$$
s_k = \rho\,f(T_1,T_2,D,f_{\text{flow}},\Delta\omega;\,u_k) + \epsilon_k,
$$

where $u_k$ contains pulse controls and $\epsilon_k$ denotes complex noise.

## 3. Protocol Blocks
### 3.1 Cardiac Imaging Block
Variable-cine schedule: TR in 4.8-7.3 ms, TE in 1.9-2.6 ms, ECG and respiratory synchronization.

### 3.2 Stroke Imaging Block
Diffusion-perfusion fingerprinting with $b\in\{0,150,500,1000,1500\}\,\text{s/mm}^2$ and multi-PLD ASL.

### 3.3 Tumour Resection Robotics Block
Interoperative data exchange uses ROS2 DDS + OpenIGTLink + DICOM-RT metadata, with:
1. Pose update target: 50 Hz.
2. End-to-end latency target: <20 ms.
3. Safety channels: SAR guard, magnet-zone lock, thermal freeze gate.

## 4. Finite Mathematics Beyond Partial Fourier
Classical partial Fourier recovers missing half-space under conjugate symmetry. Here we define a finite closure operator with fractional boundary gaps.

Let $\Omega_m$ be sampled k-space indices and $\Omega_u$ unsampled indices. Reconstruction seeks

$$
\hat{x}=\arg\min_x \|A_{\Omega_m}x-y\|_2^2 + \lambda_1\|Dx\|_1 + \lambda_2\|G_\alpha x\|_2^2.
$$

$G_\alpha$ is the fractional boundary-gap operator:

$$
(G_\alpha x)_k = x_k - \sum_{j=1}^{p} w_j^{(\alpha)} x_{k-j},
\qquad
w_j^{(\alpha)} = (-1)^j \binom{\alpha}{j},
$$

with $\alpha=1.35$ and finite stencil order $p=4$.

Boundary-gap decay law:

$$
\Delta_g(k)=\frac{g_0}{(k+\varphi)^\alpha},
\qquad
g_0=0.12,\ \varphi=1.61803398875.
$$

### 4.1 Worked Numeric Example
For $k=12$,

$$
\Delta_g(12)=\frac{0.12}{(12+1.61803398875)^{1.35}}
\approx 3.52\times 10^{-3}.
$$

This value is used as the boundary mismatch tolerance for k-space extrapolation at that index.

### 4.2 Finite-Resolution Gain Estimate
If baseline partial Fourier residual is $r_{PF}=0.018$ and proposed residual is $r_{FBG}=0.012$,

$$
\text{Relative gain}=\frac{r_{PF}-r_{FBG}}{r_{PF}}=\frac{0.006}{0.018}=0.333\,(33.3\%).
$$

## 5. Quantum Phase-Resolved Thermometry Coupling
Phase change from temperature follows PRF approximation

$$
\Delta T = \frac{\Delta\phi}{\gamma B_0 \alpha_{PRF} TE},
$$

with multi-echo robust estimate

$$
\Delta\phi = \arg\min_{\phi} \sum_{e=1}^{E} w_e\,\|s_e-\hat{s}_e(\phi)\|^2,
$$

where weights $w_e$ are derived from resonator-state confidence and echo SNR.

## 6. Implementation Artifacts
Generated artifacts:
1. Sequence file: scanner-style pulse sequence with three protocol blocks and reconstruction notes.
2. This technical report with explicit finite derivations and parameter values.

## 7. Translational Notes
Before deployment on patient workflows:
1. Validate SAR, gradient duty cycle, and PNS envelopes per vendor limits.
2. Verify robotic watchdog and emergency stop with hardware-in-the-loop tests.
3. Benchmark Dual-MRF dictionary mismatch under motion and off-resonance stress tests.

## 8. Conclusion
The proposed sequence stack unifies cardiac, stroke, and interoperative tumour-resection MRI under one quantum phase-resolved Dual-MRF architecture. Finite fractional boundary-gap operators provide a mathematically explicit extension beyond partial Fourier and can reduce closure residuals while preserving stability in high-acceleration acquisitions.
