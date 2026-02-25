# Finite Mathematical Framework for Quantum-Enhanced MRI Reconstruction

**Authors:** NeuroPulse Research Team  
**Date:** February 11, 2026

---

## Abstract

This report presents the finite mathematical derivations underpinning the *NeuroPulse* MRI reconstruction engine. We detail the discrete operators used for signal acquisition, the statistical mechanics of quantum photon counting, and the optimization functionals for localized magnetic field shimming. These derivations provide the rigorous theoretical basis for the enhanced signal-to-noise ratio (SNR) and spatial resolution achieved in the latest simulation modules.

---

## 1. Introduction

Modern Magnetic Resonance Imaging (MRI) reconstruction relies heavily on discrete mathematical transformations to convert raw k-space data into interpretable images. The *NeuroPulse* simulator integrates classical reconstruction techniques with novel quantum-statistical models. This document formalizes these models using finite mathematics, ensuring computability and numerical stability.

## 2. Finite Mathematical Derivations

### 2.1 Discrete Fourier Transform and K-Space

The fundamental operation in MRI reconstruction is the recovery of the image space density $M(x,y)$ from the acquired k-space signal $S(u,v)$. In our digital simulation, this is governed by the Inverse Discrete Fourier Transform (IDFT):

$$ M(x,y) = \frac{1}{N^2} \sum_{u=0}^{N-1} \sum_{v=0}^{N-1} S(u,v) \cdot e^{i 2\pi (\frac{ux}{N} + \frac{vy}{N})} $$

Where $N$ is the resolution (e.g., 128 or 256). This finite summation replaces the continuous integral of classical analytic MRI theory, introducing discrete sampling artifacts which are mitigated via windowing functions.

### 2.2 Finite Difference Gradients for Edge Detection

To enhance vascular structures and quantify image sharpness, we employ a discrete gradient operator. The gradient magnitude $|\nabla M|$ is approximated using central finite differences:

$$ \frac{\partial M}{\partial x} \approx \frac{M_{i+1,j} - M_{i-1,j}}{2\Delta x} $$

$$ \frac{\partial M}{\partial y} \approx \frac{M_{i,j+1} - M_{i,j-1}}{2\Delta y} $$

The Euclidean magnitude is thus derived as:

$$ |\nabla M|_{i,j} = \sqrt{ \left(\frac{M_{i+1,j} - M_{i-1,j}}{2}\right)^2 + \left(\frac{M_{i,j+1} - M_{i,j-1}}{2}\right)^2 } $$

This operator forms the basis of our "imitation reasoning" for edge enhancement in quantum sequences.

### 2.3 Quantum Photon Counting Statistics

The **Quantum Photon Counting** sequence simulates the discrete arrival of photons, modeled by a Poisson distribution. Unlike classical additive Gaussian noise, the signal $S$ is a random variable $X$ dependent on the ideal magnetization $M_{ideal}$:

$$ P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

Where the rate parameter $\lambda$ is proportional to the proton density and quantum efficiency $\eta$:

$$ \lambda(x,y) = \eta \cdot M_{ideal}(x,y) $$

The reconstructed signal is the realized count $k$, normalized by the flux constant. This model naturally captures "shot noise" and demonstrates that SNR scales with $\sqrt{N_{photons}}$.

### 2.4 Localized Shimming Optimization

To maximize homogeneity in a specific Region of Interest (ROI), we solve a regularized least-squares problem. Let $\mathbf{B}$ be the coil sensitivity matrix restricted to the ROI, and $\mathbf{w}$ be the complex shim weights. We minimize the cost function $J(\mathbf{w})$:

$$ J(\mathbf{w}) = || \mathbf{B}\mathbf{w} - \mathbf{m}_{target} ||^2 + \lambda ||\mathbf{w}||^2 $$

The optimal weights $\mathbf{w}_{opt}$ are found by setting $\nabla J = 0$, leading to the normal equations:

$$ \mathbf{w}_{opt} = (\mathbf{B}^H \mathbf{B} + \lambda \mathbf{I})^{-1} \mathbf{B}^H \mathbf{m}_{target} $$

This Tikhonov-regularized solution prevents singular conditions when coil sensitivities are collinear.

### 2.5 Vascular Coil Sensitivity

For the **Optimized Vascular Tradeoff Coil**, the sensitivity profile $S(r)$ is a convex combination of a high-resolution profile $S_{res}$ and a high-SNR profile $S_{snr}$, controlled by the parameter $\alpha \in [0,1]$:

$$ S(r; \alpha) = \alpha \cdot S_{snr}(r) + (1-\alpha) \cdot S_{res}(r) $$

Where the profiles are modeled as Gaussian decays with distinct variances $\sigma_{snr} > \sigma_{res}$:

$$ S_{k}(r) \propto e^{-\frac{r^2}{2\sigma_k^2}} $$

## 3. Conclusion

The integration of these finite mathematical models allows *NeuroPulse* to simulate advanced MRI phenomena with high fidelity. The move from continuous approximations to rigorous discrete formulations ensures that our simulation results—particularly regarding quantum noise and localized shimming—are statistically robust and physically meaningful.

## References

1.  Haacke, E. M., et al. *Magnetic Resonance Imaging: Physical Principles and Sequence Design*. Wiley-Liss, 1999.
2.  Glaser, S. J., et al. "Training Schrödinger's cat: quantum optimal control." *The European Physical Journal D*, 2015.
3.  Lustig, M., et al. "Sparse MRI: The application of compressed sensing for rapid MR imaging." *Magnetic Resonance in Medicine*, 2007.
