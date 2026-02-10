
# NeuroPulse Clinical Physics Report
**Date:** January 10, 2026
**Simulation ID:** QuantumGenerativeRecon-quantum_vascular

---

## 1. Executive Summary
This report details the simulation results for the **Quantum Vascular** operating with **QuantumGenerativeRecon**.

## 2. Physics & Circuit Topology
This coil topology exploits the Berry Phase of adiabatic spin transport.

### Coil Derivation
$$ \gamma_n(C) = i \oint_C \langle \psi_n | \nabla | \psi_n \rangle \cdot d\mathbf{R} $$

### Pulse Sequence Physics
$$ S \propto \rho (1-e^{-TR/T1})e^{-TE/T2} $$

---

## 3. Finite Math & Discrete Derivations
$$ M_z^{sub} = M_z(t) \cdot e^{-\Delta t/T1} + M_0(1 - e^{-\Delta t/T1}) $$
$$ Z_{ij} = \sum \frac{\mu_0}{4\pi} \frac{\mathbf{J}_i \cdot \mathbf{J}_j}{|\mathbf{r}_{ij}|} \Delta A_k $$

---

## 4. Visual Reconstruction Data
![Reconstruction](static/report_images/recon.png)

## 5. Metrics
* **Contrast:** 0.1290
* **Sharpness:** 6.91
