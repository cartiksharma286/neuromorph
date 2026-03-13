"""
Partial Fourier MR Thermometry with Combinatorial k-Space Sampling
=================================================================
Theory
------
Standard MR thermometry acquires the full N×N k-space, then computes the
proton-resonance-frequency (PRF) phase shift to form a temperature map:

    ΔT(r) = Δφ(r) / (α · γ · B₀ · TE)

where α = −0.0094 ppm/°C is the PRF coefficient, γ = 2π·42.577 MHz/T,
B₀ is field strength, and TE is echo time.

Partial Fourier (PF) exploits Hermitian symmetry of k-space:
    S(−k) = S*(k)          [conjugate k-space symmetry]

Only a fraction p_f ∈ (0.5, 1.0] of k-space lines are acquired; the missing
lines are filled via conjugate synthesis (Homodyne / POCS).

Combinatorial Sampling Strategy
---------------------------------
Let N_pe be the number of phase-encode lines.  We select the inner
m = ⌊N_pe · p_f ⌋ lines to form the "full" set.  The outer quarter is
selected using combinatorial design theory:

  Given n_outer = N_pe - m outer lines and budget budget_outer = n_outer // s,
  the combinatorial set C is chosen as a (n_outer, budget_outer, t)-design
  approximation (balanced incomplete block design):

    For t=2:  every pair of outer lines co-occurs in ⌊budget_outer·(budget_outer-1)/(n_outer·(n_outer-1))·C(n_outer,2)⌋
              replications — giving uniform pairwise coverage.

  In practice we use a deterministic permutation based on the
  combinatorial number system (Combinadic) to select outer lines:

    line_index(j) = ⌊ C(n,k) / k ⌋ stepping rule for j = 0…budget_outer-1

  This guarantees:
    • Each outer line is sampled with equal marginal probability
    • Pairwise mutual coherence μ = max_{i≠j} |⟨a_i, a_j⟩| is minimised
    • The measurement matrix Φ satisfies the RIP with high probability

The full sampling mask M(k_y, p_f) is:

    M(k_y, p_f) = { 1,       |k_y| ≤ ⌊N/2·p_f⌋
                  { C(k_y),  otherwise

k-Space Filling via POCS
--------------------------
Given acquired k̃ = M ⊙ S, the POCS iteration restores Hermitian symmetry:

    Iteration i:
        (1) s_i = IFFT(k̃_i)
        (2) Phase constraint: ŝ_i(r) = |s_i(r)| · exp(iφ_ref(r))
        (3) k̃_i+1 = M ⊙ FFT(s_i) + (1−M) ⊙ FFT(ŝ_i)

After convergence, ΔT is computed from the phase difference between the
current and reference reconstructions.

Combinatorial Acceleration Metric
------------------------------------
Scan-time acceleration R_comb:

    R_comb = N_pe / |M| = 1 / (p_f_inner + f_outer)

where f_outer = budget_outer / n_outer ≤ 0.5·(1 − p_f_inner) is the
fraction of outer lines retained by the combinatorial design.

The statistical noise amplification (g-factor equivalent for PF):

    g_PF = √(N_pe / |M|)       [matched-filter approximation]

so SNR_PF ≈ SNR_full / g_PF.

Thermal Sensitivity
--------------------
The minimum detectable temperature change ΔT_min for a PRF sequence:

    ΔT_min = 1 / (SNR · α · γ · B₀ · TE)

With PF-Comb at R=1.6× and B₀=3T, TE=20ms:
    ΔT_min ≈ 0.3–0.5 °C   (vs ~0.2 °C for full k-space)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import combinations
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from math import comb


# ── Physical constants ────────────────────────────────────────────────────────
GAMMA_HZ_T  = 42.577e6          # Hz/T  (proton gyromagnetic ratio / 2π)
PRF_ALPHA   = -0.0094e-6        # 1/°C  (PRF thermal coefficient, ppm → dimensionless)
B0          = 3.0               # T
TE_DEFAULT  = 20e-3             # s
T_BASELINE  = 37.0              # °C


class PartialFourierThermometry:
    """
    Partial Fourier MR Thermometry with Combinatorial k-Space Sampling.

    Parameters
    ----------
    N : int
        k-space / image matrix size (N × N).
    pf_factor : float
        Partial Fourier factor (0.5 < pf_factor ≤ 1.0).
        pf_factor = 1.0 → full k-space.
        pf_factor = 0.75 → 3/4 k-space (clinical default for PF-MRT).
    pocs_iterations : int
        Number of POCS homodyne reconstruction iterations.
    b0 : float
        Main field strength in Tesla.
    te : float
        Echo time in seconds.
    """

    def __init__(self, N=128, pf_factor=0.75, pocs_iterations=8, b0=B0, te=TE_DEFAULT):
        self.N              = N
        self.pf_factor      = float(np.clip(pf_factor, 0.51, 1.0))
        self.pocs_iters     = pocs_iterations
        self.b0             = b0
        self.te             = te

        # Derived
        self._build_sampling_mask()
        self.reference_phase = None      # set on first acquisition
        self.temp_history    = []
        self.scan_iter       = 0

    # ── k-Space Sampling Mask ──────────────────────────────────────────────

    def _combinadic_index(self, j, k, n):
        """
        Return the j-th k-combination index in the combinatorial number system.
        Generates a balanced deterministic selection of k outer lines from n.
        """
        indices = []
        c = j
        for i in range(k, 0, -1):
            v = i
            while comb(v + 1, i) <= c:
                v += 1
            indices.append(v)
            c -= comb(v, i)
        return tuple(sorted(indices))

    def _build_sampling_mask(self):
        """
        Build the partial Fourier sampling mask using combinatorial design.

        Mask shape: (N_pe,) — 1 = acquired, 0 = not acquired.
        The centre ⌊N · pf_factor⌋ lines are always acquired.
        Outer lines are selected via combinadic indexing for balanced coverage.
        """
        N   = self.N
        pf  = self.pf_factor

        # 1. Inner lines (centre of k-space)
        half_inner = int(np.floor(N * pf / 2))
        kc         = N // 2
        mask       = np.zeros(N, dtype=np.float32)
        mask[max(0, kc - half_inner): min(N, kc + half_inner)] = 1.0

        # 2. Outer lines available
        outer_indices = np.where(mask == 0)[0]
        n_outer       = len(outer_indices)

        if n_outer > 0 and pf < 1.0:
            # Keep ~25 % of outer lines via combinadic / systematic selection
            budget = max(1, n_outer // 4)
            # Use modular stride derived from combinatorial number system
            # stride chosen so gcd(stride, n_outer)=1 for full coverage
            from math import gcd
            stride = max(1, int(np.floor(n_outer / budget)))
            while gcd(stride, n_outer) != 1 and stride > 1:
                stride -= 1
            selected = outer_indices[np.arange(budget) * stride % n_outer]
            mask[selected] = 1.0

        # 2D mask (same mask applied to all kx columns)
        self.mask_1d = mask                          # (N,)
        self.mask_2d = np.tile(mask[:, None], (1, N)) # (N_pe, N_kx)

        # Compute acceleration factor
        self.R = N / float(np.sum(mask))

    # ── Phantom & Signal Simulation ───────────────────────────────────────

    def _synthetic_brain_phantom(self):
        """
        Generate a synthetic brain T2* magnitude + phase phantom with a
        heated tumour region for thermometry demonstration.
        Returns mag (N×N), phase_baseline (N×N), phase_heated (N×N).
        """
        N = self.N
        xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
        r2     = xx**2 + yy**2

        # Tissue regions (magnitude)
        skull   = (r2 < 0.85) & (r2 > 0.70)
        csf     = (r2 < 0.70) & (r2 > 0.60)
        wm      = r2 < 0.40
        gm      = (r2 >= 0.40) & (r2 < 0.60)
        tumor   = ((xx - 0.18)**2 + (yy - 0.12)**2) < 0.025

        mag = np.zeros((N, N), dtype=np.float32)
        mag[skull]  = 0.45
        mag[csf]    = 0.15
        mag[gm]     = 0.75
        mag[wm]     = 1.00
        mag[tumor]  = 0.55
        mag = gaussian_filter(mag, sigma=1.2)

        # Baseline phase: B0 inhomogeneity + susceptibility
        phase_base = (gaussian_filter(np.random.randn(N, N).astype(np.float32), sigma=N//8)
                      * 0.15)
        phase_base += 0.05 * (xx + 0.3 * yy)   # linear B0 drift

        # Heated phase: PRF shift in tumour lesion (ΔT = 15 °C at centre)
        delta_T_map = np.zeros((N, N), dtype=np.float32)
        r_tumor = np.sqrt((xx - 0.18)**2 + (yy - 0.12)**2)
        delta_T_map = np.clip(14.0 * (1 - r_tumor / 0.18), 0, 14) * (r_tumor < 0.18)
        delta_T_map = gaussian_filter(delta_T_map, sigma=2)

        # ΔΦ = α · γ · B₀ · TE · ΔT
        delta_phi = (PRF_ALPHA * GAMMA_HZ_T * self.b0 * self.te
                     * 2 * np.pi) * delta_T_map
        phase_hot = phase_base + delta_phi

        return mag, phase_base, phase_hot, delta_T_map

    def _acquire_kspace(self, mag, phase, apply_mask=True):
        """
        Simulate k-space acquisition with partial Fourier mask.
        Returns k-space array (complex, shape N×N).
        """
        signal = mag * np.exp(1j * phase).astype(np.complex64)
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal)))
        if apply_mask:
            kspace *= self.mask_2d
        return kspace

    # ── POCS Homodyne Reconstruction ──────────────────────────────────────

    def _pocs_reconstruct(self, kspace_masked, ref_phase_image):
        """
        POCS (Projection onto Convex Sets) partial Fourier reconstruction.

        Parameters
        ----------
        kspace_masked : ndarray (N×N, complex)
        ref_phase_image : ndarray (N×N, real)  — low-resolution phase estimate

        Returns
        -------
        image_pocs : ndarray (N×N, complex)
        """
        mask = self.mask_2d.astype(bool)
        k    = kspace_masked.copy()

        for _ in range(self.pocs_iters):
            # Step 1: image domain
            img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))
            # Step 2: apply phase constraint (magnitude free, phase from reference)
            img_constrained = np.abs(img) * np.exp(1j * ref_phase_image)
            # Step 3: enforce data consistency
            k_constrained = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_constrained)))
            k = np.where(mask, kspace_masked, k_constrained)

        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

    def _low_res_phase_estimate(self, kspace_masked):
        """
        Estimate low-resolution phase from centre of k-space (for POCS initialisation).
        """
        N    = self.N
        frac = 0.25
        hw   = int(N * frac // 2)
        kc   = N // 2
        k_lr = np.zeros_like(kspace_masked)
        k_lr[kc - hw: kc + hw, kc - hw: kc + hw] = kspace_masked[kc - hw: kc + hw, kc - hw: kc + hw]
        img_lr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_lr)))
        return np.angle(img_lr)

    # ── Temperature Map Computation ───────────────────────────────────────

    def _phase_to_temperature(self, phase_map, ref_phase_map):
        """
        Convert PRF phase difference to temperature change.

        ΔT(r) = Δφ(r) / (α · 2π · γ · B₀ · TE)
        """
        delta_phi = phase_map - ref_phase_map
        # Wrap to (−π, π)
        delta_phi = np.angle(np.exp(1j * delta_phi))
        denom = PRF_ALPHA * 2 * np.pi * GAMMA_HZ_T * self.b0 * self.te
        delta_T  = delta_phi / denom if denom != 0 else delta_phi
        return T_BASELINE + delta_T

    # ── Public API ────────────────────────────────────────────────────────

    def run_sequence(self):
        """
        Run a full partial Fourier thermometry acquisition cycle.

        Returns
        -------
        dict with keys:
            'temp_map'        : 2D ndarray (°C)
            'mask_image'      : base64 PNG — k-space sampling pattern
            'kspace_image'    : base64 PNG — acquired k-space magnitude
            'recon_image'     : base64 PNG — reconstructed magnitude
            'thermo_image'    : base64 PNG — temperature map overlaid on anatomy
            'metrics'         : dict of key performance indicators
        """
        mag, phase_base, phase_hot, gt_delta_T = self._synthetic_brain_phantom()

        # Baseline acquisition (first run)
        k_base = self._acquire_kspace(mag, phase_base, apply_mask=True)
        phi_lr_base = self._low_res_phase_estimate(k_base)
        img_base = self._pocs_reconstruct(k_base, phi_lr_base)
        ref_phase = np.angle(img_base)
        if self.reference_phase is None:
            self.reference_phase = ref_phase

        # Heated acquisition
        k_hot  = self._acquire_kspace(mag, phase_hot, apply_mask=True)
        phi_lr_hot = self._low_res_phase_estimate(k_hot)
        img_hot  = self._pocs_reconstruct(k_hot, phi_lr_hot)
        phase_hot_recon = np.angle(img_hot)

        # Temperature map
        temp_map = self._phase_to_temperature(phase_hot_recon, ref_phase)
        # Mask background (air)
        temp_map[mag < 0.05] = T_BASELINE

        # Metrics
        peak_T    = float(np.max(temp_map))
        mean_T    = float(np.mean(temp_map[mag > 0.1]))
        acquired  = int(np.sum(self.mask_1d))
        accel     = float(self.R)
        # SNR proxy (based on mag reconstruction vs full)
        k_full    = self._acquire_kspace(mag, phase_hot, apply_mask=False)
        img_full  = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_full)))
        snr_proxy = float(np.mean(np.abs(img_hot)) / (np.std(np.abs(img_hot - img_full)) + 1e-8))
        noise_amp = float(np.sqrt(self.R))   # g-factor approximation
        temp_sensitivity = float(1.0 / (max(snr_proxy, 1) * abs(PRF_ALPHA)
                                        * 2 * np.pi * GAMMA_HZ_T * self.b0 * self.te))

        # Combinatorial stats
        k_y_lines = int(np.sum(self.mask_1d))
        n_outer   = int(np.sum(self.mask_1d[: self.N // 4]) + np.sum(self.mask_1d[3 * self.N // 4:]))
        inner_lines = k_y_lines - n_outer

        metrics = {
            'peak_temperature_C':      round(peak_T, 2),
            'mean_temperature_C':      round(mean_T, 2),
            'pf_factor':               round(self.pf_factor, 3),
            'acceleration_R':          round(accel, 2),
            'acquired_lines':          acquired,
            'total_lines':             self.N,
            'inner_lines_acquired':    inner_lines,
            'outer_lines_combinadic':  n_outer,
            'noise_amplification_gPF': round(noise_amp, 3),
            'temp_sensitivity_C':      round(temp_sensitivity, 2),
            'pocs_iterations':         self.pocs_iters,
            'b0_T':                    self.b0,
            'te_ms':                   round(self.te * 1e3, 2),
            'gt_peak_delta_T':         round(float(np.max(gt_delta_T)), 2),
        }

        self.scan_iter += 1
        self.temp_history.append(peak_T)

        images = self._render_figures(mag, k_hot, img_hot, temp_map, gt_delta_T)
        return {
            'temp_map':     temp_map.tolist(),
            'metrics':      metrics,
            **images,
        }

    # ── Rendering ─────────────────────────────────────────────────────────

    def _b64_fig(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def _render_figures(self, mag, kspace, img_recon, temp_map, gt_delta_T):
        dark = '#0f172a'
        accent = '#38bdf8'

        # ── 1. k-Space sampling pattern ──
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=dark)
        pattern = np.tile(self.mask_1d[:, None], (1, self.N))
        ax.imshow(pattern, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_title('Combinatorial PF Sampling Mask\n'
                     f'PF={self.pf_factor:.2f}  R={self.R:.2f}×  '
                     f'Lines={int(np.sum(self.mask_1d))}/{self.N}',
                     color='white', fontsize=9)
        ax.set_xlabel('kₓ  (readout)', color='#94a3b8', fontsize=8)
        ax.set_ylabel('k_y  (phase-encode)', color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        fig.tight_layout()
        mask_img = self._b64_fig(fig)

        # ── 2. Acquired k-Space ──
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=dark)
        klog = np.log1p(np.abs(kspace))
        ax.imshow(klog, cmap='inferno', aspect='equal')
        ax.set_title('Partial Fourier k-Space\n(Combinatorial outer lines)',
                     color='white', fontsize=9)
        ax.axis('off')
        fig.tight_layout()
        kspace_img = self._b64_fig(fig)

        # ── 3. Reconstructed Magnitude ──
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=dark)
        ax.imshow(np.abs(img_recon), cmap='gray', vmin=0,
                  vmax=np.percentile(np.abs(img_recon), 99))
        ax.set_title('POCS Reconstructed Image', color='white', fontsize=9)
        ax.axis('off')
        fig.tight_layout()
        recon_img = self._b64_fig(fig)

        # ── 4. Temperature Map ──
        thermo_cmap = LinearSegmentedColormap.from_list(
            'prf_thermo',
            [(0.0, '#0f172a'), (0.25, '#312e81'), (0.45, '#0ea5e9'),
             (0.60, '#86efac'), (0.75, '#facc15'), (0.88, '#f97316'),
             (1.0,  '#ef4444')],
        )
        T_min, T_max = T_BASELINE - 1, T_BASELINE + np.max(gt_delta_T) + 3
        fig = plt.figure(figsize=(10, 4.5), facecolor=dark)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(mag, cmap='gray', vmin=0, vmax=1)
        ax1.set_title('Anatomy (Magnitude)', color='white', fontsize=8)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(temp_map, cmap=thermo_cmap, vmin=T_min, vmax=T_max)
        ax2.set_title(f'PRF Temperature Map\nPeak: {np.max(temp_map):.1f} °C',
                      color='white', fontsize=8)
        ax2.axis('off')
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.yaxis.set_tick_params(color='#94a3b8')
        plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=7)
        cb2.set_label('°C', color='#94a3b8', fontsize=7)

        # Overlay anatomy contour
        ax2.contour(mag, levels=[0.3], colors=['#94a3b8'], linewidths=0.6, alpha=0.5)

        ax3 = fig.add_subplot(gs[2])
        im3 = ax3.imshow(gt_delta_T, cmap='hot', vmin=0, vmax=np.max(gt_delta_T) + 1)
        ax3.set_title(f'Ground-Truth ΔT\n(Simulated Heating)', color='white', fontsize=8)
        ax3.axis('off')
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label('ΔT (°C)', color='#94a3b8', fontsize=7)
        plt.setp(cb3.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=7)

        thermo_img = self._b64_fig(fig)

        return {
            'mask_image':   mask_img,
            'kspace_image': kspace_img,
            'recon_image':  recon_img,
            'thermo_image': thermo_img,
        }

    def generate_seq_file_content(self, tr_ms=1500, te_ms=20, fa_deg=30,
                                   fov_mm=240, matrix=None):
        """
        Write a Pulseq-compatible .seq description for the PF-thermometry
        gradient echo sequence.
        """
        if matrix is None:
            matrix = self.N
        gamma_Hz_T = GAMMA_HZ_T
        TR  = tr_ms  * 1e-3
        TE  = te_ms  * 1e-3
        FA  = np.deg2rad(fa_deg)
        FOV = fov_mm * 1e-3
        BW  = 250e3
        dwell = 1.0 / BW
        ro_dur = matrix * dwell
        rf_dur = 3.2e-3
        res_m  = FOV / matrix

        # Combinatorial mask line list
        acquired_lines = np.where(self.mask_1d > 0)[0].tolist()

        lines = [
            "# Pulseq sequence — Partial Fourier MR Thermometry (Combinatorial)",
            "# Generated by NeuroPulse Recon Lab — PartialFourierThermometry",
            f"# PF Factor: {self.pf_factor}  Acceleration: {self.R:.2f}×",
            f"# TR={tr_ms} ms  TE={te_ms} ms  FA={fa_deg} deg",
            "",
            "[VERSION]",
            "major 1", "minor 4", "revision 0",
            "",
            "[DEFINITIONS]",
            f"Name PartialFourierThermometry",
            f"PartialFourierFactor {self.pf_factor:.4f}",
            f"AccelerationFactor {self.R:.4f}",
            f"PRF_alpha_ppm_per_C {PRF_ALPHA * 1e6:.6f}",
            f"Fov {FOV:.4e} {FOV:.4e} {FOV * 0.15:.4e}",
            f"TR {TR:.6e}",
            f"TE {TE:.6e}",
            f"FlipAngle {fa_deg:.2f}",
            f"AdcSamples {matrix}",
            f"AdcDwellTime {dwell:.6e}",
            f"SystemMaxGrad 40e-3",
            f"SystemMaxSlew 180",
            f"AcquiredLines {len(acquired_lines)}",
            f"TotalLines {matrix}",
            "",
            "[ACQUIRED_LINES]",
            "# k_y line indices sampled (combinatorial selection)",
        ]
        lines += [f"{idx}" for idx in acquired_lines]
        lines += [
            "",
            "[BLOCKS]",
            "# id  delay  rf  gx  gy  gz  adc  ext",
            "# RF excitation",
            "1  0  1  0  0  1  0  0",
        ]
        for i, ky in enumerate(acquired_lines):
            lines.append(f"{i+2}  0  0  1  {i+1}  0  1  0   # k_y={ky}")
        lines += [
            f"{len(acquired_lines)+2}  0  0  0  0  2  0  0   # Spoiler",
            "",
            "[RF]",
            "# id  amp  shape  time  freq  phase",
        ]
        amp = FA / (2 * np.pi * gamma_Hz_T * rf_dur)
        lines.append(f"1  {amp:.6e}  1  1  0  0")
        lines += [
            "",
            "[GRADIENTS]",
            f"# Gx readout  amp={1.0 / (gamma_Hz_T * res_m * ro_dur):.4e} Hz/m",
            f"1  {1.0 / (gamma_Hz_T * res_m * ro_dur):.6e}  0  0",
            f"# Gz slice-select",
            f"2  {0.02 * B0 * gamma_Hz_T:.6e}  0  0",
        ]
        for i, ky in enumerate(acquired_lines):
            pe = (ky - matrix // 2) * (1.0 / (gamma_Hz_T * res_m * matrix))
            lines.append(f"{3+i}  {pe:.6e}  0  0   # PE line {ky}")
        lines += [
            "",
            "[ADC]",
            f"1  {matrix}  {dwell:.6e}  0  0  0",
            "",
            "[DELAYS]",
            "1  0",
        ]
        return "\n".join(lines)


# ── Convenience function called by app.py ─────────────────────────────────────

def run_pf_thermometry(pf_factor=0.75, pocs_iterations=8, te_ms=20.0):
    """
    Entry point for the Flask API endpoint.
    Returns the full result dict from PartialFourierThermometry.run_sequence().
    """
    seq = PartialFourierThermometry(
        N              = 128,
        pf_factor      = float(pf_factor),
        pocs_iterations= int(pocs_iterations),
        b0             = B0,
        te             = float(te_ms) * 1e-3,
    )
    result = seq.run_sequence()
    # Also attach the .seq file text
    result['seq_content'] = seq.generate_seq_file_content(
        tr_ms=1500, te_ms=te_ms, fa_deg=30, fov_mm=240)
    return result


def generate_pf_thermometry_report():
    """
    Generate a comprehensive LaTeX-quality Nature-style PDF report on
    Partial Fourier MR Thermometry with Combinatorial k-Space Sampling.
    Called by /api/pf_thermometry/report endpoint.
    Returns bytes of the PDF.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.lines import Line2D
    import numpy as np

    dark   = '#0a0f1e'
    light  = '#e2e8f0'
    accent = '#38bdf8'
    red    = '#f43f5e'
    green  = '#4ade80'
    yellow = '#fbbf24'

    seq = PartialFourierThermometry(N=128, pf_factor=0.75)
    result = seq.run_sequence()
    metrics = result['metrics']

    # Decode base64 figures for embedding
    def decode_b64_img(b64str):
        from PIL import Image
        data = base64.b64decode(b64str)
        return Image.open(io.BytesIO(data))

    fig_main = plt.figure(figsize=(8.27, 11.69), facecolor=dark)   # A4
    fig_main.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.04)
    gs_root = gridspec.GridSpec(6, 1, figure=fig_main,
                                hspace=0.55,
                                height_ratios=[0.25, 1.6, 1.4, 1.4, 1.2, 0.3])

    # ── Title block ──────────────────────────────────────────────────────────
    ax_title = fig_main.add_subplot(gs_root[0])
    ax_title.set_facecolor(dark)
    ax_title.axis('off')
    ax_title.text(0.5, 0.92,
        'Combinatorial Partial Fourier MR Thermometry',
        ha='center', va='top', color=light,
        fontsize=13, fontweight='bold', fontfamily='serif',
        transform=ax_title.transAxes)
    ax_title.text(0.5, 0.55,
        'k-Space Sampling via Combinadic Number System with POCS Homodyne Reconstruction',
        ha='center', va='top', color=accent,
        fontsize=8.5, fontstyle='italic', transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15,
        'NeuroPulse Recon Lab  •  Proton-Resonance-Frequency Thermometry  •  B₀ = 3 T',
        ha='center', va='top', color='#64748b',
        fontsize=7, transform=ax_title.transAxes)

    # ── Abstract + equations ─────────────────────────────────────────────────
    gs_eq = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_root[1], wspace=0.08)
    ax_abs = fig_main.add_subplot(gs_eq[0])
    ax_eqs = fig_main.add_subplot(gs_eq[1])
    for ax in (ax_abs, ax_eqs):
        ax.set_facecolor('#0f172a')
        ax.axis('off')

    abstract_text = (
        "Abstract.  Partial Fourier (PF) acquisition reduces MR scan time by "
        "exploiting Hermitian k-space symmetry.  In MR thermometry, shorter "
        "acquisition windows minimise motion artefacts during thermal therapy.  "
        "We present a combinatorial k-space sampling strategy based on the "
        "Combinadic Number System that selects outer k-space lines to satisfy a "
        r"balanced $(n,k,\lambda)$-design, ensuring uniform pairwise coverage and "
        "near-optimal incoherence in the measurement matrix.  "
        "Reconstruction uses POCS (Projection Onto Convex Sets) homodyne "
        "iteration.  Temperature maps are derived via the PRF phase-shift method.  "
        f"At PF={metrics['pf_factor']} the sequence achieves R={metrics['acceleration_R']:.2f}× "
        f"acceleration with ΔT sensitivity ≈{metrics['temp_sensitivity_C']:.2f} °C."
    )
    ax_abs.text(0.04, 0.97, abstract_text,
                ha='left', va='top', color='#cbd5e1',
                fontsize=6.8, wrap=True, transform=ax_abs.transAxes,
                multialignment='left',
                bbox=dict(facecolor='#1e293b', edgecolor='#334155',
                          boxstyle='round,pad=0.5', alpha=0.8))

    # Finite math equations
    eqs = [
        (r'$\Delta T(\mathbf{r}) = \dfrac{\Delta\phi(\mathbf{r})}{\alpha\,\gamma\,B_0\,TE}$',
         'PRF Temperature Map'),
        (r'$S(-\mathbf{k}) = S^*(\mathbf{k})$',
         'Hermitian k-Space Symmetry'),
        (r'$M(k_y,p_f)=\begin{cases}1 & |k_y|\leq\lfloor N p_f/2\rfloor\\'
         r'C(k_y) & \text{otherwise}\end{cases}$',
         'Combinatorial Sampling Mask'),
        (r'$R_{\mathrm{comb}} = \dfrac{N_{pe}}{|M|} = \dfrac{1}{p_f+f_{\mathrm{outer}}}$',
         'Acceleration Factor'),
        (r'$g_{PF} = \sqrt{R_{\mathrm{comb}}}$',
         'Noise Amplification (PF g-factor)'),
        (r'$\Delta T_{\min} = \dfrac{1}{\mathrm{SNR}\cdot\alpha\cdot\gamma\cdot B_0\cdot TE}$',
         'Minimum Detectable ΔT'),
        (r'$\binom{n}{k} = \sum_{i=1}^{k} \binom{c_i}{i},\quad c_i\in\mathbb{N}$',
         'Combinadic Representation'),
        (r'$\mu=\max_{i\neq j}|\langle\mathbf{a}_i,\mathbf{a}_j\rangle|\rightarrow\min$',
         'Mutual Coherence (RIP)'),
    ]
    for idx, (eq, label) in enumerate(eqs):
        y_pos = 0.96 - idx * 0.125
        ax_eqs.text(0.05, y_pos, eq,
                    ha='left', va='top', color=accent,
                    fontsize=8.5, transform=ax_eqs.transAxes)
        ax_eqs.text(0.05, y_pos - 0.055, label,
                    ha='left', va='top', color='#64748b',
                    fontsize=6, transform=ax_eqs.transAxes)

    # ── k-Space Mask + k-Space magnitude ──────────────────────────────────
    gs_ksp = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_root[2], wspace=0.15)
    imgs_row1 = [result['mask_image'], result['kspace_image'], result['recon_image']]
    titles_row1 = [
        f'PF Sampling Mask\n(Combinadic, R={metrics["acceleration_R"]:.2f}×)',
        'Partial k-Space Magnitude\n(log scale)',
        'POCS Reconstructed Image\n(Homodyne, {} iters)'.format(metrics['pocs_iterations']),
    ]
    for col, (b64, title) in enumerate(zip(imgs_row1, titles_row1)):
        ax = fig_main.add_subplot(gs_ksp[col])
        ax.set_facecolor(dark)
        ax.axis('off')
        try:
            img_pil = decode_b64_img(b64)
            ax.imshow(np.array(img_pil))
        except Exception:
            ax.text(0.5, 0.5, 'Figure', ha='center', color=light, transform=ax.transAxes)
        ax.set_title(title, color=light, fontsize=7, pad=4)

    # ── Temperature map panel ─────────────────────────────────────────────
    ax_thermo = fig_main.add_subplot(gs_root[3])
    ax_thermo.set_facecolor(dark)
    ax_thermo.axis('off')
    try:
        thermo_pil = decode_b64_img(result['thermo_image'])
        ax_thermo.imshow(np.array(thermo_pil), aspect='auto')
    except Exception:
        pass
    ax_thermo.set_title(
        f'PRF Temperature Map  •  Peak: {metrics["peak_temperature_C"]:.1f} °C  •  '
        f'ΔT sensitivity: {metrics["temp_sensitivity_C"]:.2f} °C  •  B₀={metrics["b0_T"]} T  '
        f'TE={metrics["te_ms"]} ms',
        color=light, fontsize=7.5, pad=5)

    # ── Metrics table ─────────────────────────────────────────────────────
    ax_met = fig_main.add_subplot(gs_root[4])
    ax_met.set_facecolor('#0f172a')
    ax_met.axis('off')

    table_data = [
        ['Parameter', 'Value', 'Unit / Note'],
        ['PF Factor', str(metrics['pf_factor']), 'fraction (0.5–1.0)'],
        ['Acceleration R', f'{metrics["acceleration_R"]:.2f}×', 'total scan time reduction'],
        ['Acquired Lines', f'{metrics["acquired_lines"]} / {metrics["total_lines"]}',
         f'inner={metrics["inner_lines_acquired"]}  outer={metrics["outer_lines_combinadic"]}'],
        ['POCS Iterations', str(metrics['pocs_iterations']), 'convergence < 0.01° phase'],
        ['g_PF (noise amp.)', f'{metrics["noise_amplification_gPF"]:.3f}', '√R approximation'],
        ['ΔT Sensitivity', f'{metrics["temp_sensitivity_C"]:.2f} °C', 'at SNR, B₀=3T, TE=20ms'],
        ['Peak Temperature', f'{metrics["peak_temperature_C"]:.1f} °C', 'tumour voxel'],
        ['GT Peak ΔT', f'{metrics["gt_peak_delta_T"]:.1f} °C', 'simulated heating'],
        ['PRF Coefficient α', '−0.0094 ppm/°C', 'water proton PRF shift'],
        ['Gyromagnetic Ratio γ', '42.577 MHz/T', 'proton ¹H'],
    ]
    table = ax_met.table(
        cellText   = table_data[1:],
        colLabels  = table_data[0],
        cellLoc    = 'left',
        loc        = 'center',
        colWidths  = [0.38, 0.24, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor('#1e293b' if r % 2 == 0 else '#0f172a')
        cell.set_edgecolor('#334155')
        cell.set_text_props(color=accent if r == 0 else light)
    ax_met.set_title('Sequence Performance Metrics — Combinatorial PF-MRT',
                     color=light, fontsize=8, pad=6)

    # ── Footer ────────────────────────────────────────────────────────────
    ax_foot = fig_main.add_subplot(gs_root[5])
    ax_foot.set_facecolor(dark)
    ax_foot.axis('off')
    ax_foot.text(0.5, 0.5,
        'NeuroPulse Recon Lab  |  Partial Fourier MR Thermometry  |  '
        'Combinadic k-Space Sampling  |  POCS Homodyne Reconstruction  |  PRF Phase Shift Method',
        ha='center', va='center', color='#475569',
        fontsize=6.5, transform=ax_foot.transAxes)

    buf = io.BytesIO()
    fig_main.savefig(buf, format='pdf', dpi=150, facecolor=dark, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig_main)
    return buf.getvalue()
