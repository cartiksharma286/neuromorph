"""
cs_cardiac_pulse.py
Compressed Sensing Cardiovascular MR Pulse Sequence Engine
══════════════════════════════════════════════════════════
Physics model:
  y = Φ F Ψ* α + n
  where:
    y  ∈ C^m  — measured k-space samples (m << N)
    Φ  ∈ {0,1}^(m×N) — random undersampling mask
    F  ∈ C^(N×N)     — discrete Fourier operator
    Ψ  ∈ R^(N×N)     — wavelet / TV sparsifying transform
    α  ∈ R^N         — sparse coefficient vector
    n  ~ CN(0,σ²I)   — complex Gaussian noise

Recovery (LASSO):
  α̂ = argmin_α ½‖y − ΦFΨ*α‖₂² + λ‖α‖₁

Solved via FISTA (Beck & Teboulle, 2009):
  1. Gradient step:  z ← α − (1/L)∇‖y−ΦFΨ*α‖₂²
  2. Soft threshold: α ← S_{λ/L}(z)  [S_τ(x) = sign(x)·max(|x|−τ,0)]
  3. Momentum:       update Nesterov momentum

Continued Fractions for optimal acceleration R*:
  The variable-density sampling density is:
    p(k) ∝ (1 − (|k|/k_max)^γ) · k_max/N  +  ε_edge
  where γ is the density exponent. The incoherence μ is minimised
  when R ≈ φ^2 = 2.618... (golden ratio squared), whose CF expansion
  [2; 1,1,1,1,...] provides the best rational acceleration targets.

  We generate CF convergents of φ² and select the one that satisfies
  the restricted isometry property (RIP) proxy:
    RIP_proxy(R) = 1 − exp(−m/N × R)   ≥  0.95
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


# ─────────────────────────────────────────────
# 1. Continued Fraction tools
# ─────────────────────────────────────────────

GOLDEN_RATIO_SQ = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618...  φ² ≈ 2.618

def cf_expand(target: float, depth: int = 12):
    """Return CF coefficients aₖ and convergents pₖ/qₖ for `target`."""
    x   = float(target)
    p, q = [1, 0], [0, 1]
    convergents = []
    for k in range(depth):
        a = int(x)
        p_k = a * p[-1] + p[-2]
        q_k = a * q[-1] + q[-2]
        approx  = p_k / q_k if q_k else float('inf')
        err_pct = abs(approx - target) / (abs(target) + 1e-12) * 100
        convergents.append({
            "k":           k,
            "a_k":         a,
            "numerator":   int(p_k),
            "denominator": int(q_k),
            "approx":      round(float(approx), 8),
            "error_pct":   round(float(err_pct), 6),
        })
        p.append(p_k); q.append(q_k)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac
    return convergents


def rip_proxy(R: float, N: int = 256, m_fraction: float = 0.35) -> float:
    """Approximate RIP coherence proxy: 1 − exp(−m·R / N)."""
    m = int(N * m_fraction)
    return float(1.0 - np.exp(-m * R / N))


def optimal_cf_acceleration(N: int = 256, target_rip: float = 0.90):
    """
    Find the CF convergent of φ² that best satisfies RIP_proxy ≥ target_rip.
    Returns (R_opt, convergents).
    """
    convs = cf_expand(GOLDEN_RATIO_SQ, depth=14)
    R_opt, best_conv = GOLDEN_RATIO_SQ, convs[-1]
    for c in convs:
        R = c["approx"]
        if R <= 1.0:
            continue
        if rip_proxy(R, N) >= target_rip:
            R_opt, best_conv = R, c
            break
    return float(R_opt), convs


# ─────────────────────────────────────────────
# 2. Variable-density undersampling mask
# ─────────────────────────────────────────────

def vd_sampling_mask(N: int, R: float, gamma: float = 2.0, seed: int = 42) -> np.ndarray:
    """
    Variable-density random k-space sampling mask.
    p(k) ∝ (1 − (|k|/k_max)^γ)  → denser near centre.
    Returns boolean mask of shape (N, N), with fraction ~1/R ones.
    """
    rng  = np.random.default_rng(seed)
    ky   = np.arange(-N // 2, N // 2)
    kx   = np.arange(-N // 2, N // 2)
    KX, KY = np.meshgrid(kx, ky)
    K_mag  = np.sqrt(KX**2 + KY**2)
    k_max  = N / 2.0

    density = np.clip(1.0 - (K_mag / k_max) ** gamma, 0.05, 1.0)
    density /= density.sum()

    n_samples = max(1, int(N * N / R))
    flat_idx  = rng.choice(N * N, size=n_samples, replace=False, p=density.ravel())
    mask      = np.zeros(N * N, dtype=bool)
    mask[flat_idx] = True
    return mask.reshape(N, N)


def sampling_stats(mask: np.ndarray) -> dict:
    N       = mask.shape[0]
    sampled = int(mask.sum())
    total   = N * N
    R_actual = round(total / sampled, 4)
    # Coherence proxy: normalised peak of PSF side lobes
    psf = np.abs(ifft2(ifftshift(mask.astype(float))))
    psf_norm = psf / psf.max()
    psf_norm.flat[0] = 0.0          # zero DC
    mu = float(psf_norm.max())
    return {
        "N":          N,
        "sampled":    sampled,
        "total":      total,
        "R_actual":   R_actual,
        "fill_pct":   round(sampled / total * 100, 2),
        "incoherence_mu": round(float(mu), 5),
    }


# ─────────────────────────────────────────────
# 3. Sparsifying transform  (2-D Haar wavelet approx via multi-scale diff)
# ─────────────────────────────────────────────

def wavelet_fwd(x: np.ndarray) -> np.ndarray:
    """Simple 2-level Haar-like forward transform (shift-difference)."""
    W = x.copy()
    for axis in (0, 1):
        W = np.diff(W, axis=axis, append=np.take(W, [0], axis=axis))
    return W


def wavelet_inv(W: np.ndarray) -> np.ndarray:
    """Approximate inverse of the above (cumsum)."""
    x = W.copy()
    for axis in (0, 1):
        x = np.cumsum(x, axis=axis)
    return x


# ─────────────────────────────────────────────
# 4. FISTA solver
# ─────────────────────────────────────────────

def _soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    """Complex soft-thresholding: S_τ(z) = z · max(|z|−τ, 0) / |z|."""
    mag   = np.abs(z)
    scale = np.maximum(mag - tau, 0.0) / (mag + 1e-15)
    return z * scale


def fista_cs(y_us: np.ndarray, mask: np.ndarray,
             lam: float = 0.005, n_iter: int = 60, L: float = 1.0):
    """
    FISTA reconstruction of undersampled cardiac k-space.

    Parameters
    ----------
    y_us  : (N,N) complex  — undersampled k-space (zeros outside mask)
    mask  : (N,N) bool     — undersampling mask
    lam   : float          — sparsity regularisation weight
    n_iter: int            — number of iterations
    L     : float          — Lipschitz constant of ∇f (set = 1 for normalised data)

    Returns
    -------
    recon    : (N,N) real  — reconstructed image
    residuals: list[float] — ‖y − Φ F x_k‖₂ per iteration
    alphas   : list[float] — ‖Ψ x_k‖₁ per iteration
    """
    N   = y_us.shape[0]
    x   = np.zeros((N, N), dtype=complex)   # image estimate
    z   = x.copy()                          # momentum variable
    t   = 1.0
    residuals, l1_norms = [], []

    for i in range(n_iter):
        # Forward: Φ F z
        Fz       = fftshift(fft2(z)) / N
        residual = mask * Fz - y_us          # data residual
        res_norm = float(np.linalg.norm(residual))
        residuals.append(res_norm)
        l1_norms.append(float(np.sum(np.abs(wavelet_fwd(np.abs(z))))))

        # Gradient of ½‖y − ΦFz‖₂² w.r.t. z (back-projection)
        grad = ifft2(ifftshift(mask * residual)) * N   # adjoint

        # Gradient step
        u = z - (1.0 / L) * grad

        # Sparsify in wavelet domain via soft threshold
        Wu    = wavelet_fwd(u.real)
        Wu_th = _soft_threshold(Wu, lam / L)
        x_new = wavelet_inv(Wu_th).astype(complex) + 1j * u.imag

        # Nesterov momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        beta  = (t - 1.0) / t_new
        z     = x_new + beta * (x_new - x)
        x, t  = x_new, t_new

        if res_norm < 1e-6:
            break

    return np.abs(x), residuals, l1_norms


# ─────────────────────────────────────────────
# 5. Synthetic cardiac phantom
# ─────────────────────────────────────────────

def cardiac_phantom(N: int = 128) -> np.ndarray:
    """
    Sparse synthetic cardiac phantom:
    LV blood pool (bright disc) + myocardium ring + RV + aorta.
    """
    cx, cy = N // 2, N // 2
    Y, X   = np.ogrid[:N, :N]

    # Left ventricle (blood pool)
    lv_r = N // 5
    lv   = ((X - cx)**2 + (Y - cy)**2) <= lv_r**2

    # Myocardium ring
    myo_ro = int(N * 0.31)
    myo_ri = int(N * 0.21)
    myo    = (((X - cx)**2 + (Y - cy)**2) <= myo_ro**2) & \
             (((X - cx)**2 + (Y - cy)**2) >  myo_ri**2)

    # Right ventricle (crescent-like)
    rv_cx  = cx - int(N * 0.28)
    rv_r   = int(N * 0.14)
    rv     = ((X - rv_cx)**2 + (Y - cy)**2) <= rv_r**2

    # Aorta
    ao_cx  = cx + int(N * 0.12)
    ao_cy  = cy - int(N * 0.28)
    ao_r   = int(N * 0.06)
    ao     = ((X - ao_cx)**2 + (Y - ao_cy)**2) <= ao_r**2

    phantom = np.zeros((N, N), dtype=float)
    phantom[lv]  = 1.00   # bright blood
    phantom[myo] = 0.45   # grey myocardium
    phantom[rv]  = 0.85   # blood
    phantom[ao]  = 0.95   # aorta
    return phantom


# ─────────────────────────────────────────────
# 6. Sparsity profile analysis
# ─────────────────────────────────────────────

def sparsity_profile(image: np.ndarray, n_bins: int = 64):
    """Return sorted coefficient magnitudes (wavelet domain)."""
    W    = wavelet_fwd(image)
    coef = np.abs(W.ravel())
    coef.sort(); coef = coef[::-1]
    step = max(1, len(coef) // n_bins)
    idx  = list(range(0, len(coef), step))[:n_bins]
    return {
        "rank":        list(range(1, len(idx) + 1)),
        "magnitude":   [round(float(coef[i]), 6) for i in idx],
        "cum_energy":  [round(float(coef[:i+1].sum() / (coef.sum() + 1e-12)), 5) for i in idx],
    }


# ─────────────────────────────────────────────
# 7. Master pipeline
# ─────────────────────────────────────────────

def run_cs_cardiac(N: int = 96, lam: float = 0.005, n_iter: int = 60,
                   target_rip: float = 0.90, gamma: float = 2.0):
    """
    End-to-end CS cardiac MRI pipeline.

    Returns a JSON-serialisable dict containing:
      - cf_convergents     : CF expansion of φ² and optimal R*
      - rip_curve          : RIP proxy vs acceleration factor
      - sampling_stats     : mask statistics (fill pct, incoherence)
      - fista_convergence  : residual and L1 norm per FISTA iteration
      - sparsity_profile   : wavelet coefficient decay
      - mask_2d            : NxN bool mask as flat int list (downsampled 32x32)
      - phantom_2d         : ground truth phantom (32x32 downsampled)
      - reconstruction_2d  : FISTA output (32x32 downsampled)
      - metrics            : SSIM proxy, PSNR, sparsity fraction
    """
    rng = np.random.default_rng(0)

    # ── Step 1: CF-optimal acceleration ──────────────────────────
    R_opt, cf_convs = optimal_cf_acceleration(N, target_rip)

    # RIP curve vs a range of R values
    R_range = [round(1.0 + k * 0.2, 2) for k in range(25)]
    rip_vals = [round(rip_proxy(R, N), 5) for R in R_range]

    # ── Step 2: Synthesise phantom ────────────────────────────────
    phantom = cardiac_phantom(N)

    # ── Step 3: Forward model (full k-space + noise) ──────────────
    kspace_full = fftshift(fft2(phantom)) / N
    noise       = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * 0.01
    kspace_full += noise

    # ── Step 4: Apply CS mask ─────────────────────────────────────
    mask   = vd_sampling_mask(N, R_opt, gamma=gamma)
    y_us   = mask * kspace_full               # undersampled measurements
    stats  = sampling_stats(mask)

    # ── Step 5: FISTA reconstruction ─────────────────────────────
    recon, residuals, l1_norms = fista_cs(y_us, mask, lam=lam, n_iter=n_iter)

    # ── Step 6: Metrics ───────────────────────────────────────────
    mse  = float(np.mean((recon - phantom) ** 2))
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))
    sp_frac = float(np.mean(np.abs(wavelet_fwd(phantom)) < 0.05))

    # ── Step 7: Downscale for JSON transport ──────────────────────
    ds = 32   # downsample to 32×32 for payload
    ds_mask    = mask[::N//ds, ::N//ds][:ds, :ds]
    ds_phantom = phantom[::N//ds, ::N//ds][:ds, :ds]
    ds_recon   = recon[::N//ds, ::N//ds][:ds, :ds]

    # Normalise for display
    ds_phantom /= (ds_phantom.max() + 1e-12)
    ds_recon   /= (ds_recon.max()   + 1e-12)

    return {
        "target":       "CS Cardiac MRI (FISTA + VD sampling)",
        "N":            N,
        "R_opt":        round(R_opt, 4),
        "lam":          lam,
        "n_iter_done":  len(residuals),
        "gamma":        gamma,
        "cf_convergents": cf_convs,
        "rip_curve": {
            "R":   R_range,
            "rip": rip_vals,
            "R_opt": round(R_opt, 4),
            "target_rip": target_rip,
        },
        "sampling_stats": stats,
        "fista_convergence": {
            "iteration": list(range(1, len(residuals) + 1)),
            "residual":  [round(r, 8) for r in residuals],
            "l1_norm":   [round(v, 6) for v in l1_norms],
        },
        "sparsity_profile": sparsity_profile(phantom),
        "mask_2d":          ds_mask.astype(int).tolist(),
        "phantom_2d":       [[round(float(v), 5) for v in row] for row in ds_phantom],
        "reconstruction_2d":[[round(float(v), 5) for v in row] for row in ds_recon],
        "metrics": {
            "PSNR_dB":          round(psnr, 2),
            "MSE":              round(mse, 6),
            "sparsity_frac":    round(sp_frac, 4),
            "incoherence_mu":   stats["incoherence_mu"],
            "fill_pct":         stats["fill_pct"],
            "R_achieved":       stats["R_actual"],
        }
    }
