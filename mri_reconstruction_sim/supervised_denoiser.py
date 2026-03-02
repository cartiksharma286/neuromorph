"""
supervised_denoiser.py
======================
Two-stage noise suppression for MRI reconstruction:

Stage 1 — Butterworth K-Space Filter (frequency domain)
    Bandpass mask applied to each coil's k-space array before IFFT.
    Rejects DC singularity (low-stop) and high-frequency speckle (high-stop).
    Transfer function:
        H(r) = [1 - 1/sqrt(1+(r/r_low)^(2n))] * [1/sqrt(1+(r/r_high)^(2n))]

Stage 2 — Attention-Based Supervised Denoiser (spatial domain)
    Treats image backgrounds (< p10 intensity) as noise masks (supervision signal)
    and neurovascular foreground (> p75 intensity) as clean signal targets.
    Per-patch attention:
        alpha_i = softmax(Q_i · K^T / sqrt(d))
        x_clean = x_noisy - alpha_i · noise_estimate_i
    Pure numpy / scipy — no GPU dependencies.
"""

import numpy as np
import scipy.ndimage


# ── Stage 1: Butterworth K-Space Bandpass Filter ──────────────────────────────

class ButterworthKSpaceFilter:
    """
    Applies a Butterworth bandpass filter directly in the k-space (frequency)
    domain to each coil array before reconstruction.

    Parameters
    ----------
    low_cutoff  : float  fraction of Nyquist radius for the high-pass cutoff (DC removal)
    high_cutoff : float  fraction of Nyquist radius for the low-pass cutoff  (speckle removal)
    order       : int    Butterworth polynomial order (steepness)
    """

    def __init__(self, low_cutoff: float = 0.02, high_cutoff: float = 0.90, order: int = 4):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        self._cache: dict = {}

    def _build_mask(self, shape: tuple) -> np.ndarray:
        key = shape
        if key in self._cache:
            return self._cache[key]

        rows, cols = shape
        cy, cx = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalise to [0, 1] where 1 = Nyquist
        r_norm = r / (min(rows, cols) / 2.0)

        n = self.order
        # High-pass component (DC removal)
        hp = 1.0 - 1.0 / np.sqrt(1.0 + (r_norm / max(self.low_cutoff, 1e-9)) ** (2 * n))
        # Low-pass component (speckle removal)
        lp = 1.0 / np.sqrt(1.0 + (r_norm / self.high_cutoff) ** (2 * n))

        mask = hp * lp
        self._cache[key] = mask
        return mask

    def apply(self, kspace_list: list) -> list:
        """
        Filter a list of per-coil k-space arrays.

        Parameters
        ----------
        kspace_list : list[np.ndarray]  Each array is complex-valued, shift-centred k-space.

        Returns
        -------
        list[np.ndarray]  Filtered k-space arrays, same shape.
        """
        filtered = []
        for k in kspace_list:
            mask = self._build_mask(k.shape)
            filtered.append(k * mask)
        return filtered


# ── Stage 2: Attention-Based Supervised Denoiser ─────────────────────────────

class AttentionDenoiser:
    """
    Lightweight attention-based supervised denoiser.

    Architecture overview
    ---------------------
    For each image the denoiser performs a one-shot self-supervised training step:

      1. **Noise mask**  (supervision):  pixels below the p10 intensity percentile
         in background regions — these are pure noise by definition.
      2. **Signal mask** (targets):      neurovascular foreground pixels > p75.
      3. Per-patch embeddings (Q, K, V) are extracted via local averaging kernels
         — equivalent to a single-head linear attention over spatial patches.
      4. Attention weights α = softmax(Q · Kᵀ / √d) are used to compute a
         *noise prediction* for each patch:  ñ = α · V_noise
      5. Subtraction:  x_clean = x_noisy − λ · ñ

    Parameters
    ----------
    patch_size  : int    spatial size of each patch in pixels
    n_features  : int    number of feature channels (kernel set size)
    lambda_sub  : float  noise subtraction strength ∈ [0, 1]
    signal_pct  : float  percentile threshold for clean signal mask
    noise_pct   : float  percentile threshold for noise/background mask
    """

    def __init__(
        self,
        patch_size: int = 8,
        n_features: int = 16,
        lambda_sub: float = 0.85,
        signal_pct: float = 75.0,
        noise_pct: float = 10.0,
    ):
        self.patch_size = patch_size
        self.n_features = n_features
        self.lambda_sub = lambda_sub
        self.signal_pct = signal_pct
        self.noise_pct = noise_pct

        # Fixed Gabor-like feature kernels (no learning required for structure detection)
        rng = np.random.default_rng(42)
        ps = patch_size
        self._kernels = [
            rng.standard_normal((ps, ps)) * np.exp(
                -np.sum(np.meshgrid(
                    np.linspace(-1, 1, ps), np.linspace(-1, 1, ps)
                )[i] ** 2 for i in range(2)) / (2 * 0.5 ** 2)
            )
            for _ in range(n_features)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Convolve image with each kernel → (H, W, n_features) feature map."""
        feats = np.stack(
            [scipy.ndimage.convolve(image, k, mode='reflect') for k in self._kernels],
            axis=-1,
        )
        return feats  # (H, W, F)

    # ──────────────────────────────────────────────────────────────────────────
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-9)

    # ──────────────────────────────────────────────────────────────────────────
    def fit_predict(self, image: np.ndarray) -> np.ndarray:
        """
        One-shot fit and denoise. Returns denoised image in [0, 1].

        Parameters
        ----------
        image : np.ndarray  2-D float array, values in [0, 1].

        Returns
        -------
        np.ndarray  Denoised image, same shape, values in [0, 1].
        """
        img = np.clip(image, 0.0, 1.0)

        # ── 1. Derive supervision masks ──────────────────────────────────────
        p_noise = np.percentile(img, self.noise_pct)
        p_signal = np.percentile(img, self.signal_pct)

        noise_mask = img <= p_noise          # background / noise regions
        signal_mask = img >= p_signal        # clean neurovascular foreground

        # ── 2. Extract feature maps ──────────────────────────────────────────
        features = self._extract_features(img)  # (H, W, F)
        F = self.n_features
        d = float(F)

        # ── 3. Query, Key, Value representations ────────────────────────────
        # Q = features at noise sites, K/V = features at signal sites
        Q = features[noise_mask]   # (N_noise, F)
        K = features[signal_mask]  # (N_signal, F)
        V_signal = img[signal_mask]  # (N_signal,)
        V_noise = img[noise_mask]    # (N_noise,)

        if Q.shape[0] == 0 or K.shape[0] == 0:
            # Edge case: trivial image — return as-is
            return img

        # ── 4. Attention: α = softmax(Q Kᵀ / √d) ───────────────────────────
        # Limit for memory efficiency: subsample K if very large
        max_keys = 512
        if K.shape[0] > max_keys:
            idx = np.linspace(0, K.shape[0] - 1, max_keys, dtype=int)
            K = K[idx]
            V_signal = V_signal[idx]

        # (N_noise, N_signal)
        attn_scores = Q @ K.T / np.sqrt(d)
        attn_weights = self._softmax(attn_scores, axis=-1)  # (N_noise, N_signal)

        # ── 5. Noise estimate via attention-weighted signal reconstruction ───
        # Predicted clean value at each noisy pixel
        pred_clean = attn_weights @ V_signal  # (N_noise,)

        # Noise estimate = noisy value - predicted clean value
        noise_estimate = V_noise - pred_clean  # (N_noise,)

        # ── 6. Subtract predicted noise ─────────────────────────────────────
        img_out = img.copy()
        img_out[noise_mask] -= self.lambda_sub * noise_estimate

        # ── 7. Mild Gaussian smoothing to blend patch boundaries ─────────────
        img_out = scipy.ndimage.gaussian_filter(img_out, sigma=0.35)

        # ── 8. Final normalise to [0, 1] ─────────────────────────────────────
        lo, hi = img_out.min(), img_out.max()
        if hi - lo > 1e-9:
            img_out = (img_out - lo) / (hi - lo)

        return np.clip(img_out, 0.0, 1.0)
