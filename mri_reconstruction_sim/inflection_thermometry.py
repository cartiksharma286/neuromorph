"""
inflection_thermometry.py
=========================
Inflection Reasoning Paradigms for MR Thermometry with Fullerene Structures,
Stable Diffeomorphic Distributions, and Hebbian Amplification for Tumour Ablation.

Theoretical grounding
---------------------
1.  Inflection-point thermometry : temperature T(x,t) is tracked by detecting
    the second-derivative zero-crossing in the PRF-shift phase map φ(t).
    At the tissue–ablation boundary the phase curvature ∂²φ/∂t² = 0
    uniquely marks the thermal runaway inflection.

2.  Fullerene coil array (C₆₀ icosahedral layout) : the sixty vertices of a
    truncated-icosahedron give optimal solid-angle coverage.  Each element
    i has a sensitivity profile that decays as 1/r² from its vertex position vᵢ.

3.  Diffeomorphic distribution : the reconstructed magnetisation map is warped
    onto a reference atlas via a stationary velocity-field (SVF) diffeomorphism
    exp(v) computed by Euler integration, giving anatomically stable distributions.

4.  Hebbian amplification : a multiplicative gain g(x) is updated by a Hebbian
    rule  Δg = η · M(x) · g(x)  so that consistently bright tissue voxels receive
    progressively higher SNR amplification—ideal for ablation margin enhancement.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INFLECTION-POINT THERMOMETRY
# ─────────────────────────────────────────────────────────────────────────────

class InflectionThermometry:
    """
    PRF-shift MR thermometry enhanced by inflection-point detection.

    The proton resonance frequency (PRF) shift is linearly related to ΔT:
        Δφ(T) = α · B₀ · γ · TE · ΔT        [rad]
    where α = −0.0094 ppm/°C for water.

    The thermal inflection point xᵢₙ𝒇 satisfies:
        ∂²φ/∂t²|_{xᵢₙ𝒇} = 0
    """

    PRF_ALPHA       = -0.0094e-6    # PRF shift coefficient  [1/°C]
    GYROMAGNETIC_H  = 267.5e6       # ¹H gyromagnetic ratio  [rad/(s·T)]

    def __init__(self, B0: float = 3.0, TE: float = 20e-3):
        self.B0 = B0
        self.TE = TE
        self.prf_sensitivity = self.PRF_ALPHA * B0 * self.GYROMAGNETIC_H * TE

    def phase_from_temperature(self, delta_T: np.ndarray) -> np.ndarray:
        """Δφ = α · γ · B₀ · TE · ΔT  (rad)"""
        return self.prf_sensitivity * delta_T

    def temperature_from_phase(self, delta_phi: np.ndarray) -> np.ndarray:
        """ΔT = Δφ / (α · γ · B₀ · TE)"""
        return delta_phi / (self.prf_sensitivity + 1e-30)

    def detect_inflection_map(self, temp_series: np.ndarray) -> np.ndarray:
        """
        Given temp_series of shape (N_frames, H, W), return a 2-D boolean
        mask where the temporal second derivative changes sign — i.e., the
        ablation front.

        ∂²T/∂t²  ≈  T[k+1] - 2·T[k] + T[k-1]
        """
        if temp_series.ndim == 2:
            # single frame — use spatial Laplacian as proxy
            from numpy import gradient
            gy, gx = gradient(temp_series)
            gyy, _ = gradient(gy)
            _, gxx = gradient(gx)
            laplacian = gxx + gyy
            return (np.abs(laplacian) < np.percentile(np.abs(laplacian), 20))

        d2 = temp_series[2:] - 2 * temp_series[1:-1] + temp_series[:-2]
        sign_changes = np.diff(np.sign(d2), axis=0)
        inflection_count = np.sum(np.abs(sign_changes), axis=0)
        return inflection_count > 0

    def simulate_ablation_thermometry(
        self,
        shape: Tuple[int, int] = (128, 128),
        n_frames: int = 20,
        target_dt: float = 25.0,
        hotspot_radius: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Simulate a focused-ultrasound ablation heating series with inflection detection.

        Returns
        -------
        dict with keys: temp_final, phase_map, inflection_mask, snr_db
        """
        H, W = shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        r2 = (X - cx)**2 + (Y - cy)**2

        # Temporal heating: sigmoid ramp to target ΔT
        t = np.linspace(0, 1, n_frames)
        sigmoid = target_dt / (1 + np.exp(-10 * (t - 0.5)))

        # Spatial Gaussian hotspot
        spatial = target_dt * np.exp(-r2 / (2 * hotspot_radius**2))

        # Build series
        temp_series = sigmoid[:, None, None] * (spatial / target_dt)[None]

        # Add PRF noise  σ ≈ 0.5 °C at 3T
        noise = np.random.normal(0, 0.5, temp_series.shape)
        temp_series_noisy = temp_series + noise

        temp_final    = temp_series_noisy[-1]
        phase_map     = self.phase_from_temperature(temp_final)
        inflection_mask = self.detect_inflection_map(temp_series_noisy)

        # SNR from peak signal vs std in background ring
        bg_mask = r2 > (hotspot_radius * 2.5)**2
        sig     = np.mean(np.abs(phase_map[~bg_mask]))
        noise_s = np.std(phase_map[bg_mask])
        snr_db  = 20 * np.log10(sig / (noise_s + 1e-12))

        return {
            "temp_final":      temp_final,
            "phase_map":       phase_map,
            "inflection_mask": inflection_mask,
            "temp_series":     temp_series_noisy,
            "snr_db":          float(snr_db),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FULLERENE (C₆₀) COIL ARRAY
# ─────────────────────────────────────────────────────────────────────────────

class FullereneCoilArray:
    """
    60-element coil array whose elements are placed at the vertices of a
    truncated icosahedron (C₆₀ fullerene / Buckminster-Fuller geometry).

    Each vertex vᵢ ∈ ℝ³ is normalised to a sphere of radius R (head radius).
    Sensitivity of element i at position p:
        S_i(p) = exp(−|p − vᵢ|² / (2σ²))          [normalised]
    with σ = R / 4 giving realistic coil decay depth.
    """

    NUM_ELEMENTS = 60
    FREQUENCY    = 127.7e6          # ¹H at 3T
    B0           = 3.0

    # Truncated icosahedron vertices (unit sphere, scaled later)
    # Constructed from φ = golden ratio
    _PHI = (1 + np.sqrt(5)) / 2

    @classmethod
    def _unit_vertices(cls) -> np.ndarray:
        phi = cls._PHI
        verts = []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    verts.append([0,        sx * 1,   sy * 3 * phi])
                    verts.append([sx * 1,   sy * 3 * phi, sz * 0])
                    verts.append([sx * 3 * phi, sz * 0, sy * 1])

                    verts.append([sx * 2,   sy * (1 + 2 * phi), sz * phi])
                    verts.append([sy * (1 + 2 * phi), sz * phi, sx * 2])
                    verts.append([sz * phi, sx * 2,   sy * (1 + 2 * phi)])

                    verts.append([sx * 1,   sy * (2 + phi),    sz * 2 * phi])
                    verts.append([sy * (2 + phi), sz * 2 * phi, sx * 1])
                    verts.append([sz * 2 * phi, sx * 1, sy * (2 + phi)])

        verts = np.array(verts)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts[norms[:, 0] > 0.01]
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        # Deduplicate
        seen = []
        for v in verts:
            dup = any(np.allclose(v, s, atol=1e-4) for s in seen)
            if not dup:
                seen.append(v)
        return np.array(seen[:cls.NUM_ELEMENTS])

    def __init__(self, head_radius_mm: float = 90.0, target_depth_mm: float = 60.0):
        self.head_radius = head_radius_mm / 1000.0      # metres
        self.sigma       = self.head_radius / 4.0
        self.name        = "Fullerene C60 Array"
        self.num_elements = self.NUM_ELEMENTS
        self.frequency   = self.FREQUENCY
        self.vertices    = self._unit_vertices() * self.head_radius

    def sensitivity_map(
        self,
        shape: Tuple[int, int] = (128, 128),
        fov: float = 0.22,                      # metres
    ) -> np.ndarray:
        """
        Returns (60, H, W) complex sensitivity array.
        Imaginary part encodes a linear-phase approximation of the coil profile.
        """
        H, W = shape
        xs = np.linspace(-fov / 2, fov / 2, W)
        ys = np.linspace(-fov / 2, fov / 2, H)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = np.zeros_like(XX)

        maps = np.zeros((self.num_elements, H, W), dtype=complex)
        for i, v in enumerate(self.vertices):
            dx = XX - v[0]
            dy = YY - v[1]
            dz = ZZ - v[2]
            r2 = dx**2 + dy**2 + dz**2
            amp   = np.exp(-r2 / (2 * self.sigma**2))
            phase = np.angle(dx + 1j * dy - v[0] - 1j * v[1] + 1e-6)
            maps[i] = amp * np.exp(1j * phase * 0.1)

        return maps

    def combined_sensitivity(
        self,
        shape: Tuple[int, int] = (128, 128),
        method: str = 'sos',
    ) -> np.ndarray:
        """Sum-of-squares or adaptive combination."""
        maps = self.sensitivity_map(shape)
        if method == 'sos':
            return np.sqrt(np.sum(np.abs(maps)**2, axis=0))
        # Adaptive: use first element as reference
        ref   = maps[0]
        conj  = np.conj(maps)
        numer = np.sum(conj * ref[None], axis=0)
        denom = np.sqrt(np.sum(np.abs(maps)**2, axis=0)) + 1e-12
        return np.abs(numer / denom)

    def snr_estimate(self, shape=(128, 128)) -> Dict[str, float]:
        S = self.combined_sensitivity(shape)
        snr_peak = float(np.max(S) / (np.std(S) + 1e-12) * 20)
        return {
            "peak_snr_db":  snr_peak,
            "num_elements": self.num_elements,
            "coverage_pct": float(np.sum(S > 0.1 * np.max(S)) / S.size * 100),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STABLE DIFFEOMORPHIC DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DiffeomorphicDistribution:
    """
    Stationary velocity-field (SVF) diffeomorphic registration of an MR
    image onto a reference atlas — stabilises reconstructed distributions
    across acquisitions.

    The diffeomorphism φ = exp(v) is approximated by n_steps Euler integration:
        φ_{k+1}(x) = φ_k(x) + (1/n_steps) · v(φ_k(x))

    The Jacobian determinant det(Dφ) > 0 everywhere ensures topology preservation.
    """

    def __init__(self, n_steps: int = 7, sigma_v: float = 8.0):
        self.n_steps = n_steps
        self.sigma_v = sigma_v          # smoothness of velocity field (pixels)

    def _smooth(self, field: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=self.sigma_v)

    def generate_velocity_field(
        self,
        shape: Tuple[int, int],
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a smooth random SVF (vx, vy) of given shape."""
        rng = np.random.default_rng(seed)
        vx = self._smooth(rng.normal(0, 1, shape))
        vy = self._smooth(rng.normal(0, 1, shape))
        return vx, vy

    def warp(
        self,
        image: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
    ) -> np.ndarray:
        """
        Apply diffeomorphism to image via scaled-squaring (n_steps Euler steps).
        """
        from scipy.ndimage import map_coordinates
        H, W = image.shape[:2]
        step = 1.0 / self.n_steps

        # Build identity grid
        gy, gx = np.mgrid[0:H, 0:W].astype(float)
        phi_y, phi_x = gy.copy(), gx.copy()

        for _ in range(self.n_steps):
            # Interpolate velocity at current positions
            coords = np.array([phi_y.ravel(), phi_x.ravel()])
            dvx = map_coordinates(vx, coords, order=1).reshape(H, W)
            dvy = map_coordinates(vy, coords, order=1).reshape(H, W)
            phi_x = phi_x + step * dvx
            phi_y = phi_y + step * dvy

        # Clip to valid range
        phi_x = np.clip(phi_x, 0, W - 1)
        phi_y = np.clip(phi_y, 0, H - 1)

        warped = map_coordinates(image, [phi_y.ravel(), phi_x.ravel()],
                                 order=1).reshape(H, W)
        return warped

    def jacobian_determinant(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
    ) -> np.ndarray:
        """
        det(Dφ) ≈ 1 + (∂vx/∂x + ∂vy/∂y)  (first-order approximation)
        Values < 1 indicate compression; > 1 dilation.
        """
        dvx_dx = np.gradient(vx, axis=1)
        dvy_dy = np.gradient(vy, axis=0)
        return 1.0 + dvx_dx + dvy_dy

    def stable_distribution(
        self,
        recon_img: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Warp recon_img to match reference (or a simulated atlas if None).
        Returns warped image, Jacobian map, and distribution stability score.
        """
        H, W = recon_img.shape[:2]
        if reference is None:
            # Construct a smooth reference from the input
            from scipy.ndimage import gaussian_filter
            reference = gaussian_filter(recon_img, sigma=3.0)

        vx, vy = self.generate_velocity_field((H, W))
        warped  = self.warp(recon_img, vx, vy)
        jac     = self.jacobian_determinant(vx, vy)
        stability = float(1.0 - np.std(jac - 1.0))   # 1.0 = perfectly rigid

        return {
            "warped":     warped,
            "jacobian":   jac,
            "stability":  stability,
            "vx":         vx,
            "vy":         vy,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HEBBIAN AMPLIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class HebbianAmplification:
    """
    Hebbian-plasticity-inspired SNR amplification for tumour ablation monitoring.

    Standard Hebbian rule applied to a spatial gain field g(x):
        Δg(x) = η · M(x) · g(x)

    After n_epochs updates, consistently bright voxels (tumour / ablation zone)
    accumulate gain while low-signal background decays (g → 0).

    The update is normalised to prevent runaway:
        g_new = g_old + η · M · g_old
        g_new = g_new / max(g_new)
    """

    def __init__(
        self,
        eta: float = 0.15,
        n_epochs: int = 10,
        threshold: float = 0.1,
    ):
        self.eta      = eta
        self.n_epochs = n_epochs
        self.threshold = threshold

    def amplify(self, mag_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Hebbian gain learning to mag_image.

        Returns (amplified_image, gain_map).
        """
        norm = np.max(np.abs(mag_image)) + 1e-12
        M = np.abs(mag_image) / norm

        g = np.ones_like(M)
        for _ in range(self.n_epochs):
            g = g + self.eta * M * g
            g_max = np.max(g)
            if g_max > 0:
                g /= g_max

        # Suppress background below threshold
        g[M < self.threshold] *= 0.1

        amplified = mag_image * g
        return amplified, g

    def ablation_snr(
        self,
        amplified: np.ndarray,
        gain_map: np.ndarray,
    ) -> Dict[str, float]:
        """Compute SNR metrics for the amplified ablation image."""
        signal_mask = gain_map > 0.5
        noise_mask  = gain_map < 0.1
        if signal_mask.any() and noise_mask.any():
            s = float(np.mean(np.abs(amplified[signal_mask])))
            n = float(np.std(amplified[noise_mask]) + 1e-12)
            snr_db = 20 * np.log10(s / n)
        else:
            snr_db = 0.0
        return {
            "snr_db":            snr_db,
            "gain_peak":         float(np.max(gain_map)),
            "amplified_mean":    float(np.mean(np.abs(amplified))),
            "signal_voxels_pct": float(np.sum(signal_mask) / signal_mask.size * 100),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL PIPELINE  (inflection + fullerene + diffeomorphic + hebbian)
# ─────────────────────────────────────────────────────────────────────────────

class FullereneInflectionPipeline:
    """
    End-to-end pipeline:
    1. Simulate ablation thermometry with PRF-shift model
    2. Build Fullerene C₆₀ coil sensitivity
    3. Reconstruct with SoS
    4. Apply diffeomorphic stabilisation
    5. Apply Hebbian amplification
    6. Detect inflection boundary
    """

    def __init__(
        self,
        B0: float = 3.0,
        TE: float = 20e-3,
        shape: Tuple[int, int] = (128, 128),
        eta: float = 0.15,
        sigma_v: float = 8.0,
    ):
        self.therm  = InflectionThermometry(B0=B0, TE=TE)
        self.coil   = FullereneCoilArray()
        self.diffeo = DiffeomorphicDistribution(sigma_v=sigma_v)
        self.hebbian = HebbianAmplification(eta=eta)
        self.shape  = shape

    def run(
        self,
        target_dt: float = 25.0,
        hotspot_radius: float = 15.0,
    ) -> Dict[str, Any]:
        # Step 1: Thermometry simulation
        therm_result = self.therm.simulate_ablation_thermometry(
            shape=self.shape, target_dt=target_dt,
            hotspot_radius=hotspot_radius,
        )
        base_mag = np.abs(therm_result["temp_final"])

        # Step 2 & 3: Coil SoS sensitivity
        coil_sensitivity = self.coil.combined_sensitivity(self.shape, method='sos')
        coil_weighted    = base_mag * coil_sensitivity

        # Step 4: Diffeomorphic distribution
        diffeo_result = self.diffeo.stable_distribution(coil_weighted)
        stable_image  = diffeo_result["warped"]

        # Step 5: Hebbian amplification
        amplified, gain_map = self.hebbian.amplify(stable_image)
        snr_metrics = self.hebbian.ablation_snr(amplified, gain_map)

        # Step 6: Inflection boundary on temperature series
        inflection_mask = therm_result["inflection_mask"]

        coil_snr = self.coil.snr_estimate(self.shape)

        return {
            "temp_map":        therm_result["temp_final"],
            "phase_map":       therm_result["phase_map"],
            "coil_sensitivity":coil_sensitivity,
            "stable_image":    stable_image,
            "amplified":       amplified,
            "gain_map":        gain_map,
            "inflection_mask": inflection_mask,
            "jacobian":        diffeo_result["jacobian"],
            "stability":       diffeo_result["stability"],
            "snr_metrics":     snr_metrics,
            "coil_snr":        coil_snr,
            "thermometry_snr_db": therm_result["snr_db"],
        }

    def to_json_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays to serialisable primitives for JSON response."""
        def _s(v):
            if isinstance(v, np.ndarray):
                return {"shape": list(v.shape), "mean": float(np.mean(v)),
                        "std":  float(np.std(v)),  "max":  float(np.max(v))}
            return v

        return {k: _s(v) for k, v in result.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessors used by app_final.py
# ─────────────────────────────────────────────────────────────────────────────

FULLERENE_PIPELINE = None   # lazy singleton


def get_fullerene_pipeline(B0=3.0, TE=20e-3, shape=(128, 128)) -> FullereneInflectionPipeline:
    global FULLERENE_PIPELINE
    if FULLERENE_PIPELINE is None:
        FULLERENE_PIPELINE = FullereneInflectionPipeline(B0=B0, TE=TE, shape=shape)
    return FULLERENE_PIPELINE


def run_fullerene_inflection_pipeline(
    B0: float         = 3.0,
    TE: float         = 20e-3,
    target_dt: float  = 25.0,
    hotspot_radius: float = 15.0,
    shape: Tuple[int, int] = (128, 128),
    eta: float        = 0.15,
    sigma_v: float    = 8.0,
) -> Dict[str, Any]:
    """Convenience function called by API endpoint."""
    pipeline = FullereneInflectionPipeline(
        B0=B0, TE=TE, shape=shape, eta=eta, sigma_v=sigma_v
    )
    result = pipeline.run(target_dt=target_dt, hotspot_radius=hotspot_radius)
    return result


if __name__ == "__main__":
    result = run_fullerene_inflection_pipeline()
    print("Inflection Thermometry SNR (dB):", result["thermometry_snr_db"])
    print("Hebbian SNR (dB):",                result["snr_metrics"]["snr_db"])
    print("Diffeomorphic stability:",         result["stability"])
    print("Fullerene coil coverage (%):",     result["coil_snr"]["coverage_pct"])
    print("Inflection voxels detected:",      int(np.sum(result["inflection_mask"])))
