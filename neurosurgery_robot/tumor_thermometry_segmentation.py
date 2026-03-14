"""
MR Thermometry Tumor Segmentation
====================================
Segments tumor boundaries from MRI temperature maps using:
  - PRF (Proton Resonance Frequency) shift-based differential temperature mapping
  - Adaptive thresholding on delta-T above physiological baseline (37 °C)
  - Fast level-set (Chan–Vese) active contour refinement
  - Ablation-zone tracking via thermal dose (CEM43 > 240 equivalent minutes)
  - End-effector laser targeting from thermometry-derived centroid
"""

import numpy as np
from typing import Optional
from scipy.ndimage import (
    gaussian_filter, label, distance_transform_edt,
    binary_fill_holes, binary_dilation, binary_erosion,
)


class MRThermometrySegmenter:
    """
    Segment tumor from MR thermometry temperature maps.

    In real MR thermometry the PRF phase shift encodes temperature changes.
    Here we simulate that signal by computing delta-T relative to the
    physiological baseline (37 °C) and deriving tumor boundaries from the
    differential heating pattern that tumours produce due to their elevated
    metabolic heat generation and altered perfusion.
    """

    # PRF thermometry constants (brain tissue at 3 T)
    ALPHA_PRF = -0.01e-6   # PRF coefficient  °C⁻¹
    GAMMA     = 2.675e8    # gyromagnetic ratio  rad s⁻¹ T⁻¹
    B0        = 3.0        # main field strength  T
    TE        = 0.020      # echo time  s

    def __init__(self, width: int = 128, height: int = 128,
                 baseline_temp: float = 37.0):
        self.width = width
        self.height = height
        self.baseline_temp = baseline_temp

        # Detection thresholds
        self.hot_spot_threshold = 2.0    # delta-T (°C) above baseline → potential tumour
        self.ablation_threshold = 43.0   # temperature (°C) for active ablation zone
        self.necrosis_cem43    = 240.0   # CEM43-minutes for full necrosis

        # Segmentation outputs (updated each call to segment_from_thermometry)
        self.tumor_mask    = np.zeros((height, width), dtype=bool)
        self.ablation_mask = np.zeros((height, width), dtype=bool)
        self.necrosis_mask = np.zeros((height, width), dtype=bool)
        self.boundary      = np.zeros((height, width), dtype=bool)

        # centroid in normalised [0, 1] coords  (cx, cy)
        self.centroid    = np.array([0.5, 0.5])
        self.centroid_px = np.array([width // 2, height // 2], dtype=float)

        # Level-set function φ  (φ < 0 ↔ inside tumour)
        self.phi = np.ones((height, width), dtype=float) * 5.0

        # Metrics history
        self.ablation_coverage_history = []  # type: list
        self.tumor_volume_history = []        # type: list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_from_thermometry(
        self,
        temp_map: np.ndarray,
        damage_map: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Segment tumour from the MR-thermometry temperature map.

        Parameters
        ----------
        temp_map   : 2-D array  temperature field (°C)
        damage_map : 2-D array  cumulative CEM43 thermal dose (optional)

        Returns
        -------
        dict with keys:
            tumor_mask, ablation_mask, necrosis_mask, boundary,
            centroid (normalised), centroid_px,
            delta_T (smoothed), tumor_volume_mm2,
            ablation_coverage (0–1), necrosis_fraction (0–1)
        """
        # --- 1.  PRF-equivalent differential temperature map ----------------
        # Simulate PRF phase map: Δφ = α · γ · B₀ · TE · ΔT
        delta_T_raw = temp_map - self.baseline_temp
        delta_T_raw = np.clip(delta_T_raw, 0.0, None)

        # Spatial smoothing (models finite MR resolution ~1 mm voxel)
        delta_T = gaussian_filter(delta_T_raw, sigma=1.5)

        # --- 2.  Hot-spot detection ------------------------------------------
        hot_mask = delta_T > self.hot_spot_threshold
        if np.any(hot_mask):
            hot_mask = binary_fill_holes(hot_mask)
            labeled, n = label(hot_mask)
            if n > 0:
                sizes = [(labeled == i).sum() for i in range(1, n + 1)]
                largest = int(np.argmax(sizes)) + 1
                hot_mask = labeled == largest

        self.tumor_mask = hot_mask

        # --- 3.  Chan–Vese level-set refinement (fast, 5 iterations) --------
        self._update_level_set(delta_T, hot_mask, iterations=5)

        # --- 4.  Active ablation zone  (T ≥ 43 °C) -------------------------
        self.ablation_mask = temp_map >= self.ablation_threshold

        # --- 5.  Necrosis zone  (CEM43 > 240) --------------------------------
        if damage_map is not None:
            self.necrosis_mask = damage_map > self.necrosis_cem43
        else:
            self.necrosis_mask = np.zeros_like(hot_mask)

        # --- 6.  Boundary extraction -----------------------------------------
        dilated = binary_dilation(self.tumor_mask)
        eroded  = binary_erosion(self.tumor_mask)
        self.boundary = dilated & ~eroded

        # --- 7.  Centroid in normalised & pixel coordinates ------------------
        if np.any(self.tumor_mask):
            ys, xs = np.where(self.tumor_mask)
            cy, cx = float(ys.mean()), float(xs.mean())
            self.centroid_px = np.array([cx, cy])
            self.centroid    = np.array([cx / self.width, cy / self.height])

        # --- 8.  Coverage metrics -------------------------------------------
        tumor_vol   = float(np.sum(self.tumor_mask))
        ablation_cov = (
            float(np.sum(self.ablation_mask & self.tumor_mask)) / max(tumor_vol, 1.0)
        )
        necrosis_frac = (
            float(np.sum(self.necrosis_mask & self.tumor_mask)) / max(tumor_vol, 1.0)
        )

        self.ablation_coverage_history.append(ablation_cov)
        self.tumor_volume_history.append(tumor_vol)

        return {
            "tumor_mask":       self.tumor_mask,
            "ablation_mask":    self.ablation_mask,
            "necrosis_mask":    self.necrosis_mask,
            "boundary":         self.boundary,
            "centroid":         self.centroid,
            "centroid_px":      self.centroid_px,
            "delta_T":          delta_T,
            "tumor_volume_mm2": tumor_vol,
            "ablation_coverage": ablation_cov,
            "necrosis_fraction": necrosis_frac,
        }

    def get_laser_target(self) -> np.ndarray:
        """Return normalised (x, z) target for the laser end-effector."""
        return self.centroid.copy()

    def get_ablation_progress(self) -> float:
        """Fraction of tumour volume that has been ablated (CEM43 > 240)."""
        if not np.any(self.tumor_mask):
            return 0.0
        return float(np.sum(self.necrosis_mask & self.tumor_mask)) / float(np.sum(self.tumor_mask))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_level_set(
        self,
        field: np.ndarray,
        initial_mask: np.ndarray,
        iterations: int = 5,
        dt: float = 0.30,
    ) -> None:
        """
        Fast Chan–Vese level-set update driven by the thermometry gradient.

        Uses a simplified version of the Chan–Vese energy functional:
            F = -(field - μ) · g
        where g is the edge-stopping function and μ is the mean between
        the inside/outside regions.
        """
        # Initialise φ from mask (signed-distance function)
        if np.any(initial_mask):
            dist_out = distance_transform_edt(~initial_mask).astype(float)
            dist_in  = distance_transform_edt(initial_mask).astype(float)
            self.phi = np.where(initial_mask, -dist_in, dist_out)

        # Edge-stopping function from temperature gradient
        gy, gx   = np.gradient(field)
        edge_mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        g        = 1.0 / (1.0 + edge_mag * 5.0)

        for _ in range(iterations):
            phi_y, phi_x = np.gradient(self.phi)
            phi_mag = np.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)

            # Unit-normal components
            nx = phi_x / phi_mag
            ny = phi_y / phi_mag

            # Mean curvature (divergence of unit normal)
            kappa = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)

            # Chan–Vese region speed term
            inside  = self.phi <  0
            outside = self.phi >= 0
            mu_in  = field[inside].mean()  if inside.any()  else 0.0
            mu_out = field[outside].mean() if outside.any() else 0.0

            F = -(field - (mu_in + mu_out) * 0.5) * g

            # Level-set update
            self.phi = self.phi + dt * (0.5 * kappa + F) * phi_mag

        self.tumor_mask = self.phi < 0
