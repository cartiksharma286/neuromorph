"""
Conformal Skull-Surface Head Coil Designer
============================================
Designs an N-element receive coil array whose geometry is *conformal*
to a simplified ellipsoidal skull surface model and applies
combinatorial B0/B1 shimming to maximise signal-to-noise ratio (SNR)
simultaneously in tumour and healthy-tissue regions of interest (ROIs).

Physics
-------
  • Coil sensitivity:  B1⁺ ∝  (μ₀/4π) · ∮ (dl × r̂)/r²  (Biot-Savart)
  • Combined sensitivity array: S(r) = Σᵢ wᵢ * Sᵢ(r)
  • Shim weights w optimised by solving the constrained problem:
        maximise   SNR_tumour  subject to  SNR_healthy ≥ β · SNR_tumour
    via combinatorial search over ±weight lattice + L-BFGS-B refinement.
  • State-transfer Hamiltonian (spin-½ × field modes, Jaynes-Cummings):
        H = (ħω₀/2)σ_z + ħg(a†σ⁻ + aσ⁺) + ħωₐ†a
    Used to compute coherence time for each coil element and derive
    optimal coupling strength g for maximum energy-to-spin-state transfer.

API
---
    coil = ConformatSkullCoil(n_elements=8)
    result = coil.optimise(tumour_center=[0,0,0], tumour_radius=0.025)
    # result contains: shim_weights, snr_tumour, snr_healthy,
    #                  coherence_times, state_transfer_efficiency,
    #                  geometry_dict, plot_b64
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product
import io, base64

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ELLIPSOIDAL SKULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def skull_surface_points(n_elements: int,
                          semi_axes: tuple = (0.10, 0.12, 0.095),
                          offset_r: float = 0.005) -> np.ndarray:
    """
    Distribute n_elements uniformly on an ellipsoid + offset_r (coil shell).
    Uses a Fibonacci spiral to achieve quasi-uniform coverage.

    Returns
    -------
    positions : (n_elements, 3) array  in metres
    """
    a, b, c = semi_axes
    a_off    = a + offset_r
    b_off    = b + offset_r
    c_off    = c + offset_r

    golden   = (1 + np.sqrt(5)) / 2.0
    points   = []
    for i in range(n_elements):
        theta = np.arccos(1 - 2*(i+0.5)/n_elements)          # polar
        phi   = 2.0 * np.pi * i / golden                      # azimuthal
        x     = a_off * np.sin(theta) * np.cos(phi)
        y     = b_off * np.sin(theta) * np.sin(phi)
        z     = c_off * np.cos(theta)
        points.append([x, y, z])
    return np.array(points)


def skull_normals(positions: np.ndarray,
                  semi_axes: tuple = (0.10, 0.12, 0.095)) -> np.ndarray:
    """Surface normals on the ellipsoid at each coil position."""
    a, b, c = semi_axes
    n = positions / np.array([a**2, b**2, c**2])
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BIOT-SAVART LOOP SENSITIVITY  (simplified 1-loop, analytic)
# ─────────────────────────────────────────────────────────────────────────────

MU0 = 4 * np.pi * 1e-7   # H/m

def loop_sensitivity(coil_pos: np.ndarray, coil_normal: np.ndarray,
                     loop_radius: float,
                     target_points: np.ndarray) -> np.ndarray:
    """
    Approximate |B1⁺| at each target point from a circular loop coil
    using the on-axis analytic formula extended via dipole approximation
    for off-axis points.

    Returns |B1⁺| in T / (A·m²)  for each target point.
    """
    r_vec     = target_points - coil_pos[np.newaxis, :]         # (N,3)
    r_mag     = np.linalg.norm(r_vec, axis=1)                   # (N,)
    cos_theta = (r_vec @ coil_normal) / np.maximum(r_mag, 1e-9) # (N,)

    # Magnetic dipole approximation  B ≈ (μ₀m/4π) * (3cosθ·r̂ − ẑ)/r³
    # effective dipole moment m = I·π·R²
    m_eff = np.pi * loop_radius**2   # unit current I=1 A
    B_axial = MU0 * m_eff / (4*np.pi) * \
              np.abs(2*cos_theta) / np.maximum(r_mag**3, 1e-18)

    return B_axial                  # shape (N,)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROI DEFINITION  (tumour + healthy shell)
# ─────────────────────────────────────────────────────────────────────────────

def roi_points(center: np.ndarray, radius: float,
               n_pts: int = 64) -> np.ndarray:
    """Uniformly sample a spherical ROI."""
    golden = (1+np.sqrt(5))/2
    pts    = []
    for i in range(n_pts):
        theta = np.arccos(1 - 2*(i+0.5)/n_pts)
        phi   = 2*np.pi*i/golden
        r     = radius * (0.3 + 0.7*((i+1)/n_pts)**(1/3))   # radial jitter
        pts.append(center + r * np.array([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)]))
    return np.array(pts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SENSITIVITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_sensitivity_matrix(coil_positions: np.ndarray,
                              coil_normals: np.ndarray,
                              target_points: np.ndarray,
                              loop_radius: float = 0.04) -> np.ndarray:
    """
    S  shape (n_targets, n_elements)
    Each column is the sensitivity profile of one coil element.
    """
    n_tgt  = len(target_points)
    n_coil = len(coil_positions)
    S      = np.zeros((n_tgt, n_coil))
    for j in range(n_coil):
        S[:, j] = loop_sensitivity(coil_positions[j], coil_normals[j],
                                   loop_radius, target_points)
    return S


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINATORIAL SHIMMING  (coarse → fine)
# ─────────────────────────────────────────────────────────────────────────────

def combined_signal(weights: np.ndarray,
                    S: np.ndarray,
                    noise_sigma: float = 1e-9) -> float:
    """Root-sum-of-squares combined signal across ROI points."""
    combined = S @ weights             # (n_pts,)
    combined = np.abs(combined)
    signal   = np.sqrt(np.mean(combined**2))
    return signal / max(noise_sigma, 1e-30)


def combinatorial_shim(S_tumour: np.ndarray,
                        S_healthy: np.ndarray,
                        beta: float = 0.7,
                        n_coarse_levels: int = 3) -> np.ndarray:
    """
    Coarse combinatorial search over n_coarse_levels equal-spaced weights
    in [0, 1], then L-BFGS-B refinement.

    Objective: maximise SNR_tumour subject to SNR_healthy ≥ β·SNR_tumour
    Implemented as: minimise −SNR_tumour + λ·max(0, β·SNR_tumour − SNR_healthy)
    """
    n = S_tumour.shape[1]
    levels = np.linspace(0.1, 1.0, n_coarse_levels)

    best_obj  = -np.inf
    best_w    = np.ones(n) / np.sqrt(n)

    # Limit exhaustive search to ≤ n_coarse_levels^min(n,4) combinations
    search_dim = min(n, 4)
    for combo in product(levels, repeat=search_dim):
        w = np.array(list(combo) + [0.5]*(n - search_dim))
        w = w / (np.linalg.norm(w) + 1e-12)
        snr_t = combined_signal(w, S_tumour)
        snr_h = combined_signal(w, S_healthy)
        obj   = snr_t - 10.0 * max(0.0, beta*snr_t - snr_h)
        if obj > best_obj:
            best_obj = obj
            best_w   = w.copy()

    # Refine with gradient-based optimiser
    noise_sig = 1e-9

    def neg_snr(w_):
        w_ = np.abs(w_)
        w_ = w_ / (np.linalg.norm(w_) + 1e-12)
        snr_t = combined_signal(w_, S_tumour, noise_sig)
        snr_h = combined_signal(w_, S_healthy, noise_sig)
        penalty = 10.0 * max(0.0, beta * snr_t - snr_h)
        return -(snr_t - penalty)

    res = minimize(neg_snr, best_w,
                   method="L-BFGS-B",
                   bounds=[(0.0, 1.0)]*n,
                   options={"maxiter": 400, "ftol": 1e-10})
    w_opt = np.abs(res.x)
    w_opt = w_opt / (np.linalg.norm(w_opt) + 1e-12)
    return w_opt


# ─────────────────────────────────────────────────────────────────────────────
# 6.  JAYNES-CUMMINGS STATE-TRANSFER MODEL
# ─────────────────────────────────────────────────────────────────────────────

HBAR   = 1.0545718e-34   # J·s
OMEGA0 = 2*np.pi * 127.73e6   # ¹H Larmor at 3 T  (rad/s)


def state_transfer_efficiency(coupling_g: float,
                               omega_cavity: float,
                               detuning_delta: float,
                               t_interact_us: float) -> dict:
    """
    Jaynes-Cummings vacuum Rabi oscillation (single excitation subspace).
    
    Rabi frequency: Ω_R = sqrt(g² + δ²)
    Population inversion:  P_e(t) = (g/Ω_R)² · sin²(Ω_R · t)

    Parameters
    ----------
    coupling_g      : vacuum Rabi coupling [rad/s]
    omega_cavity    : cavity mode frequency [rad/s]
    detuning_delta  : ω₀ − ω_cav [rad/s]
    t_interact_us   : interaction time [µs]

    Returns
    -------
    dict with Rabi_freq, max_transfer, t_pi_us, coherence_time_us
    """
    t = t_interact_us * 1e-6
    Omega_R   = np.sqrt(coupling_g**2 + detuning_delta**2)
    P_excited = (coupling_g / max(Omega_R, 1e-20))**2 * \
                np.sin(Omega_R * t)**2

    # π-pulse time (complete state transfer)
    t_pi_s   = np.pi / (2 * max(Omega_R, 1e-20))
    
    # Effective coherence time (decay at κ = g/10 approximation)
    kappa    = coupling_g / 10.0
    T_coh_us = 1.0 / max(kappa, 1e-20) * 1e6

    return {
        "Rabi_freq_MHz":        float(Omega_R / (2*np.pi) * 1e-6),
        "state_transfer":       float(P_excited),
        "t_pi_us":              float(t_pi_s * 1e6),
        "coherence_time_us":    float(T_coh_us),
    }


def per_element_coupling(coil_pos: np.ndarray,
                          tumour_center: np.ndarray,
                          base_g: float = 2*np.pi*1e6) -> float:
    """
    Coupling g scales as 1/r³ (dipole coupling falloff).
    """
    r = np.linalg.norm(coil_pos - tumour_center)
    return base_g / max(r * 100, 1.0)**3


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN COIL DESIGNER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ConformatSkullCoil:
    """
    Conformally shimmed skull-surface head coil array.

    Parameters
    ----------
    n_elements   : number of receive coil elements (default 8)
    semi_axes    : skull ellipsoid semi-axes (m)  (a,b,c)
    loop_radius  : radius of each individual loop element (m)
    """

    def __init__(self, n_elements: int = 8,
                 semi_axes: tuple = (0.100, 0.120, 0.095),
                 loop_radius: float = 0.040):
        self.n_el         = n_elements
        self.semi_axes    = semi_axes
        self.loop_radius  = loop_radius

        self.positions    = skull_surface_points(n_elements, semi_axes)
        self.normals      = skull_normals(self.positions, semi_axes)

    # ── public API ────────────────────────────────────────────────────────────

    def optimise(self,
                 tumour_center: list   = [0.02, -0.01, 0.03],
                 tumour_radius: float  = 0.025,
                 healthy_shell_factor: float  = 2.0,
                 beta: float           = 0.7,
                 n_roi_pts: int        = 64,
                 omega_cavity: float   = 2*np.pi*127.73e6,
                 detuning_delta: float = 2*np.pi*1e4) -> dict:
        """
        Run the full coil-design and shimming pipeline.

        Returns
        -------
        dict with all results + plot_b64
        """
        tc = np.array(tumour_center)

        # ROIs
        pts_tumour  = roi_points(tc, tumour_radius, n_roi_pts)
        pts_healthy = roi_points(tc, tumour_radius * healthy_shell_factor,
                                  n_roi_pts)

        # Sensitivity matrices
        S_all    = build_sensitivity_matrix(self.positions, self.normals,
                                             np.vstack([pts_tumour,
                                                        pts_healthy]),
                                             self.loop_radius)
        S_t = S_all[:n_roi_pts, :]
        S_h = S_all[n_roi_pts:, :]

        # Combinatorial shim
        w_opt = combinatorial_shim(S_t, S_h, beta=beta)

        # SNR metrics
        noise_sig = 1e-9
        snr_t = combined_signal(w_opt, S_t, noise_sig)
        snr_h = combined_signal(w_opt, S_h, noise_sig)

        # State-transfer per element
        transfers = []
        for j in range(self.n_el):
            g  = per_element_coupling(self.positions[j], tc)
            st = state_transfer_efficiency(
                    coupling_g     = g,
                    omega_cavity   = omega_cavity,
                    detuning_delta = detuning_delta,
                    t_interact_us  = 100.0)
            transfers.append(st)

        # Combined efficiency (weighted by shim weight)
        eff_combined = float(np.dot(w_opt, [t["state_transfer"] for t in transfers]))

        # Geometry summary
        geometry = {
            "n_elements":    self.n_el,
            "semi_axes_m":   list(self.semi_axes),
            "loop_radius_m": self.loop_radius,
            "positions_m":   self.positions.tolist(),
            "normals":       self.normals.tolist(),
            "shim_weights":  w_opt.tolist(),
        }

        plot_b64 = self._render_plot(pts_tumour, pts_healthy, w_opt,
                                     snr_t, snr_h, transfers, tc)

        return {
            "shim_weights":              w_opt.tolist(),
            "snr_tumour":                float(snr_t),
            "snr_healthy":               float(snr_h),
            "tumour_to_healthy_ratio":   float(snr_t / max(snr_h, 1e-9)),
            "state_transfers":           transfers,
            "state_transfer_efficiency": eff_combined,
            "geometry":                  geometry,
            "plot_b64":                  plot_b64,
        }

    # ── visualisation ─────────────────────────────────────────────────────────

    def _render_plot(self, pts_t, pts_h, w_opt, snr_t, snr_h,
                     transfers, tumour_center) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D   # noqa

            fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")

            COL   = "#00c8ff"
            col2  = "#ff6f61"
            col3  = "#b2ff59"
            colW  = "#ffd700"

            # ── 1  3-D coil layout ────────────────────────────────────────────
            ax1 = fig.add_subplot(1, 3, 1, projection="3d",
                                  facecolor="#161b22")
            ax1.set_facecolor("#161b22")
            ax1.scatter(*self.positions.T, c=w_opt,
                        cmap="cool", s=120*w_opt+30, depthshade=True,
                        edgecolors="white", lw=0.5)
            ax1.scatter(*pts_t.T, c=col2, s=6, alpha=0.4, label="Tumour ROI")
            ax1.scatter(*pts_h.T, c=col3, s=4, alpha=0.2, label="Healthy ROI")
            ax1.scatter(*tumour_center, c=colW, s=200, marker="*",
                        label="Tumour centre")
            ax1.set_title("Conformal Skull Array\n(size ∝ shim weight)",
                          color=COL, fontweight="bold")
            ax1.tick_params(colors="white")
            ax1.xaxis.pane.fill = False
            ax1.yaxis.pane.fill = False
            ax1.zaxis.pane.fill = False
            ax1.legend(fontsize=7, labelcolor="white",
                       facecolor="#161b22", edgecolor="#30363d", loc="upper left")

            # ── 2  Shim-weight bar ────────────────────────────────────────────
            ax2 = fig.add_subplot(1, 3, 2, facecolor="#161b22")
            bars = ax2.bar(range(1, self.n_el+1), w_opt,
                           color=[plt.cm.cool(w) for w in w_opt],
                           edgecolor="none")
            ax2.set_xlabel("Coil element", color="white")
            ax2.set_ylabel("Shim weight", color="white")
            ax2.set_title("Combinatorial Shim Weights", color=COL,
                          fontweight="bold")
            ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#30363d")
            ax2.set_facecolor("#161b22")
            ax2.axhline(np.mean(w_opt), color="yellow", linestyle="--",
                        lw=1.2, label=f"mean={np.mean(w_opt):.2f}")
            ax2.legend(fontsize=8, labelcolor="white",
                       facecolor="#161b22", edgecolor="#30363d")

            # ── 3  State-transfer barh ────────────────────────────────────────
            ax3 = fig.add_subplot(1, 3, 3, facecolor="#161b22")
            st_vals = [t["state_transfer"] for t in transfers]
            coh_vals = [t["coherence_time_us"] for t in transfers]
            ax3.barh(range(1, self.n_el+1), st_vals,
                     color=col3, edgecolor="none", label="State transfer P_e")
            ax3b = ax3.twiny()
            ax3b.plot(coh_vals, range(1, self.n_el+1),
                      "o-", color=colW, lw=1.5, label="Coherence (µs)")
            ax3b.tick_params(colors=colW)
            ax3b.spines[:].set_color("#30363d")
            ax3.set_xlabel("Transfer probability", color="white")
            ax3.set_ylabel("Coil element", color="white")
            ax3.set_title(
                f"JC State Transfer\nEfficiency (combined): {np.dot(w_opt, st_vals):.3f}",
                color=COL, fontweight="bold")
            ax3.tick_params(colors="white")
            ax3.spines[:].set_color("#30363d")
            ax3.set_facecolor("#161b22")

            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1+lines2, labels1+labels2, fontsize=7,
                       labelcolor="white", facecolor="#161b22",
                       edgecolor="#30363d")

            # Super-title
            fig.suptitle(
                f"Conformal Skull Coil  |  SNR tumour: {snr_t:.2e}  "
                f"healthy: {snr_h:.2e}  ratio: {snr_t/max(snr_h,1e-9):.2f}×",
                color="white", fontsize=11, fontweight="bold", y=1.01
            )

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except Exception:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    coil = ConformatSkullCoil(n_elements=8)
    res  = coil.optimise(tumour_center=[0.02, -0.01, 0.03],
                          tumour_radius=0.025)
    print(f"SNR tumour : {res['snr_tumour']:.3e}")
    print(f"SNR healthy: {res['snr_healthy']:.3e}")
    print(f"Tumour/healthy ratio: {res['tumour_to_healthy_ratio']:.2f}×")
    print(f"State-transfer efficiency: {res['state_transfer_efficiency']:.4f}")
    print(f"Shim weights: {[f'{w:.3f}' for w in res['shim_weights']]}")
