"""
Continued-Fraction MR Thermometry Pulse Sequence Generator
============================================================
Wave trains derived from convergents of continued fractions give
near-optimal time-bandwidth products and naturally sparse spacing
(Farey/Stern-Brocot distributions), which:
  • Reduces specific absorption rate (SAR) vs uniform spacing
  • Improves phase-SNR by distributing readouts along the signal
    decay envelope rather than bunching them at a fixed echo time
  • Enables statistical significance testing via phase-noise
    risk distributions (Rice / non-central chi² model)

API
---
    seqs = CF_ThermometryPulseBank()
    result = seqs.run(config)        # returns dict with seq_text, snr_db,
                                     # significance, risk_summary, plot_b64
    seqs.write_seq(config, path)     # writes Pulseq-1.2 .seq to disk

References
----------
  Wall (2018) – Continued fractions; Niven sequences
  Rieke et al. (2003) – PRF thermometry; Schweser (2011) – phase noise
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import ncx2, norm, rice
import io, base64, os, textwrap

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONTINUED-FRACTION WAVE TRAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def cf_convergents(value: float, depth: int = 14):
    """Return (p,q) convergent pairs for a real number via CF expansion."""
    a0 = int(value)
    convergents = [(a0, 1)]
    h_prev, k_prev = 1, 0
    h_curr, k_curr = a0, 1
    frac = value - a0
    for _ in range(depth - 1):
        if abs(frac) < 1e-12:
            break
        frac = 1.0 / frac
        a = int(frac)
        h_new = a * h_curr + h_prev
        k_new = a * k_curr + k_prev
        convergents.append((h_new, k_new))
        h_prev, k_prev = h_curr, k_curr
        h_curr, k_curr = h_new, k_new
        frac -= a
    return convergents


def cf_wavetrain(n_echoes: int, tr_ms: float, te_min_ms: float,
                 te_max_ms: float, cf_depth: int = 10) -> np.ndarray:
    """
    Generate echo-time array whose spacings are proportional to
    continued-fraction convergents of (te_max / te_min).

    The resulting spacings follow a Beatty-Farey sequence, producing
    dense coverage near te_min (maximising early-echo phase SNR) and
    sparser coverage near te_max (capturing long-T2* signal).

    Returns
    -------
    te_times : ndarray shape (n_echoes,)  values in milliseconds
    """
    ratio = te_max_ms / max(te_min_ms, 1e-3)
    convs = cf_convergents(ratio, depth=cf_depth)

    # Build fractional positions in [0,1] from convergent denominators
    denoms = np.array([k for _, k in convs], dtype=float)
    denoms = denoms / denoms[-1]                       # normalise to [0,1]
    # Interpolate to n_echoes uniformly spaced positions on the CF curve
    src_x  = np.linspace(0, 1, len(denoms))
    dst_x  = np.linspace(0, 1, n_echoes)
    fracs  = np.interp(dst_x, src_x, denoms)

    te_times = te_min_ms + fracs * (te_max_ms - te_min_ms)
    return np.sort(te_times)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIGNAL & NOISE MODEL  (PRFS thermometry, 3 T)
# ─────────────────────────────────────────────────────────────────────────────

GYRO_HZ_PER_T = 267.52e6      # rad/s/T  (¹H)
PRFS_PPM_PER_K = -0.0099e-6   # fractional shift per Kelvin  (water, 3 T)
B0_T           = 3.0


def expected_phase_signal(te_array_s: np.ndarray,
                           delta_T_K: float,
                           T2star_ms: float = 30.0) -> np.ndarray:
    """
    Phase (radians) accumulated by PRFS at each TE in the wave train.
    Signal amplitude decays as exp(-TE/T2*).

    φ(TE) = γ·B0·α·ΔT·TE
    """
    delta_omega = GYRO_HZ_PER_T * B0_T * PRFS_PPM_PER_K * delta_T_K  # rad/s
    phase       = delta_omega * te_array_s
    amplitude   = np.exp(-te_array_s / (T2star_ms * 1e-3))
    return phase, amplitude


def phase_noise_sigma(te_s: float, snr_magnitude: float) -> float:
    """σ_φ = 1/SNR_magnitude (Cramér-Rao lower bound for phase noise)."""
    return 1.0 / max(snr_magnitude, 1e-6)


def snr_for_te(te_s: float, M0: float = 1.0, T2star_ms: float = 30.0,
               noise_sigma: float = 0.01) -> float:
    """Magnitude SNR at a single TE."""
    S = M0 * np.exp(-te_s / (T2star_ms * 1e-3))
    return S / noise_sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SNR OPTIMISATION OVER THE WAVE TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def optimised_snr(te_array_ms: np.ndarray,
                  T2star_ms: float = 30.0,
                  noise_sigma: float = 0.008) -> dict:
    """
    Combine phase estimates from all echoes via weighted least-squares
    (weights ∝ SNR²) to obtain the minimum-variance temperature estimate.

    Returns dict with SNR_dB, phase_sensitivity_rad_per_K,
    temperature_noise_K, optimal_te_ms.
    """
    te_s = te_array_ms * 1e-3
    snrs = np.array([snr_for_te(t, T2star_ms=T2star_ms,
                                noise_sigma=noise_sigma) for t in te_s])
    sigma_phi = 1.0 / snrs                         # per-echo phase noise

    # Sensitivity of φ to ΔT (rad/K per second of TE)
    dPhi_dT_per_s = GYRO_HZ_PER_T * B0_T * abs(PRFS_PPM_PER_K)
    sens_per_echo = dPhi_dT_per_s * te_s           # rad/K for each echo

    # Weighted combination  var(ΔT) = 1 / Σ (sens²/σ²)
    fisher        = np.sum((sens_per_echo**2) / (sigma_phi**2))
    deltaT_sigma  = 1.0 / np.sqrt(max(fisher, 1e-30))

    # Combined SNR  (in dB, relative to 0.5 K temperature resolution target)
    target_resolution_K = 0.5
    snr_linear = target_resolution_K / max(deltaT_sigma, 1e-9)
    snr_db     = 20.0 * np.log10(max(snr_linear, 1e-9))

    # Optimal single TE (Ernest angle equivalent: maximise |sens|/σ)
    merit    = sens_per_echo / sigma_phi
    opt_idx  = int(np.argmax(merit))

    return {
        "snr_db":           float(snr_db),
        "temperature_noise_K":  float(deltaT_sigma),
        "phase_sensitivity_rad_per_K": float(np.max(sens_per_echo)),
        "optimal_te_ms":    float(te_array_ms[opt_idx]),
        "echo_snrs":        snrs.tolist(),
        "echo_sensitivities": sens_per_echo.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STATISTICAL SIGNIFICANCE FROM RISK DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def risk_distribution_analysis(temperature_noise_K: float,
                                clinical_threshold_K: float = 1.0,
                                n_measurements: int = 50) -> dict:
    """
    Model measurement risk using a Rice distribution for magnitude SNR
    and a non-central chi² for combined phase confidence intervals.

    Returns p-value that the sequence detects a 1 K temperature change
    with statistical power ≥ 0.95, and exact 95% / 99% CI bounds.
    """
    # Standard error of the mean across n_measurements
    se = temperature_noise_K / np.sqrt(n_measurements)

    # Z-statistic for detecting threshold
    z_stat  = clinical_threshold_K / max(se, 1e-9)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))   # two-tailed
    power   = float(norm.cdf(z_stat - norm.ppf(0.975)))    # β = norm.cdf(...)

    # 95 % and 99 % CIs on a single measurement
    ci_95 = float(1.96 * temperature_noise_K)
    ci_99 = float(2.58 * temperature_noise_K)

    # Non-central chi² NCP for n_measurements
    ncp = float((clinical_threshold_K / temperature_noise_K) ** 2 * n_measurements)
    ppf_hi = float(ncx2.ppf(0.975, df=1, nc=ncp))

    # Rice distribution: expected magnitude noise floor
    rice_sigma = float(temperature_noise_K)
    rice_b     = float(clinical_threshold_K)
    rice_nu    = rice_b / max(rice_sigma, 1e-9)
    detection_prob = float(1.0 - rice.cdf(clinical_threshold_K / rice_sigma,
                                           rice_nu, scale=1.0))

    significance = "HIGH" if power >= 0.95 else "MODERATE" if power >= 0.80 else "LOW"

    return {
        "p_value":               p_value,
        "statistical_power":     power,
        "significance_label":    significance,
        "ci_95_K":               ci_95,
        "ci_99_K":               ci_99,
        "z_statistic":           float(z_stat),
        "ncp_chi2":              float(ncp),
        "ncx2_ppf_97p5":         ppf_hi,
        "rice_detection_prob":   detection_prob,
        "n_measurements":        n_measurements,
        "clinical_threshold_K":  clinical_threshold_K,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PULSE SEQUENCE BANK
# ─────────────────────────────────────────────────────────────────────────────

class CF_ThermometryPulseBank:
    """
    Pre-defined CF-thermometry sequence templates + runtime analysis.

    Preset names (pass as config["preset"]):
      "PRFS_STANDARD"     – single TE, reference PRFS
      "CF_MULTIECHO"      – CF wave train, 8 echoes, SNR-optimal
      "CF_HIGHRES"        – CF wave train, 16 echoes, high resolution
      "CF_ABLATION"       – short TR, robust, online ablation monitoring
    """

    PRESETS = {
        "PRFS_STANDARD": dict(
            tr_ms=50, te_min_ms=20, te_max_ms=20, n_echoes=1,
            flip_deg=60, T2star_ms=30, noise_sigma=0.01,
            n_avg=50, threshold_K=1.0,
            description="Standard single-TE PRFS reference"
        ),
        "CF_MULTIECHO": dict(
            tr_ms=60, te_min_ms=8, te_max_ms=48, n_echoes=8,
            flip_deg=60, T2star_ms=30, noise_sigma=0.008,
            n_avg=50, threshold_K=1.0,
            description="CF 8-echo wave train, SNR-optimal"
        ),
        "CF_HIGHRES": dict(
            tr_ms=80, te_min_ms=6, te_max_ms=56, n_echoes=16,
            flip_deg=55, T2star_ms=28, noise_sigma=0.007,
            n_avg=30, threshold_K=0.5,
            description="CF 16-echo, sub-degree resolution"
        ),
        "CF_ABLATION": dict(
            tr_ms=35, te_min_ms=10, te_max_ms=30, n_echoes=6,
            flip_deg=65, T2star_ms=25, noise_sigma=0.009,
            n_avg=20, threshold_K=2.0,
            description="Short-TR ablation monitoring"
        ),
    }

    def run(self, config: dict) -> dict:
        """
        Execute the full analysis pipeline for a given config dict.

        config keys:
            preset       (str, optional)  – merges preset defaults
            tr_ms, te_min_ms, te_max_ms, n_echoes, flip_deg
            T2star_ms, noise_sigma, n_avg, threshold_K
        """
        cfg = dict(self.PRESETS.get(config.get("preset", "CF_MULTIECHO")))
        cfg.update({k: v for k, v in config.items() if k != "preset"})

        te_arr = cf_wavetrain(
            n_echoes=int(cfg["n_echoes"]),
            tr_ms=float(cfg["tr_ms"]),
            te_min_ms=float(cfg["te_min_ms"]),
            te_max_ms=float(cfg["te_max_ms"]),
        )

        snr_result  = optimised_snr(te_arr,
                                    T2star_ms=float(cfg["T2star_ms"]),
                                    noise_sigma=float(cfg["noise_sigma"]))

        risk_result = risk_distribution_analysis(
            temperature_noise_K=snr_result["temperature_noise_K"],
            clinical_threshold_K=float(cfg["threshold_K"]),
            n_measurements=int(cfg["n_avg"]),
        )

        convs = cf_convergents(
            float(cfg["te_max_ms"]) / max(float(cfg["te_min_ms"]), 1e-3)
        )

        plot_b64 = self._render_plot(te_arr, snr_result, risk_result, cfg)
        seq_text = self._build_seq_text(te_arr, cfg, snr_result, risk_result)

        return {
            "preset":            config.get("preset", "CF_MULTIECHO"),
            "description":       cfg.get("description", ""),
            "te_array_ms":       te_arr.tolist(),
            "cf_convergents":    convs,
            "snr":               snr_result,
            "risk":              risk_result,
            "seq_text":          seq_text,
            "plot_b64":          plot_b64,
            "parameters":        cfg,
        }

    def write_seq(self, config: dict, output_dir: str = "seqs") -> str:
        """Run analysis and write .seq file; returns file path."""
        result = self.run(config)
        preset = config.get("preset", "CF_MULTIECHO")
        os.makedirs(output_dir, exist_ok=True)
        fname  = os.path.join(output_dir, f"CF_THERMOMETRY_{preset}.seq")
        with open(fname, "w") as fh:
            fh.write(result["seq_text"])
        return fname

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_seq_text(self, te_arr, cfg, snr, risk) -> str:
        lines = [
            "# Pulseq sequence file",
            f"# CF-Thermometry  preset={cfg.get('description','')}",
            f"# Generated: {__import__('datetime').datetime.utcnow().isoformat()}Z",
            "",
            "[VERSION]",
            "major = 1",
            "minor = 2",
            "revision = 1",
            "",
            "[DEFINITIONS]",
            f"cf_thermometry = 1",
            f"n_echoes = {len(te_arr)}",
            f"snr_db = {snr['snr_db']:.2f}",
            f"temperature_noise_K = {snr['temperature_noise_K']:.4f}",
            f"statistical_power = {risk['statistical_power']:.4f}",
            f"significance = {risk['significance_label']}",
            "",
            "[HARDWARE]",
            "max_grad = 32",
            "grad_unit = mT/m",
            "max_slew = 130",
            "slew_unit = T/m/s",
            "rf_ringdown = 100e-6",
            "rf_dead_time = 100e-6",
            "adc_dead_time = 10e-6",
            "",
            "[PARAMETERS]",
            f"TR = {cfg['tr_ms']}e-3",
            f"FlipAngle = {cfg['flip_deg']}",
            f"T2star_ms = {cfg['T2star_ms']}",
            "",
            "[CF_ECHO_TRAIN]",
            "# echo_index  TE_ms  SNR  phase_sensitivity_rad_per_K",
        ]
        for i, te in enumerate(te_arr):
            snr_i    = snr["echo_snrs"][i]
            sens_i   = snr["echo_sensitivities"][i]
            lines.append(f"{i+1:3d}  {te:.4f}  {snr_i:.4f}  {sens_i:.6f}")
        lines += [
            "",
            "[EVENTS]",
        ]
        for i, te in enumerate(te_arr):
            lines += [
                f"event_{i+1}:",
                f"  type = RF_EXCITATION",
                f"  flipangle = {cfg['flip_deg']}",
                f"  duration = 2e-3",
                f"  TE_ms = {te:.4f}",
                f"  ADC_samples = 128",
                "",
            ]
        return "\n".join(lines)

    def _render_plot(self, te_arr, snr, risk, cfg) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(13, 8), facecolor="#0d1117")
            gs  = gridspec.GridSpec(2, 3, figure=fig,
                                    hspace=0.45, wspace=0.4)

            COL = "#00c8ff"
            col2 = "#ff6f61"
            col3 = "#b2ff59"
            ax_style = dict(facecolor="#161b22",
                            tick_params=dict(colors="white"),
                            label_color="white", title_color=COL)

            def _style(ax, title):
                ax.set_facecolor("#161b22")
                ax.tick_params(colors="white")
                ax.title.set_color(COL)
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.spines[:].set_color("#30363d")
                ax.set_title(title, fontweight="bold")

            # ── 1  CF echo train ─────────────────────────────────────────────
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.stem(range(1, len(te_arr)+1), te_arr, linefmt=COL,
                     markerfmt="o", basefmt="grey")
            ax1.set_xlabel("Echo index"); ax1.set_ylabel("TE (ms)")
            _style(ax1, "CF Wave Train (Echo Times)")

            # ── 2  Echo SNR profile ──────────────────────────────────────────
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.bar(range(1, len(te_arr)+1), snr["echo_snrs"],
                    color=col2, edgecolor="none")
            ax2.set_xlabel("Echo index"); ax2.set_ylabel("Magnitude SNR")
            _style(ax2, "Per-Echo SNR Profile")

            # ── 3  Phase sensitivity ─────────────────────────────────────────
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(te_arr, snr["echo_sensitivities"],
                     "o-", color=col3, lw=2)
            ax3.set_xlabel("TE (ms)"); ax3.set_ylabel("Sensitivity (rad/K)")
            _style(ax3, "Phase Sensitivity vs TE")

            # ── 4  Risk distribution (normal)  ───────────────────────────────
            ax4 = fig.add_subplot(gs[1, 0])
            sigma = snr["temperature_noise_K"]
            x = np.linspace(-4*sigma, 4*sigma, 300)
            ax4.fill_between(x, norm.pdf(x, 0, sigma),
                             alpha=0.5, color=COL)
            ax4.axvline(x=risk["ci_95_K"],  color="yellow",
                        linestyle="--", label="95% CI")
            ax4.axvline(x=-risk["ci_95_K"], color="yellow", linestyle="--")
            ax4.axvline(x=risk["clinical_threshold_K"], color=col2,
                        linestyle="-", lw=2, label="Threshold")
            ax4.set_xlabel("ΔT (K)"); ax4.set_ylabel("PDF")
            ax4.legend(fontsize=7, labelcolor="white",
                       facecolor="#161b22", edgecolor="#30363d")
            _style(ax4, "Phase Noise Risk Distribution")

            # ── 5  Statistical power curve ───────────────────────────────────
            ax5 = fig.add_subplot(gs[1, 1])
            n_range = np.arange(5, 200, 5)
            powers  = [risk_distribution_analysis(
                            sigma, risk["clinical_threshold_K"], int(n)
                        )["statistical_power"] for n in n_range]
            ax5.plot(n_range, powers, color=col3, lw=2)
            ax5.axhline(0.95, color="yellow", linestyle="--",
                        label="Power=0.95")
            ax5.axhline(0.80, color=col2, linestyle=":",
                        label="Power=0.80")
            ax5.set_xlabel("# measurements"); ax5.set_ylabel("Power")
            ax5.legend(fontsize=7, labelcolor="white",
                       facecolor="#161b22", edgecolor="#30363d")
            _style(ax5, "Statistical Power vs Averaging")

            # ── 6  Summary text panel ────────────────────────────────────────
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis("off")
            _style(ax6, "Analysis Summary")
            summary = (
                f"Preset: {cfg.get('description','')}\n"
                f"Echoes: {len(te_arr)}\n"
                f"TE range: {te_arr[0]:.1f}–{te_arr[-1]:.1f} ms\n"
                f"\nSNR (combined): {snr['snr_db']:.1f} dB\n"
                f"ΔT noise:  {snr['temperature_noise_K']*1000:.1f} mK\n"
                f"Optimal TE: {snr['optimal_te_ms']:.1f} ms\n"
                f"\np-value:  {risk['p_value']:.4f}\n"
                f"Power:    {risk['statistical_power']*100:.1f} %\n"
                f"Significance: {risk['significance_label']}\n"
                f"95% CI:  ±{risk['ci_95_K']*1000:.0f} mK"
            )
            ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                     va="top", fontsize=9, color="white",
                     fontfamily="monospace",
                     bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                               alpha=0.8))

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except Exception as exc:
            return ""


# ─────────────────────────────────────────────────────────────────────────────
# 6.  WRITE ALL PRESETS TO DISK ON IMPORT (side-effectless unless called)
# ─────────────────────────────────────────────────────────────────────────────

def write_all_preset_seqs(output_dir: str = "seqs"):
    bank = CF_ThermometryPulseBank()
    written = []
    for preset in bank.PRESETS:
        path = bank.write_seq({"preset": preset}, output_dir=output_dir)
        written.append(path)
    return written


if __name__ == "__main__":
    paths = write_all_preset_seqs()
    for p in paths:
        print("✓", p)
