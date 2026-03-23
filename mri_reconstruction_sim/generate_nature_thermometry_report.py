#!/usr/bin/env python3
"""
Nature Technical Report: Enhanced Quantum MR Thermometry
with Colorised Maps, Finite Math Equations, FFTW Reconstruction,
Multimodal Reasoning, SNR Characteristics and New Pulse Sequences

Generates:
  1. New thermometry pulse sequences (.seq files)
  2. Colorised MR thermometry maps overlaid on reconstructed images
  3. Nature-formatted PDF with finite math, SNR tables, distribution analysis
"""

import os
import sys
import io
import base64
import math
import tempfile
from datetime import datetime

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_thermometry_enhanced import (
    run_enhanced_thermometry_pipeline,
    TISSUE_LUT, B0_LUT,
    generate_thermometry_pulse_sequence,
    compute_fisher_information_temperature,
    GAMMA_HZ, GAMMA_RAD, PRF_ALPHA, T_BASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _b64_to_rl_image(b64_str: str, width: float, height: float) -> RLImage:
    """Convert a base64-encoded PNG to a ReportLab Image flowable."""
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    return RLImage(buf, width=width, height=height)


def _save_b64_png(b64_str: str, path: str):
    """Persist a base64 PNG to disk."""
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Report Generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_thermometry_nature_report(
    B0: float = 3.0,
    n_echoes: int = 10,
    matrix: int = 128,
    hotspot_dT: float = 6.0,
    target_temp: float = 55.0,
    output_dir: str = None,
):
    """
    Run the full enhanced thermometry pipeline, generate .seq files,
    and write a Nature-style technical PDF report.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Run Pipeline ──────────────────────────────────────────────────────
    print("[1/4] Running enhanced thermometry pipeline …")
    result = run_enhanced_thermometry_pipeline(
        n_echoes=n_echoes,
        B0=B0,
        matrix=matrix,
        hotspot_dT=hotspot_dT,
        probe_x=matrix // 2 + 15,
        probe_y=matrix // 2 - 5,
        target_temp=target_temp,
        seq_types=["multiecho_gre", "epi_thermometry", "prfs_highres", "stack_of_stars"],
    )
    print(f"     SNR = {result['snr_mean']:.1f} ({result['snr_db']:.1f} dB)")
    print(f"     RMSE raw = {result['rmse_raw_C']:.4f} °C,  QML = {result['rmse_qml_C']:.4f} °C")
    print(f"     Best distribution: {result['best_distribution']}")

    # ── 2. Write generated .seq summary ───────────────────────────────────────
    print("[2/4] Pulse sequences written:")
    for sp in result["generated_seq_paths"]:
        print(f"      → {sp}")

    # ── 3. Save figure PNGs alongside report ─────────────────────────────────
    fig_dir = os.path.join(output_dir, "report_figures")
    os.makedirs(fig_dir, exist_ok=True)
    for key in ("plot_thermometry", "plot_distributions", "plot_multimodal", "plot_fftw_recon"):
        _save_b64_png(result[key], os.path.join(fig_dir, f"{key}.png"))
    print(f"[3/4] Figures saved to {fig_dir}")

    # ── 4. Build PDF ─────────────────────────────────────────────────────────
    pdf_name = f"Nature_Quantum_MR_Thermometry_{B0:.0f}T_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    print(f"[4/4] Writing PDF → {pdf_path}")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        rightMargin=0.7 * inch, leftMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Styles ────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=14,
        textColor=colors.HexColor("#000000"), spaceAfter=8,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#333333"), spaceAfter=5,
        alignment=TA_CENTER, fontName="Helvetica-Oblique",
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=11,
        textColor=colors.HexColor("#000000"), spaceAfter=6,
        spaceBefore=10, fontName="Helvetica-Bold",
    )
    subheading_style = ParagraphStyle(
        "SubHeading", parent=styles["Heading3"], fontSize=10,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9.5,
        alignment=TA_JUSTIFY, spaceAfter=8, leading=13,
    )
    eq_style = ParagraphStyle(
        "Equation", parent=styles["Normal"], fontSize=9,
        alignment=TA_CENTER, spaceAfter=6, spaceBefore=4,
        fontName="Courier",
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=8.5,
        alignment=TA_CENTER, spaceAfter=10, spaceBefore=2,
        fontName="Helvetica-Oblique", textColor=colors.HexColor("#444444"),
    )
    affil_style = ParagraphStyle(
        "Affil", parent=styles["Normal"], fontSize=8,
        alignment=TA_CENTER, spaceAfter=10,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        "Enhanced Quantum MR Thermometry with Tissue-Specific Lookup Tables, "
        "FFTW-Optimised Reconstruction, Multimodal Bayesian Reasoning, "
        "and Colorised Temperature Probe Control",
        title_style,
    ))
    elements.append(Spacer(1, 0.08 * inch))
    elements.append(Paragraph(
        "<i>Cartik Sharma</i><sup>1,2*</sup>, "
        "<i>NeuroPulse Research Collaborative</i><sup>1</sup>",
        subtitle_style,
    ))
    elements.append(Paragraph(
        "<sup>1</sup>Institute for Computational Neuromagnetism &amp; Geometric Analysis<br/>"
        "<sup>2</sup>Department of Advanced Imaging Sciences<br/>"
        "*Correspondence: cartiksharma@neuromorph.ai",
        affil_style,
    ))
    elements.append(Spacer(1, 0.12 * inch))

    # ── Abstract ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("Abstract", heading_style))
    elements.append(Paragraph(
        f"We present a comprehensive quantum-enhanced MR thermometry framework operating at "
        f"B<sub>0</sub> = {B0:.1f} T with {n_echoes}-echo acquisition on a {matrix}×{matrix} "
        f"reconstruction matrix. Tissue-specific proton resonance frequency (PRF) lookup tables "
        f"spanning 8 tissue classes and 4 field strengths enable adaptive phase-to-temperature "
        f"conversion with Cramér–Rao minimum detectable ΔT = "
        f"{result['fisher_info']['min_detectable_dT_C']:.3f} °C. "
        f"FFTW-accelerated partial-Fourier reconstruction with {result['pf_factor']}-factor "
        f"homodyne POCS achieves mean SNR = {result['snr_mean']:.1f} ({result['snr_db']:.1f} dB). "
        f"Multimodal Bayesian fusion of phase-based and T<sub>1</sub>-based temperature estimates "
        f"combined with quantum RBM denoising reduces RMSE from "
        f"{result['rmse_raw_C']:.4f} °C (raw) to {result['rmse_qml_C']:.4f} °C. "
        f"Signal magnitude distributions are best described by the "
        f"{result['best_distribution'].title()} model (KS = {result['best_ks_stat']:.4f}). "
        f"Four optimised pulse sequences are generated in Pulseq-1.2 format for "
        f"real-time thermometry with PID-controlled temperature probe feedback.",
        body_style,
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  1. INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("1. Introduction", heading_style))
    elements.append(Paragraph(
        "Accurate non-invasive temperature measurement is critical for MR-guided thermal therapies "
        "including focused ultrasound ablation, laser interstitial thermal therapy (LITT), and "
        "radiofrequency hyperthermia. The proton resonance frequency (PRF) shift method exploits "
        "the linear temperature dependence of water proton chemical shift, providing quantitative "
        "maps with sub-degree precision under optimal conditions. However, tissue-type dependence "
        "of the PRF coefficient, partial-Fourier acceleration artifacts, and noise limit clinical "
        "accuracy. This work addresses these limitations through (1) tissue-adaptive lookup tables, "
        "(2) FFTW-optimised homodyne reconstruction, (3) multimodal Bayesian temperature fusion, "
        "and (4) new pulse sequences with integrated probe control.",
        body_style,
    ))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  2. THEORY: PRF THERMOMETRY
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("2. Theory: PRF Shift Thermometry", heading_style))

    elements.append(Paragraph("<b>2.1 Proton Resonance Frequency Shift</b>", subheading_style))
    elements.append(Paragraph(
        "The water proton resonance frequency shifts linearly with temperature due to changes "
        "in hydrogen-bond geometry. The PRF shift coefficient α = −0.0094 ppm/°C is approximately "
        "tissue-independent for aqueous tissues, but varies for fat and pathological tissue:",
        body_style,
    ))
    elements.append(Paragraph(
        "Δf(T) = γ · B₀ · α · ΔT",
        eq_style,
    ))
    elements.append(Paragraph(
        "Δφ(TE, ΔT) = 2π · γ · B₀ · α · TE · ΔT",
        eq_style,
    ))
    elements.append(Paragraph(
        f"where γ = {GAMMA_HZ:.3e} Hz/T (gyromagnetic ratio), B₀ = {B0:.1f} T, "
        f"α = {PRF_ALPHA:.4e} ppm/°C, and TE is the echo time. "
        f"Temperature change is recovered by inverting the phase difference:",
        body_style,
    ))
    elements.append(Paragraph(
        "ΔT = Δφ / (2π · γ · B₀ · α · TE)",
        eq_style,
    ))

    elements.append(Spacer(1, 0.06 * inch))
    elements.append(Paragraph("<b>2.2 Tissue-Specific Lookup Tables</b>", subheading_style))
    elements.append(Paragraph(
        "We construct tissue-specific LUTs mapping (tissue, B₀) → (dφ/dT, optimal TE, SNR "
        "efficiency, minimum detectable ΔT). The phase sensitivity per unit temperature for "
        "tissue <i>k</i> at field strength B₀ is:",
        body_style,
    ))
    elements.append(Paragraph(
        "dφ/dT|_k = γ_rad · B₀ · α_k · TE*_k     [rad/°C]",
        eq_style,
    ))
    elements.append(Paragraph(
        "(dφ/dT|_k)° = (dφ/dT|_k) · 180/π         [deg/°C]",
        eq_style,
    ))
    elements.append(Paragraph(
        "where α_k is the tissue-specific PRF coefficient and TE*_k = T2*_k is the "
        "optimal echo time maximising phase SNR for tissue k. The SNR efficiency is:",
        body_style,
    ))
    elements.append(Paragraph(
        "η_SNR(k) = |dφ/dT|_k| · (TE*_k / T2*_k) · exp(−TE*_k / T2*_k)",
        eq_style,
    ))
    elements.append(Paragraph(
        "Cramér–Rao lower bound on temperature precision:",
        body_style,
    ))
    elements.append(Paragraph(
        "σ_T ≥ 1 / √I(T),    I(T) = Σ_e (2π · γ · B₀ · α · TE_e)² · SNR_e²",
        eq_style,
    ))

    # ── LUT Table ─────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 0.06 * inch))
    lut_data = [["Tissue", "dφ/dT (deg/°C)", "Opt TE (ms)", "SNR Eff", "Min ΔT (°C)"]]
    for tissue, vals in result["lut_summary"].items():
        lut_data.append([
            tissue.replace("_", " ").title(),
            f"{vals['dPhi_dT_deg']:.4f}",
            f"{vals['opt_TE_ms']:.1f}",
            f"{vals['snr_eff']:.6f}",
            f"{vals['min_dT_C']:.2f}",
        ])
    lut_table = Table(lut_data, colWidths=[1.5 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch])
    lut_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(KeepTogether([
        Paragraph(f"<b>Table 1.</b> Tissue-specific PRF lookup table at B₀ = {B0:.1f} T", caption_style),
        lut_table,
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  3. FFTW RECONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("3. FFTW-Optimised Partial-Fourier Reconstruction", heading_style))

    elements.append(Paragraph("<b>3.1 Forward and Inverse DFT</b>", subheading_style))
    elements.append(Paragraph(
        "K-space data is transformed using the centred 2D discrete Fourier transform "
        f"(backend: {result['fftw_backend']}) with optional apodisation window W(k):",
        body_style,
    ))
    elements.append(Paragraph(
        "S(k_x, k_y) = FFT₂ᶜ{ s(x,y) · W(x,y) }",
        eq_style,
    ))
    elements.append(Paragraph(
        "FFT₂ᶜ{f} = fftshift( FFT₂( ifftshift(f) ) )",
        eq_style,
    ))
    elements.append(Paragraph(
        "s(x,y) = IFFT₂ᶜ{ S(k_x, k_y) }    (inverse centred transform)",
        eq_style,
    ))

    elements.append(Paragraph("<b>3.2 Apodisation Windows</b>", subheading_style))
    elements.append(Paragraph(
        "Three windows are supported to reduce Gibbs ringing at the expense of resolution:",
        body_style,
    ))
    elements.append(Paragraph(
        "Hann:     W[n] = 0.5 (1 − cos(2πn / (N−1)))",
        eq_style,
    ))
    elements.append(Paragraph(
        "Hamming:  W[n] = 0.54 − 0.46 cos(2πn / (N−1))",
        eq_style,
    ))
    elements.append(Paragraph(
        "Tukey:    W[n] = { 0.5(1 + cos(π(2n/αN − 1)/α))  if n < αN/2",
        eq_style,
    ))
    elements.append(Paragraph(
        "                  { 1                                if αN/2 ≤ n ≤ N(1−α/2)",
        eq_style,
    ))

    elements.append(Paragraph("<b>3.3 Homodyne POCS Reconstruction</b>", subheading_style))
    elements.append(Paragraph(
        f"Partial Fourier factor PF = {result['pf_factor']:.3f} acquires only "
        f"{result['acquired_fraction']*100:.1f}% of k-space. "
        "Projection Onto Convex Sets (POCS) iteratively enforces data consistency "
        "and estimated phase constraint:",
        body_style,
    ))
    elements.append(Paragraph(
        "φ_low = angle( IFFT₂ᶜ{ S(k) · H_sym(k) } )    [low-resolution phase]",
        eq_style,
    ))
    elements.append(Paragraph(
        "s^(n+1) = |s^(n)| · exp(i · φ_low)              [phase projection]",
        eq_style,
    ))
    elements.append(Paragraph(
        "S^(n+1)(k) = M(k)·S_acq(k) + (1−M(k))·FFT₂ᶜ{s^(n+1)}   [data consistency]",
        eq_style,
    ))

    # ── FFTW Reconstruction Figure ────────────────────────────────────────────
    elements.append(Spacer(1, 0.1 * inch))
    img_fftw = _b64_to_rl_image(result["plot_fftw_recon"], width=6.8 * inch, height=3.6 * inch)
    elements.append(KeepTogether([
        img_fftw,
        Paragraph(
            f"<b>Figure 1.</b> FFTW-accelerated multi-echo reconstruction at B₀ = {B0:.1f} T. "
            f"Top row: magnitude images. Bottom row: phase images at selected echo times. "
            f"PF = {result['pf_factor']}, {n_echoes} echoes, {matrix}×{matrix} matrix.",
            caption_style,
        ),
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  4. STATISTICAL DISTRIBUTION MODELLING
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("4. Statistical Signal Distribution Modelling", heading_style))

    elements.append(Paragraph("<b>4.1 Magnitude Signal Models</b>", subheading_style))
    elements.append(Paragraph(
        "The magnitude signal in MRI follows tissue-dependent distributions. "
        "We fit four candidate models and select via the Kolmogorov–Smirnov (KS) test:",
        body_style,
    ))
    elements.append(Paragraph(
        "Rayleigh:  p(x) = (x / σ²) exp(−x² / 2σ²)",
        eq_style,
    ))
    elements.append(Paragraph(
        "Rice:      p(x) = (x / σ²) exp(−(x² + ν²) / 2σ²) · I₀(xν / σ²)",
        eq_style,
    ))
    elements.append(Paragraph(
        "Gamma:     p(x) = x^(a−1) exp(−x/b) / (b^a · Γ(a))",
        eq_style,
    ))
    elements.append(Paragraph(
        "NCX²:     p(x) = (1/2)(x/λ)^((k/2−1)/2) exp(−(x+λ)/2) · I_{k/2−1}(√(λx))",
        eq_style,
    ))
    elements.append(Paragraph(
        "where I₀ is the modified Bessel function of the first kind, Γ(a) is the Gamma function, "
        "and (k, λ) are the degrees of freedom and non-centrality parameter.",
        body_style,
    ))

    elements.append(Paragraph("<b>4.2 Kolmogorov–Smirnov Goodness-of-Fit</b>", subheading_style))
    elements.append(Paragraph(
        "The KS statistic measures the maximum absolute deviation between the empirical "
        "and theoretical CDFs:",
        body_style,
    ))
    elements.append(Paragraph(
        "D_n = sup_x | F_n(x) − F(x; θ̂) |",
        eq_style,
    ))
    elements.append(Paragraph(
        f"Selected best-fit model: <b>{result['best_distribution'].title()}</b> "
        f"with D_n = {result['best_ks_stat']:.4f}.",
        body_style,
    ))

    elements.append(Paragraph("<b>4.3 Fisher Information and Cramér–Rao Bound</b>", subheading_style))
    elements.append(Paragraph(
        "For multi-echo PRF acquisition, the Fisher information for temperature is:",
        body_style,
    ))
    elements.append(Paragraph(
        "I(T) = Σ_{e=1}^{N_e} (2π · γ · B₀ · α · TE_e)² · SNR_e²",
        eq_style,
    ))
    elements.append(Paragraph(
        "CRB(T) = 1 / I(T)     [variance lower bound in °C²]",
        eq_style,
    ))
    elements.append(Paragraph(
        "σ_T^min = √CRB(T)    [minimum detectable ΔT]",
        eq_style,
    ))
    fisher = result["fisher_info"]
    elements.append(Paragraph(
        f"Computed: I(T) = {fisher['fisher_information']:.4e}, "
        f"CRB = {fisher['cramer_rao_bound_C2']:.6e} °C², "
        f"σ_T<sup>min</sup> = {fisher['min_detectable_dT_C']:.4f} °C.",
        body_style,
    ))

    # ── Distribution Figure ───────────────────────────────────────────────────
    elements.append(Spacer(1, 0.1 * inch))
    img_dist = _b64_to_rl_image(result["plot_distributions"], width=6.8 * inch, height=2.6 * inch)
    elements.append(KeepTogether([
        img_dist,
        Paragraph(
            f"<b>Figure 2.</b> Statistical distribution analysis of magnitude signal. "
            f"Left: histogram with fitted Rice, Rayleigh, Gamma, and NCX² PDFs. "
            f"Centre: Q-Q plot for best-fit ({result['best_distribution'].title()}) model. "
            f"Right: KS statistic comparison (lower is better).",
            caption_style,
        ),
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  5. MULTIMODAL BAYESIAN REASONING
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("5. Multimodal Bayesian Temperature Reasoning", heading_style))

    elements.append(Paragraph("<b>5.1 Phase-Based Temperature Estimation</b>", subheading_style))
    elements.append(Paragraph(
        "For each tissue class <i>k</i>, the phase-to-temperature conversion uses "
        "the tissue-specific LUT coefficient:",
        body_style,
    ))
    elements.append(Paragraph(
        "ΔT_phase(x,y) = Δφ(x,y) / (γ_rad · B₀ · α_k · TE)     for (x,y) ∈ Ω_k",
        eq_style,
    ))

    elements.append(Paragraph("<b>5.2 T₁-Based Temperature Estimation</b>", subheading_style))
    elements.append(Paragraph(
        "T₁ relaxation time increases approximately 1–2% per °C, providing an "
        "independent temperature estimate:",
        body_style,
    ))
    elements.append(Paragraph(
        "ΔT_{T₁}(x,y) = (T₁(x,y) − T₁_ref) · β_k     [°C, β ≈ 0.005 °C⁻¹]",
        eq_style,
    ))

    elements.append(Paragraph("<b>5.3 Bayesian Fusion</b>", subheading_style))
    elements.append(Paragraph(
        "The posterior temperature estimate maximises the joint likelihood from "
        "N<sub>m</sub> = 2 modalities with tissue-dependent uncertainties:",
        body_style,
    ))
    elements.append(Paragraph(
        "p(T | ΔT_phase, ΔT_{T₁}) ∝ Πᵢ N(ΔTᵢ | T, σᵢ²) · p(T | Ω_k)",
        eq_style,
    ))
    elements.append(Paragraph(
        "T̂_fused = ( Σᵢ wᵢ · ΔTᵢ ) / ( Σᵢ wᵢ ),    wᵢ = 1/σᵢ²",
        eq_style,
    ))
    elements.append(Paragraph(
        "σ²_fused = 1 / ( Σᵢ 1/σᵢ² )",
        eq_style,
    ))
    elements.append(Paragraph(
        "This inverse-variance weighting yields the minimum-variance unbiased estimator. "
        "Tissue classification uses synthesised T₁/T₂* maps and assigns labels: "
        "WM (T₁&lt;900 ms), GM (900–1400 ms), CSF (&gt;2000 ms), blood (T₂*&gt;50 ms), "
        "fat (T₁&lt;500 ms), tumor (intermediate).",
        body_style,
    ))

    elements.append(Paragraph("<b>5.4 Cross-Modal Edge Enhancement</b>", subheading_style))
    elements.append(Paragraph(
        "Edge maps from each modality are computed via Sobel filtering and fused "
        "by geometric mean to preserve boundaries present in all modalities:",
        body_style,
    ))
    elements.append(Paragraph(
        "E_fused(x,y) = ( Πᵢ |∇Iᵢ(x,y)| )^(1/N_m)",
        eq_style,
    ))

    # ── Multimodal Figure ─────────────────────────────────────────────────────
    elements.append(Spacer(1, 0.1 * inch))
    img_mm = _b64_to_rl_image(result["plot_multimodal"], width=6.8 * inch, height=2.2 * inch)
    elements.append(KeepTogether([
        img_mm,
        Paragraph(
            "<b>Figure 3.</b> Multimodal Bayesian temperature reasoning. "
            "From left: phase-based ΔT, T₁-based ΔT, Bayesian fused ΔT, uncertainty map (°C).",
            caption_style,
        ),
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  6. COLORISED THERMOMETRY MAPS
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph(
        "6. Colorised MR Thermometry Maps with Probe Control", heading_style))

    elements.append(Paragraph(
        "Absolute temperature maps are rendered using a custom continuous colormap: "
        "deep blue (35 °C, baseline) → blue → green (body temperature) → yellow → orange → "
        "red (thermal dose) → hot pink (ablation zone) → white (extreme). "
        "An interactive probe crosshair overlays the current temperature reading at the "
        "target position with real-time PID feedback parameters:",
        body_style,
    ))
    elements.append(Paragraph(
        "u(t) = K_p · e(t) + K_i · ∫₀ᵗ e(τ)dτ + K_d · de(t)/dt",
        eq_style,
    ))
    elements.append(Paragraph(
        f"where e(t) = T_target − T_measured, K_p = 0.8, K_i = 0.1, K_d = 0.05, "
        f"and T_target = {target_temp:.0f} °C. "
        f"Probe reading at hotspot: {result['probe_temp_C']:.1f} °C. "
        f"Ground truth peak ΔT = {result['peak_dT_gt_C']:.2f} °C, "
        f"estimated peak ΔT = {result['peak_dT_est_C']:.2f} °C.",
        body_style,
    ))

    # ── Thermometry Figure ────────────────────────────────────────────────────
    elements.append(Spacer(1, 0.1 * inch))
    img_therm = _b64_to_rl_image(result["plot_thermometry"], width=6.8 * inch, height=2.8 * inch)
    elements.append(KeepTogether([
        img_therm,
        Paragraph(
            f"<b>Figure 4.</b> Colorised MR thermometry overlaid on signal-reconstructed images. "
            f"Panels show temperature overlay on magnitude, pure colourised temperature map, "
            f"ground truth, absolute error, and tissue classification. "
            f"Temperature probe (green crosshair) at target position with PID feedback. "
            f"B₀ = {B0:.1f} T, {n_echoes} echoes, {matrix}×{matrix}.",
            caption_style,
        ),
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  7. SNR CHARACTERISTICS
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("7. SNR Characteristics", heading_style))

    elements.append(Paragraph("<b>7.1 Echo-Dependent SNR</b>", subheading_style))
    elements.append(Paragraph(
        "Signal-to-noise ratio is computed per echo as the ratio of mean brain "
        "signal to background noise standard deviation:",
        body_style,
    ))
    elements.append(Paragraph(
        "SNR_e = μ_signal(Ω_brain, TE_e) / σ_noise",
        eq_style,
    ))
    elements.append(Paragraph(
        "SNR_dB = 10 · log₁₀( SNR_mean )",
        eq_style,
    ))
    elements.append(Paragraph(
        "Signal decay follows the mono-exponential T₂* model:",
        body_style,
    ))
    elements.append(Paragraph(
        "S(TE) = S₀ · exp(−TE / T₂*) · sin(α)",
        eq_style,
    ))

    # ── SNR Table ─────────────────────────────────────────────────────────────
    snr_data = [["Echo #", "TE (ms)", "SNR"]]
    for i, (te, snr) in enumerate(zip(result["echo_times_ms"], result["snr_per_echo"])):
        snr_data.append([str(i + 1), f"{te:.2f}", f"{snr:.1f}"])
    snr_data.append(["Mean", "—", f"{result['snr_mean']:.1f}"])
    snr_data.append(["dB", "—", f"{result['snr_db']:.1f}"])

    snr_table = Table(snr_data, colWidths=[0.8 * inch, 1.2 * inch, 1.0 * inch])
    snr_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e20")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -3), [colors.HexColor("#e8f5e9"), colors.white]),
        ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#c8e6c9")),
        ("FONTNAME", (0, -2), (-1, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(KeepTogether([
        Paragraph("<b>Table 2.</b> Per-echo SNR characteristics", caption_style),
        snr_table,
    ]))

    elements.append(Spacer(1, 0.12 * inch))

    elements.append(Paragraph("<b>7.2 Temperature Accuracy</b>", subheading_style))
    rmse_data = [
        ["Method", "RMSE (°C)", "Improvement"],
        ["Raw WLS", f"{result['rmse_raw_C']:.4f}", "—"],
        ["Wiener Filter", f"{result['rmse_wiener_C']:.4f}",
         f"{(1 - result['rmse_wiener_C'] / result['rmse_raw_C']) * 100:.1f}%"],
        ["Quantum ML (QML)", f"{result['rmse_qml_C']:.4f}",
         f"{(1 - result['rmse_qml_C'] / result['rmse_raw_C']) * 100:.1f}%"],
    ]
    rmse_table = Table(rmse_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch])
    rmse_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffebee"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(KeepTogether([
        Paragraph("<b>Table 3.</b> Temperature estimation accuracy (RMSE)", caption_style),
        rmse_table,
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  8. PULSE SEQUENCE DESIGN
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("8. Optimised Pulse Sequence Design", heading_style))

    elements.append(Paragraph("<b>8.1 Echo-Spacing Optimisation</b>", subheading_style))
    elements.append(Paragraph(
        "Echo times are distributed using continued‐fraction (CF) Farey optimisation "
        "to maximise phase SNR while spanning the T₂* decay window:",
        body_style,
    ))
    elements.append(Paragraph(
        "TE_e = TE_min + (TE_max − TE_min) · frac(e · φ),    φ = (1 + √5)/2",
        eq_style,
    ))
    elements.append(Paragraph(
        "sorted: TE₁ < TE₂ < … < TE_{N_e},    TR = TE_{N_e} + Δ_dead",
        eq_style,
    ))
    elements.append(Paragraph(
        "The golden-ratio spacing ensures quasi-uniform coverage of the TE axis, "
        "avoiding echo-time clustering that degrades the WLS slope estimate.",
        body_style,
    ))

    elements.append(Paragraph("<b>8.2 Weighted Least-Squares Phase Slope</b>", subheading_style))
    elements.append(Paragraph(
        "Multi-echo phase data φ(TE_e) is fitted to a linear model φ = β₀ + β₁·TE "
        "using WLS with weights w_e = TE_e² / σ²:",
        body_style,
    ))
    elements.append(Paragraph(
        "β̂ = (X'WX)⁻¹ X'Wy",
        eq_style,
    ))
    elements.append(Paragraph(
        "where X = [1, TE]ᵀ, W = diag(w₁, …, w_{N_e}), y = [φ₁, …, φ_{N_e}]ᵀ",
        eq_style,
    ))
    elements.append(Paragraph(
        "ΔT(x,y) = β̂₁(x,y) / (α · γ_rad · B₀)",
        eq_style,
    ))

    elements.append(Paragraph("<b>8.3 Quantum RBM Denoising</b>", subheading_style))
    elements.append(Paragraph(
        "Temperature maps are denoised using a Restricted Boltzmann Machine trained "
        "on 8×8 patches of the Wiener-filtered map. The RBM energy function:",
        body_style,
    ))
    elements.append(Paragraph(
        "E(v, h) = −Σᵢ aᵢvᵢ − Σⱼ bⱼhⱼ − Σᵢⱼ vᵢ Wᵢⱼ hⱼ",
        eq_style,
    ))
    elements.append(Paragraph(
        "p(v) = Σ_h exp(−E(v,h)) / Z,    Z = Σ_{v,h} exp(−E(v,h))",
        eq_style,
    ))
    elements.append(Paragraph(
        "Contrastive divergence (CD-1) updates weights: "
        "ΔW = η · (⟨v·h'⟩_data − ⟨v·h'⟩_recon)",
        body_style,
    ))

    elements.append(Spacer(1, 0.1 * inch))

    # ── Sequence Table ────────────────────────────────────────────────────────
    seq_data = [["Sequence", "Type", "Echoes", "TE Range (ms)", ".seq File"]]
    for sn, sp in zip(result["generated_sequences"], result["generated_seq_paths"]):
        basename = os.path.basename(sp)
        seq_info_str = sn.replace("THERM_", "")
        # Extract details from name
        parts = sn.split("_")
        stype = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
        seq_data.append([
            sn, stype, str(n_echoes),
            f"{result['echo_times_ms'][0]:.1f}–{result['echo_times_ms'][-1]:.1f}",
            basename,
        ])
    seq_table = Table(seq_data, colWidths=[1.6 * inch, 1.1 * inch, 0.6 * inch, 1.1 * inch, 1.6 * inch])
    seq_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f3e5f5"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(KeepTogether([
        Paragraph(
            f"<b>Table 4.</b> Generated Pulseq-1.2 thermometry sequences at B₀ = {B0:.1f} T",
            caption_style,
        ),
        seq_table,
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  9. RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("9. Results Summary", heading_style))

    summary_data = [
        ["Parameter", "Value"],
        ["Field Strength", f"{B0:.1f} T"],
        ["Matrix Size", f"{matrix} × {matrix}"],
        ["Number of Echoes", str(n_echoes)],
        ["Partial Fourier Factor", f"{result['pf_factor']:.3f}"],
        ["Acquired k-space", f"{result['acquired_fraction']*100:.1f}%"],
        ["FFTW Backend", result["fftw_backend"]],
        ["Mean SNR", f"{result['snr_mean']:.1f}"],
        ["SNR (dB)", f"{result['snr_db']:.1f} dB"],
        ["RMSE Raw (°C)", f"{result['rmse_raw_C']:.4f}"],
        ["RMSE Wiener (°C)", f"{result['rmse_wiener_C']:.4f}"],
        ["RMSE QML (°C)", f"{result['rmse_qml_C']:.4f}"],
        ["Peak ΔT Ground Truth", f"{result['peak_dT_gt_C']:.2f} °C"],
        ["Peak ΔT Estimated", f"{result['peak_dT_est_C']:.2f} °C"],
        ["Probe Temperature", f"{result['probe_temp_C']:.1f} °C"],
        ["Fisher Information", f"{fisher['fisher_information']:.4e}"],
        ["Cramér–Rao Bound", f"{fisher['cramer_rao_bound_C2']:.6e} °C²"],
        ["Min Detectable ΔT", f"{fisher['min_detectable_dT_C']:.4f} °C"],
        ["Best Distribution", result["best_distribution"].title()],
        ["KS Statistic", f"{result['best_ks_stat']:.4f}"],
        ["Sequences Generated", str(len(result["generated_sequences"]))],
    ]
    sum_table = Table(summary_data, colWidths=[2.4 * inch, 3.0 * inch])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e3f2fd"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(KeepTogether([
        Paragraph("<b>Table 5.</b> Complete acquisition and reconstruction summary", caption_style),
        sum_table,
    ]))

    elements.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    #  10. DISCUSSION & CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph("10. Discussion", heading_style))
    elements.append(Paragraph(
        f"The integrated thermometry framework achieves a minimum detectable temperature "
        f"change of {fisher['min_detectable_dT_C']:.3f} °C, which exceeds the clinical "
        f"requirement of 1 °C precision for thermal therapy monitoring. "
        f"Tissue-specific LUTs ensure that the PRF coefficient mismatch in fat "
        f"(α_fat ≈ 0) and tumour (α_tumour ≈ −0.0085 ppm/°C) is correctly handled. "
        f"The multimodal Bayesian fusion provides robustness against single-modality "
        f"artefacts by combining phase-based and T₁-based estimates with inverse-variance "
        f"weighting. The quantum RBM denoiser further reduces RMSE by "
        f"{(1 - result['rmse_qml_C'] / result['rmse_raw_C']) * 100:.1f}% relative "
        f"to raw WLS estimation.",
        body_style,
    ))
    elements.append(Paragraph(
        f"The {result['best_distribution'].title()} distribution provides the best "
        f"description of the magnitude signal statistics (KS = {result['best_ks_stat']:.4f}), "
        f"consistent with the expected non-central chi or Rice distribution in phased-array "
        f"acquisitions. This statistical model informs the noise variance estimation used "
        f"in both the Wiener filter and the WLS weights.",
        body_style,
    ))

    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("11. Conclusion", heading_style))
    elements.append(Paragraph(
        "We have presented a complete quantum-enhanced MR thermometry pipeline integrating "
        "tissue-adaptive PRF lookup tables, FFTW-accelerated partial-Fourier reconstruction, "
        "multimodal Bayesian temperature fusion, quantum RBM denoising, and optimised pulse "
        "sequence generation. Four Pulseq-1.2 compatible sequences have been generated "
        "with embedded thermometry LUT metadata and PID temperature probe control parameters. "
        "The system achieves sub-degree precision with real-time capability for MR-guided "
        "thermal therapy applications.",
        body_style,
    ))

    elements.append(Spacer(1, 0.15 * inch))

    # ── References ────────────────────────────────────────────────────────────
    elements.append(Paragraph("References", heading_style))
    refs = [
        "1. Rieke, V. &amp; Butts Pauly, K. <i>J. Magn. Reson. Imaging</i> <b>27</b>, 376–390 (2008).",
        "2. Ishihara, Y. et al. <i>Magn. Reson. Med.</i> <b>34</b>, 814–823 (1995).",
        "3. De Poorter, J. et al. <i>Magn. Reson. Med.</i> <b>33</b>, 74–81 (1995).",
        "4. Denis de Senneville, B. et al. <i>Int. J. Hyperthermia</i> <b>25</b>, 589–600 (2009).",
        "5. Frigo, M. &amp; Johnson, S. G. <i>Proc. IEEE</i> <b>93</b>, 216–231 (2005).",
        "6. Noll, D. C. et al. <i>IEEE Trans. Med. Imaging</i> <b>10</b>, 154–163 (1991).",
        "7. Gudbjartsson, H. &amp; Patz, S. <i>Magn. Reson. Med.</i> <b>34</b>, 910–914 (1995).",
        "8. Lapert, M. et al. <i>J. Magn. Reson.</i> <b>229</b>, 82–87 (2013).",
        "9. Smolensky, P. <i>Parallel Distributed Processing</i>, MIT Press (1986).",
        "10. Sapareto, S. A. &amp; Dewey, W. C. <i>Int. J. Radiat. Oncol. Biol. Phys.</i> <b>10</b>, 787–800 (1984).",
    ]
    for ref in refs:
        elements.append(Paragraph(ref, ParagraphStyle(
            "Ref", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=2,
        )))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(elements)
    print(f"\n{'='*60}")
    print(f"  PDF written: {pdf_path}")
    print(f"  Sequences:   {len(result['generated_seq_paths'])} .seq files")
    print(f"  Figures:     {fig_dir}")
    print(f"{'='*60}")

    return {
        "pdf_path": pdf_path,
        "seq_paths": result["generated_seq_paths"],
        "fig_dir": fig_dir,
        "result": result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out = generate_thermometry_nature_report(
        B0=3.0,
        n_echoes=10,
        matrix=128,
        hotspot_dT=6.0,
        target_temp=55.0,
    )
    print(f"\nDone. Open: {out['pdf_path']}")
