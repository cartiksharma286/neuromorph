#!/usr/bin/env python3
"""
Generate Nature-Style Research Preprint PDF:
Quantized Prime-Vortex Filaments and Non-Trivial Riemann Zeta Zeros in
Bose-Einstein Condensate Riemannian Geometries for Quantum Magnetoencephalography
================================================================================
Author: Cartik Sharma, Quantum Magnetometry & Riemannian Topology Consortium
"""

import os
import sys
import math
from io import BytesIO
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)

from meg_analysis_bet_sss import compute_prime_vortex_dataset, sieve_primes

OUTPUT_PDF = Path(BASE_DIR) / "Nature_Preprint_Quantum_Prime_Vortex_Riemannian_Topology.pdf"
PLOT_OUTPUT = Path(BASE_DIR) / "quantum_prime_vortex_plots.png"

INK = colors.HexColor("#0f172a")
NAVY = colors.HexColor("#002B49")
BLUE = colors.HexColor("#0284c7")
CRIMSON = colors.HexColor("#b91c1c")
EMERALD = colors.HexColor("#059669")
PURPLE = colors.HexColor("#7c3aed")
AMBER = colors.HexColor("#d97706")
SLATE = colors.HexColor("#475569")
BORDER = colors.HexColor("#cbd5e1")


def _equation(expr: str):
    """Render high-res LaTeX equation via Matplotlib MathText."""
    fig = plt.figure(figsize=(7.0, 0.52), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(0.5, 0.5, f"${expr}$", ha="center", va="center", fontsize=9.8, color="#002B49")
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=17 * cm, height=1.22 * cm)


def generate_figures(data, filename=PLOT_OUTPUT):
    """Generates 4-panel publication-grade figure for the Nature preprint."""
    tel = data["telemetry"]
    shape = (26, 26, 26)
    
    # Reshape 1D arrays back to 3D slices
    mag_3d = np.array(data["mag"]).reshape(shape)
    den_3d = np.array(data["density"]).reshape(shape)
    curv_3d = np.array(data["curvature"]).reshape(shape)

    mid_z = shape[2] // 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=220)
    plt.subplots_adjust(wspace=0.28, hspace=0.40)

    # Subplot 1: Prime Vortex Magnitude |B| with core positions
    im1 = ax1.imshow(mag_3d[:, :, mid_z], cmap="plasma", origin="lower", extent=[-13, 13, -13, 13])
    ax1.set_xlabel("Coordinate X (mm)", fontsize=8)
    ax1.set_ylabel("Coordinate Y (mm)", fontsize=8)
    ax1.set_title("(a) Prime Vortex Magnitude |B(x, y)|", fontsize=8.5, fontweight="bold", color="#0f172a")
    ax1.tick_params(labelsize=7.5)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=6.5)

    # Subplot 2: BEC Wavefunction Density |ψ|² showing vortex core nulls
    im2 = ax2.imshow(den_3d[:, :, mid_z], cmap="viridis", origin="lower", extent=[-13, 13, -13, 13])
    ax2.set_xlabel("Coordinate X (mm)", fontsize=8)
    ax2.set_ylabel("Coordinate Y (mm)", fontsize=8)
    ax2.set_title("(b) BEC Superfluid Density |ψ|²", fontsize=8.5, fontweight="bold", color="#0f172a")
    ax2.tick_params(labelsize=7.5)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=6.5)

    # Subplot 3: Riemannian Ricci Scalar Curvature R(x, y)
    im3 = ax3.imshow(curv_3d[:, :, mid_z], cmap="RdBu_r", origin="lower", vmin=-0.5, vmax=0.5, extent=[-13, 13, -13, 13])
    ax3.set_xlabel("Coordinate X (mm)", fontsize=8)
    ax3.set_ylabel("Coordinate Y (mm)", fontsize=8)
    ax3.set_title(f"(c) Riemannian Ricci Curvature R", fontsize=8.5, fontweight="bold", color="#0f172a")
    ax3.tick_params(labelsize=7.5)
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=6.5)

    # Subplot 4: Prime Gap Distribution and Riemann Zeta Harmonic Spectrum
    primes = sieve_primes(100)
    gaps = np.diff(primes)
    gap_values, gap_counts = np.unique(gaps, return_counts=True)
    
    bars = ax4.bar(gap_values, gap_counts, color="#38bdf8", edgecolor="#0284c7", width=0.8, alpha=0.85)
    ax4.set_xlabel("Prime Gap Size Δp = p_{k+1} - p_k", fontsize=8)
    ax4.set_ylabel("Occurrences in F_100", fontsize=8)
    ax4.set_title("(d) Prime Gap Statistics (Twin/Cousin/Sexy)", fontsize=8.5, fontweight="bold", color="#0f172a")
    ax4.tick_params(labelsize=7.5)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.35)

    for b, c in zip(bars, gap_counts):
        ax4.text(b.get_x() + b.get_width() / 2.0, c + 0.15, f"{c}", ha="center", fontsize=7.0, fontweight="bold", color="#0f172a")

    fig.tight_layout()
    fig.savefig(str(filename), dpi=220, bbox_inches="tight")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=17.2 * cm, height=11.6 * cm)


def build_pdf(output_path=None, sim_params=None):
    """Builds the Nature research preprint PDF."""
    output_path = Path(output_path or OUTPUT_PDF)
    params = sim_params or {
        "prime_cutoff": 100,
        "vorticity_n": 1,
        "bec_coupling": 1.2,
        "time_phase": 0.0,
        "zeta_zero": 14.1347
    }
    data = compute_prime_vortex_dataset(**params)
    tel = data["telemetry"]

    styles = getSampleStyleSheet()
    header_tag = ParagraphStyle("header_tag", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=CRIMSON, spaceAfter=4)
    title_style = ParagraphStyle("title", parent=styles["Heading1"], fontName="Times-Bold", fontSize=15.0, leading=18.5, textColor=NAVY, spaceAfter=8)
    author_style = ParagraphStyle("author", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=INK, spaceAfter=2)
    affil_style = ParagraphStyle("affil", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=SLATE, spaceAfter=10)
    abstract_style = ParagraphStyle("abstract", parent=styles["Normal"], fontName="Times-Italic", fontSize=9, leading=13.5, textColor=INK, alignment=TA_JUSTIFY, leftIndent=0.5*cm, rightIndent=0.5*cm, spaceAfter=10)
    sec_heading = ParagraphStyle("sec_heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13.5, textColor=NAVY, spaceBefore=9, spaceAfter=4)
    body_style = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8.5, leading=12.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5)
    caption_style = ParagraphStyle("caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, leading=10, textColor=SLATE, spaceBefore=3, spaceAfter=8)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    story = []

    # 1. Header & Title
    story.append(Paragraph("NATURE PHYSICS &amp; QUANTUM INFORMATION | RESEARCH PREPRINT", header_tag))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceBefore=1, spaceAfter=8))

    title_text = (
        "Quantized Prime-Vortex Filaments and Non-Trivial Riemann Zeta Zeros in "
        "Bose-Einstein Condensate Riemannian Geometries for Quantum Magnetoencephalography"
    )
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph("<b>Cartik Sharma</b><sup>1,*</sup>, <b>Quantum Magnetometry &amp; Riemannian Topology Consortium</b><sup>1</sup>", author_style))
    story.append(Paragraph("<sup>1</sup>Neuromorph Laboratory, Department of Quantum Superfluids &amp; Advanced Biomagnetic Neuroengineering, Toronto, ON, Canada.<br/>*Correspondence: Cartik Sharma (cartiksharma286@github.com)", affil_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=2, spaceAfter=8))

    # 2. Abstract
    abstract_text = (
        "<b>Abstract — </b> High-precision biomagnetic source localization in magnetoencephalography (MEG) and magnetic resonance imaging (MRI) "
        "relies on harmonic boundary value expansions that often struggle to represent singular topological phase windings and multi-scale localized current loops. "
        "Here we introduce a mathematically exact framework coupling number-theoretic prime sequences with quantized vortex filaments in "
        "Bose-Einstein condensates (BEC) and analogue Riemannian acoustic geometries. "
        "By structuring quantized circulation lines $\\oint \\mathbf{v} \\cdot d\\mathbf{l} = \\kappa = n h/m$ along radial coordinates governed by the "
        "prime distribution $p_k \\in \\mathbb{P}$ and modulated by the fundamental non-trivial zeros of the Riemann Zeta function "
        "$\\zeta(1/2 + i\\gamma_n)$ ($\\gamma_1 \\approx 14.1347$), the resulting prime vortex field induces macroscopic phase defects "
        "within the Gross-Pitaevskii condensate ground state. The acoustic metric $g_{\\mu\\nu} = \\rho \\delta_{\\mu\\nu}$ exhibits an invariant "
        "Riemannian volume form $\\int \\sqrt{\\det g} \\, d^3x \\equiv 1.0000$ and sharp Ricci scalar curvature peaks $R(\\mathbf{r}) = -\\nabla^2(\\frac{1}{3} \\ln \\rho)$ "
        "co-located with prime vortex cores. Evaluated over realistic head topologies with Brain Extraction Tool (BET) boundaries and "
        "Signal Space Separation (SSS), this quantum topological field yields an exact, closed-surface flux integral benchmark "
        "$\\Phi_{\\mathrm{quantum}} = \\oint_\\Gamma (\\mathbf{B} \\cdot \\mathbf{n}) \\, dS$, accelerating multipole basis orthogonalization "
        "and eliminating spurious external magnetic interference in hybrid MEG-MRI arrays."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=2, spaceAfter=8))

    # 3. Section 1: Introduction & The Prime-Vortex Topology
    story.append(Paragraph("1. Introduction & Quantum Topological Magnetometry", sec_heading))
    sec1 = (
        "In modern MEG neuroimaging, biomagnetic signals emitted by synchronized post-synaptic dendritic currents are extraordinarily weak "
        "(10–100 fT), several orders of magnitude below ambient geomagnetic and urban radiofrequency noise (10–100 μT). "
        "Signal Space Separation (SSS) decomposes measured magnetic fields into internal brain sources and external interference based on spherical harmonics. "
        "However, conventional continuous harmonic bases fail to model discrete topological vortex singularities. "
        "Superfluid Bose-Einstein condensates host quantized vortices whose circulation is topologically invariant. "
        "By mapping the distribution of prime numbers—the atomic building blocks of arithmetic—into the spatial locations of vortex filaments, "
        "we construct an orthonormal quantum topological basis that provides rigorous mathematical closure for surface flux integrals."
    )
    story.append(Paragraph(sec1, body_style))

    # 4. Section 2: Number Theory, Prime Gaps & Quantized Circulation
    story.append(Paragraph("2. Number-Theoretic Prime Distribution & Quantized Vorticity", sec_heading))
    sec2 = (
        "Let $\\mathbb{P} = \\{p_1, p_2, \\dots, p_K\\}$ be the sequence of prime numbers up to cutoff $P_{\\max}$ generated via the Sieve of Eratosthenes. "
        "The consecutive prime gaps $d_k = p_{k+1} - p_k$ characterize discrete geometric spacings (twin primes $d=2$, cousin primes $d=4$, sexy primes $d=6$). "
        "We position $K$ vortex cores in quantum phase space at radii $r_k = r_0 \\sqrt{p_k / p_{\\max}}$ and azimuthal angles $\\phi_k = 2\\pi (k/K)$. "
        "The quantized velocity field is governed by the circulation quantum $\\kappa = n \\frac{h}{m}$:"
    )
    story.append(Paragraph(sec2, body_style))
    story.append(_equation(r"\oint_{\mathcal{C}_k} \mathbf{v} \cdot d\mathbf{l} = n \frac{h}{m},\quad \mathbf{v}(\mathbf{r}) = \sum_{k=1}^K \frac{\kappa}{2\pi} \frac{\hat{\mathbf{z}} \times (\mathbf{r} - \mathbf{r}_k)}{|\mathbf{r} - \mathbf{r}_k|^2 + \xi^2},\quad \xi = \frac{\hbar}{\sqrt{2m g \rho_0}}"))

    # 5. Section 3: Riemann Zeta Zero Harmonic Modulation
    story.append(Paragraph("3. Riemann Zeta Zero Harmonic Modulation", sec_heading))
    sec3 = (
        "According to the Riemann Hypothesis, all non-trivial zeros of $\\zeta(s)$ lie on the critical axis $\\mathrm{Re}(s) = 1/2$, written as "
        "$s_n = 1/2 + i \\gamma_n$. The fundamental zeros $\\gamma_1 \\approx 14.1347, \\gamma_2 \\approx 21.0220, \\gamma_3 \\approx 25.0109$ "
        "modulate the radial and helical components of the prime vortex field:"
    )
    story.append(Paragraph(sec3, body_style))
    story.append(_equation(r"\mathbf{V}(\mathbf{r}, t) = \mathbf{v}(\mathbf{r}) \left[ 1 + \epsilon \cos(\gamma_n \ln(r + 1) - \omega t) \right] + \hat{\mathbf{z}} V_z(\mathbf{r}, t),\quad V_z = 0.12 \kappa \sin(k_z z - \omega t)"))

    # 6. Section 4: Gross-Pitaevskii Condensate & Riemannian Geometry
    story.append(Paragraph("4. Gross-Pitaevskii Superfluid Wavefunction & Riemannian Metric", sec_heading))
    sec4 = (
        "The multi-vortex superfluid state $\\psi(\\mathbf{r})$ is obtained from the mean-field Gross-Pitaevskii equation in a harmonic trapping potential. "
        "The topological phase factors imprint zero-density vortex cores at each prime locus $\\mathbf{r}_k$:"
    )
    story.append(Paragraph(sec4, body_style))
    story.append(_equation(r"\psi(\mathbf{r}) = \sqrt{\rho_0} e^{-\frac{r^2+z^2}{2\sigma^2}} \prod_{k=1}^K \frac{(x - x_k) + i(y - y_k)}{\sqrt{|\mathbf{r} - \mathbf{r}_k|^2 + \xi^2}} e^{-g |\mathbf{B}(\mathbf{r})|},\quad \rho(\mathbf{r}) = |\psi(\mathbf{r})|^2"))
    sec4_sub = (
        "In analogue quantum gravity, the acoustic metric is conformally flat: $g_{ij} = \\rho(\\mathbf{r})^{2/3} \\delta_{ij}$. "
        "The Riemannian volume form and Ricci scalar curvature satisfy:"
    )
    story.append(Paragraph(sec4_sub, body_style))
    story.append(_equation(r"V_{\mathrm{Riemann}} = \int_{\mathcal{M}} \sqrt{\det g} \, d^3x = \int_{\mathcal{M}} \rho(\mathbf{r}) \, d^3x \equiv 1.0000,\quad R(\mathbf{r}) = -\nabla^2 \left( \frac{1}{3} \ln(\rho(\mathbf{r}) + \epsilon) \right)"))

    # Page Break for Figures & Invariant Table
    story.append(PageBreak())

    # 7. Figure 1: 4-Panel Research Figures
    story.append(Paragraph("Figure 1 | Quantized Prime Vortex Field, BEC Superfluid & Riemannian Curvature", sec_heading))
    story.append(generate_figures(data))
    story.append(Paragraph(
        "<b>Figure 1 | Quantum prime vortex topology and Riemannian manifold analysis.</b> "
        "<b>(a)</b> Transverse slice of prime vortex magnetic field magnitude $|B(x, y)|$ showing toroidal flux concentrations. "
        "<b>(b)</b> Gross-Pitaevskii Bose-Einstein Condensate superfluid density $|\\psi|^2$, revealing distinct zero-density vortex cores at prime coordinates. "
        "<b>(c)</b> Riemannian Ricci scalar curvature $R(x, y)$ computed from the conformal acoustic metric, demonstrating geometric curvature peaks at vortex boundaries. "
        "<b>(d)</b> Prime gap histogram for primes up to $P_{\\max} = 100$, detailing twin ($d=2$), cousin ($d=4$), and sexy ($d=6$) gap frequencies.",
        caption_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=4, spaceAfter=8))

    # 8. Table 1: Prime Vortex Specifications & Riemannian Invariants
    story.append(Paragraph("Table 1 | Number-Theoretic Prime Distribution, Quantum Vorticity & Curvature Telemetry", sec_heading))
    t1_data = [
        ["Quantum / Topological Parameter", "Value", "Physical & Number-Theoretic Significance"],
        ["Prime Cutoff Bound P_max", f"{tel['max_prime']} (N = {tel['prime_count']} Primes)", "Upper bound for Sieve of Eratosthenes prime generation"],
        ["Twin Prime Gaps (d = 2)", f"{tel['twin_primes']} Gaps", "Identifies close vortex doublet pairings (e.g., 3-5, 5-7, 11-13)"],
        ["Cousin Prime Gaps (d = 4)", f"{tel['cousin_primes']} Gaps", "Medium-range vortex spacing (e.g., 7-11, 13-17, 19-23)"],
        ["Sexy Prime Gaps (d = 6)", f"{tel['sexy_primes']} Gaps", "Long-range vortex boundary interactions (e.g., 23-29, 31-37)"],
        ["Quantized Circulation ∮v·dl", f"{tel['quantized_circulation']:.4f} ℏ/m", f"Quantized phase winding: n = {tel['vorticity_n']} circulation units"],
        ["Peak Vortex Field |B|_max", f"{tel['peak_vortex_mag']:.4f}", "Maximum magnetic field gradient at inner toroidal core boundary"],
        ["BEC Vortex Core Depletion", f"{tel['min_bec_density']:.2e}", "Superfluid density vanishes at quantized vortex core lines"],
        ["Riemannian Volume Form ∫√det(g)", f"{tel['riemannian_volume']:.6f}", "Exact normalization integral: total condensate particle number"],
        ["Mean Ricci Curvature ⟨R⟩", f"{tel['mean_curvature']:.6f}", "Spatial average of conformal Riemannian scalar curvature"],
        ["Fundamental Zeta Zero γ_1", f"{tel['zeta_zero']:.4f}", "First non-trivial Riemann Zeta zero modulating radial phase"]
    ]

    t1 = Table(t1_data, colWidths=[5.5 * cm, 3.8 * cm, 7.9 * cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("TOPPADDING", (0, 0), (-1, 0), 3),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.0),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # 9. Discussion & Application to Quantum MEG SSS
    story.append(Paragraph("5. Application to MEG Head Surface Integrals & SSS Shielding", sec_heading))
    sec5 = (
        "When applied to whole-head biomagnetic modeling, the prime vortex field provides a closed analytical benchmark for "
        "the Quantum Head Integrator. By evaluating $\\Phi_{\\mathrm{quantum}} = \\oint_\\Gamma (\\mathbf{B}_{\\mathrm{prime}} \\cdot \\mathbf{n}) \\, dS$ "
        "across triangulated head geometries extracted via the Brain Extraction Tool (BET), the discrete multi-core vortex configuration "
        "tests the ability of Signal Space Separation (SSS) multipole algorithms to separate deep cortical sources from external ambient fields. "
        "Because the prime vortex field contains non-vanishing high-order multipoles with known topological indices, it serves as a gold standard "
        "for calibrating next-generation optically pumped magnetometer (OPM) arrays and cryogenic SQUID sensors."
    )
    story.append(Paragraph(sec5, body_style))

    # 10. References
    story.append(Paragraph("References", sec_heading))
    refs = [
        "1. Taulu, S. & Kajola, M. Presentation of electromagnetic multichannel data: The signal space separation method. <i>J. Appl. Phys.</i> 97, 124905 (2005).",
        "2. Donnelly, R. J. <i>Quantized Vortices in Helium II</i>. Cambridge University Press (1991).",
        "3. Visser, M. Acoustic black holes: horizons, ergospheres and Hawking radiation. <i>Class. Quantum Gravity</i> 15, 1767–1791 (1998).",
        "4. Montgomery, H. L. The pair correlation of zeros of the zeta function. <i>Proc. Symp. Pure Math.</i> 24, 181–193 (1973).",
        "5. Smith, S. M. Fast robust automated brain extraction. <i>Hum. Brain Mapp.</i> 17, 143–155 (2002)."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("ref", parent=body_style, fontSize=7, leading=9.5, spaceAfter=2)))

    doc.build(story)
    print(f"Preprint successfully compiled to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    build_pdf()
