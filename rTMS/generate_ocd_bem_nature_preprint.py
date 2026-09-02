#!/usr/bin/env python3
"""Generate a research-only Nature-style preprint from the OCD tab simulations."""

from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
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
    TableStyle,
)

from logic.moduli_bem_paradigm import get_moduli_bem_paradigm
from logic.rtms_engine import get_ocd_fea_simulation

OUTPUT = Path(__file__).with_name("Nature_Preprint_OCD_BEM_Finite_Math.pdf")
INK = colors.HexColor("#17212b")
BLUE = colors.HexColor("#145a7a")
TEAL = colors.HexColor("#008f83")
GOLD = colors.HexColor("#c58b00")
RED = colors.HexColor("#bd3b35")
MIST = colors.HexColor("#eef4f5")
GREY = colors.HexColor("#60717d")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Times-Bold", fontSize=19,
                                leading=23, textColor=INK, alignment=TA_CENTER, spaceAfter=8),
        "byline": ParagraphStyle("byline", parent=base["Normal"], fontName="Helvetica", fontSize=9,
                                 leading=12, textColor=GREY, alignment=TA_CENTER, spaceAfter=2),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=11, leading=14, textColor=BLUE, spaceBefore=13, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Times-Roman", fontSize=9,
                                leading=13, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
        "abstract": ParagraphStyle("abstract", parent=base["BodyText"], fontName="Times-Roman", fontSize=9,
                                    leading=13, textColor=INK, alignment=TA_JUSTIFY, leftIndent=12,
                                    rightIndent=12, spaceAfter=8),
        "equation": ParagraphStyle("equation", parent=base["Code"], fontName="Courier", fontSize=8.3,
                                    leading=12, textColor=INK, alignment=TA_CENTER, backColor=MIST,
                                    borderColor=colors.HexColor("#c9dadd"), borderWidth=0.4,
                                    borderPadding=6, spaceBefore=5, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.5,
                                   leading=10, textColor=GREY, alignment=TA_JUSTIFY, spaceAfter=8),
        "small": ParagraphStyle("small", parent=base["Normal"], fontName="Helvetica", fontSize=7.5,
                                 leading=10, textColor=GREY, alignment=TA_CENTER, spaceAfter=4),
    }


def figure_image(fig, height_cm=7.0):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="#fbfcfc")
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=16.0 * cm, height=height_cm * cm)


def figure_bem(bem):
    heatmap = bem["bem_heatmap"]
    attenuation = bem["bem_attenuation"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#fbfcfc")
    potential = np.asarray(heatmap["potential"])
    image = left.imshow(potential, cmap="viridis", aspect="auto", origin="lower")
    left.set_title("Single-layer BEM cortical potential", fontsize=9, fontweight="bold")
    left.set_xlabel("Azimuthal boundary node", fontsize=8)
    left.set_ylabel("Polar boundary node", fontsize=8)
    left.tick_params(labelsize=7)
    colorbar = fig.colorbar(image, ax=left, fraction=0.045, pad=0.04)
    colorbar.set_label("Scaled potential", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)

    depths = attenuation["depths_mm"]
    fields = attenuation["field_pct"]
    right.plot(depths, fields, color="#008f83", linewidth=2.6)
    right.fill_between(depths, fields, color="#008f83", alpha=0.16)
    right.set_title("Layered attenuation", fontsize=9, fontweight="bold")
    right.set_xlabel("Modeled depth (mm)", fontsize=8)
    right.set_ylabel("Relative field (%)", fontsize=8)
    right.set_ylim(bottom=0)
    right.tick_params(labelsize=7)
    right.grid(alpha=0.18)
    fig.tight_layout()
    return figure_image(fig)


def figure_stability(bem):
    grid = bem["moduli_grid"]
    protocol = bem["optimal_protocol"]
    fig, axis = plt.subplots(figsize=(8.2, 3.1))
    fig.patch.set_facecolor("#fbfcfc")
    contour = axis.contourf(grid["freq_axis"], grid["intensity_axis"], grid["stability"], levels=18, cmap="YlGnBu")
    axis.contour(grid["freq_axis"], grid["intensity_axis"], grid["stability"], levels=7, colors="#ffffff", linewidths=0.45, alpha=0.55)
    axis.scatter(protocol["frequency_hz"], protocol["intensity_pct"], s=70, marker="*", color="#bd3b35", edgecolor="white", linewidth=0.8, zorder=3, label="Finite-grid maximum")
    axis.set_title("Finite protocol-stability landscape", fontsize=9, fontweight="bold")
    axis.set_xlabel("Frequency (Hz)", fontsize=8)
    axis.set_ylabel("Drive intensity (% MSO)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=7, loc="lower right", frameon=True)
    colorbar = fig.colorbar(contour, ax=axis, pad=0.02)
    colorbar.set_label("Stability score", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return figure_image(fig)


def figure_response(ocd):
    fea = ocd["fea_surface"]
    trajectory = ocd["ybocs_trajectory"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    fig.patch.set_facecolor("#fbfcfc")
    image = left.imshow(np.asarray(fea["z"]), cmap="magma", origin="lower", aspect="auto")
    left.set_title("Illustrative dual-lobe field map", fontsize=9, fontweight="bold")
    left.set_xlabel("Lateral mesh coordinate", fontsize=8)
    left.set_ylabel("Anterior-posterior mesh coordinate", fontsize=8)
    left.tick_params(labelsize=7)
    colorbar = fig.colorbar(image, ax=left, fraction=0.045, pad=0.04)
    colorbar.set_label("Modeled field (V/m)", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)

    sessions = trajectory["sessions"]
    scores = trajectory["ybocs_scores"]
    right.plot(sessions, scores, color="#bd3b35", linewidth=2.4, marker="o", markersize=3)
    right.fill_between(sessions, scores, max(scores), color="#bd3b35", alpha=0.10)
    right.set_title("Illustrative finite-session trajectory", fontsize=9, fontweight="bold")
    right.set_xlabel("Modeled session", fontsize=8)
    right.set_ylabel("Synthetic severity index", fontsize=8)
    right.tick_params(labelsize=7)
    right.grid(alpha=0.18)
    fig.tight_layout()
    return figure_image(fig)


def paragraph(text, style):
    return Paragraph(text, style)


def equation(text, label, document_styles):
    return [paragraph(text, document_styles["equation"]), paragraph(label, document_styles["caption"])]


def characteristics_table(bem, ocd, document_styles):
    protocol = bem["optimal_protocol"]
    attenuation = bem["bem_attenuation"]
    heatmap = bem["bem_heatmap"]
    final_field = attenuation["field_pct"][-1]
    rows = [
        ["Characteristic", "Finite-model result", "Interpretation"],
        ["Sweep domain", "18-22 Hz; 80-120% MSO", "OCD tab scenario bounds"],
        ["Grid maximum", f"{protocol['frequency_hz']:.2f} Hz; {protocol['intensity_pct']:.2f}%", "Discrete stability argmax"],
        ["Stability score", f"{protocol['stability_score']:.6f}", "Dimensionless model score"],
        ["Peak cortical potential", f"{heatmap['peak_potential']:.4f}", "Scaled BEM surface quantity"],
        ["Depth endpoint", f"{final_field:.4g}% at {attenuation['depths_mm'][-1]:.0f} mm", "Layered attenuation output"],
        ["Finite session series", f"{len(ocd['ybocs_trajectory']['sessions'])} simulated sessions", "Illustrative, not clinical outcome"],
    ]
    table = Table(rows, colWidths=[4.0 * cm, 5.0 * cm, 6.5 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.7),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, MIST]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c6d2d7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf():
    ocd = get_ocd_fea_simulation()
    bem = get_moduli_bem_paradigm("ocd", freq_range=(18.0, 22.0), intensity_range=(80.0, 120.0))
    document_styles = styles()
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.7 * cm,
        title="Finite-Math Boundary-Element Modeling for OCD Deep TMS: a research preprint",
        author="NeuroMorph Computational Modeling Group",
    )
    story = [
        HRFlowable(width="100%", thickness=2.2, color=RED),
        Spacer(1, 8),
        paragraph("A finite-math boundary-element modeling workflow for OCD deep TMS", document_styles["title"]),
        paragraph("NeuroMorph Computational Modeling Group", document_styles["byline"]),
        paragraph(f"Research preprint | Generated {date.today().isoformat()} | Computational demonstration", document_styles["byline"]),
        Spacer(1, 8),
        paragraph("ABSTRACT", document_styles["section"]),
        paragraph(
            "We describe a finite, reproducible visualization workflow for an obsessive-compulsive disorder (OCD) tab in a neuromodulation research interface. The workflow combines a bounded frequency-intensity grid, a single-layer boundary-element (BEM) cortical potential calculation, layered attenuation, and illustrative session-indexed response curves. All results are outputs of a simplified computational model. They are not patient-specific field estimates, clinical recommendations, evidence of efficacy, or a substitute for regulatory-cleared planning systems.",
            document_styles["abstract"]),
        paragraph("Keywords: boundary element method; finite grid optimization; deep TMS visualization; OCD research; computational modeling", document_styles["small"]),
        paragraph("1. Model scope", document_styles["section"]),
        paragraph(
            "The model intentionally constrains the OCD scenario to a finite 18-22 Hz by 80-120% MSO parameter rectangle. Each lattice point is reduced by a modular stability function and evaluated as a visualization aid. The cortical mesh is a deformed spherical boundary under a dipole source; it is not an anatomical head model and does not encode a patient MRI, coil registration, stimulation threshold, or treatment response.",
            document_styles["body"]),
        paragraph("2. Finite mathematical formulation", document_styles["section"]),
    ]
    story += equation(
        "phi_i = (mu_0 / 4pi) [m dot (x_i - x_coil)] / max(||x_i - x_coil||^3, epsilon),  i = 1,...,N",
        "Equation 1 | Discrete single-layer BEM potential evaluated at N boundary nodes.", document_styles)
    story += equation(
        "E(d_j) = I_0 product_(ell: d_j >= b_ell) exp[-4 sigma_ell (d_j - b_ell)],  j = 1,...,50",
        "Equation 2 | Finite layered attenuation through the configured conductivity layers.", document_styles)
    story += equation(
        "(f*, I*) = argmax_{f in F, I in J} Phi(R(f + iI/100)),  |F| = |J| = 46",
        "Equation 3 | Finite-grid selection of the modular stability maximum after reduction R.", document_styles)
    story += equation(
        "Z_pq = E_max [exp(-a r_left,pq) + exp(-a r_right,pq)] [1 + 0.15 cos(2X_pq) sin(Y_pq)]",
        "Equation 4 | Finite dual-lobe field visualization over the tab's 50 x 50 mesh.", document_styles)
    story += equation(
        "s_k = max(8, 34 - 0.85k + eta_k),  k = 1,...,29",
        "Equation 5 | Illustrative finite-session severity series; eta_k is synthetic perturbation.", document_styles)
    story += [
        paragraph("3. Data-derived model characteristics", document_styles["section"]),
        characteristics_table(bem, ocd, document_styles),
        paragraph("Table 1 | Characteristics computed during this preprint run from the same bounded BEM pathway displayed by the OCD treatment tab.", document_styles["caption"]),
        figure_bem(bem),
        paragraph("Figure 1 | The left panel visualizes the BEM potential over the discrete cortical boundary mesh. The right panel reports the model's layered depth attenuation. Values are normalized/scaled implementation outputs and are not in-vivo dosimetry.", document_styles["caption"]),
        figure_stability(bem),
        paragraph("Figure 2 | Bounded stability landscape used by the OCD tab. The star identifies the maximum on the 46 by 46 finite grid, not a patient-level optimum.", document_styles["caption"]),
        figure_response(ocd),
        paragraph("Figure 3 | Existing OCD-tab field visualization and synthetic 29-session series. The session plot is a display model and makes no clinical efficacy claim.", document_styles["caption"]),
        PageBreak(),
        paragraph("4. Interpretation and limitations", document_styles["section"]),
        paragraph(
            "The figures demonstrate that a finite BEM pathway can make its assumptions visible: a source-to-boundary potential, a layered attenuation curve, and a bounded search landscape. The apparent maximum is determined by the defined objective and mesh, so it should be interpreted only as a property of this model. A clinically usable field model would require MRI-derived segmentation, validated tissue conductivities, coil placement and orientation, calibrated stimulator waveforms, uncertainty quantification, and independent validation.",
            document_styles["body"]),
        paragraph(
            "The illustrative response series is purposefully separated from the field calculations. It is synthetic, includes randomly generated perturbations, and has no patient data, comparator arm, p-value, or outcome-validation basis. It must not be used to predict symptom change, choose treatment parameters, or support a clinical decision.",
            document_styles["body"]),
        paragraph("5. Reproducibility", document_styles["section"]),
        paragraph(
            "This document was produced from the repository functions get_ocd_fea_simulation() and get_moduli_bem_paradigm(condition='ocd', freq_range=(18.0, 22.0), intensity_range=(80.0, 120.0)). The generator records computed values in Table 1 and embeds the resulting plots directly in the PDF. The finite-grid and mesh sizes are part of the source implementation, allowing the visualization to be rerun under the same assumptions.",
            document_styles["body"]),
        paragraph("Data and code availability", document_styles["section"]),
        paragraph(
            "The simulation code and PDF generator are included in the local NeuroMorph rTMS workspace. No human participant data were used to create this preprint.",
            document_styles["body"]),
        HRFlowable(width="100%", thickness=0.6, color=GREY),
        Spacer(1, 6),
        paragraph("Research-only computational preprint. Not peer reviewed. Not medical advice or a treatment protocol.", document_styles["small"]),
    ]
    document.build(story)
    print(f"PDF written to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
