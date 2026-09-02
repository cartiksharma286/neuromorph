#!/usr/bin/env python3
"""Render a research-only RLS recurrent-design finite-math preprint."""

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
from reportlab.platypus import HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from logic.rls_rtms import simulate_rls_rtms

OUTPUT = Path(__file__).with_name("Nature_Preprint_RLS_Recurrent_rTMS_Finite_Math.pdf")
INK, BLUE, TEAL, RED, GREY, MIST = (colors.HexColor(value) for value in ("#17212b", "#145a7a", "#008f83", "#bd3b35", "#60717d", "#eef4f5"))


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Times-Bold", fontSize=19, leading=23, textColor=INK, alignment=TA_CENTER, spaceAfter=8),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=3),
        "head": ParagraphStyle("head", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=BLUE, spaceBefore=13, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Times-Roman", fontSize=9, leading=13, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
        "eq": ParagraphStyle("eq", parent=base["Code"], fontName="Courier", fontSize=8.2, leading=12, textColor=INK, alignment=TA_CENTER, backColor=MIST, borderColor=colors.HexColor("#c9dadd"), borderWidth=.4, borderPadding=6, spaceBefore=5, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, leading=10, textColor=GREY, alignment=TA_JUSTIFY, spaceAfter=8),
    }


def image(fig, height=7.0):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="#fbfcfc")
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=16 * cm, height=height * cm)


def fig_recurrence(data):
    fig, axis = plt.subplots(figsize=(8.2, 3.0)); fig.patch.set_facecolor("#fbfcfc")
    palette = ["#008f83", "#145a7a", "#c58b00", "#8064a2", "#bd3b35"]
    for curve, color in zip(data["recurrent_curves"], palette): axis.plot(data["weeks"], curve["values"], label=curve["name"], lw=2, color=color)
    axis.set(title="Finite recurrent target activation", xlabel="Study week", ylabel="Normalized activation", ylim=(0, 1.05))
    axis.grid(alpha=.18); axis.legend(fontsize=6.5, ncol=2); axis.tick_params(labelsize=7); fig.tight_layout()
    return image(fig)


def fig_field_distribution(data):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.0)); fig.patch.set_facecolor("#fbfcfc")
    heat = left.imshow(np.asarray(data["bem_field"]["z"]), cmap="viridis", origin="lower", aspect="auto")
    left.set(title="Synthetic multi-target BEM field", xlabel="Boundary X", ylabel="Boundary Y"); left.tick_params(labelsize=7)
    cb = fig.colorbar(heat, ax=left, fraction=.045, pad=.04); cb.set_label("Normalized field", fontsize=7); cb.ax.tick_params(labelsize=7)
    right.hist(data["experimental_design"]["response_samples"], bins=28, color="#145a7a", alpha=.9)
    right.axvline(50, color="#c58b00", ls="--", lw=1.4); right.set(title="Synthetic response-score distribution", xlabel="Response score (%)", ylabel="Draw count")
    right.tick_params(labelsize=7); right.grid(axis="y", alpha=.18); fig.tight_layout()
    return image(fig)


def fig_outcomes(data):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.0)); fig.patch.set_facecolor("#fbfcfc")
    left.plot(data["weeks"], data["recovery_pct"], color="#008f83", lw=2.6, marker="o", ms=3); left.fill_between(data["weeks"], data["recovery_pct"], color="#008f83", alpha=.12)
    left.set(title="Illustrative recovery from baseline", xlabel="Study week", ylabel="Recovery (%)", ylim=(0, 100)); left.grid(alpha=.18); left.tick_params(labelsize=7)
    years = ["Y1", "Y2", "Y3", "Y4", "Y5"]; right.bar(years, data["economics"]["revenue_5yr"], color=["#145a7a", "#008f83", "#c58b00", "#8064a2", "#bd3b35"])
    right.set(title="Hypothetical market revenue scenario", xlabel="Fiscal year", ylabel="Revenue ($)"); right.tick_params(labelsize=7); right.grid(axis="y", alpha=.18); fig.tight_layout()
    return image(fig)


def add_equation(story, text, label, styles):
    story.extend([Paragraph(text, styles["eq"]), Paragraph(label, styles["caption"])])


def build_pdf():
    data = simulate_rls_rtms()
    s = make_styles()
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.7*cm, title="Finite recurrent rTMS RLS research preprint")
    design, economics = data["experimental_design"], data["economics"]
    rows = [["Characteristic", "Finite-model output", "Scope"]]
    rows += [
        ["Cortical targets", str(len(data["lobe_protocol"])), "Predefined research allocation"],
        ["Experimental design", f"{design['best']['frequency_hz']} Hz / {design['best']['intensity_pct']}%", "Finite-grid maximum"],
        ["Response-score likelihood", f"{design['response_likelihood_pct']}%", "Synthetic draw threshold >= 50%"],
        ["Horizon recovery", f"{data['recovery_pct'][-1]}%", "Synthetic model output"],
        ["Market scenario", f"${economics['total_revenue_5yr']:,.0f} / 5 years", "Hypothetical clinic model"],
    ]
    table = Table(rows, colWidths=[4.2*cm, 5.1*cm, 6.2*cm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.8), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, MIST]), ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#c6d2d7")), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
    story = [HRFlowable(width="100%", thickness=2.2, color=RED), Spacer(1,8), Paragraph("A finite recurrent optimal-design workflow for RLS rTMS research", s["title"]), Paragraph("NeuroMorph Computational Modeling Group", s["meta"]), Paragraph("Nature-style research preprint | Synthetic simulation | Not peer reviewed", s["meta"]), Spacer(1,8), Paragraph("ABSTRACT", s["head"]), Paragraph("This preprint documents a finite, synthetic modeling workflow for a restless legs syndrome (RLS) rTMS research dashboard. A five-target cortical allocation is propagated through a bounded recurrent update, a finite experimental-design grid, a normalized boundary-field visualization, simulated response-score draws, and a hypothetical market scenario. The implementation is intended for software visualization and reproducibility only. It does not establish efficacy, estimate cure probability, prescribe stimulation, or support a clinical decision.", s["body"]), Paragraph("1. Finite model", s["head"]), Paragraph("The model represents five predefined target allocations spanning medial frontal, frontal, parietal, temporal, and occipital cortical regions. These labels are design variables for a visualization experiment, not a proposal to stimulate all cortical lobes clinically. Each target receives a bounded weight and evolves through a finite recurrence over the study horizon.", s["body"])]
    add_equation(story, "a_(ell,k+1) = min(1, 0.63 a_(ell,k) + w_ell [1 - exp(-0.42k)]),  ell = 1,...,5", "Equation 1 | Finite recurrent state update for the lobe allocation visualization.", s)
    add_equation(story, "D(f,I) = exp(-((f-10)/7)^2 - ((I-100)/18)^2);  (f*,I*) = argmax_(f,I) D(f,I)", "Equation 2 | Bounded 4 by 4 experimental-design score; it is not a treatment optimizer.", s)
    add_equation(story, "F_pq = sum_(ell=1)^5 w_ell exp(-((x_p-c_x,ell)^2 + (y_q-c_y,ell)^2)/0.13)", "Equation 3 | Finite normalized multi-source boundary-field visualization.", s)
    add_equation(story, "R_k = 100 [B - S_k] / B,  k = 0,...,K", "Equation 4 | Synthetic recovery percentage from baseline index B.", s)
    story += [Paragraph("2. Characteristics and figures", s["head"]), table, Paragraph("Table 1 | Characteristics generated during this run. Each result is restricted to the model's finite assumptions.", s["caption"]), fig_recurrence(data), Paragraph("Figure 1 | Recurrent activations for the predefined target allocation. Curves are synthetic state variables.", s["caption"]), fig_field_distribution(data), Paragraph("Figure 2 | Normalized BEM-like target-field visualization and synthetic response-score distribution. Neither panel is in-vivo dosimetry or an outcome forecast.", s["caption"]), fig_outcomes(data), Paragraph("Figure 3 | Illustrative model recovery and hypothetical five-year clinic-market revenue scenario.", s["caption"]), PageBreak(), Paragraph("3. Equipment and economic assumptions", s["head"]), Paragraph("The dashboard models a clinic with figure-8 TMS systems, neuronavigation, EMG/accelerometry, cooling, and safety monitoring. Quantities are operational placeholders. Revenue is computed from modeled annual course capacity and modeled course price, while costs are represented through explicit capital and operating assumptions. These financial outputs are scenario calculations rather than market forecasts.", s["body"]), add_equation(story, "Rev_y = N_courses P_course g_y;  NPV = -C_0 + sum_(y=1)^5 (Rev_y - O) / (1+r)^y", "Equation 5 | Finite five-year revenue and discounted-cash-flow scenario.", s), Paragraph("4. Limitations", s["head"]), Paragraph("No participant data, patient anatomy, stimulator waveform calibration, clinical comparator, safety monitoring data, regulatory clearance, or external outcome validation were used. The response distribution is sampled from a fixed beta distribution and must not be interpreted as a probability of recovery or cure. Any clinical study would require appropriate ethical review, registered protocol, validated endpoints, and qualified clinical oversight.", s["body"]), Paragraph("Code availability", s["head"]), Paragraph("The local workspace contains the RLS simulation module and this PDF generator. Running the generator recalculates the finite values and embeds the resulting figures.", s["body"]), HRFlowable(width="100%", thickness=.6, color=GREY), Spacer(1,5), Paragraph("Research-only computational preprint. Not medical advice, not a clinical protocol, and not evidence of efficacy.", s["meta"])]
    document.build(story)
    print(f"PDF written to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
