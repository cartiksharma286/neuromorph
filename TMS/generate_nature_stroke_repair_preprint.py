"""Build a Nature-style, research-only stroke-repair rTMS preprint PDF."""

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from logic.stroke_repair_rtms import simulate_stroke_repair_rtms

OUTPUT_PDF = Path(__file__).with_name("Nature_Preprint_Stroke_Repair_rTMS_Finite_Staging.pdf")


def _equation(expr: str):
    fig = plt.figure(figsize=(7.0, 0.48), facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(0.5, 0.5, f"${expr}$", ha="center", va="center", fontsize=11, color="#0f172a")
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=17 * cm, height=1.15 * cm)


def build_pdf(output_path=None, sim_params=None):
    output_path = Path(output_path or OUTPUT_PDF)
    data = simulate_stroke_repair_rtms(**(sim_params or {}))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Times-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#102a43"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=13, textColor=colors.HexColor("#243b53"), alignment=4)
    section = ParagraphStyle("section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#0b7285"), spaceBefore=8)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.4, leading=10, textColor=colors.HexColor("#52606d"))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=1.55 * cm, rightMargin=1.55 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    story = [
        Paragraph("ARTICLE | RESEARCH PREPRINT", ParagraphStyle("tag", parent=small, textColor=colors.HexColor("#b91c1c"))),
        Spacer(1, 0.12 * cm),
        Paragraph("Statistical Distributions and Continued-Fraction Staging for Long-Term Stroke Repair with Repetitive Transcranial Magnetic Stimulation", title),
        Paragraph("Cartik Sharma, NeuroMorph Computational Neuromodulation Group | September 2026", small),
        Spacer(1, 0.25 * cm),
        Paragraph("<b>Abstract.</b> We present a hypothesis-generating computational framework for staging repetitive transcranial magnetic stimulation (rTMS) after stroke. The model combines distributional uncertainty around a motor-recovery trajectory with finite-state stage transitions and rational continued-fraction approximants for phase scheduling. Outputs are simulations, not clinical evidence or treatment instructions.", body),
        Paragraph("1. Model and finite staging", section),
        Paragraph("A bounded Fugl-Meyer motor score is represented by a normal distribution whose mean follows saturating recovery. A prime-modulus state tracks discrete stage transitions, while convergents of a target synchronization ratio provide reproducible scheduling fractions.", body),
        _equation(r"Y_t \sim \mathcal{N}(\mu_t,\sigma_t^2),\quad \mu_t=B+(A-B)(1-e^{-kt})"),
        _equation(r"s_{t+1}=(s_t+\lfloor \gamma r_t\rfloor)\mathrm{mod}\ p,\quad s_t\in\mathbb{F}_p"),
        _equation(r"\rho=[a_0;a_1,\ldots,a_n]\approx p_n/q_n"),
        Paragraph("2. Simulated trajectories", section),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    weeks = data["weeks"]
    ax.plot(weeks, data["trajectories"]["usual_care"], label="Usual care", color="#64748b", linestyle=":")
    ax.plot(weeks, data["trajectories"]["fixed_rtms"], label="Fixed rTMS", color="#f97316")
    ax.plot(weeks, data["trajectories"]["adaptive_rtms"], label="Adaptive staged rTMS", color="#0f766e", linewidth=2.2)
    ax.fill_between(weeks, data["distribution"]["adaptive_lower"], data["distribution"]["adaptive_upper"], color="#0f766e", alpha=0.14, label="95% model interval")
    ax.set_xlabel("Weeks after index event"); ax.set_ylabel("Fugl-Meyer motor score")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=220, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    story.extend([Image(buf, width=17 * cm, height=6.8 * cm), Paragraph("Figure 1 | Simulated long-term trajectories with a distributional uncertainty band. Parameters are illustrative and require prospective validation.", small)])

    story.append(Paragraph("3. Optimal staging protocol", section))
    table_data = [["Stage", "Weeks", "Hz", "% MSO", "Sessions/wk", "CF sync"]]
    for row in data["stage_protocol"]:
        table_data.append([row["stage"], row["duration_weeks"], row["frequency_hz"], row["intensity_pct_mso"], row["sessions_per_week"], row["sync_fraction"]])
    table = Table(table_data, colWidths=[3.0 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 2.4 * cm, 3.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdfa")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([table, Spacer(1, 0.25 * cm), Paragraph("4. Limitations and translation", section), Paragraph("This preprint is a computational design study. Statistical distributions encode model uncertainty rather than patient-specific confidence intervals. Continued fractions provide numerical scheduling structure, not evidence of biological optimality. Any clinical protocol must be established by qualified clinicians under appropriate ethics, safety, and prospective trial oversight.", body), Paragraph("Reproducibility: deterministic seed 20260924; default horizon 52 weeks; prime modulus p = 13; six continued-fraction terms.", small)])
    doc.build(story)
    return str(output_path)


if __name__ == "__main__":
    build_pdf()