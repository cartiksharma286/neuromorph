#!/usr/bin/env python3
"""Generate a clearly labeled, research-only mathematical concept preprint."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from hiv_research_repair.hiv_repair_research_app import OUTPUT_DIR, write_plots


def generate_pdf() -> Path:
    figures = write_plots()
    pdf_path = Path(__file__).parent / "HIV_Repair_Research_Concept_Preprint.pdf"
    document = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=.75*inch, rightMargin=.75*inch,
                                topMargin=.7*inch, bottomMargin=.7*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=16, leading=20)
    body = ParagraphStyle("Body", parent=styles["BodyText"], alignment=TA_JUSTIFY, fontSize=10, leading=14, spaceAfter=8)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=12, spaceAfter=6)
    equation = ParagraphStyle("Equation", parent=body, alignment=TA_CENTER, fontName="Times-Italic", textColor=colors.HexColor("#0a6f73"), fontSize=12)
    story = [Paragraph("A Finite Mathematical Framework for Exploratory HIV Reservoir-Repair Modeling", title),
             Spacer(1, .12*inch), Paragraph("Concept preprint | Computational research artifact | 29 August 2026", ParagraphStyle("Authors", parent=body, alignment=TA_CENTER)),
             Paragraph("<b>Important limitation.</b> This document reports a synthetic computational concept. It does not establish efficacy or safety, is not a clinical prediction, and provides no treatment, experimental, sequence, delivery, or dosing instructions.", body),
             Paragraph("Abstract", heading), Paragraph("We describe an illustrative, bounded model joining sequencing confidence, hypothetical inhibitor-gated editing concepts, conjugate variability, and long-term uncertainty. Feynman-style path weights are used only as an abstract aggregation metaphor; continued fractions impose finite convergence; and elliptic-integral-inspired scaling represents bounded transport heterogeneity. Outputs are normalized indices, not biological repair rates.", body),
             Paragraph("Finite Model", heading), Paragraph("For a horizon H and time t, the illustrative index uses a finite exponential trajectory:", body),
             Paragraph("R(t) = 100 [1 - exp(-&alpha;t/H)]", equation), Paragraph("where &alpha; = A q E(k) C_n. Here A is a conceptual activity coefficient, q is a data-quality factor, E(k) is a bounded elliptic-scale term, and C_n is a truncated continued fraction. A path-weight summary is finite by construction:", body),
             Paragraph("Z_N = &Sigma;_{j=1..N} w_j exp(-S_j) ;  C_n = a_1 + 1/(a_2 + ... + 1/a_n)", equation),
             Paragraph("Statistical Treatment", heading), Paragraph("Illustrative conjugate variability is represented by a beta distribution on [0,1]. The uncertainty band combines scenario dispersion with the square-root horizon factor. This is a visual sensitivity analysis, not a confidence interval from patient data.", body)]
    story += [Image(str(OUTPUT_DIR / figures[0]), width=6.4*inch, height=3.6*inch), Paragraph("Figure 1 | Illustrative normalized trajectory with synthetic uncertainty envelope.", body),
              Image(str(OUTPUT_DIR / figures[1]), width=6.4*inch, height=3.6*inch), Paragraph("Figure 2 | Synthetic beta-distribution visualization for conjugate-associated variability.", body), PageBreak(),
              Paragraph("Scenario Characteristics", heading)]
    table = Table([["Component", "Representation", "Interpretation"], ["Sequencing", "q in [0.5, 1.0]", "Data quality proxy"], ["Conjugate", "Beta-distributed index", "Bounded synthetic variability"], ["Horizon", "1-50 years", "Scenario visualization window"], ["Output", "0-100 index", "Not a repair percentage or outcome"]], colWidths=[1.25*inch, 2.0*inch, 2.7*inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a6f73")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#b8c9c7")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f8f6")), ("PADDING", (0, 0), (-1, -1), 7)]))
    story += [table, Paragraph("Discussion", heading), Paragraph("Any potential HIV cure strategy must undergo rigorous preclinical validation, independent replication, clinical trials, regulatory review, and long-term safety monitoring. The model intentionally excludes clinical endpoints and should only be used for mathematical visualization and communication of uncertainty.", body), Paragraph("References", heading), Paragraph("1. Feynman, R. P. Space-time approach to non-relativistic quantum mechanics. Rev. Mod. Phys. 20, 367-387 (1948).<br/>2. Abramowitz, M. & Stegun, I. Handbook of Mathematical Functions (1972).<br/>3. UNAIDS. Global AIDS Update (accessed 2026).", body)]
    document.build(story)
    return pdf_path


if __name__ == "__main__":
    print(f"Generated: {generate_pdf()}")