#!/usr/bin/env python3
"""Build a research-only preprint for the recurrent-prime cavity simulation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from .drill_path_planner import RecurrentPrimeDrillPathEngine
except ImportError:
    from drill_path_planner import RecurrentPrimeDrillPathEngine


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "finite_element_drill_cavity_recurrent_primes_preprint.pdf"


class CavitySchematic(Flowable):
    """Draw a compact schematic of the simulation domain."""

    def __init__(self, width: float = 6.75 * inch, height: float = 2.2 * inch) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        x0, y0 = 0.25 * inch, 0.18 * inch
        width, height = self.width - 0.5 * inch, self.height - 0.42 * inch

        canvas.setFillColor(colors.HexColor("#e8f1fa"))
        canvas.setStrokeColor(colors.HexColor("#1f4e79"))
        canvas.roundRect(x0, y0, width, height, 7, fill=1, stroke=1)

        pulp_y = y0 + 0.22 * height
        canvas.setFillColor(colors.HexColor("#f7d9de"))
        canvas.setStrokeColor(colors.HexColor("#a73552"))
        canvas.roundRect(x0 + 0.2 * inch, pulp_y, width - 0.4 * inch, 0.22 * inch, 4, fill=1, stroke=1)

        cavity_x = x0 + 0.54 * width
        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(colors.HexColor("#0f172a"))
        canvas.setLineWidth(1.2)
        canvas.rect(cavity_x - 0.13 * inch, y0 + 0.45 * inch, 0.26 * inch, height - 0.7 * inch, fill=1, stroke=1)

        canvas.setStrokeColor(colors.HexColor("#0e7490"))
        canvas.setLineWidth(1.4)
        canvas.line(cavity_x, y0 + height + 0.22 * inch, cavity_x, y0 + height - 0.05 * inch)
        canvas.setFillColor(colors.HexColor("#0e7490"))
        canvas.circle(cavity_x, y0 + height + 0.26 * inch, 0.055 * inch, fill=1, stroke=0)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#172554"))
        canvas.drawString(x0 + 0.12 * inch, y0 + height - 0.2 * inch, "hard-tissue proxy")
        canvas.setFillColor(colors.HexColor("#7f1d1d"))
        canvas.drawString(x0 + 0.12 * inch, pulp_y + 0.07 * inch, "pulp-protection boundary")
        canvas.setFillColor(colors.HexColor("#0f172a"))
        canvas.drawString(cavity_x + 0.18 * inch, y0 + 0.75 * inch, "0.52 mm drill envelope")
        canvas.setFillColor(colors.HexColor("#0e7490"))
        canvas.drawString(cavity_x + 0.1 * inch, y0 + height + 0.22 * inch, "feed")


def equation(lines: list[str], number: int, caption: str, styles: dict[str, ParagraphStyle]) -> list:
    formula = Paragraph("<br/>".join(lines), styles["equation"])
    marker = Paragraph(f"<b>({number})</b>", styles["equation_number"])
    block = Table([[formula, marker]], colWidths=[6.1 * inch, 0.55 * inch])
    block.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fc")),
                ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#9fc4df")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [Spacer(1, 0.04 * inch), block, Paragraph(caption, styles["caption"]), Spacer(1, 0.04 * inch)]


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#1f4e79"))
    canvas.setLineWidth(0.55)
    canvas.line(doc.leftMargin, 0.49 * inch, letter[0] - doc.rightMargin, 0.49 * inch)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#52657a"))
    canvas.drawString(doc.leftMargin, 0.34 * inch, "Research-only simulation preprint | Not peer reviewed | Not for clinical planning")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, f"{doc.page}")
    canvas.restoreState()


def build_preprint(output_path: Path = OUTPUT_PATH) -> Path:
    """Create the PDF and return its absolute output path."""
    result = RecurrentPrimeDrillPathEngine().simulate()
    model = result["model"]
    characteristics = result["characteristics"]
    terms = model["prime_terms"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.66 * inch,
        title="Finite-element simulation of a dental drill cavity with recurrent-prime path perturbations",
        author="Neuromorph Dental Robotics Simulation Group",
        subject="Research-only finite-element formulation and deterministic 2-D surrogate results",
    )
    stylesheet = getSampleStyleSheet()
    navy = colors.HexColor("#153b5b")
    blue = colors.HexColor("#147aa9")
    ink = colors.HexColor("#182534")
    styles = {
        "banner": ParagraphStyle(
            "banner", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=8,
            textColor=blue, leading=10, alignment=TA_LEFT, spaceAfter=5
        ),
        "title": ParagraphStyle(
            "title", parent=stylesheet["Title"], fontName="Helvetica-Bold", fontSize=17,
            leading=21, alignment=TA_CENTER, textColor=navy, spaceAfter=6
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=stylesheet["Normal"], fontName="Helvetica-Oblique", fontSize=9,
            leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#52657a"), spaceAfter=7
        ),
        "author": ParagraphStyle(
            "author", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=9,
            leading=12, alignment=TA_CENTER, textColor=ink, spaceAfter=3
        ),
        "affiliation": ParagraphStyle(
            "affiliation", parent=stylesheet["Normal"], fontName="Helvetica", fontSize=8,
            leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#52657a"), spaceAfter=9
        ),
        "abstract": ParagraphStyle(
            "abstract", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=8.8,
            leading=12.6, alignment=TA_JUSTIFY, textColor=ink, leftIndent=0.22 * inch,
            rightIndent=0.22 * inch, spaceAfter=7
        ),
        "heading": ParagraphStyle(
            "heading", parent=stylesheet["Heading2"], fontName="Helvetica-Bold", fontSize=10.8,
            leading=14, textColor=navy, spaceBefore=8, spaceAfter=4
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=stylesheet["Heading3"], fontName="Helvetica-Bold", fontSize=9.1,
            leading=12, textColor=blue, spaceBefore=5, spaceAfter=2
        ),
        "body": ParagraphStyle(
            "body", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=8.7,
            leading=12.4, alignment=TA_JUSTIFY, textColor=ink, spaceAfter=5
        ),
        "small": ParagraphStyle(
            "small", parent=stylesheet["BodyText"], fontName="Helvetica", fontSize=7.7,
            leading=10.2, alignment=TA_JUSTIFY, textColor=ink, spaceAfter=3
        ),
        "equation": ParagraphStyle(
            "equation", parent=stylesheet["Normal"], fontName="Courier-Bold", fontSize=8.2,
            leading=11.1, alignment=TA_CENTER, textColor=navy
        ),
        "equation_number": ParagraphStyle(
            "equation_number", parent=stylesheet["Normal"], fontName="Helvetica-Bold", fontSize=8.2,
            leading=10, alignment=TA_RIGHT, textColor=colors.HexColor("#52657a")
        ),
        "caption": ParagraphStyle(
            "caption", parent=stylesheet["Normal"], fontName="Helvetica-Oblique", fontSize=7.3,
            leading=9.4, alignment=TA_CENTER, textColor=colors.HexColor("#52657a"), spaceBefore=2, spaceAfter=2
        ),
        "reference": ParagraphStyle(
            "reference", parent=stylesheet["Normal"], fontName="Helvetica", fontSize=7.4,
            leading=9.5, textColor=colors.HexColor("#3e5060"), leftIndent=0.14 * inch,
            firstLineIndent=-0.14 * inch, spaceAfter=2
        ),
    }

    story: list = [
        Paragraph("COMPUTATIONAL BIOMECHANICS | PREPRINT", styles["banner"]),
        HRFlowable(width="100%", thickness=1.3, color=navy, spaceAfter=7),
        Paragraph("Finite-element formulation of a dental drill cavity with recurrent-prime path perturbations", styles["title"]),
        Paragraph(
            "A finite, reproducible 2-D material-field surrogate with a bounded continued-fraction path law",
            styles["subtitle"],
        ),
        Paragraph("Neuromorph Dental Robotics Simulation Group", styles["author"]),
        Paragraph(
            f"Research software preprint | Version 1.0 | {date(2026, 10, 24).strftime('%d %B %Y')}<br/>"
            "Prepared from the repository's deterministic drill-path simulation; not affiliated with a journal.",
            styles["affiliation"],
        ),
        Paragraph(
            "<b>Abstract.</b> Dental drilling requires a cavity trajectory that is geometrically bounded, reproducible and interpretable against a material model. "
            "We formulate a finite-element problem on a cavity-excised domain and instantiate its geometry using a deterministic 2-D material-field surrogate. "
            "The centreline is perturbed by a finite set of eight prime harmonics, whose phases are fixed by convergents of the continued fractions of square roots. "
            "The terminal taper makes both entry and terminal lateral offsets exactly zero. For the repository default (4.0 mm planned depth, 0.075 mm amplitude), "
            "the 38 x 52-cell field reports a maximum lateral correction of 6.92 micrometres, peak non-drill tissue stress of 1.183 MPa, peak tool stress of "
            "83.65 MPa and peak cutting force of 6.42 N. These are deterministic surrogate outputs, not measurements, clinical predictions or evidence of safety. "
            "The manuscript provides the finite equations, boundary definitions and parameter values needed to reproduce the calculation.",
            styles["abstract"],
        ),
        HRFlowable(width="92%", thickness=0.5, color=colors.HexColor("#b9c9d6"), spaceAfter=5),
        Paragraph(
            "<b>Scope statement.</b> This document describes research software only. It does not use patient data, is not connected to hardware, and must not be used "
            "to select a drilling trajectory or make clinical decisions. The present implementation is a deterministic 2-D material-field surrogate, rather than a "
            "validated patient-specific three-dimensional finite-element analysis.",
            styles["small"],
        ),
        Paragraph("1. Problem definition", styles["heading"]),
        Paragraph(
            "Let the rectangular computational domain be &Omega; = [&minus;4.5, 4.5] mm &times; [&minus;0.8, 5.2] mm, with depth increasing from the entrance "
            "surface toward the cavity floor. At normalized progress u &isin; [0, 1], the planned axial coordinate is z(u) = Du with D = 4.0 mm. "
            "A 0.52 mm-wide moving drill envelope is removed from the hard-tissue proxy. The remaining finite domain is divided into soft-tissue boundary, "
            "hard-tissue proxy and pulp-protection boundary labels. Figure 1 gives the coordinate convention.",
            styles["body"],
        ),
        CavitySchematic(),
        Paragraph(
            "<b>Figure 1 | Finite cavity domain and moving envelope.</b> The blue region is the hard-tissue proxy; pink marks the protected boundary. "
            "The central white channel is the 0.52 mm drill envelope. This is a schematic of the deterministic material map, not anatomical geometry.",
            styles["caption"],
        ),
        Paragraph("2. Finite-element formulation", styles["heading"]),
        Paragraph(
            "For an elastic interpretation, the cavity-excised domain &Omega;<sub>h</sub> is partitioned into a finite mesh T<sub>h</sub> of elements e. "
            "The displacement approximation uses finite nodal basis functions N<sub>a</sub>. Essential boundary conditions are imposed on "
            "&Gamma;<sub>u</sub>; a distributed cutting load acts on the cavity boundary &Gamma;<sub>c</sub>. This standard weak form makes explicit what a "
            "full 3-D implementation would solve; the current code evaluates the 2-D deterministic stress proxy described in Section 4 instead.",
            styles["body"],
        ),
    ]
    story.extend(equation(
        [
            "V<sub>h</sub> = { <b>v</b><sub>h</sub> = &Sigma;<sub>a=1</sub><sup>N</sup> N<sub>a</sub><b>v</b><sub>a</sub> : <b>v</b><sub>h</sub>|&Gamma;<sub>u</sub> = 0 }, "
            "&nbsp; &Omega;<sub>h</sub> = &Omega; \\ B<sub>r</sub>(<b>c</b>(u))",
            "&Sigma;<sub>e&isin;T<sub>h</sub></sub> &int;<sub>&Omega;<sub>e</sub></sub> &epsilon;(<b>v</b><sub>h</sub>)<sup>T</sup><b>C</b><sub>e</sub>&epsilon;(<b>u</b><sub>h</sub>) d&Omega; "
            "= &int;<sub>&Gamma;<sub>c</sub></sub> <b>v</b><sub>h</sub><sup>T</sup><b>t</b><sub>c</sub> d&Gamma;"
        ],
        1,
        "Equation 1 | Finite weak form on the cavity-excised domain. B<sub>r</sub> is the drill envelope, <b>C</b><sub>e</sub> is an element constitutive matrix, and <b>t</b><sub>c</sub> is a boundary traction.",
        styles,
    ))
    story.extend(equation(
        [
            "<b>K</b><b>U</b> = <b>F</b>, &nbsp; <b>K</b> = &Sigma;<sub>e&isin;T<sub>h</sub></sub> <b>K</b><sub>e</sub>, &nbsp; "
            "<b>K</b><sub>e</sub> = &int;<sub>&Omega;<sub>e</sub></sub> <b>B</b><sub>e</sub><sup>T</sup><b>C</b><sub>e</sub><b>B</b><sub>e</sub> d&Omega;",
            "&sigma;<sub>vM</sub> = &radic;{ [&sigma;<sub>xx</sub> &minus; &sigma;<sub>yy</sub>]<sup>2</sup> + &sigma;<sub>yy</sub><sup>2</sup> "
            "+ &sigma;<sub>xx</sub><sup>2</sup> + 6&tau;<sub>xy</sub><sup>2</sup> \\over 2 }"
        ],
        2,
        "Equation 2 | Assembly of the finite linear system and plane-stress von Mises post-processing expression.",
        styles,
    ))
    story.extend([
        Paragraph("3. Recurrent-prime centreline", styles["heading"]),
        Paragraph(
            "The centreline is a bounded, finite trigonometric sum. The prime set is P = {2, 3, 5, 7, 11, 13, 17, 19}. "
            "For each p<sub>k</sub>, a six-term continued fraction for &radic;p<sub>k</sub> produces a convergent n<sub>k</sub>/d<sub>k</sub>; "
            "the phase c<sub>k</sub> is the fractional component of that convergent. Weights are w<sub>k</sub> = ln(p<sub>k</sub>)/&radic;p<sub>k</sub>. "
            "The taper u(1&minus;u)<sup>2</sup> forces x(0) = x(1) = 0 independently of the phases, so the perturbation does not change the entry or terminal coordinate.",
            styles["body"],
        ),
    ])
    story.extend(equation(
        [
            "c<sub>k</sub> = (n<sub>k</sub> mod d<sub>k</sub>) / d<sub>k</sub>, &nbsp; w<sub>k</sub> = ln(p<sub>k</sub>) / &radic;p<sub>k</sub>, &nbsp; z(u) = Du",
            "x(u) = A u(1&minus;u)<sup>2</sup> { &Sigma;<sub>k=1</sub><sup>8</sup> w<sub>k</sub> sin[2&pi;p<sub>k</sub>u + 2&pi;c<sub>k</sub>] \\over "
            "&Sigma;<sub>k=1</sub><sup>8</sup> w<sub>k</sub> }, &nbsp; A = 0.075 mm, &nbsp; D = 4.0 mm"
        ],
        3,
        "Equation 3 | Bounded recurrent-prime path law used by the simulation. All sums, phases and parameters are finite.",
        styles,
    ))
    prime_rows = [["p<sub>k</sub>", "six-term convergent", "phase c<sub>k</sub>"]]
    for term in terms:
        prime_rows.append([str(term["prime"]), term["convergent"], f'{term["phase_fraction"]:.7f}'])
    prime_table = Table(
        [[Paragraph(cell, styles["small"]) for cell in row] for row in prime_rows],
        colWidths=[1.0 * inch, 2.5 * inch, 2.3 * inch],
    )
    prime_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c9d6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([KeepTogether([prime_table, Paragraph(
        "<b>Table 1 | Prime phases fixed by square-root continued-fraction convergents.</b> These values are generated deterministically by the repository simulator.",
        styles["caption"],
    )])])
    story.extend([
        Paragraph("4. Implemented material-field surrogate", styles["heading"]),
        Paragraph(
            "The current implementation samples 52 lateral locations and 38 depth locations. Each cell receives a categorical material label and a scalar stress proxy. "
            "The drill envelope receives zero tissue stress. Outside this envelope, two compact kernels concentrate the field near the advancing tip and entrance interface. "
            "This is a 2-D deterministic field used for visualization and sensitivity checks; it is not a substitute for material calibration, contact mechanics, "
            "thermal modelling, crack propagation or a converged 3-D mesh study.",
            styles["body"],
        ),
    ])
    story.extend(equation(
        [
            "I<sub>drill</sub>(x,z;u) = 1{ 0 &le; z &le; D and |x&minus;x(u)| &le; 0.26 mm },",
            "s(x,z;u) = [1&minus;I<sub>drill</sub>] [0.08 + 1.15 exp(&minus;d<sub>tip</sub><sup>2</sup>) + 0.34 exp(&minus;d<sub>entry</sub><sup>2</sup>)] MPa,",
            "d<sub>tip</sub><sup>2</sup> = ((|x&minus;x(u)|)/0.48)<sup>2</sup> + ((z&minus;D)/0.72)<sup>2</sup>, "
            "&nbsp; d<sub>entry</sub><sup>2</sup> = ((|x&minus;x(u)|)/0.88)<sup>2</sup> + ((z&minus;0.12)/0.55)<sup>2</sup>"
        ],
        4,
        "Equation 4 | Finite 2-D stress-proxy field evaluated on the 38 x 52 grid. The dimensions in the kernels are millimetres.",
        styles,
    ))
    story.append(PageBreak())
    story.extend([
        Paragraph("5. Default simulation output", styles["heading"]),
        Paragraph(
            "One deterministic run was generated with the source defaults. Because the model contains no random sampling, repeated runs with the same code and parameters "
            "yield the same values. The peak tissue result excludes the drill envelope, consistent with Equation 4. Reported decimal values are implementation outputs and "
            "should be interpreted only within the surrogate's stated scope.",
            styles["body"],
        ),
    ])
    results = [
        ["Quantity", "Default output", "Interpretation"],
        ["Planned depth", f'{model["planned_depth_mm"]:.1f} mm', "Axial path extent"],
        ["Lateral amplitude A", f'{model["recurrence_amplitude_mm"]:.3f} mm', "Bound before terminal taper"],
        ["Grid", "38 x 52 cells", "Deterministic 2-D material field"],
        ["Maximum correction", f'{characteristics["maximum_lateral_correction_um"]:.2f} micrometres', "max |x(u)| along sampled path"],
        ["Entry / terminal offset", "0.00 / 0.00 micrometres", "Guaranteed by the taper"],
        ["Peak tissue stress proxy", f'{characteristics["peak_tissue_stress_mpa"]:.3f} MPa', "Maximum outside drill cells"],
        ["Peak tool stress proxy", f'{characteristics["peak_tool_stress_mpa"]:.2f} MPa', "Trajectory trace maximum"],
        ["Peak cutting-force proxy", f'{characteristics["peak_cutting_force_n"]:.2f} N', "Trajectory trace maximum"],
    ]
    result_table = Table(
        [[Paragraph(cell, styles["small"]) for cell in row] for row in results],
        colWidths=[2.05 * inch, 1.55 * inch, 2.2 * inch],
    )
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c9d6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([
        result_table,
        Paragraph("<b>Table 2 | Reproducible default outputs.</b> Values were produced by <font name='Courier'>RecurrentPrimeDrillPathEngine().simulate()</font>.", styles["caption"]),
        Paragraph("6. Verification and limitations", styles["heading"]),
        Paragraph(
            "The following algebraic checks are directly verifiable from the model: (i) the finite prime list has cardinality eight; (ii) every phase is a rational number "
            "obtained from a finite convergent; (iii) the normalization denominator is positive because p<sub>k</sub> &gt; 1; and (iv) x(0) = x(1) = 0. "
            "The last identity follows from the taper and does not depend on harmonic cancellation. These checks establish reproducibility of the path definition, "
            "not biomechanical validity.",
            styles["body"],
        ),
    ])
    story.extend(equation(
        [
            "x(0) = A&middot;0&middot;(1&minus;0)<sup>2</sup>R(0) = 0, &nbsp; "
            "x(1) = A&middot;1&middot;(1&minus;1)<sup>2</sup>R(1) = 0,",
            "R(u) = { &Sigma;<sub>k=1</sub><sup>8</sup> w<sub>k</sub> sin(2&pi;p<sub>k</sub>u+2&pi;c<sub>k</sub>) \\over "
            "&Sigma;<sub>k=1</sub><sup>8</sup> w<sub>k</sub> }, &nbsp; &Sigma;<sub>k=1</sub><sup>8</sup>w<sub>k</sub> &gt; 0"
        ],
        5,
        "Equation 5 | Endpoint invariance of the recurrent-prime perturbation.",
        styles,
    ))
    story.extend([
        Paragraph(
            "The results should not be extrapolated to tooth-specific morphology, enamel, dentin, cementum, pulp, periodontal ligament, irrigation, heat generation, "
            "burr wear, operator control, robotic positioning error or fracture risk. A clinically relevant study would require image-derived geometry, material "
            "calibration, mesh-convergence analysis, loading and thermal boundary conditions, uncertainty quantification, independent verification and appropriate "
            "regulatory validation.",
            styles["body"],
        ),
        Paragraph("7. Reproducibility", styles["heading"]),
        Paragraph(
            "The generator is self-contained apart from ReportLab and the local <font name='Courier'>drill_path_planner.py</font> simulation engine. "
            "Run <font name='Courier'>python3 dental_robot_surgery/generate_fea_cavity_preprint.py</font> from the repository root to regenerate this document. "
            "The source specifies the default depth, amplitude, finite prime set, continued-fraction length, grid and equations; no external data are required.",
            styles["body"],
        ),
        Paragraph("References", styles["heading"]),
    ])
    references = [
        "1. Hughes, T. J. R. <i>The Finite Element Method: Linear Static and Dynamic Finite Element Analysis</i>. Dover (2000).",
        "2. Zienkiewicz, O. C., Taylor, R. L. &amp; Zhu, J. Z. <i>The Finite Element Method: Its Basis and Fundamentals</i>, 7th edn. Elsevier (2013).",
        "3. Khinchin, A. Ya. <i>Continued Fractions</i>. Dover (1997).",
        "4. RecurrentPrimeDrillPathEngine. Local deterministic research simulation in <font name='Courier'>dental_robot_surgery/drill_path_planner.py</font> (accessed 24 October 2026).",
    ]
    story.extend(Paragraph(reference, styles["reference"]) for reference in references)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path.resolve()


if __name__ == "__main__":
    print(build_preprint())
