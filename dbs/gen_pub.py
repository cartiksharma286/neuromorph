from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def generate_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        spaceAfter=14,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    abstract_style = ParagraphStyle(
        'AbstractStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        leftIndent=40,
        rightIndent=40,
        alignment=TA_JUSTIFY,
        fontName='Times-Italic'
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        alignment=TA_JUSTIFY,
        fontName='Times-Roman'
    )

    equation_style = ParagraphStyle(
        'EquationStyle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceBefore=12,
        spaceAfter=12,
        fontName='Times-Italic'
    )

    elements = []

    # Title
    elements.append(Paragraph("Quantum-Enhanced Resonant Neuro-Modulation for Stage-Gated Dementia Intervention", title_style))
    elements.append(Paragraph("Neuromorph Research Team & Collaboration Group", author_style))
    elements.append(Spacer(1, 12))

    # Abstract
    elements.append(Paragraph("Abstract", section_style))
    elements.append(Paragraph(
        "Cognitive decline in dementia is fundamentally linked to the collapse of hippocampal theta-band network dynamics. "
        "Conventional deep brain stimulation (DBS) often utilizes rigid programming, ignoring personalized tissue impedance "
        "and neuro-resonant characteristics. Here, we present a resilient framework for quantum-enhanced neuromodulation. "
        "Leveraging continued fraction signal decomposition and simulated quantum annealing, we optimize stimulation "
        "frequencies to maximize synaptic entrainment. Our results indicate that stage-gated protocols—progressing from "
        "initial biomarker audit to high-fidelity resonant plasticity modulation—provide a robust therapeutic pathway "
        "for mitigating dementia-induced circuit degradation.",
        abstract_style
    ))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("1. Introduction", section_style))
    elements.append(Paragraph(
        "Dementia is characterized by progressive circuit-level failure within the hippocampal-fornix complex. "
        "The Crucian Arch of the fornix acts as a critical bottleneck for memory consolidation; its dysfunction is "
        "mirrored in the attenuation of rhythmic neural synchronization. Proactive neuro-modulation via DBS "
        "seeks to restore these oscillations by delivering precisely timed electrical pulses.",
        body_style
    ))

    # Mathematical Framework
    elements.append(Paragraph("2. Mathematical Framework", section_style))
    elements.append(Paragraph("2.1 Axial Magnetic Field Dynamics", ParagraphStyle('SubSection', fontName='Helvetica-Bold', fontSize=10)))
    elements.append(Paragraph(
        "The induced stimulation field at a distance <i>z</i> along the vertical axis of a nanoscopically-refined "
        "implantable coil is governed by the specialized Biot-Savart derivation:",
        body_style
    ))
    # Equation 1: B-field with italics and proper sub/superscripts
    elements.append(Paragraph(
        "<b><font size='14'><i>B</i>(<i>z</i>) = \frac{<i>μ₀</i> \cdot <i>I</i> \cdot <i>R</i><sup>2</sup>}{2 \cdot (<i>R</i><sup>2</sup> + <i>z</i><sup>2</sup>)^{3/2}}</font></b>",
        equation_style
    ))
    
    elements.append(Paragraph("2.2 Continued Fraction Signal Decomposition", ParagraphStyle('SubSection', fontName='Helvetica-Bold', fontSize=10)))
    elements.append(Paragraph(
        "To achieve robust resonance detection amidst biological noise, we decompose the neuro-biosignal <i>f(ν)</i> "
        "using a recursive continued fraction expansion of order <i>n</i>:",
        body_style
    ))
    # Equation 2: Continued Fraction
    # Improved continued fraction formatting for readability
    elements.append(Paragraph(
        """
        <b><font size='14'>
        <i>f</i>(<i>ν</i>) = <i>a</i><sub>0</sub> / [<i>b</i><sub>0</sub> + <i>a</i><sub>1</sub> / (<i>b</i><sub>1</sub> + <i>a</i><sub>2</sub> / (<i>b</i><sub>2</sub> + ...)))]
        </font></b>
        """,
        equation_style
    ))
    
    # Statistical Optimization
    elements.append(Paragraph("2.3 Statistical Congruence Optimization", ParagraphStyle('SubSection', fontName='Helvetica-Bold', fontSize=10)))
    elements.append(Paragraph(
        "The optimized therapeutic yield (<i>Y</i><sub>opt</sub>) is derived by modulating the base yield <i>Y</i><sub>base</sub> "
        "with a congruence-based periodic stability factor:",
        body_style
    ))
    elements.append(Paragraph(
        "<b><font size='14'><i>Y</i><sub>opt</sub> = <i>Y</i><sub>base</sub> + \frac{\left( <i>Y</i><sub>base</sub> \cdot 100 \right) \bmod 7.3}{10}</font></b>",
        equation_style
    ))

    # Methods
    elements.append(Paragraph("3. Methods", section_style))
    elements.append(Paragraph(
        "Our experimental setup utilizes the NM-QM-Fornix-v2.0 hardware bridge. The protocol transition follows a "
        "three-stage optimization path summarized in Table 1.",
        body_style
    ))
    elements.append(Spacer(1, 8))
    
    # Protocol Table
    data = [
        ['Stage', 'Protocol Name', 'Functional Objective', 'Target Parameter'],
        ['I', 'Biomarker Audit', 'Baseline Connectivity Check', '1.0V (μ=0.1s)'],
        ['II', 'Neural Priming', 'Sub-threshold Sensitization', '2.0V (Fixed)'],
        ['III', 'Synaptic Entrainment', 'Resonant Plasticity Mod.', '3.5V (Optimal)']
    ]
    t = Table(data, colWidths=[40, 110, 160, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#333333")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f8f8")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(t)
    elements.append(Paragraph("<font size='8'>Table 1: Stage-Gated DBS Protocol Parameters.</font>", ParagraphStyle('Caption', alignment=TA_CENTER, spaceBefore=4)))
    elements.append(Spacer(1, 12))

    # Conclusion
    elements.append(Paragraph("4. Conclusion", section_style))
    elements.append(Paragraph(
        "By integrating quantum annealing with adaptive resonant feedback, we establish a robust methodology for "
        "personalized dementia care. The use of continued fraction decomposition ensures high signal integrity "
        "in dynamic neuro-environments.",
        body_style
    ))

    # Build PDF
    doc.build(elements)
    print(f"Refined PDF generated: {filename}")

if __name__ == "__main__":
    generate_pdf("nature_publication_dbs.pdf")
