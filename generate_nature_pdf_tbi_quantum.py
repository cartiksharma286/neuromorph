#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_tbi_nature_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_TBI_Quantum_Repair.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5*inch, rightIndent=0.5*inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=11, leading=16, alignment=TA_CENTER, fontName='Times-Italic', spaceBefore=6, spaceAfter=6, textColor=colors.HexColor('#1a365d'))

    elements = []
    elements.append(Paragraph("Quantum Repair of TBI via Continued Fractions and Mock Modular Forms: A Finite Math Framework", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = """
    Abstract: We present a quantum-theoretic protocol for traumatic brain injury (TBI) repair, leveraging continued fraction expansions and mock modular forms to accelerate neural convergence. Our approach models synaptic connectivity as a finite sequence, with each repair step governed by discrete mathematical transformations. This framework enables rapid restoration of network function, validated by finite math equations and quantum statistical metrics.
    """
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Continued Fraction Expansion for Neural Repair", 
         "The neural repair process is modeled as a continued fraction expansion, where each term represents a discrete synaptic update:",
         "$$x = a_0 + \\cfrac{1}{a_1 + \\cfrac{1}{a_2 + \\cfrac{1}{a_3 + \ldots}}}$$",
         "Here, $a_k$ are integer coefficients determined by the current neural state. The convergence of this expansion reflects the restoration of healthy connectivity."),

        ("2. Mock Modular Form Correction", 
         "To accelerate convergence, we introduce a mock modular form correction at each step, inspired by the eta function:",
         "$$\\eta(n) = e^{-\\pi n/12} \\cdot \\sin(\\pi n/2)$$",
         "The total synaptic boost at step $k$ is given by:",
         "$$\\Delta S_k = (\\log(a_k+1) \\cdot 2.5) + \\frac{10}{a_k+1} + 8 \\cdot \\eta(a_k)$$",
         "This correction ensures rapid, stable convergence to the target network state."),

        ("3. Finite Math Convergence Guarantee", 
         "The repair protocol is bounded by a finite number of steps $N$, with the final connectivity:",
         "$$C_{final} = C_{base} + \\sum_{k=1}^N \\Delta S_k$$",
         "where $C_{base}$ is the initial connectivity. The process halts when $C_{final} \\geq 100$% or the sequence converges."),

        ("4. Quantum Statistical Metrics", 
         "We validate the repair using quantum statistical measures, including the Ramanujan congruence ratio and KAM stability index:",
         "$$R = \\frac{\\text{# congruent links}}{\\text{total links}}, \quad K = 1 - \\frac{1}{N} \\sum_{k=1}^N |w_k - \\varphi|$$",
         "where $w_k$ are synaptic weights and $\\varphi$ is the golden ratio.")
    ]

    for section in sections:
        for i, part in enumerate(section):
            if i == 0:
                elements.append(Paragraph(part, heading_style))
            elif part.startswith("$$"):
                elements.append(Paragraph(part.replace("$$", ""), math_style))
            else:
                elements.append(Paragraph(part, body_style))
        elements.append(Spacer(1, 0.1*inch))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == "__main__":
    generate_tbi_nature_pdf()
