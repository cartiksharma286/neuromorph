import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Demographics_Finite_Math.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5*inch, rightIndent=0.5*inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=12, leading=18, alignment=TA_CENTER, fontName='Times-BoldItalic', spaceBefore=10, spaceAfter=10, textColor=colors.HexColor('#1a365d'))

    elements = []

    elements.append(Paragraph("Finite Mathematical Topologies and Quantum Surface Integrals for Strategic Demographic Expansion: A GDP-Optimized Growth Model", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Large-scale demographic shifts require rigorous mathematical scaffolding to ensure socioeconomic stability. In this report, we propose a unified finite mathematical model employing quantum phase-space geometries to coordinate a target 30% aggregate population increase across Canada. By utilizing mineral-ore probability distribution manifolds and finite sum absorption formulas, we establish predictive algorithms mapping population densities onto high-yield GDP vectors. This methodology maximizes labor allocation efficiency in critical mining corridors while strictly adhering to finite topological constraints."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("1. Introduction", heading_style))
    intro_txt = ("Strategic population densification necessitates optimizing regional carrying capacities against natural resource availability. We deploy a discrete, deterministic quantum surface approach, shifting from classical continuous diffusion to finite step absorption functions. This ensures precise calculation of regional intakes. The state space modeling projects a 30% deterministic multiplier overlaying standard stochastic growth metrics.")
    elements.append(Paragraph(intro_txt, body_style))

    elements.append(Paragraph("2. Finite Quantum Surface Integrals", heading_style))
    math1_txt = ("The probability amplitude framework initiates by constructing a base scalar distance metric R from a discrete magnetic mesh grid coordinates (x, y):")
    elements.append(Paragraph(math1_txt, body_style))
    elements.append(Paragraph("R<sub>i,j</sub> = \u221A(x<sub>i</sub><sup>2</sup> + y<sub>j</sub><sup>2</sup>)", math_style))
    
    math2_txt = ("Instead of generic probability densities, the mineral potential distribution explicitly factors an interference pattern mapped across the finite topology:")
    elements.append(Paragraph(math2_txt, body_style))
    elements.append(Paragraph("Z<sup>min</sup><sub>i,j</sub> = sin(R<sub>i,j</sub>) \u00B7 exp(-0.1 \u00B7 R<sub>i,j</sub><sup>2</sup>)", math_style))

    math3_txt = ("Squaring the amplitude yields the final phase-space density matrix for population probability D<sub>p</sub>:")
    elements.append(Paragraph(math3_txt, body_style))
    elements.append(Paragraph("D<sub>p</sub>(x<sub>i</sub>, y<sub>j</sub>) = | Z<sup>min</sup><sub>i,j</sub> |<sup>2</sup> \u00B7 min( M<sub>target</sub>, 1.5 )", math_style))

    elements.append(Paragraph("3. Finite Sum Allocation and GDP Synthesis", heading_style))
    math4_txt = ("Translating the surface integral into discrete demographic figures relies on finite proportional scaling. For any region k with a quantified mineral index m_k and a prior base population B_k, the updated sum N_k requires calculating the finite pool of unallocated growth:")
    elements.append(Paragraph(math4_txt, body_style))
    elements.append(Paragraph("\u0394<sub>pool</sub> = P<sub>total</sub> \u00B7 M<sub>target</sub> - \u03A3<sub>k=1</sub><sup>n</sup> B<sub>k</sub>", math_style))
    
    math5_txt = ("The assignment into finite discrete blocks dynamically factors the regional mineral weighting:")
    elements.append(Paragraph(math5_txt, body_style))
    elements.append(Paragraph("N<sub>k</sub> = 1.05 \u00B7 B<sub>k</sub> + \u0394<sub>pool</sub> \u00B7 ( m<sub>k</sub> / \u03A3<sub>j=1</sub><sup>n</sup> m<sub>j</sub> )", math_style))

    math6_txt = ("Subsequently, the scalar mapping onto projected Gross Domestic Product per capita (G_k) establishes strict economic boundary conditions. Using finite intercepts:")
    elements.append(Paragraph(math6_txt, body_style))
    elements.append(Paragraph("G<sub>k</sub> = 55000 + m<sub>k</sub> \u00B7 35000", math_style))

    elements.append(Paragraph("4. Conclusion", heading_style))
    conclusion_txt = ("The transformation of stochastic demographic expansion into a deterministic quantum phase-space integration allows robust algorithmic control over multi-regional policy. By applying finite discrete operators directly to resource vectors, labor mobility perfectly overlays economic stimulus clusters, theoretically preserving capital stability amidst extreme 30% demographic injection multipliers.")
    elements.append(Paragraph(conclusion_txt, body_style))

    doc.build(elements)
    print(f"Successfully generated {pdf_path}")

if __name__ == '__main__':
    generate_pdf()