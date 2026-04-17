#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Pulse_Sequences_Finite_Math.pdf')
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

    elements.append(Paragraph("Optimized Quantum Machine Learning Pulse Sequences: A Finite Mathematical Framework for Neuromorphic Imaging", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Recent advancements in quantum algorithm-driven radiofrequency pulse sequences necessitate robust theoretical foundations. Here we present a set of finite mathematical formulations to formalize the most recent dual spin-echo generated waveforms. We utilize discrete spatial Laplacians for inflection-point detection and Euler integration methods. Crucially, we investigate the amplitude modulation observed in the last two pulse sequences synthesized in our environment, establishing a definitive discrete topological framework for neurovascular imaging architectures."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Finite Mathematical Analysis of the Last Two Pulse Sequences", 
         "The latest phase-amplitude arrays reflect a stable progression of optimized RF transitions. Utilizing the discrete time formulation, the RF envelope B1 is modeled over N finite steps:",
         "B<sub>1,k+1</sub> = B<sub>1,k</sub> + &alpha; &middot; cos(&omega; &middot; t<sub>k</sub>) &middot; &Delta;t",
         "where &alpha; is the quantum control scalar. This formulation is explicitly applied to the last two pulse variations (custom_sequence and custom_sequenceA) recently executed, indicating stable performance limits."),
        
        ("2. Finite Difference Laplacian Operators", 
         "To compute optimal readout intervals and structural gradients during the sequence execution, the spatial Laplacian is approximated finitely:",
         "&nabla;<sup>2</sup>M<sub>i,j</sub> &approx; [ M<sub>i+1,j</sub> + M<sub>i-1,j</sub> + M<sub>i,j+1</sub> + M<sub>i,j-1</sub> - 4M<sub>i,j</sub> ] / h<sup>2</sup>",
         "where M represents the discrete transverse magnetization. This prevents the computational instability and singularities characteristic of continuum gradient-descent."),
        
        ("3. Diffeomorphic Mappings using Discrete Euler Integration", 
         "The spatial mapping &phi; of the pulse responses maintains topological tissue integrity, ensuring no physical folding occurs. A stationary finite velocity field v(x) drives the diffeomorphism:",
         "&phi;<sub>k+1</sub>(x) = &phi;<sub>k</sub>(x) + (1/N) &middot; v(&phi;<sub>k</sub>(x))",
         "The positive determinant Jacobian generated strictly tracks structural deformation induced by the latest temporal TR/TE settings.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()
