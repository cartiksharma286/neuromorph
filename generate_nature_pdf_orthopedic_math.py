#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Orthopedic_Imaging_Finite_Math.pdf')
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

    elements.append(Paragraph("Advanced Orthopedic Pulse Sequences and Conformal Coils: A Finite Mathematical Framework", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Specialized orthopedic MRI necessitates precision geometries to capture the nuances of articular cartilage, cortical bone, and dense connective tissue interfaces. In this communication, we introduce a novel finite mathematical framework leveraging discrete difference operators and polar conformal coordinate mappings to optimize signal extraction. By combining custom sequences, such as the FiniteMathElbowPulse, with topographically aligned multi-channel arrays (ElbowJointCoil, OrthopedicKneeCoil), we observe preserved boundary topologies and enhanced signal-to-noise ratios (SNR)."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Conformal Polar Mappings in Orthopedic Coil Design", 
         "Standard volume arrays produce suboptimal sensitivity profiles around asymmetrical joint hubs such as the elbow or shoulder. Our specialized joint coils utilize polar coordinates dynamically mapped to the joint radius vector. The sensitivity field is defined functionally as:",
         "S(x,y) = &Beta; &middot; exp(-((x - r &middot; cos(&theta;))<sup>2</sup> + (y - r &middot; sin(&theta;))<sup>2</sup>) / 2&sigma;<sup>2</sup>) &middot; exp(i&phi;)",
         "This phase-modulated profile ensures deep penetration into the ulnar groove and meniscal interfaces without saturation artifact constraints."),
        
        ("2. Finite Difference Laplacians in Pulse Envelopes", 
         "To compute localized boundary edges dynamically within the RF spin-echo sequences, we synthesize discrete difference Laplacians directly into the signal formation vector. The transverse magnetization M computes as a spatial Laplacian approximated finitely across the domain lattice:",
         "&nabla;<sup>2</sup>M<sub>i,j</sub> &approx; M<sub>i+1,j</sub> + M<sub>i-1,j</sub> + M<sub>i,j+1</sub> + M<sub>i,j-1</sub> - 4M<sub>i,j</sub>",
         "Implementing this discrete topological diff matrix removes continuous differentiation approximations and enables perfect edge preservation corresponding to tissue interfaces (such as bone-cartilage transitions)."),
        
        ("3. Dynamic Amplitude Modulation", 
         "Finally, signal enhancement relies on modulating the final baseline echo via the discrete Laplacian field mapped through a quantum control scalar &alpha;. The final observed magnitude operates under:",
         "M<sub>opt</sub> = | M<sub>base</sub> + &alpha; &middot; &nabla;<sup>2</sup>M<sub>base</sub> |",
         "When applied to the FiniteMathElbowPulse sequence, this results in significant structural delineation algorithms with empirical validation showcasing amplified joint visualization.")
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