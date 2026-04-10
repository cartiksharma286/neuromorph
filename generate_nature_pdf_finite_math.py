#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Fullerene_Coils_Finite_Math.pdf')
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

    elements.append(Paragraph("Finite Mathematical Foundations of Diffeomorphic Image Registration and Inflection-Point MR Thermometry with C60 Fullerene Arrays", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Robust tracking of thermal ablation boundaries requires precise spatiotemporal gradient calculations. In this report, we detail the finite mathematical and discrete numerical methods underpinning our novel C60 fullerene coil thermometry pipeline. We explicitly formulate the discrete Laplacian operators for inflection-point detection, the finite Euler integration steps for stationary velocity field diffeomorphisms, and the discrete-time Hebbian signal amplification updates. These finite implementations guarantee computational stability and strict topological compliance during continuous MRI resonance reconstruction."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Finite Difference Laplacian for Inflection Points", 
         "To locate the ablation margin, we identify the inflection boundaries of the temperature-induced phase shift. This is achieved by computing the discrete spatial Laplacian. Using a finite difference method on the 2D pixel grid, the second spatial derivative is evaluated as:",
         "&nabla;<sup>2</sup>T<sub>i,j</sub> &approx; [ T<sub>i+1,j</sub> + T<sub>i-1,j</sub> + T<sub>i,j+1</sub> + T<sub>i,j-1</sub> - 4T<sub>i,j</sub> ] / h<sup>2</sup>",
         "where h represents the discrete spatial resolution (voxel size). The zero-crossings of this operator definitively map the maximal gradient of the thermal boundary."),
        
        ("2. Discrete Diffeomorphic Flow (Euler Integration)", 
         "Motion compensation is enforced via a stationary velocity field (SVF), denoted v(x). To ensure the transformation &phi; remains a diffeomorphism (no tissue folding), we compute the flow via discrete Euler integration over N finite steps:",
         "&phi;<sub>k+1</sub>(x) = &phi;<sub>k</sub>(x) + (1/N) &middot; v(&phi;<sub>k</sub>(x))",
         "for k = 0, 1, ..., N-1, where &phi;<sub>0</sub>(x) = x. The discrete Jacobian determinant |J| is strictly maintained above zero, ensuring stable bijective mapping."),
        
        ("3. Discrete-Time Hebbian Amplification", 
         "We isolate the ablation zone using an unsupervised Hebbian gain modifier. Implemented as a discrete-time dynamical system, the gain g at time step t+1 for a given voxel's magnitude M is updated using the discrete rule:",
         "g<sub>t+1</sub> = g<sub>t</sub> + &eta; &middot; M<sub>t</sub> &middot; g<sub>t</sub>",
         "where &eta; is the finite learning rate. Voxels below the background threshold are damped via exponential decay, segregating target tissues structurally."),
         
        ("4. C60 Fullerene Topology as a Finite Point Set", 
         "The 60-element coil array is modeled as a discrete non-Euclidean geometry mapped to the vertices of a truncated icosahedron. The finite set V of coil nodes is generated using the golden ratio &Phi; = (1+&radic;5)/2:",
         "V = { (0, &plusmn;1, &plusmn;3&Phi;), (&plusmn;1, &plusmn;3&Phi;, 0), ... }",
         "The signal is combined using a discrete sum-of-squares (SoS) across the finite sensor set n &isin; {1..60}, ensuring uniform signal-to-noise coverage.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("5. Conclusion", heading_style))
    elements.append(Paragraph("By employing explicit finite mathematical transforms—ranging from discrete Euler manifolds to finite-difference topological locators—we ensure that our neurovascular ablation metrics are computationally tractable, physically accurate, and unconditionally stable during execution.", body_style))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()