#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Cellular_Tracking_Finite_Math.pdf')
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

    elements.append(Paragraph("Finite Mathematical Formulation of Discrete Cellular Tracking and Lineage Inference", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: High-fidelity spatiotemporal tracking of cellular morphology and division requires robust discrete mathematics. In this report, we detail the finite equations driving our automated cellular tracking framework. By formulating discrete state-space estimators, graph-theoretic global assignments, and finite morphological representations, we avoid cumulative integration errors and maintain deterministic lineage histories across arbitrarily deep imaging sequences."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Discrete State-Space Kinematics", 
         "Cellular centroids and velocities are modeled using a finite Markov decision process. For a given cell i, its kinematic state vector x at discrete time step k evolves linearly according to the finite transition matrix A:",
         "x<sub>i,k+1</sub> = A &middot; x<sub>i,k</sub> + w<sub>k</sub>",
         "where w_k represents discrete Gaussian process noise. The transition matrix restricts the kinematics to locally rigid translations, ensuring numerical bounds during prolonged tracking sequences."),
        
        ("2. Finite Bipartite Assignment", 
         "Assigning detections at time k to existing tracks from time k-1 is modeled as a minimum-weight bipartite matching problem. We construct a finite cost matrix C where element C_ij is the spatial Mahalanobis distance between track i and detection j. The optimal discrete assignment is found via:",
         "&Sigma;<sub>i,j</sub> C<sub>i,j</sub> X<sub>i,j</sub> &rarr; min,  subject to &Sigma;<sub>i</sub> X<sub>i,j</sub> = 1, &Sigma;<sub>j</sub> X<sub>i,j</sub> = 1",
         "where X_ij &isin; {0, 1} is the Boolean assignment matrix. This strict discrete combinatorics prevents track merging and splitting ambiguities."),
        
        ("3. Discrete Morphological Curvature", 
         "The boundaries of individual cells are extracted as discrete polygons with N vertices {p_1, p_2, ..., p_N}. To identify morphological inflection states indicative of cellular differentiation or division, discrete approximation of the contour curvature &kappa; at vertex n is computed:",
         "&kappa;<sub>n</sub> &approx; 2 &middot; det(p<sub>n+1</sub> - p<sub>n</sub>, p<sub>n-1</sub> - p<sub>n</sub>) / |p<sub>n+1</sub> - p<sub>n-1</sub>|<sup>3</sup>",
         "This finite difference strategy operates strictly over the set of ordered pixel coordinates, rendering it invariant to digital rasterization noise."),
         
        ("4. Graph-Theoretic Lineage Trees", 
         "Cell division events construct a directed acyclic graph (DAG) where nodes represent discrete states in time. The finite graph G = (V, E) evolves under the constraint that the out-degree of any non-terminal vertex v &isin; V satisfies:",
         "deg<sup>+</sup>(v) &isin; {1, 2}",
         "where an out-degree of 2 strictly enforces mitotic division conservation. This finite topological rule encapsulates the biological reality of cellular bifurcation.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("5. Conclusion", heading_style))
    elements.append(Paragraph("Encoding biological tracking phenomena strictly through finite mathematics—including Boolean assignment combinatorics and directed non-cyclic graphs—yields an unconditionally stable discrete framework tailored for longitudinal high-throughput microscopy.", body_style))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()
