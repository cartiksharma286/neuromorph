import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem

def generate_grant():
    output_path = os.path.join(os.path.dirname(__file__), "NSERC_CREATE_Pulse_Sequences.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    body_style = styles['Normal']
    body_style.spaceAfter = 10
    
    story = []
    
    # Title Block
    story.append(Paragraph("<b>NSERC CREATE Grant Application</b>", title_style))
    story.append(Paragraph("<font color='gray'>Program: Collaborative Research and Training Experience (CREATE)</font>", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Project Title: Advanced Training in Next-Generation MRI: Commercialization of 40 Novel Pulse Sequences for Healthcare and Geospatial Sensing", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color="black", spaceBefore=10, spaceAfter=20))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", subheading_style))
    story.append(Paragraph(
        "This NSERC CREATE program aims to bridge the gap between theoretical quantum magnetic resonance research and clinical/geospatial application. "
        "By leveraging a comprehensive library of 40 proprietary, mathematically optimized MRI and NMR pulse sequences—ranging from statistical manifold distributions for dementia tracking to hyperpolarized reconstructions and quantum measure theory for geological oil/ore reservoirs—this program will train the next generation of Highly Qualified Personnel (HQP). "
        "Trainees will gain interdisciplinary expertise in quantum physics, software engineering, RF hardware, and commercial translation.", body_style))
    
    # 2. Excellence of the Team of Researchers
    story.append(Paragraph("2. Excellence of the Team of Researchers", subheading_style))
    story.append(Paragraph(
        "Led by Cartik Sharma and the NeuroMorph ecosystem, the applicant pool represents a unique convergence of quantum computing, medical biophysics, and mathematical modeling. "
        "The team has demonstrated exceptional ability in generating high-impact algorithmic sequences for complex neuromodulation (e.g., Deep Brain Stimulation stage gating via Radon-Nikodym mapping) and geospatial Continued Fraction mapping.", body_style))
    
    # 3. Merit of the Proposed Training Program
    story.append(Paragraph("3. Merit of the Proposed Training Program", subheading_style))
    story.append(Paragraph(
        "The core of the program will revolve around the testing, validation, and eventual health/industrial qualification of 40 distinct pulse sequences. Trainees will engage in advanced laboratory and computational modules:", body_style))
    
    components = [
        ListItem(Paragraph("<b>Pulse Sequence Programming:</b> Hands-on development using custom simulators, optimizing RF sweeps and gradients.", body_style)),
        ListItem(Paragraph("<b>Quantum Machine Learning (QML):</b> Translating QML Ansatz models and continued fraction mathematics into actionable imaging and modulation protocols.", body_style)),
        ListItem(Paragraph("<b>Clinical and Geospatial Translation:</b> Porting sequences to identify mineral/hydrocarbon reservoirs or neural degenerative anomalies like Tau and Amyloid-beta oligomers.", body_style))
    ]
    story.append(ListFlowable(components, bulletType='bullet', leftIndent=15))
    story.append(Spacer(1, 10))
    
    # 4. Need for HQP in the Field
    story.append(Paragraph("4. Need for Highly Qualified Personnel (HQP)", subheading_style))
    story.append(Paragraph(
        "Canada's advanced sensing and medical imaging sectors face a critical shortage of engineers and physicists capable of bridging abstract mathematical concepts with radiofrequency (RF) hardware implementation. "
        "This CREATE program addresses the national deficit by generating industry-ready innovators fully proficient in creating deployable IP around next-generation sensors.", body_style))
    
    # 5. Program Management
    story.append(Paragraph("5. Program Management and Sustainability", subheading_style))
    story.append(Paragraph(
        "The program will enforce a strict, multi-stage gating process for sequence validation. "
        "Intellectual property generated surrounding the 40 pulse sequences will be co-licensed to partner medical facilities and natural resource coalitions, ensuring long-term sustainability of the CREATE training ecosystem and continuing Canadian leadership in quantum sensing.", body_style))
    
    doc.build(story)
    print(f"Publication successfully generated at {output_path}")

if __name__ == '__main__':
    generate_grant()