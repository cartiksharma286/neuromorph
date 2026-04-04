import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem

def generate_detailed_grant():
    output_path = os.path.join(os.path.dirname(__file__), "NSERC_CREATE_Advanced_MRI_Robotics_Grant.pdf")
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=A4, 
        leftMargin=2.5*cm, rightMargin=2.5*cm, 
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1A5276"), alignment=TA_CENTER, spaceAfter=15)
    grant_subtitle = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, textColor=colors.gray, spaceAfter=20)
    h1_style = ParagraphStyle('H1', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2C3E50"), spaceBefore=15, spaceAfter=10)
    h2_style = ParagraphStyle('H2', parent=styles['Heading3'], fontSize=12, textColor=colors.HexColor("#C0392B"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle('BodyJ', parent=styles['Normal'], fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=10)
    math_style = ParagraphStyle('Math', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=11, alignment=TA_CENTER, textColor=colors.HexColor("#2471A3"), spaceBefore=10, spaceAfter=15)
    
    story = []
    
    # Header
    story.append(Paragraph("<b>NSERC CREATE Grant Application: Comprehensive Proposal</b>", title_style))
    story.append(Paragraph("Integrating Next-Generation MRI Pulse Sequences, Quantum Machine Learning, and Optimal Surgical Robotics", grant_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#D5D8DC"), spaceBefore=10, spaceAfter=20))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "Supported by groundbreaking theories published in high-impact journals (Nature Neuroscience, Nature Machine Intelligence), this NSERC CREATE training ecosystem aims to pioneer the next frontier of medical physics and intraoperative neurosurgery. By converging finite mathematical pulse sequence optimization, Quantum Machine Learning (QML) Ansatz models, and 6-Degrees-of-Freedom (6-DOF) surgical robotics, this program will generate a cohort of Highly Qualified Personnel (HQP) capable of developing precision, sub-millimeter surgical interventions driven by state-of-the-art MRI reconstructions.", body_style))
    
    # 2. Theoretical Framework & Nature-Backed Science
    story.append(Paragraph("2. Theoretical Framework (Nature Publications Integration)", h1_style))
    story.append(Paragraph(
        "Recent <i>Nature</i> literature highlights the necessity of dynamic, model-driven MRI acquisition overriding static sequences. By computing Riemannian manifolds of cognitive decline and physiological anomalies (e.g., Tau tangles), we direct surgical robotics with unparalleled mathematical precision. Our HQP will be trained to implement and test 40 novel pulse sequences leveraging rigorous finite difference and differential geometry calculations.", body_style))
    
    # 3. Finite Mathematics & Pulse Sequence Optimization
    story.append(Paragraph("3. Module A: Finite Math in Pulse Sequence Optimizations", h1_style))
    story.append(Paragraph(
        "Traditional MRI sequences suffer from Specific Absorption Rate (SAR) limits and spatial artifacts. Our program trains HQP in formulating pulse sequence generation as an Optimal Control problem. The objective function minimizes RF energy while forcing the magnetization vector <b>M</b>(t) to match the target <b>M</b><sub>target</sub>:", body_style))
    story.append(Paragraph(
        "J(<b>B</b><sub>1</sub>) = &int; ||<b>B</b><sub>1</sub>(t)||<sup>2</sup> dt + &lambda; ||<b>M</b>(T) - <b>M</b><sub>target</sub>||<sup>2</sup>", math_style))
    story.append(Paragraph(
        "To achieve finite pulse generation, trainees solve the discretized Bloch equations matrix recursively:", body_style))
    story.append(Paragraph(
        "<b>M</b><sub>k+1</sub> = e<sup>&Delta;t <b>A</b>(<b>B</b><sub>1,k</sub>, <b>G</b><sub>k</sub>)</sup> <b>M</b><sub>k</sub> + (I - e<sup>&Delta;t <b>A</b></sup>) <b>A</b><sup>-1</sup> <b>C</b>", math_style))
    story.append(Paragraph(
        "Under this theoretical limit, the 40 proprietary sequences are mathematically calibrated to dynamically adjust bandwidths &Delta;&omega; corresponding precisely to non-linear neuro-tissues.", body_style))

    # 4. Quantum Machine Learning
    story.append(Paragraph("4. Module B: Quantum Machine Learning (QML) Driven Diagnostics", h1_style))
    story.append(Paragraph(
        "QML provides exponential speedups in processing volumetric NIfTI data and complex biomarker segmentations. Training integrates Quantum Variational Circuits (Ansatz) optimized locally using parameterized phase shifts <i>&theta;</i>:", body_style))
    story.append(Paragraph(
        "|&Psi;(&theta;)> = &Pi;<sub>i=1</sub><sup>N</sup> U<sub>i</sub>(&theta;<sub>i</sub>) |0>", math_style))
    story.append(Paragraph(
        "The HQP will utilize gradient descent on real quantum hardware pipelines to update probabilistic clusters mapping the patient's pathological manifold, tracking parameter optimization via: <i>&theta;<sub>n+1</sub> = &theta;<sub>n</sub> - &eta; &nabla;C(&theta;)</i>.", body_style))

    # 5. Surgical Robotics
    story.append(Paragraph("5. Module C: Sub-Millimeter Precision Robotics for Neurosurgery", h1_style))
    story.append(Paragraph(
        "Surgical interventions rely on intraoperative MRI (iMRI). We map the coordinate geometry of the robotic stereotactic manipulator into the QML-derived Riemannian manifold. Inverse kinematics algorithms guide the surgical effector over geodesic pathways:", body_style))
    story.append(Paragraph(
        "d<sup>2</sup>x<sup>&alpha;</sup>/dt<sup>2</sup> + &Gamma;<sup>&alpha;</sup><sub>&beta;&gamma;</sub> (dx<sup>&beta;</sup>/dt)(dx<sup>&gamma;</sup>/dt) = &epsilon;(t)", math_style))
    story.append(Paragraph(
        "Where &Gamma;<sup>&alpha;</sup><sub>&beta;&gamma;</sub> represents the Christoffel symbols determined by tissue elasticity and avoidance zones (e.g., major cerebral vasculature). The robot's joint velocity <code>q&#x0307;</code> is computed dynamically against the Jacobian tracking error:", body_style))
    story.append(Paragraph(
        "&Delta;q = <b>J</b><sup>&dagger;</sup>(q) (x<sub>target</sub> - f(q)) + (I - <b>J</b><sup>&dagger;</sup><b>J</b>) z<sub>null</sub>", math_style))
    story.append(Paragraph(
        "This ensures minimal tissue deformation, nullifying tremor-based error, a direct application translating physics to surgical oncology and DBS electrode implantations.", body_style))

    # 6. Training & Impact
    story.append(Paragraph("6. Program Implementation and HQP Skill Acquisition", h1_style))
    story.append(Paragraph(
        "The CREATE ecosystem will foster interdisciplinary skills. Graduates will be fluent in advanced topological math, quantum-circuit building, MRI scanner interfacing (GE/Siemens/Siemens IDEA), and mechanical impedance control for robotics.", body_style))
    story.append(Paragraph("<b>Outcomes:</b> Accelerated commercialization of the 40 next-gen pulse sequences, integration into existing neurosurgical platforms (e.g., Monteris, Medtronic), and a newly forged Canadian vanguard in quantum-robotic medical biophysics.", body_style))

    doc.build(story)
    print(f"Detailed NSERC Grant Publication successfully generated at {output_path}")

if __name__ == '__main__':
    generate_detailed_grant()
