#!/usr/bin/env python3
"""
Generates a Nature-style preprint PDF summarizing the finite mathematical
framework behind the Pelvic Floor Implant BEM / HPC / NVQLink Design Studio:
boundary element formulation, HPC Amdahl scaling, NVQLink hybrid quantum
acceleration, chamfer geometry, and continued-fraction manifold repair.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors


def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Pelvic_Floor_BEM_HPC_NVQLink.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12,
                                  alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14,
                                     alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5 * inch,
                                     rightIndent=0.5 * inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8,
                                    spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14,
                                 alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=11, leading=16,
                                 alignment=TA_CENTER, fontName='Times-Italic', spaceBefore=6, spaceAfter=6,
                                 textColor=colors.HexColor('#1a365d'))

    elements = []
    elements.append(Paragraph(
        "Boundary Element Simulation of Chamfered Pelvic Floor Implants on HPC/NVQLink "
        "Hybrid Infrastructure: A Continued-Fraction Manifold Repair Framework", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2 * inch))

    abstract = (
        "Abstract: We present a finite mathematical framework for the design and biomechanical "
        "validation of chamfered pelvic floor reconstruction implants. Implant boundary geometry is "
        "generated with a parametric chamfer profile and repaired to a closed two-manifold using "
        "discrete Euler-characteristic and Gauss&ndash;Bonnet consistency checks. A continued-fraction "
        "convergent expansion of an irrational rotation number parameterizes a quasi-periodic curvature "
        "blending field across the chamfer-to-body transition, analogous to KAM-type irrational winding, "
        "eliminating periodic stress risers. The resulting closed-manifold surface is discretized into "
        "constant boundary panels and solved via the Boundary Element Method (BEM) using the Kelvin "
        "fundamental solution under cyclic Valsalva pressure loading. The dense BEM influence system is "
        "distributed across an HPC cluster using SLURM-scheduled MPI domain decomposition, with strong-scaling "
        "behaviour projected via Amdahl's law, and further accelerated through a simulated NVQLink hybrid "
        "GPU&ndash;QPU interconnect implementing an HHL-class quantum linear-solver preconditioner."
    )
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2 * inch))

    sections = [
        ("1. Chamfer Geometry and Manifold Repair",
         "The implant boundary is offset inward by the chamfer width w<sub>c</sub> and projected to depth "
         "along the bevel angle &theta;<sub>c</sub>, giving the chamfer depth:",
         "d<sub>c</sub> = w<sub>c</sub> &middot; tan(&theta;<sub>c</sub>)",
         "Mesh repair enforces manifold closure by fan-triangulating detected boundary loops and verifying "
         "the discrete Euler characteristic &chi; = V &minus; E + F against the Gauss&ndash;Bonnet identity "
         "(1/2&pi;)&oint;<sub>M</sub> K dA + &oint;<sub>&part;M</sub> &kappa;<sub>g</sub> ds = &chi;, "
         "confirming genus-0 closure of the repaired implant surface."),

        ("2. Continued-Fraction Quasi-Periodic Curvature Blending",
         "The regular continued fraction of a rotation number x (e.g. the golden ratio &phi;, chosen for its "
         "maximally irrational Diophantine properties) is expanded as:",
         "x = a<sub>0</sub> + 1/(a<sub>1</sub> + 1/(a<sub>2</sub> + 1/(a<sub>3</sub> + &hellip;)))",
         "with convergents p<sub>k</sub>/q<sub>k</sub> satisfying p<sub>k</sub> = a<sub>k</sub>p<sub>k-1</sub> "
         "+ p<sub>k-2</sub>, q<sub>k</sub> = a<sub>k</sub>q<sub>k-1</sub> + q<sub>k-2</sub>. Superposing convergent "
         "harmonics along the boundary arclength s produces the curvature blending field "
         "&kappa;(s) = &Sigma;<sub>k</sub> cos(2&pi;q<sub>k</sub>s + p<sub>k</sub>) / (|q<sub>k</sub>| + 1), "
         "which is non-resonant by construction and evenly distributes curvature transition across the "
         "chamfer-to-body junction."),

        ("3. Boundary Element Formulation",
         "Implant biomechanics are evaluated via the Somigliana boundary integral identity for 3D "
         "elastostatics on the closed surface &Gamma;:",
         "c<sub>ij</sub>(x) u<sub>j</sub>(x) + &int;<sub>&Gamma;</sub> T<sub>ij</sub>(x,y) u<sub>j</sub>(y) d&Gamma; "
         "= &int;<sub>&Gamma;</sub> U<sub>ij</sub>(x,y) t<sub>j</sub>(y) d&Gamma;",
         "with the Kelvin fundamental solution U<sub>ij</sub>(x,y) = [1/(16&pi;&mu;(1&minus;&nu;)r)] "
         "&middot; [(3&minus;4&nu;)&delta;<sub>ij</sub> + r&#770;<sub>i</sub>r&#770;<sub>j</sub>], discretized "
         "into constant boundary panels and collocated to assemble the dense linear system Hu = Gt under "
         "cyclic Valsalva intra-abdominal pressure loading (3.5&ndash;25 kPa)."),

        ("4. HPC Domain Decomposition and Amdahl Scaling",
         "The O(N&sup3;) dense BEM solve is distributed across P MPI ranks via SLURM-scheduled domain "
         "decomposition. Strong-scaling speedup and parallel efficiency follow Amdahl's law:",
         "S(P) = 1 / [(1&minus;f) + f/P],  &nbsp; E(P) = S(P)/P",
         "where f is the parallelizable fraction of the assembly/solve workload. Communication overhead from "
         "boundary-panel halo exchange is modeled to scale with &radic;P, bounding the achievable efficiency "
         "at large rank counts."),

        ("5. NVQLink Hybrid Quantum-Classical Acceleration",
         "A simulated NVQLink interconnect couples HPC GPU nodes to a QPU co-processor for preconditioning "
         "the ill-conditioned Kelvin-kernel subspace. HHL-class quantum linear solvers achieve complexity:",
         "T<sub>quantum</sub> &sim; O(log(N) &middot; &kappa;<sup>2</sup> / &epsilon;)  vs.  T<sub>classical</sub> &sim; O(N&sup3;)",
         "where &kappa; is the system condition number and &epsilon; the target solution tolerance. The "
         "resulting hybrid pipeline &mdash; QPU-assisted deflation followed by GPU-side iterative refinement "
         "across the NVQLink fabric &mdash; is projected to reduce wall-clock time for well-conditioned "
         "implant subspaces relative to the classical direct solve."),

        ("6. Conclusion",
         "Combining chamfer-aware manifold repair, continued-fraction quasi-periodic curvature blending, "
         "boundary element biomechanical simulation, and HPC/NVQLink hybrid acceleration provides a "
         "computationally tractable, mathematically grounded pipeline for the design and clinical validation "
         "of next-generation pelvic floor reconstruction implants.", "", "")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        if math_eq:
            elements.append(Paragraph(math_eq, math_style))
        if post_text:
            elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")


if __name__ == '__main__':
    generate_pdf()
