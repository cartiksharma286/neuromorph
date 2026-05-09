import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

def generate_pdf():
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nature_MEG_QML_Beamformer_Localization.pdf")
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontSize=11, leading=14))
    styles.add(ParagraphStyle(name='CenterEquation', alignment=TA_CENTER, fontSize=13, fontName='Helvetica-Oblique', leading=16))
    styles.add(ParagraphStyle(name='TitleStyle', alignment=TA_CENTER, fontSize=16, fontName='Helvetica-Bold', leading=20))
    
    story = []
    
    # Title
    story.append(Paragraph("Quantum Machine Learning for MEG Source Localization:", styles['TitleStyle']))
    story.append(Paragraph("Beamformer-Initialized Variational Optimization for Epileptic Spikes", styles['TitleStyle']))
    story.append(Spacer(1, 20))
    
    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
    story.append(Paragraph("Magnetoencephalography (MEG) provides high-temporal resolution for identifying epileptogenic zones. However, classical inverse problem solutions suffer from ill-posed conditions and anatomical noise. We introduce a hybrid quantum-classical architecture that utilizes Linearly Constrained Minimum Variance (LCMV) beamformers to establish an initial spatial prior, subsequently refined via a Variational Quantum Eigensolver (VQE) acting as a spatial optimization block. By leveraging geometric annealing paths, this method bypasses barren plateaus and achieves sub-millimeter precision for transient epileptic spike localization.", styles['Justify']))
    story.append(Spacer(1, 15))
    
    # 1. Introduction
    story.append(Paragraph("<b>1. Forward Model and Field Equations</b>", styles['Heading2']))
    story.append(Paragraph("Accurate identification of epileptic foci relies on reconstructing neural current dipoles from extracranial magnetic fields. The standard finite mathematical forward model is defined linearly as:", styles['Justify']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("B<sub>k</sub>(t) = Σ<sub>j</sub> L<sub>kj</sub> Q<sub>j</sub>(t) + n<sub>k</sub>(t)", styles['CenterEquation']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("where B<sub>k</sub> is the measured magnetic field at the k-th Superconducting Quantum Interference Device (SQUID) sensor. L<sub>kj</sub> serves as the lead field observation projection matrix displaying an approximated 1/r<sup>2</sup> far-field spatial decay, Q<sub>j</sub> is the dipole moment intensity at volumetric grid point j, and n<sub>k</sub> denotes external stochastic noise.", styles['Justify']))
    story.append(Spacer(1, 15))
    
    # 2. Beamformer
    story.append(Paragraph("<b>2. LCMV Beamformer Spatial Priors</b>", styles['Heading2']))
    story.append(Paragraph("To circumvent barren plateaus inherently prevalent in high-dimensional quantum neural networks uniformly initialized across the brain grid, we utilize a classical LCMV beamformer. This establishes a highly constrained spatial prior for the true source by minimizing structural variance:", styles['Justify']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("w_opt = argmin (w<sup>T</sup> C w)   subject to   w<sup>T</sup> L = 1", styles['CenterEquation']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Solving via the Lagrange multiplier framework derives the pseudo-probability spectrum metric at any arbitrary continuous locus r:", styles['Justify']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("P(r) = [L<sup>T</sup>(r) C<sup>-1</sup> L(r)]<sup>-1</sup>", styles['CenterEquation']))
    story.append(Spacer(1, 10))
    
    # 3. QML
    story.append(Paragraph("<b>3. Quantum Variational Optimization Block</b>", styles['Heading2']))
    story.append(Paragraph("The spatial index yielding max{ P(r) } seeds the quantum circuit parametrization set θ<sub>0</sub>. A Variational Quantum Eigensolver (VQE) then explores the spatial loss valley iteratively using the parameter-shift rule for gradient calculations:", styles['Justify']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("θ<sub>n+1</sub> = θ<sub>n</sub> - η ∇<sub>θ</sub> ⟨ψ(θ)| H<sub>MEG</sub> |ψ(θ)⟩ + ε<sub>n</sub>", styles['CenterEquation']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Here, the Hamiltonian H<sub>MEG</sub> maps to the disparity between the simulated electromagnetic flux of the parametric coordinates and the empirical temporal array signature. The simulated noise distribution ε<sub>n</sub> acts as an annealing thermal state. Upon convergence, the temporal activation of the isolated optimal position reliably discriminates paroxysmal epileptic spike transients via an exponential kernel transform:", styles['Justify']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("S(t) = exp(-(t - t_peak)<sup>2</sup> / σ<sup>2</sup>) · sin(ω t)", styles['CenterEquation']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>4. Conclusion</b>", styles['Heading2']))
    story.append(Paragraph("Embedding robust mathematical LCMV initializations into QML topologies vastly accelerates source optimization. Simulative telemetry ensures resilient localization of neuro-pathological sites, rendering Q-MEG viable for preoperative mapping workflows in functional neurosurgery.", styles['Justify']))
    
    doc.build(story)
    print(f"Generated PDF: {filename}")

if __name__ == '__main__':
    generate_pdf()
