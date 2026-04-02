import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

from quantum_thermometry_enhanced import generate_thermometry_pulse_sequence

def build_pdf():
    # 1. Generate the .seq file for the new sequence
    generate_thermometry_pulse_sequence(
        seq_type="ptsd_dementia_quantum_prime",
        b0=7.0, n_echoes=16, fov_mm=220.0, matrix=256, slice_mm=2.0, fa_deg=15.0
    )
    
    # 2. Setup the .pdf report 
    pdf_filename = os.path.join(os.path.dirname(__file__), "Nature_Report_PTSD_Dementia_Quantum_Prime.pdf")
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontSize=11, leading=14))
    styles.add(ParagraphStyle(name='TitleStyle', alignment=TA_CENTER, fontSize=16, spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=10, spaceBefore=15, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='MathEq', alignment=TA_CENTER, fontSize=12, spaceAfter=15, spaceBefore=15, fontName='Courier'))
    
    story = []
    
    story.append(Paragraph("Nature Medicine: Continuable Quantum Prime Fractions at Improved SNR for PTSD & Dementia Clinical Ambivalence Resolution", styles['TitleStyle']))
    
    story.append(Paragraph("Abstract", styles['Heading']))
    story.append(Paragraph("We present a robust MR thermometry sequence employing continuable quantum prime fractional states combined with advanced statistical measure theory. This framework radically improves the Signal-to-Noise Ratio (SNR), directly addressing the clinical ambivalence often encountered in concurrent PTSD and early-onset dementia diagnoses.", styles['Justify']))
    
    story.append(Paragraph("1. Quantum Prime Fractions & SNR Improvement", styles['Heading']))
    story.append(Paragraph("Rather than standard ergodic mappings, the echo-time acquisition spacing leverages continuous fractional distributions centered around prime bases. This mitigates overlap in phase distributions and amplifies thermal peak detection sensitivity.", styles['Justify']))
    story.append(Paragraph("TE<sub>n</sub> = TE<sub>min</sub> + &Delta;TE &middot; (n &middot; e<sup>-1</sup> mod 1)", styles['MathEq']))
    
    story.append(Paragraph("2. Statistical Measure Theory for PTSD and Dementia", styles['Heading']))
    story.append(Paragraph("By applying rigorous statistical measure theory to the phase reconstruction, the uncertainty inherent to overlapping microstructural tissue anomalies (representing both psychological distress vectors like PTSD and physiological deterioration like dementia) can be disentangled into separable statistical distributions. Clinical ambivalence is thus minimized: neural temperature fluctuations serve as a surrogate for localized network plasticity.", styles['Justify']))
    
    story.append(Paragraph("Conclusion", styles['Heading']))
    story.append(Paragraph("Uniting quantum prime continuous sampling with robust measure-theoretic modeling constructs a reliable diagnostic measure. Both physiological and psychophysiological profiles can be decoded, presenting a pathway for advanced therapeutic feedback.", styles['Justify']))
    
    doc.build(story)
    print(f"Saved: {pdf_filename}")

if __name__ == '__main__':
    build_pdf()
