import math
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm

# Import sequences
from cfb_thermometry_sequences import cfb_fid_isotopes

def create_fid_plot(filename):
    plt.figure(figsize=(8, 5))
    t = np.linspace(0, 100, 500)
    # T2* for typical tissues
    t2star_13c = 30.0
    t2star_129xe = 22.0
    
    T = 37.0 + 0.5 * t / 10.0 # temperature changes over time
    
    # Simulate FID signals
    fid_13c = np.exp(-t / t2star_13c) * np.cos(2*np.pi*t*0.1)
    fid_129xe = np.exp(-t / t2star_129xe) * np.cos(2*np.pi*t*0.12)
    
    plt.plot(t, fid_13c, label=r'$^{13}$C FID Signal', color='#004488', linewidth=2)
    plt.plot(t, fid_129xe, label=r'$^{129}$Xe FID Signal', color='#dd5500', linewidth=2)
    
    plt.title('Free Induction Decay & Thermal Progression', fontsize=14, pad=15)
    plt.xlabel('Echo Time (ms)', fontsize=12)
    plt.ylabel('Signal Intensity (A.U.)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def build_pdf():
    doc = SimpleDocTemplate("Nature_CFB_FID_Isotopes.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, textColor=colors.darkblue)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, spaceAfter=10)
    
    story = []
    
    story.append(Paragraph("Quantum MRI Pulse Sequences: Free Induction Decay with Continued Fractions and Isotopic n+/n Atomicity", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Abstract</b>", styles['Heading3']))
    story.append(Paragraph("This report explores the implementation of Free Induction Decay (FID) thermometry driven by continued fractional bounds corresponding to the n+/n atomicity ratios of emerging medical isotopes (e.g., 129Xe, 13C, 3He).", body_style))
    
    story.append(Spacer(1, 10))
    
    plot_file = "fid_plot_temp.png"
    create_fid_plot(plot_file)
    story.append(Image(plot_file, width=6*inch, height=3.75*inch))
    
    doc.build(story)
    
    if os.path.exists(plot_file):
        os.remove(plot_file)
    print("Generated Nature_CFB_FID_Isotopes.pdf successfully.")
        
if __name__ == "__main__":
    build_pdf()
