import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

pdf_path = "Dementia_DBS_Technical_Report.pdf"

# Initialize PDF
with PdfPages(pdf_path) as pdf:
    # Title Page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.5, 0.9, "Nature Technical Report:\nOptimization of Deep Brain Stimulation\nfor Dementia and Smart Aging Paradigms", fontsize=18, fontweight='bold', ha='center', va='top')
    
    abstract = ("This report documents the architectural specifications and mathematical manifolds\n"
                "governing the advanced Deep Brain Stimulation (DBS) framework for dementia deceleration.")
    ax.text(0.1, 0.75, abstract, fontsize=12, ha='left', va='top', style='italic')

    # Electrical Specs
    ax.text(0.1, 0.6, "1. Comprehensive Electrical Specifications", fontsize=14, fontweight='bold', ha='left')
    specs = (
        "- Voltage Range: 0.0 V - 10.5 V\n"
        "- Current Range: 0.0 mA -import matplotlib.pyplot as plt
from y)from matplotlib.backends.backePuimport os

pdf_path = "Dementia_DBS_Technical_Repornt
pdf_pat Ra
# Initialize PDF
with PdfPages(pdf_path) as elewith PdfPages(pCS   02-405 MHz), ISM (2.4 GHz), m    fig,
            ax.axis('off')
    ax.text(0.5, 0.9, "Naant (~6.78 MHz)\n"
      
    abstract = ("This report documents the architectural specifications and mathematical manifolds\n"
                "governing the advanced Deep Brain Stimulation (DBS) framewo    # M                "governing the advanced Deep Brain Stimulation (DBS) framework for dementia decelera.     ax.text(0.1, 0.75, abstract, fontsize=12, ha='left', va='top', style='italic')

    # Electrical Specssc
    # Electrical Specs
    ax.text(0.1, 0.6, "1. Comprehensive Electrical Specifnni    ax.text(0.1, 0.6,      specs = (
        "- Voltage Range: 0.0 V - 10.5 V\n"
        "- Current Range: 0.0 mA -import matplotlib0.        "- Vc,        "=12, ha='left', va='top')

    eq1 from y)from matplotlib.backends.backePuimportcdot (\sigma(r) \nabl
pdf_path = "Dementia_DBS_Technical_Repornt
pdfextpdf_pat Ra
# Initialize PDF
with PdfPages
    text_subwith PdfPages(pWh            ax.axis('off')
    ax.text(0.5, 0.9, "Naant (~6.78 MHz)\n"
      
    abst       r"denotes the conductivity tensor bounded by a prime modulus $p_k$." "\n"                "governing the advanced Deep Brain Stimulation (DBS) framewo    # M                "ex
    # Electrical Specssc
    # Electrical Specs
    ax.text(0.1, 0.6, "1. Comprehensive Electrical Specifnni    ax.text(0.1, 0.6,      specs = (
        "- Voltage Range: 0.0 V - 10.5 V\n"
        "- Current Range: 0.0 mA -import matplotlib0.        "- Vc,        "rin    # Electrical Specrated {pdf_path}")
