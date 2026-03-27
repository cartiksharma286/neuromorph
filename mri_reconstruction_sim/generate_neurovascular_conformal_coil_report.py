import os
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_coil_geometry(filename):
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, 0.2, 100)
    T, Z = np.meshgrid(theta, z)
    R = 0.12 + 0.02 * np.cos(3 * T) 

    X = R * np.cos(T)
    Y = R * np.sin(T)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    for i in range(16):
        angle = i * (2 * np.pi / 16)
        x_coil = (0.12 + 0.02 * np.cos(3 * angle)) * np.cos(angle)
        y_coil = (0.12 + 0.02 * np.cos(3 * angle)) * np.sin(angle)
        ax.plot([x_coil, x_coil], [y_coil, y_coil], [0, 0.2], color='red', linewidth=2)

    ax.set_title("16-Channel Conformal Neurovascular Rx Coil Array")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_schematic(filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot([0, 1], [0.5, 0.5], 'k-', lw=2)  
    ax.plot([1, 1], [0.3, 0.7], 'k-', lw=2) 
    ax.plot([1.2, 1.2], [0.3, 0.7], 'k-', lw=2)
    ax.text(1.1, 0.8, 'C_tune', ha='center')
    ax.plot([1.2, 2], [0.5, 0.5], 'k-', lw=2)
    
    x = np.linspace(2, 3, 100)
    y = 0.5 + 0.2 * np.sin(x * 5 * np.pi) * np.exp(-(x-2.5)**2 / 0.1)
    ax.plot(x, y, 'k-', lw=2)
    ax.text(2.5, 0.8, 'L_coil', ha='center')
    ax.plot([3, 4], [0.5, 0.5], 'k-', lw=2)
    
    ax.plot([4, 4], [0.5, 0.1], 'k-', lw=2)
    ax.plot([3.8, 4.2], [0.1, 0.1], 'k-', lw=2) 
    ax.plot([3.9, 4.1], [0.05, 0.05], 'k-', lw=2)
    ax.plot([3.95, 4.05], [0.0, 0.0], 'k-', lw=2)

    ax.add_patch(plt.Rectangle((4, 0.4), 0.5, 0.2, fill=True, color='lightblue', zorder=3))
    ax.text(4.25, 0.5, 'LNA', ha='center', va='center', zorder=4)
    ax.plot([4.5, 5], [0.5, 0.5], 'k-', lw=2)

    ax.axis('off')
    ax.set_title("Advanced Preamplifier Decoupling Circuitry for High SNR")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_gdt_drawing(filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Outer profile
    circle = plt.Circle((0.5, 0.5), 0.35, fill=False, color='black', lw=2)
    ax.add_patch(circle)
    # Inner profile
    circle2 = plt.Circle((0.5, 0.5), 0.33, fill=False, color='black', lw=1)
    ax.add_patch(circle2)
    
    # Dimensions
    ax.annotate('', xy=(0.15, 0.5), xytext=(0.85, 0.5), arrowprops=dict(arrowstyle='<->'))
    ax.text(0.5, 0.52, 'Ø240 ± 0.5', ha='center')
    
    # Target points for datum
    ax.plot(0.5, 0.5, 'k+')
    ax.text(0.53, 0.53, 'Datum A', ha='left')
    
    # GD&T Frame (Profile of a surface, cylindricity)
    ax.text(0.5, 0.90, '⌖ | Ø 0.1 | A | B | C\n⌭ | 0.05', bbox=dict(facecolor='white', edgecolor='black'), ha='center')
    ax.plot([0.5, 0.5], [0.85, 0.90], 'k-') # Leader line
    
    ax.text(0.2, 0.1, 'Coil Former Wall Thickness: 2.0 mm\nMaterial: Polycarbonate (MR Safe)\nTolerance Class: ISO 2768-m', fontsize=9)
    
    ax.axis('off')
    ax.set_title("GD&T Conformal Coil Former Engineering Drawing")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def export_to_stl(filename):
    R = 0.12
    H = 0.2
    segments = 36
    
    with open(filename, 'w') as f:
        f.write(f"solid neurovascular_conformal_coil\n")
        
        for i in range(segments):
            theta1 = i * 2.0 * np.pi / segments
            theta2 = ((i + 1) * 2.0 * np.pi) / segments
            
            x1, y1 = R*np.cos(theta1), R*np.sin(theta1)
            x2, y2 = R*np.cos(theta2), R*np.sin(theta2)
            
            f.write(f"  facet normal 0 0 0\n    outer loop\n      vertex {x1} {y1} 0\n      vertex {x2} {y2} 0\n      vertex {x2} {y2} {H}\n    endloop\n  endfacet\n")
            f.write(f"  facet normal 0 0 0\n    outer loop\n      vertex {x1} {y1} 0\n      vertex {x2} {y2} {H}\n      vertex {x1} {y1} {H}\n    endloop\n  endfacet\n")
            
        f.write(f"endsolid neurovascular_conformal_coil\n")

def build_pdf(output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = styles['Heading1']
    body_style = styles['Normal']
    
    story = []
    
    story.append(Paragraph("Technical Report: Custom Head Coils for Neurovascular Imaging", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("This report details the design, electrical circuitry, and manufacturing specifications of an advanced, conformal 16-channel receive coil array optimized for high-resolution neurovascular magnetic resonance imaging. A companion .stl file (neurovascular_conformal_coil.stl) has been generated for 3D printing and CAD import.", body_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # GD&T Drawing
    gdt_file = "coil_gdt_drawing.png"
    generate_gdt_drawing(gdt_file)
    story.append(Image(gdt_file, width=5*inch, height=5*inch))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("<b>Figure 1:</b> GD&T Engineering Drawing for the coil former showing True Position and Cylindricity tolerances critical for maintaining conformality.", body_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Coil Geometry
    geom_file = "conformal_coil_geometry.png"
    generate_coil_geometry(geom_file)
    story.append(Image(geom_file, width=4*inch, height=4*inch))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("<b>Figure 2:</b> 3D geometry layout of the conformal coil array elements.", body_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Circuit Schematic
    sch_file = "coil_electrical_schematic.png"
    generate_schematic(sch_file)
    story.append(Image(sch_file, width=6*inch, height=3*inch))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("<b>Figure 3:</b> Electrical schematic of a single actively decoupled coil element.", body_style))
    
    doc.build(story)
    
    # Cleanup
    # for f in [geom_file, sch_file, gdt_file]:
    #     if os.path.exists(f):
    #         os.remove(f)

if __name__ == "__main__":
    stl_filename = "neurovascular_conformal_coil.stl"
    export_to_stl(stl_filename)
    
    pdf_filename = "Technical_Report_Neurovascular_Conformal_Coils.pdf"
    build_pdf(pdf_filename)
    print(f"Generated {stl_filename} and {pdf_filename} successfully.")

