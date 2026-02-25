import os
import shutil
from circuit_schematic_generator import CircuitSchematicGenerator
import generate_pdf
import base64

def generate_comprehensive_report():
    print("Initializing Comprehensive MRI Coil & Physics Report Generation...")
    
    # 1. Generate Schematics for ALL Coils
    gen = CircuitSchematicGenerator()
    schematics = gen.generate_all()
    
    # Save schematic images
    image_paths = {}
    for name, b64_str in schematics.items():
        filename = f"report_schematic_{name}.png"
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(b64_str))
        image_paths[name] = filename
        print(f"Generated schematic: {filename}")

    # 2. Define Markdown Content
    md_content = """# NeuroPulse Comprehensive Physics & Coil Report
**Date:** January 14, 2026
**System:** Gemini MRI Reconstruction Simulator

---

## 1. Executive Summary
This document provides a holistic overview of the RF Coil topologies available in the NeuroPulse system, alongside the finite mathematical derivations for the supported pulse sequences. It serves as a verification of the physical reliability of the simulator.

---

## 2. Pulse Sequence Finite Math Derivations

The simulator operates on the solution of the Bloch Equations in the rotating frame.

### Spin Echo (SE)
The Spin Echo sequence uses a $90^\\circ$ excitation pulse followed by a $180^\\circ$ refocusing pulse at $TE/2$.

**Signal Equation:**
$$ S(t) = M_0 \\left( 1 - 2e^{-(TR-TE/2)/T1} + e^{-TR/T1} \\right) e^{-TE/T2} $$

**Finite Difference (Bloch):**
$$ \\frac{{d\\mathbf{{M}}}}{{dt}} = \\gamma \\mathbf{{M}} \\times \\mathbf{{B}} - \\frac{{M_x \\mathbf{{i}} + M_y \\mathbf{{j}}}}{{T2}} - \\frac{{(M_z - M_0)\\mathbf{{k}}}}{{T1}} $$

### Gradient Echo (GRE)
GRE utilizes a variable flip angle $\\alpha$ and lacks a $180^\\circ$ refocusing pulse, making it sensitive to $T2^*$ effects.

**Signal Equation (Steady State):**
$$ S = M_0 \\frac{{ (1 - e^{-TR/T1}) \\sin\\alpha }}{{ 1 - e^{-TR/T1} \\cos\\alpha }} e^{-TE/T2^*} $$

**Ernst Angle Optimization:**
$$ \\alpha_E = \\arccos(e^{-TR/T1}) $$

### Inversion Recovery (IR)
IR begins with a $180^\\circ$ inversion pulse.

**Signal Equation:**
$$ M_z(TI) = M_0 (1 - 2e^{-TI/T1} + e^{-TR/T1}) $$
*Null Point:* $TI_{{null}} = T1 \\ln 2$

---

## 3. RF Coil Circuit Schematics

### 3.1 Birdcage Coil (Standard)
**Topology:** Ladder Network (High-Pass/Low-Pass)
**Application:** Volume imaging (Head/Body)
**Homogeneity:** High ($\propto 1 - \cos(N\phi)$)

![Birdcage Schematic](report_schematic_birdcage.png)

### 3.2 Solenoid / Loop Coil
**Topology:** Inductive Solenoid with Capacitive Tuning/Matching
**Application:** Extremities, Surface localizers
**Feature:** High Sensitivity near surface

![Solenoid Schematic](report_schematic_solenoid.png)

### 3.3 Surface Phased Array
**Topology:** Overlapping Geometric Loops to minimize mutual inductance ($M \approx 0$).
**Application:** Parallel Imaging (SENSE/GRAPPA)
**Feature:** High SNR + Accelerated Acquisition

![Surface Array Schematic](report_schematic_surface_array.png)

### 3.4 Quantum Lattice Coil topology
**Topology:** Non-local flux-entangled Josephson Junction lattice.
**Application:** Neurovascular Imaging
**Physics:** Based on Berry Phase accumulation along topological defects in the lattice.

$$ \\gamma_n(C) = i \\oint_C \\langle \\psi_n | \\nabla | \\psi_n \\rangle \\cdot d\\mathbf{{R}} $$

![Quantum Lattice Schematic](report_schematic_quantum_lattice.png)

---

## 4. Finite Element Implementation (BEM)
The simulation employs Boundary Element Method for calculating magnetic fields.

$$ \\mathbf{{B}}(\\mathbf{{r}}) = \\frac{{\\mu_0}}{{4\\pi}} \\int_S \\mathbf{{J}}(\\mathbf{{r}}') \\times \\frac{{\\mathbf{{r}} - \\mathbf{{r}}'}}{{ |\\mathbf{{r}} - \\mathbf{{r}}'|^3 }} dS' $$

Generated automatically by NeuroPulse Physics Engine v3.0
"""

    # 3. Write Markdown
    md_file = "comprehensive_report.md"
    with open(md_file, "w") as f:
        f.write(md_content)
    print(f"Written markdown to {md_file}")
    
    # 4. Convert to PDF
    pdf_file = "NeuroPulse_Comprehensive_Report.pdf"
    try:
        generate_pdf.md_to_pdf(md_file, pdf_file)
        print(f"Values exported to: {pdf_file}")
    except Exception as e:
        print(f"PDF Generation Failed: {e}")

if __name__ == "__main__":
    generate_comprehensive_report()
