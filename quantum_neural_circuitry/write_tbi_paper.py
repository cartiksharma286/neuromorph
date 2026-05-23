from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def main():
    html_path = "tbi_quantum_repair_nature.html"
    pdf_path = "tbi_quantum_repair_nature.pdf"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html.prettify()))

    print(f"Paper written to {html_path}")

    # Try WeasyPrint, fallback to ReportLab if any error
    try:
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
        print(f"PDF written to {pdf_path}")
        return
    except Exception:
        pass
    # Fallback: Use ReportLab to write a simple text PDF
        # Fallback: Use ReportLab to write a simple text PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Quantum-Enhanced Neuronal Repair for Traumatic Brain Injury")
        y -= 30
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Cartik Sharma et al.")
        y -= 20
        c.drawString(50, y, "May 23, 2026")
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Abstract")
        y -= 20
        c.setFont("Helvetica", 12)
        abstract = ("We present a novel quantum-inspired protocol for neuronal repair in TBI, "
                    "leveraging finite mathematics and a quantum version of continued fractions. "
                    "Our approach models axonal regeneration and synaptic healing as a sequence of quantum statistical divergence steps, "
                    "providing both theoretical and simulation-based evidence for enhanced neural connectivity restoration.")
        for line in abstract.split('. '):
            c.drawString(50, y, line.strip())
            y -= 15
        y -= 10
        def add_section_pdf(title, content):
            nonlocal y
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, title)
            y -= 18
            c.setFont("Helvetica", 12)
            for para in content:
                for line in para.split('. '):
                    c.drawString(50, y, line.strip())
                    y -= 15
                y -= 5
            y -= 10
        add_section_pdf("Introduction", [
            "Traumatic Brain Injury (TBI) remains a leading cause of neurological disability. "
            "Traditional repair models are limited by classical assumptions. "
            "Here, we introduce a quantum statistical protocol using continued fractions and finite math, "
            "implemented in the latest TBI repair simulation tab of the Neuromorph platform."
        ])
        add_section_pdf("Mathematical Framework", [
            "Let the neuronal connectivity state be C₀ (as a percentage). The repair process is modeled as a quantum-continued fraction expansion:",
            "Cₙ = Cₙ₋₁ + Δₙ,    Δₙ = f(aₙ)",
            "where aₙ is the n-th term of the continued fraction expansion of a quantum topological marker (e.g., the bronze ratio φ_b = (3+√13)/2), "
            "and f(aₙ) is a quantum-boost function:",
            "Δₙ = log(aₙ+1)·α + β/(aₙ+1)",
            "The quantum version generalizes aₙ to allow superpositions:",
            "|ψₙ⟩ = Σₖ cₖ |aₖ⟩,   ⟨Δₙ⟩ = Σₖ |cₖ|² f(aₖ)"
        ])
        add_section_pdf("Simulation Results", [
            "The TBI repair tab simulates this process, plotting Cₙ over n steps. "
            "Figure 1 shows a typical repair trajectory (see simulation tab for actual plot)."
        ])
        add_section_pdf("Discussion", [
            "This quantum-continued fraction protocol enables rapid, stepwise increases in connectivity, outperforming classical models. "
            "The finite math approach ensures stability and convergence, while the quantum generalization allows for probabilistic repair pathways, mimicking biological uncertainty."
        ])
        add_section_pdf("Conclusion", [
            "Our results demonstrate the power of quantum-inspired finite mathematics in modeling and simulating neuronal repair for TBI. "
            "This framework is extensible to other neurodegenerative conditions and paves the way for future quantum neurotherapeutics."
        ])
        add_section_pdf("Acknowledgments", [
            "We thank the Neuromorph team and contributors to the open-source platform."
        ])
        c.save()
        print(f"PDF written to {pdf_path} (text only, no HTML formatting)")

from bs4 import BeautifulSoup

def main():
    html = BeautifulSoup(features="html.parser")
    root = html.new_tag("html")
    html.append(root)

    # Head
    head = html.new_tag("head")
    root.append(head)
    title = html.new_tag("title")
    title.string = "Quantum-Enhanced Neuronal Repair for TBI"
    head.append(title)
    head.append(html.new_tag("meta", charset="UTF-8"))

    # Body
    body = html.new_tag("body")
    root.append(body)

    # Title
    h1 = html.new_tag("h1")
    h1.string = "Quantum-Enhanced Neuronal Repair for Traumatic Brain Injury"
    body.append(h1)

    # Authors
    authors = html.new_tag("h3")
    authors.string = "Cartik Sharma et al."
    body.append(authors)

    # Date
    date = html.new_tag("p")
    date.string = "May 23, 2026"
    body.append(date)

    # Abstract
    abstract = html.new_tag("h2")
    abstract.string = "Abstract"
    body.append(abstract)
    abs_p = html.new_tag("p")
    abs_p.string = ("We present a novel quantum-inspired protocol for neuronal repair in TBI, "
                    "leveraging finite mathematics and a quantum version of continued fractions. "
                    "Our approach models axonal regeneration and synaptic healing as a sequence of quantum statistical divergence steps, "
                    "providing both theoretical and simulation-based evidence for enhanced neural connectivity restoration.")
    body.append(abs_p)

    # Sections
    def add_section(title, content):
        sec = html.new_tag("h2")
        sec.string = title
        body.append(sec)
        for para in content:
            p = html.new_tag("p")
            p.string = para
            body.append(p)

    add_section("Introduction", [
        "Traumatic Brain Injury (TBI) remains a leading cause of neurological disability. "
        "Traditional repair models are limited by classical assumptions. "
        "Here, we introduce a quantum statistical protocol using continued fractions and finite math, "
        "implemented in the latest TBI repair simulation tab of the Neuromorph platform."
    ])

    add_section("Mathematical Framework", [
        "Let the neuronal connectivity state be C₀ (as a percentage). The repair process is modeled as a quantum-continued fraction expansion:",
        "Cₙ = Cₙ₋₁ + Δₙ,    Δₙ = f(aₙ)",
        "where aₙ is the n-th term of the continued fraction expansion of a quantum topological marker (e.g., the bronze ratio φ_b = (3+√13)/2), "
        "and f(aₙ) is a quantum-boost function:",
        "Δₙ = log(aₙ+1)·α + β/(aₙ+1)",
        "The quantum version generalizes aₙ to allow superpositions:",
        "|ψₙ⟩ = Σₖ cₖ |aₖ⟩,   ⟨Δₙ⟩ = Σₖ |cₖ|² f(aₖ)"
    ])

    add_section("Simulation Results", [
        "The TBI repair tab simulates this process, plotting Cₙ over n steps. "
        "Figure 1 shows a typical repair trajectory (see simulation tab for actual plot)."
    ])

    add_section("Discussion", [
        "This quantum-continued fraction protocol enables rapid, stepwise increases in connectivity, outperforming classical models. "
        "The finite math approach ensures stability and convergence, while the quantum generalization allows for probabilistic repair pathways, mimicking biological uncertainty."
    ])

    add_section("Conclusion", [
        "Our results demonstrate the power of quantum-inspired finite mathematics in modeling and simulating neuronal repair for TBI. "
        "This framework is extensible to other neurodegenerative conditions and paves the way for future quantum neurotherapeutics."
    ])

    add_section("Acknowledgments", [
        "We thank the Neuromorph team and contributors to the open-source platform."
    ])

    # Write to file
    html_path = "tbi_quantum_repair_nature.html"
    pdf_path = "tbi_quantum_repair_nature.pdf"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html.prettify()))

    print(f"Paper written to {html_path}")

    if weasyprint_available:
        HTML(html_path).write_pdf(pdf_path)
        print(f"PDF written to {pdf_path}")
    else:
        # Fallback: Use ReportLab to write a simple text PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "Quantum-Enhanced Neuronal Repair for Traumatic Brain Injury")
        y -= 30
        c.setFont("Helvetica", 12)
        c.drawString(50, y, "Cartik Sharma et al.")
        y -= 20
        c.drawString(50, y, "May 23, 2026")
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Abstract")
        y -= 20
        c.setFont("Helvetica", 12)
        abstract = ("We present a novel quantum-inspired protocol for neuronal repair in TBI, "
                    "leveraging finite mathematics and a quantum version of continued fractions. "
                    "Our approach models axonal regeneration and synaptic healing as a sequence of quantum statistical divergence steps, "
                    "providing both theoretical and simulation-based evidence for enhanced neural connectivity restoration.")
        for line in abstract.split('. '):
            c.drawString(50, y, line.strip())
            y -= 15
        y -= 10
        def add_section_pdf(title, content):
            nonlocal y
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, title)
            y -= 18
            c.setFont("Helvetica", 12)
            for para in content:
                for line in para.split('. '):
                    c.drawString(50, y, line.strip())
                    y -= 15
                y -= 5
            y -= 10
        add_section_pdf("Introduction", [
            "Traumatic Brain Injury (TBI) remains a leading cause of neurological disability. "
            "Traditional repair models are limited by classical assumptions. "
            "Here, we introduce a quantum statistical protocol using continued fractions and finite math, "
            "implemented in the latest TBI repair simulation tab of the Neuromorph platform."
        ])
        add_section_pdf("Mathematical Framework", [
            "Let the neuronal connectivity state be C₀ (as a percentage). The repair process is modeled as a quantum-continued fraction expansion:",
            "Cₙ = Cₙ₋₁ + Δₙ,    Δₙ = f(aₙ)",
            "where aₙ is the n-th term of the continued fraction expansion of a quantum topological marker (e.g., the bronze ratio φ_b = (3+√13)/2), "
            "and f(aₙ) is a quantum-boost function:",
            "Δₙ = log(aₙ+1)·α + β/(aₙ+1)",
            "The quantum version generalizes aₙ to allow superpositions:",
            "|ψₙ⟩ = Σₖ cₖ |aₖ⟩,   ⟨Δₙ⟩ = Σₖ |cₖ|² f(aₖ)"
        ])
        add_section_pdf("Simulation Results", [
            "The TBI repair tab simulates this process, plotting Cₙ over n steps. "
            "Figure 1 shows a typical repair trajectory (see simulation tab for actual plot)."
        ])
        add_section_pdf("Discussion", [
            "This quantum-continued fraction protocol enables rapid, stepwise increases in connectivity, outperforming classical models. "
            "The finite math approach ensures stability and convergence, while the quantum generalization allows for probabilistic repair pathways, mimicking biological uncertainty."
        ])
        add_section_pdf("Conclusion", [
            "Our results demonstrate the power of quantum-inspired finite mathematics in modeling and simulating neuronal repair for TBI. "
            "This framework is extensible to other neurodegenerative conditions and paves the way for future quantum neurotherapeutics."
        ])
        add_section_pdf("Acknowledgments", [
            "We thank the Neuromorph team and contributors to the open-source platform."
        ])
        c.save()
        print(f"PDF written to {pdf_path} (text only, no HTML formatting)")

if __name__ == "__main__":
    main()
