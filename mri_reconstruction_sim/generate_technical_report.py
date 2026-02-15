#!/usr/bin/env python3
"""
Technical Nature Report Generator for Neurovascular Coils
=========================================================
Generates a PDF report with finite math derivations for all 26 quantum vascular
coil geometries and pulse sequences.
"""
import os
import sys
import inspect
import re
from datetime import datetime
import numpy as np
import matplotlib
from xml.sax.saxutils import escape
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                PageBreak, Table, TableStyle, Preformatted, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Application imports
sys.path.append(os.getcwd())
from quantum_vascular_coils import QUANTUM_VASCULAR_COIL_LIBRARY, QuantumVascularCoil
from statistical_adaptive_pulse import ADAPTIVE_SEQUENCES

def parse_markdown_to_flowables(md_content, styles):
    """
    Parses Markdown content into ReportLab flowables.
    Handles headers, paragraphs, and standard LaTeX-style math blocks ($$...$$).
    """
    story = []
    lines = md_content.split('\n')
    
    # Styles
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    h3 = styles['Heading3']
    body = styles['BodyText']
    code = styles['MathCode']
    
    current_block = []
    in_code_block = False
    in_math_block = False
    
    for line in lines:
        line = line.strip()
        
        # Headers
        if line.startswith('# '):
            story.append(Paragraph(line[2:], h1))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], h2))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], h3))
        
        elif line.startswith('$$') and line.endswith('$$'):
            math_content = line[2:-2].strip()
            story.append(Paragraph(escape(math_content), styles['Equation']))
        elif line.startswith('$$'):
            in_math_block = True
            current_block = [line[2:]]
        elif line.endswith('$$') and in_math_block:
            in_math_block = False
            current_block.append(line[:-2])
            # Join with <br/> for Paragraph to handle line breaks while allowing centering
            # FIRST escape the lines, THEN join with <br/>
            escaped_block = [escape(l) for l in current_block]
            full_math = '<br/>'.join(escaped_block)
            story.append(Paragraph(full_math, styles['Equation']))
            current_block = []
        elif in_math_block:
            current_block.append(line)
            
        # Lists (Bullet)
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:]
            # Process inline formatting in list item
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            bullet_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', bullet_text)
            bullet_text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', bullet_text)
            # Highlight inline math
            bullet_text = re.sub(r'\$(.*?)\$', r'<font color="#2e86c1">\1</font>', bullet_text)
            story.append(Paragraph(bullet_text, body, bulletText='•'))
            
        # Lists (Numbered) -- Simple detection of "1. "
        elif re.match(r'^\d+\.\s', line):
            # Extract number and text
            match = re.match(r'^(\d+\.)\s(.*)', line)
            if match:
                num_marker = match.group(1)
                item_text = match.group(2)
                item_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item_text)
                item_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', item_text)
                item_text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', item_text)
                item_text = re.sub(r'\$(.*?)\$', r'<font color="#2e86c1">\1</font>', item_text)
                story.append(Paragraph(item_text, body, bulletText=num_marker))
            
        # Standard Text / Empty Lines
        elif line == '':
            story.append(Spacer(1, 0.1*inch))
        elif line.startswith('---'):
            story.append(PageBreak())
        else:
            # Inline formatting for standard paragraphs
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            formatted_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_line)
            formatted_line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', formatted_line)
            formatted_line = re.sub(r'\$(.*?)\$', r'<font color="#2e86c1">\1</font>', formatted_line)
            story.append(Paragraph(formatted_line, body))
            
    return story

def create_coil_spec_table(coil_class):
    """Generates a specification table for a coil class."""
    coil = coil_class()
    data = [
        ['Parameter', 'Value'],
        ['Name', coil.name],
        ['Elements', str(coil.num_elements)],
        ['Frequency', f"{coil.frequency/1e6:.2f} MHz"],
        ['Omega', f"{coil.omega:.2e} rad/s"]
    ]
    return data

def extract_math_from_docstring(docstring):
    """Extracts math content from docstrings."""
    if not docstring:
        return "No mathematical derivation provided."
    
    lines = docstring.split('\n')
    math_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('='):
            math_lines.append(stripped)
            
    return '\n'.join(math_lines)

def generate_report():
    output_filename = "Neurovascular_Coil_Technical_Report.pdf"
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='MathCode', fontName='Courier', fontSize=9, 
                              backColor=colors.HexColor('#f0f0f0'), borderPadding=10,
                              spaceBefore=10, spaceAfter=10))
    styles.add(ParagraphStyle(name='Equation', fontName='Courier-Bold', fontSize=10,
                              alignment=TA_CENTER,
                              backColor=colors.HexColor('#f5f5f5'), borderPadding=10,
                              spaceBefore=12, spaceAfter=12,
                              textColor=colors.HexColor('#1a5276')))
    
    story = []
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Neurovascular Coil Technical Report", styles['Title']))
    story.append(Paragraph("Finite Math Derivations & Pulse Sequences", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(PageBreak())
    
    # Part 1: Theory (from Markdown)
    md_path = "Quantum_Vascular_Coil_Theory.md"
    if os.path.exists(md_path):
        with open(md_path, 'r') as f:
            md_content = f.read()
            story.extend(parse_markdown_to_flowables(md_content, styles))
    else:
        story.append(Paragraph("Theory file not found.", styles['Normal']))
        
    story.append(PageBreak())

    # Part 1.5: Advanced Sequences Supplement
    story.append(Paragraph("Supplement: Advanced Quantum Sequences", styles['Heading1']))
    supp_path = "Quantum_Advanced_Pulse_Sequences.md"
    if os.path.exists(supp_path):
        with open(supp_path, 'r') as f:
            supp_content = f.read()
            story.extend(parse_markdown_to_flowables(supp_content, styles))
    else:
        story.append(Paragraph("Supplement file not found.", styles['Normal']))
        
    story.append(PageBreak())
    
    # Part 2: Coil Library Derivations
    story.append(Paragraph("Part II: Coil Implementation & Derivations", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    for idx, coil_class in sorted(QUANTUM_VASCULAR_COIL_LIBRARY.items()):
        coil = coil_class()
        
        # Header
        story.append(Paragraph(f"{idx}. {coil.name}", styles['Heading2']))
        
        # specs
        t = Table(create_coil_spec_table(coil_class), colWidths=[2*inch, 3*inch], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.black),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # Derivation (Docstring)
        story.append(Paragraph("Mathematical Derivation:", styles['Heading3']))
        math_text = extract_math_from_docstring(coil_class.__doc__)
        # Convert newlines to <br/> for CENTERED Paragraph execution
        # Escape first!
        math_text_formatted = escape(math_text).replace('\n', '<br/>')
        story.append(Paragraph(math_text_formatted, styles['Equation']))
        
        # Additional Methods Highlighting
        methods = inspect.getmembers(coil_class, predicate=inspect.isfunction)
        for name, func in methods:
            if name.startswith('__'): continue
            if func.__doc__:
                 story.append(Paragraph(f"Method: {name}", styles['Heading4']))
                 story.append(Preformatted(func.__doc__.strip(), styles['Code']))
                 
        story.append(Spacer(1, 0.3*inch))
        # Keep together to avoid orphaned headers? simpler to just stream flowables
        
    story.append(PageBreak())
    
    # Part 3: Pulse Sequences
    story.append(Paragraph("Part III: Pulse Sequence Math", styles['Heading1']))
    
    for seq_id, seq_class in ADAPTIVE_SEQUENCES.items():
        seq = seq_class()
        story.append(Paragraph(f"Sequence: {seq.sequence_name}", styles['Heading2']))
        
        story.append(Paragraph("Description & Math:", styles['Heading3']))
        math_text = extract_math_from_docstring(seq_class.__doc__)
        story.append(Paragraph(escape(math_text).replace('\n', '<br/>'), styles['Equation']))
        
        # Methods
        methods = inspect.getmembers(seq_class, predicate=inspect.isfunction)
        for name, func in methods:
             if name in ['generate_sequence', 'adapt_parameters']:
                 story.append(Paragraph(f"Algorithm: {name}", styles['Heading4']))
                 if func.__doc__:
                    story.append(Preformatted(func.__doc__.strip(), styles['Code']))
                 
        story.append(Spacer(1, 0.2*inch))

    doc.build(story)
    print(f"Report generated: {output_filename}")

if __name__ == "__main__":
    generate_report()
