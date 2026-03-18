#!/usr/bin/env python3
"""
Convert markdown technical report to PDF using reportlab.
Handles headers, paragraphs, equations, lists, and formatting.
"""
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

def md_to_pdf_technical_report(md_path, pdf_path):
    """
    Converts a Markdown technical report to PDF.
    Handles headers, paragraphs, equations (as code blocks), and lists.
    """
    try:
        if not os.path.exists(md_path):
            print(f"Error: Markdown file not found at {md_path}")
            return False

        doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor='#1f4788',
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_CENTER
        )
        
        h2_style = ParagraphStyle(
            'CustomH2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#2d5aa8',
            spaceAfter=8,
            spaceBefore=10,
            alignment=TA_LEFT
        )
        
        h3_style = ParagraphStyle(
            'CustomH3',
            parent=styles['Heading3'],
            fontSize=12,
            textColor='#3d6ab8',
            spaceAfter=6,
            spaceBefore=8,
            alignment=TA_LEFT
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        story = []

        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        skip_next = False
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if skip_next:
                skip_next = False
                i += 1
                continue

            stripped = line.rstrip()
            
            # Empty line -> add spacer
            if not stripped:
                story.append(Spacer(1, 0.1*inch))
                i += 1
                continue

            # H1 - Title
            if stripped.startswith('# '):
                title_text = stripped[2:].strip()
                p = Paragraph(title_text, title_style)
                story.append(p)
                story.append(Spacer(1, 0.15*inch))
                i += 1
                continue

            # H2 - Section
            if stripped.startswith('## '):
                heading_text = stripped[3:].strip()
                p = Paragraph(heading_text, h2_style)
                story.append(p)
                story.append(Spacer(1, 0.1*inch))
                i += 1
                continue

            # H3 - Subsection
            if stripped.startswith('### '):
                heading_text = stripped[4:].strip()
                p = Paragraph(heading_text, h3_style)
                story.append(p)
                story.append(Spacer(1, 0.08*inch))
                i += 1
                continue

            # Horizontal rule
            if stripped.startswith('---'):
                story.append(Spacer(1, 0.15*inch))
                i += 1
                continue

            # Math display (latex between $$)
            if stripped.startswith('$$'):
                # Collect the equation block
                eq_lines = [stripped[2:]]
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if next_line.endswith('$$'):
                        eq_lines.append(next_line[:-2])
                        i += 1
                        break
                    eq_lines.append(next_line)
                    i += 1
                
                eq_text = ' '.join(eq_lines).strip()
                # Display as monospace code-like block (simplified representation)
                p = Paragraph(f"<i>[Equation: {eq_text[:100]}...]</i>", styles['Italic'])
                story.append(p)
                story.append(Spacer(1, 0.08*inch))
                continue

            # Unordered list
            if stripped.startswith('- ') or stripped.startswith('* '):
                list_text = stripped[2:].strip()
                p = Paragraph(f"<b>•</b> {list_text}", body_style)
                story.append(p)
                story.append(Spacer(1, 0.04*inch))
                i += 1
                continue

            # Ordered list
            if re.match(r'^\d+\.\s', stripped):
                list_item = re.sub(r'^\d+\.\s', '', stripped).strip()
                match = re.match(r'^(\d+)\.\s', stripped)
                num = match.group(1) if match else "•"
                p = Paragraph(f"<b>{num}.</b> {list_item}", body_style)
                story.append(p)
                story.append(Spacer(1, 0.04*inch))
                i += 1
                continue

            # Regular paragraph
            # Clean up inline markdown
            para_text = stripped
            para_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', para_text)  # Bold
            para_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', para_text)      # Italic
            para_text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', para_text)  # Code
            
            p = Paragraph(para_text, body_style)
            story.append(p)
            story.append(Spacer(1, 0.06*inch))
            i += 1

        doc.build(story)
        print(f"PDF generated successfully: {pdf_path}")
        return True

    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    
    # Default paths
    cwd = os.getcwd()
    md_file = os.path.join(cwd, 'Technical_Report_Head_Coil_Cost_Economics.md')
    pdf_file = os.path.join(cwd, 'Technical_Report_Head_Coil_Cost_Economics.pdf')
    
    # Allow override via command line
    if len(sys.argv) > 1:
        md_file = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_file = sys.argv[2]
    
    print(f"Converting {md_file} to {pdf_file}...")
    success = md_to_pdf_technical_report(md_file, pdf_file)
    sys.exit(0 if success else 1)
