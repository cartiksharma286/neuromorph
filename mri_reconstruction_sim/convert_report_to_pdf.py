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


def _latex_to_readable(text):
    """Convert common LaTeX math tokens to a readable plain-text form for PDF rendering."""
    if not text:
        return ""

    out = text
    out = out.replace('\\\\', ' ')
    out = out.replace('\\left', '').replace('\\right', '')
    out = out.replace('\\cdot', ' * ')
    out = out.replace('\\times', ' x ')
    out = out.replace('\\approx', '~=')
    out = out.replace('\\arg\\min', 'arg min')
    out = out.replace('\\min', 'min')
    out = out.replace('\\max', 'max')
    out = out.replace('\\sum', 'sum')
    out = out.replace('\\in', 'in')
    out = out.replace('\\text', '')
    out = out.replace('\\Omega', 'Omega')
    out = out.replace('\\omega', 'omega')
    out = out.replace('\\theta', 'theta')
    out = out.replace('\\pi', 'pi')
    out = out.replace('\\rho', 'rho')
    out = out.replace('\\epsilon', 'epsilon')
    out = out.replace('\\Delta', 'Delta')
    out = out.replace('\\alpha', 'alpha')
    out = out.replace('\\gamma', 'gamma')
    out = out.replace('\\varphi', 'varphi')
    out = out.replace('\\pmod', 'mod')

    # Convert common superscript/subscript wrappers to inline readable forms.
    out = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', out)
    out = re.sub(r'\^\{([^{}]+)\}', r'^(\1)', out)
    out = re.sub(r'_\{([^{}]+)\}', r'_(\1)', out)

    # Remove remaining braces and collapse whitespace.
    out = out.replace('{', '').replace('}', '')
    out = re.sub(r'\s+', ' ', out).strip()
    return out

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

        equation_style = ParagraphStyle(
            'EquationStyle',
            parent=styles['BodyText'],
            fontName='Courier',
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=4
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
                # Collect equation lines until closing $$ marker.
                eq_lines = []
                first = stripped[2:].strip()
                if first:
                    eq_lines.append(first)
                i += 1
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if next_line.endswith('$$'):
                        tail = next_line[:-2].strip()
                        if tail:
                            eq_lines.append(tail)
                        i += 1
                        break
                    if next_line.strip():
                        eq_lines.append(next_line.strip())
                    i += 1

                # Render full equation without truncation, converting common LaTeX tokens.
                if eq_lines:
                    for eq_line in eq_lines:
                        readable_eq = _latex_to_readable(eq_line)
                        if readable_eq:
                            story.append(Paragraph(readable_eq, equation_style))
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
            para_text = re.sub(
                r'\$(.*?)\$',
                lambda m: f"<font face=\"Courier\">{_latex_to_readable(m.group(1))}</font>",
                para_text
            )
            
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
