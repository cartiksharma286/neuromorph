#!/usr/bin/env python3
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def clean_latex_math(text):
    # Map LaTeX math sequences to clean unicode/plain-text representation for ReportLab
    replacements = [
        (r'\\nabla\^2', '∇²'),
        (r'\\nabla', '∇'),
        (r'\\approx', '≈'),
        (r'\\bar', 'bar'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'[\1] / \2'),
        (r'\\quad', '   '),
        (r'\\forall', '∀'),
        (r'\\in', '∈'),
        (r'\\dots', '...'),
        (r'\\mathbb\{R\}', 'ℝ'),
        (r'\\mathbb\{N\}', 'ℕ'),
        (r'\\mathbb\{Z\}', 'ℤ'),
        (r'\\times', '×'),
        (r'\\{', '{'),
        (r'\\}', '}'),
        (r'\\sum_\{([^}]+)\}', r'∑_{\1}'),
        (r'\\min_\{([^}]+)\}', r'min_{\1}'),
        (r'\\mathcal\{C\}', 'C'),
        (r'\\mathcal\{([^}]+)\}', r'\1'),
        (r'\\omega', 'ω'),
        (r'\\Omega', 'Ω'),
        (r'\\phi', 'φ'),
        (r'\\Phi', 'Φ'),
        (r'\\alpha', 'α'),
        (r'\\beta', 'β'),
        (r'\\gamma', 'γ'),
        (r'\\delta', 'δ'),
        (r'\\epsilon', 'ε'),
        (r'\\theta', 'θ'),
        (r'\\lambda', 'λ'),
        (r'\\sigma', 'σ'),
        (r'\\tau', 'τ'),
        (r'\\pi', 'π'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\begin\{cases\}', '\n'),
        (r'\\end\{cases\}', ''),
        (r'([^&]+)&\s*([^\\\n]+)\\\\?', r'\1 \2\n'), # cases styling
    ]
    
    cleaned = text
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned)
        
    # Remove remaining backslashes
    cleaned = cleaned.replace('\\', '')
    
    # Escape XML special characters to make them safe inside ReportLab Paragraph blocks
    cleaned = cleaned.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return cleaned.strip()

def parse_tex_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip latex comments
    content = re.sub(r'%.*?\n', '\n', content)

    # Extract title
    title_match = re.search(r'\\title\{\\textbf\{\\Large\s*(.*?)\s*\}\}', content, re.DOTALL)
    if not title_match:
        title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Quantum Matrix Partitioning for FEA of Torispherical Reactor Vessels"
    # clean formatting from title
    title = re.sub(r'\\textbf|\\Large|\\large|\\Huge|\\huge|\{|\}', '', title)

    # Extract author
    author_match = re.search(r'\\author\{\\textbf\{\s*(.*?)\s*\}\}', content, re.DOTALL)
    if not author_match:
        author_match = re.search(r'\\author\{(.*?)\}', content, re.DOTALL)
    author = author_match.group(1).strip() if author_match else "Antigravity AI Division"
    author = re.sub(r'\\textbf|\{|\}', '', author)

    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}', content, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    abstract = re.sub(r'\\noindent\s*\\textbf\{Abstract:\}\s*', '', abstract)
    # Escape HTML characters in abstract
    abstract = abstract.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Get body (content between \begin{document} and \end{document})
    body_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', content, re.DOTALL)
    body_content = body_match.group(1) if body_match else content

    # Parse sections, subsections, paragraphs, and equations
    elements = []
    lines = body_content.split('\n')
    
    current_section = None
    i = 0
    in_equation = False
    eq_buffer = []
    
    text_buffer = []
    
    def flush_text_buffer():
        if text_buffer:
            paragraph_text = ' '.join(text_buffer)
            
            # Split the paragraph by inline math to escape non-math text safely
            parts = re.split(r'(\$.+?\$.*?)', paragraph_text)
            for idx in range(len(parts)):
                # If this part is an inline math block, strip $ and clean the latex
                if parts[idx].startswith('$') and parts[idx].endswith('$'):
                    parts[idx] = clean_latex_math(parts[idx][1:-1])
                else:
                    # Escape non-math text XML characters
                    parts[idx] = parts[idx].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            paragraph_text = ''.join(parts)
            
            # Clean general latex styling and map to HTML tags
            paragraph_text = re.sub(r'\\noindent\s*', '', paragraph_text)
            paragraph_text = re.sub(r'\\Large|\\large|\\Huge|\\huge', '', paragraph_text)
            paragraph_text = re.sub(r'\\textbf\{([^}]+)\}', r'<b>\1</b>', paragraph_text)
            paragraph_text = re.sub(r'\\textit\{([^}]+)\}', r'<i>\1</i>', paragraph_text)
            paragraph_text = paragraph_text.replace('~', ' ')
            
            elements.append(('body', paragraph_text))
            text_buffer.clear()

    while i < len(lines):
        line = lines[i].strip()
        
        # Skip doc structure commands
        if not line or line.startswith('\\maketitle') or line.startswith('\\twocolumn') or line.startswith('\\begin{abstract}') or line.startswith('\\end{abstract}') or line.startswith('\\begin{@twocolumnfalse}') or line.startswith('\\end{@twocolumnfalse}'):
            i += 1
            continue
            
        if line.startswith('\\begin{equation}'):
            flush_text_buffer()
            in_equation = True
            eq_buffer = []
            i += 1
            continue
            
        if line.startswith('\\end{equation}'):
            in_equation = False
            eq_text = '\n'.join(eq_buffer)
            elements.append(('equation', clean_latex_math(eq_text)))
            eq_buffer = []
            i += 1
            continue
            
        if in_equation:
            eq_buffer.append(line)
            i += 1
            continue
            
        if line.startswith('\\section'):
            flush_text_buffer()
            sec_title = re.search(r'\\section\*?\{([^}]+)\}', line)
            if sec_title:
                elements.append(('section', sec_title.group(1)))
            i += 1
            continue
            
        if line.startswith('\\subsection'):
            flush_text_buffer()
            subsec_title = re.search(r'\\subsection\*?\{([^}]+)\}', line)
            if subsec_title:
                elements.append(('subsection', subsec_title.group(1)))
            i += 1
            continue
            
        # Standard text line
        text_buffer.append(line)
        i += 1
        
    flush_text_buffer()
    
    return title, author, abstract, elements

def compile_pdf(tex_path, pdf_path):
    title, author, abstract, parsed_elements = parse_tex_file(tex_path)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles matching the publication aesthetics
    title_style = ParagraphStyle(
        'NatureTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#111111'),
        alignment=TA_CENTER,
        spaceAfter=15,
        fontName='Helvetica-Bold'
    )
    
    author_style = ParagraphStyle(
        'NatureAuthor',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#444444'),
        spaceAfter=15,
        fontName='Helvetica-Bold'
    )
    
    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        fontName='Helvetica-Bold',
        spaceAfter=4
    )
    
    abstract_style = ParagraphStyle(
        'NatureAbstract',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        alignment=TA_JUSTIFY,
        fontName='Helvetica-Oblique',
        leftIndent=0.25*inch,
        rightIndent=0.25*inch,
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'NatureSection',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#111111'),
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    subsection_style = ParagraphStyle(
        'NatureSubSection',
        parent=styles['Heading3'],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#333333'),
        spaceBefore=8,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'NatureBody',
        parent=styles['BodyText'],
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        textColor=colors.HexColor('#222222')
    )
    
    math_style = ParagraphStyle(
        'NatureMath',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        fontName='Times-Italic',
        spaceBefore=6,
        spaceAfter=6,
        textColor=colors.HexColor('#1a365d')
    )

    story = []
    
    # Title & Author
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(author, author_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Abstract
    if abstract:
        story.append(Paragraph("<b>Abstract</b>", abstract_heading_style))
        story.append(Paragraph(abstract, abstract_style))
        story.append(Spacer(1, 0.1*inch))
        
    # Render parsed sections and text
    for el_type, val in parsed_elements:
        if el_type == 'section':
            story.append(Paragraph(val, section_style))
        elif el_type == 'subsection':
            story.append(Paragraph(val, subsection_style))
        elif el_type == 'equation':
            # Format equations with slight indent/styling
            formatted_val = val.replace('\n', '<br/>')
            story.append(Paragraph(formatted_val, math_style))
        elif el_type == 'body':
            story.append(Paragraph(val, body_style))
            
    doc.build(story)
    print(f"Successfully generated PDF: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tex_file = os.path.join(base_dir, 'nature_fea_report.tex')
    pdf_file = os.path.join(base_dir, 'nature_fea_report.pdf')
    
    if os.path.exists(tex_file):
        compile_pdf(tex_file, pdf_file)
    else:
        print(f"Error: {tex_file} not found!")
