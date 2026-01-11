#!/usr/bin/env python3
"""
Generate PDF report with embedded simulation images.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re
import os

import matplotlib.pyplot as plt

IMAGE_DIR = '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/report_images'

def render_equation_img(tex, path):
    """Renders a LaTeX equation to an image file using Matplotlib."""
    try:
        # Create figure
        # Estimate width/height based on length? hard...
        # Matplotlib bbox_inches='tight' handles it mostly.
        fig = plt.figure(figsize=(6, 1))
        
        # Clean tex
        tex = tex.strip()
        
        # Render
        # We need to wrap in $ if not present, but usually block math uses $ or $$.
        # Matplotlib expects $...$ for math mode.
        # If user provided $$ ... $$, remove them and wrap in $ ... $ for rendering in matplotlib text
        clean_tex = tex.replace('$$', '').replace('\\[', '').replace('\\]', '').strip()
        
        plt.text(0.5, 0.5, f"${clean_tex}$", fontsize=16, ha='center', va='center')
        plt.axis('off')
        plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to render eq: {e}")
        return False

def md_to_pdf(md_path, pdf_path):
    # Read markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title_Custom', fontSize=18, leading=22, spaceAfter=20, alignment=TA_CENTER, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='Heading1_Custom', fontSize=14, leading=18, spaceAfter=12, spaceBefore=16, textColor=colors.darkblue, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Heading2_Custom', fontSize=12, leading=15, spaceAfter=8, spaceBefore=10, textColor=colors.Color(0.2, 0.3, 0.5), fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Body_Custom', fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='Equation', fontSize=10, leading=14, fontName='Courier', backColor=colors.Color(0.95, 0.95, 0.98), leftIndent=20, rightIndent=20, spaceBefore=8, spaceAfter=8, borderPadding=6))
    styles.add(ParagraphStyle(name='BulletItem', fontSize=10, leading=14, leftIndent=20, bulletIndent=10))
    styles.add(ParagraphStyle(name='Separator', fontSize=6, spaceAfter=10, spaceBefore=10, textColor=colors.grey))
    styles.add(ParagraphStyle(name='Caption', fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=12))
    
    story = []
    
    # Process lines
    lines = md_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Horizontal rule
        if line.strip() == '---':
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph('─' * 60, styles['Separator']))
            story.append(Spacer(1, 0.1*inch))
        # Title
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Title_Custom']))
        # H2
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading1_Custom']))
        # H3
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading2_Custom']))
        # Equation block ($$)
        elif line.strip().startswith('$$'):
            eq_text = line.strip().replace('$$', '')
            # If it opens and closes on same line
            if line.count('$$') == 2:
                pass # eq_text is ready
            else:
                # Multi-line
                while i + 1 < len(lines) and not'$$' in lines[i+1]:
                    i += 1
                    eq_text += ' ' + lines[i].strip() # Join with space for matplotlib
                if i + 1 < len(lines):
                    i += 1
                    eq_text += ' ' + lines[i].replace('$$', '').strip()
            
            # Render Image
            eq_filename = f"eq_{i}.png"
            eq_path = os.path.join(os.path.dirname(pdf_path), eq_filename)
            
            if render_equation_img(eq_text, eq_path):
                img = Image(eq_path, height=0.5*inch, kind='proportional') # Let width match proportional
                # Check aspect to avoid tiny images? or Huge?
                # reportlab Image kind='proportional' works well if we set one dim.
                # Actually, set width to something reasonable, let height scale?
                # Or set height cap.
                # Let's set max width 6 inch, max height 2 inch?
                # Image(path, width=.., height=..)
                
                # We don't know the aspect ratio without opening it, assume matplotlib saved it well.
                # Let's try inserting it without fixed size? ReportLab might complain.
                # Better: fix width to None (auto) and height to None? No.
                
                # Let reportlab figure it out? "width=6*inch, height=2*inch, kind='proportional'"
                # It will fit within that box.
                story.append(Spacer(1, 0.1*inch))
                story.append(Image(eq_path, width=4*inch, height=1.5*inch, kind='proportional'))
                story.append(Spacer(1, 0.1*inch))
            else:
                # Fallback
                story.append(Preformatted(eq_text.strip(), styles['Equation']))

        # Equation block (indented with 4 spaces) - Legacy support
        elif line.startswith('    ') and line.strip():
            eq_text = line.strip()
            while i + 1 < len(lines) and lines[i+1].startswith('    ') and lines[i+1].strip():
                i += 1
                eq_text += '\n' + lines[i].strip()
            story.append(Preformatted(eq_text, styles['Equation']))
        # Bullet points
        elif line.strip().startswith('• ') or line.strip().startswith('* '):
            bullet_text = line.strip()[2:]
            story.append(Paragraph(f"• {bullet_text}", styles['BulletItem']))
        # Table detection
        elif line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                table_lines.append(lines[i])
                i += 1
            i -= 1
            
            data = []
            for tl in table_lines:
                if '---' in tl:
                    continue
                cells = [c.strip() for c in tl.split('|')[1:-1]]
                data.append(cells)
            
            if data:
                t = Table(data, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.95, 0.98)),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.Color(0.95, 0.95, 0.98), colors.white])
                ]))
                story.append(Spacer(1, 0.2*inch))
                story.append(t)
                story.append(Spacer(1, 0.2*inch))
        # Bold text
        elif line.strip().startswith('**') and line.strip().endswith('**'):
            text = line.strip()[2:-2]
            story.append(Paragraph(f"<b>{text}</b>", styles['Body_Custom']))
        # Image detection ![Alt](Path)
        elif re.match(r'!\[.*?\]\(.*?\)', line.strip()):
            match = re.search(r'!\[(.*?)\]\((.*?)\)', line.strip())
            if match:
                alt = match.group(1)
                img_path = match.group(2)
                
                # Resolve relative path
                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(img_path)
                    
                if os.path.exists(img_path):
                    # Scale image to fit page width roughly
                    # Try to maintain aspect ratio
                    img = Image(img_path, width=6*inch, height=4.5*inch, kind='proportional')
                    story.append(Spacer(1, 0.1*inch))
                    story.append(img)
                    story.append(Paragraph(alt, styles['Caption'])) 
                    story.append(Spacer(1, 0.2*inch))
                else:
                    story.append(Paragraph(f"[Image not found: {os.path.basename(img_path)}]", styles['Caption']))

        # Regular body text
        elif line.strip():
            formatted = line
            formatted = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', formatted)
            story.append(Paragraph(formatted, styles['Body_Custom']))
        else:
            story.append(Spacer(1, 0.08*inch))
        
        i += 1
    
    # Add page break before images
    # story.append(PageBreak()) 
    # (Removed hardcoded appendix as the new report has inline images)
    
    doc.build(story)
    print(f"PDF generated: {pdf_path}")

if __name__ == '__main__':
    md_to_pdf(
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/coil_circuits_report.md',
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/NeuroPulse_Circuit_Design_Report.pdf'
    )
