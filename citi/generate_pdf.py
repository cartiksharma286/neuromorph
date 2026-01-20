from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from html import escape

def render_equation(equation_text):
    """Renders a LaTeX equation to a PNG image in memory."""
    try:
        # Create a figure
        # approximate size, will be cropped
        fig = plt.figure(figsize=(8, 1))
        
        # Add text - standard matplotlib mathtext uses $...$
        # The input already is stripped of $$, so we wrap it in $ for display math style
        # We assume the user's latex is compatible with matplotlib mathtext
        plt.text(0.5, 0.5, f"${equation_text}$", fontsize=16, ha='center', va='center')
        
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Failed to render equation: {equation_text} -> {e}")
        return None

def create_pdf(markdown_file, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles
    styles.add(ParagraphStyle(name='MathFallback', parent=styles['Code'], backColor=colors.whitesmoke, borderColor=colors.lightgrey, borderWidth=1, leftIndent=20, rightIndent=20, spaceBefore=10, spaceAfter=10, fontName='Courier'))
    
    with open(markdown_file, 'r') as f:
        lines = f.readlines()

    in_table = False
    table_data = []

    for line in lines:
        line = line.strip()
        
        if line.startswith('# '):
            style = styles['Title']
            text = line[2:]
            story.append(Paragraph(escape(text), style))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):
            style = styles['Heading2']
            text = line[3:]
            story.append(Spacer(1, 10))
            story.append(Paragraph(escape(text), style))
            story.append(Spacer(1, 6))
        elif line.startswith('### '):
            style = styles['Heading3']
            text = line[4:]
            story.append(Spacer(1, 10))
            story.append(Paragraph(escape(text), style))
            story.append(Spacer(1, 6))
        elif line.startswith('$$'):
            # Math Block
            eqn_text = line.replace('$$', '').strip()
            img_buf = render_equation(eqn_text)
            
            if img_buf:
                # Create ReportLab Image
                # We need to determine width/height to scale properly if needed
                # For now, let's trust auto-sizing or set a constrained width
                img = Image(img_buf)
                
                # Simple scaling to ensure it fits on page (approx 500pt width available)
                # If image is too wide, scale down
                max_width = 450
                if img.drawWidth > max_width:
                    ratio = max_width / img.drawWidth
                    img.drawWidth = max_width
                    img.drawHeight = img.drawHeight * ratio
                
                story.append(Spacer(1, 6))
                story.append(img)
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(escape(eqn_text), styles['MathFallback']))
                
        elif line.startswith('|'):
            in_table = True
            # Split and clean
            row = [cell.strip() for cell in line.strip('|').split('|')]
            if '---' in row[0]: # Skip separator line
                continue
            
            # Simple handling of internal math in table cells? 
            # Rendering inline math in table cells is hard with RL + Matplotlib images
            # stick to text for table cells, maybe simplistic replace
            processed_row = []
            for cell in row:
                # minimal replacements
                c_text = escape(cell)
                c_text = c_text.replace('$t_1$', 't1').replace('$$', '$') # simplified
                processed_row.append(Paragraph(c_text, styles['Normal']))
                
            table_data.append(processed_row)
            
        elif line == '---':
            story.append(Spacer(1, 12))
            story.append(Paragraph('_' * 60, styles['Normal']))
            story.append(Spacer(1, 12))
        elif line.startswith('* '):
            text = '• ' + escape(line[2:])
            # Inline math hack for bullets?
            # Ideally regex replace $...$ with italic or something, but let's leave as is for robustness
            story.append(Paragraph(text, styles['Normal']))
        else:
            if in_table and table_data:
                # Render Table
                # For reportlab Table, we need the flowables (Paragraphs) or strings
                # We used Paragraphs above
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BOX', (0, 0), (-1, -1), 1, colors.black),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(t)
                in_table = False
                table_data = []
            
            if line:
                text = escape(line)
                # Bold
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                # Italic
                text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
                # Inline code
                text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
                
                # Inline Math (basic approximation)
                # Replace $...$ with italics for visual distinction
                text = re.sub(r'\$(.*?)\$', r'<i>\1</i>', text)

                story.append(Paragraph(text, styles['Normal']))

    # Flush table if at end
    if in_table and table_data:
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)

    doc.build(story)
    print(f"PDF Generated: {output_pdf}")

if __name__ == "__main__":
    create_pdf('Citi_Optimizer_Finite_Math_Report.md', 'Citi_Optimizer_Finite_Math_Report.pdf')
