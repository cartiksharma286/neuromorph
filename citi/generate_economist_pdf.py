from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from html import escape

def render_equation(equation_text):
    try:
        fig = plt.figure(figsize=(8, 1))
        plt.text(0.5, 0.5, f"${equation_text}$", fontsize=14, ha='center', va='center', color='black')
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
    doc = SimpleDocTemplate(output_pdf, pagesize=LETTER,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    economist_title = ParagraphStyle(
        name='EcoTitle',
        fontName='Times-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.black,
        spaceAfter=14
    )
    economist_banner = ParagraphStyle(
        name='EcoBanner',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.white,
        backColor=colors.red,
        borderPadding=(4, 8, 4, 8),
        spaceAfter=15,
        alignment=1
    )
    economist_subtitle = ParagraphStyle(
        name='EcoSub',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.darkgrey,
        spaceAfter=14
    )
    economist_body = ParagraphStyle(
        name='EcoBody',
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        spaceAfter=10,
        alignment=4
    )
    economist_h3 = ParagraphStyle(
        name='EcoH3',
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black
    )

    story.append(Paragraph("THE ECONOMIST | QUANTITATIVE FINANCE", economist_banner))

    with open(markdown_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        if line.startswith('# '):
            text = line[2:]
            story.append(Paragraph(escape(text), economist_title))
        elif line.startswith('## '):
            text = line[3:]
            story.append(Paragraph(escape(text), economist_subtitle))
        elif line.startswith('### '):
            text = line[4:]
            story.append(Paragraph(escape(text), economist_h3))
        elif line.startswith('$$'):
            eqn_text = line.replace('$$', '').strip()
            img_buf = render_equation(eqn_text)
            if img_buf:
                img = Image(img_buf)
                max_width = 450
                if img.drawWidth > max_width:
                    ratio = max_width / img.drawWidth
                    img.drawWidth = max_width
                    img.drawHeight = img.drawHeight * ratio
                story.append(Spacer(1, 10))
                story.append(img)
                story.append(Spacer(1, 10))
        elif line:
            text = escape(line)
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
            text = re.sub(r'\$(.*?)\$', r'<i>\1</i>', text)
            story.append(Paragraph(text, economist_body))

    doc.build(story)
    print(f"PDF Generated: {output_pdf}")

if __name__ == "__main__":
    create_pdf('economist_portfolio_report.md', 'nature_economist_portfolio_report.pdf')