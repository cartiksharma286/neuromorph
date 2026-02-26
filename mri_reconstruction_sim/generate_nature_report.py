#!/usr/bin/env python3
import os
import re
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch

# Matplotlib configuration for equation rendering
plt.rc('text', usetex=False) # Use internal mathtext parser (no external latex needed)
plt.rc('font', family='serif')

def render_equation(eq_text, filename, dpi=300):
    """
    Renders a LaTeX equation to an image file using Matplotlib.
    """
    # Create a figure
    fig = plt.figure(figsize=(6, 1))
    # Add text (equation)
    # Wrap in $...$ for inline math if not already present, 
    # but usually display math is passed without $$ in some parsers.
    # Here we expect the raw tex content.
    
    # Strip $$ if present
    clean_eq = eq_text.strip()
    if clean_eq.startswith('$$') and clean_eq.endswith('$$'):
        clean_eq = clean_eq[2:-2]
    
    # Matplotlib needs $ for math mode
    # Replace \pmod which is unsupported in Matplotlib mathtext
    clean_eq = clean_eq.replace(r'\pmod', r'\text{ mod }')
    tex = f"${clean_eq}$"
    
    try:
        fig.text(0.5, 0.5, tex, fontsize=14, ha='center', va='center')
        plt.axis('off')
        
        # Save to buffer/file
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to render equation: {clean_eq}. Error: {e}")
        plt.close(fig)
        return False

def parse_markdown(md_content, story, styles):
    """
    Parses markdown and adds flowables to the story.
    """
    lines = md_content.split('\n')
    
    # Text Styles
    title_style = styles['NatureTitle']
    h1_style = styles['NatureH1']
    h2_style = styles['NatureH2']
    body_style = styles['NatureBody']
    caption_style = styles['NatureCaption']
    
    buffer_text = []
    
    def flush_buffer():
        if buffer_text:
            text = ' '.join(buffer_text)
            # Basic bold/italic parsing
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            story.append(Paragraph(text, body_style))
            buffer_text.clear()

    img_counter = 0

    for line in lines:
        line = line.strip()
        
        if not line:
            flush_buffer()
            story.append(Spacer(1, 0.1*inch))
            continue
            
        if line.startswith('# '):
            flush_buffer()
            story.append(Paragraph(line[2:], title_style))
            story.append(Spacer(1, 0.2*inch))
            
        elif line.startswith('## '):
            flush_buffer()
            story.append(Paragraph(line[3:], h1_style))
            
        elif line.startswith('### '):
            flush_buffer()
            story.append(Paragraph(line[4:], h2_style))
            
        elif line.startswith('$$') and line.endswith('$$'):
            flush_buffer()
            # Equation Block
            eq_tex = line
            img_filename = f"eq_{img_counter}.png"
            if render_equation(eq_tex, img_filename):
                # Add Image
                try:
                    img = Image(img_filename)
                    # Scale if too wide?
                    max_width = 4*inch
                    ratio = max_width / img.drawWidth
                    img.drawWidth = max_width
                    img.drawHeight *= ratio
                    story.append(img)
                    img_counter += 1
                except:
                    story.append(Paragraph(f"[Equation Render Error]", caption_style))
            else:
                story.append(Paragraph(f"[Equation Error: {line}]", caption_style))
                
        elif line.startswith('![') and '](' in line:
            flush_buffer()
            # Image Reference: ![caption](file:///path/to/image.png)
            match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if match:
                caption, img_path = match.groups()
                # Remove file:// prefix if present
                if img_path.startswith('file://'):
                    img_path = img_path.replace('file://', '')
                
                if os.path.exists(img_path):
                    try:
                        img = Image(img_path)
                        max_width = 5.5*inch
                        if img.drawWidth > max_width:
                            ratio = max_width / img.drawWidth
                            img.drawWidth = max_width
                            img.drawHeight *= ratio
                        story.append(img)
                        if caption:
                            story.append(Paragraph(caption, caption_style))
                        story.append(Spacer(1, 0.1*inch))
                    except Exception as e:
                        story.append(Paragraph(f"[Image Error: {e}]", caption_style))
                else:
                    story.append(Paragraph(f"[Image Not Found: {img_path}]", caption_style))

        elif line.startswith('- ') or line.startswith('* '):
            flush_buffer()
            item_text = line[2:]
            item_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item_text)
            story.append(Paragraph(f"• {item_text}", body_style))
            
        else:
            buffer_text.append(line)
            
    flush_buffer()

def generate_nature_report():
    output_filename = "Nature_Quantum_Pulse_Sequences.pdf"
    
    # Nature-style page setup
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles mimicking Nature
    styles.add(ParagraphStyle(name='NatureTitle', fontName='Times-Bold', fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name='NatureH1', fontName='Times-Bold', fontSize=12, leading=14, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='NatureH2', fontName='Times-Italic', fontSize=11, leading=13, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name='NatureBody', fontName='Times-Roman', fontSize=10, leading=12, alignment=0)) # 0=Left, 4=Justify
    styles.add(ParagraphStyle(name='NatureCaption', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.gray))
    
    story = []
    
    # Read Markdown
    if os.path.exists("Nature_Quantum_Pulse_Sequences.md"):
        with open("Nature_Quantum_Pulse_Sequences.md", 'r') as f:
            md_content = f.read()
        parse_markdown(md_content, story, styles)
    else:
        story.append(Paragraph("Error: Source markdown file not found.", styles['NatureBody']))
    
    doc.build(story)
    print(f"Generated: {output_filename}")
    
    # Cleanup images
    # for f in os.listdir('.'):
    #     if f.startswith('eq_') and f.endswith('.png'):
    #         os.remove(f)

if __name__ == "__main__":
    generate_nature_report()
