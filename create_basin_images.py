import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_psb_image(filename):
    width, height = 1200, 700
    img = Image.new('RGB', (width, height), color='#070f1e')
    draw = ImageDraw.Draw(img)

    # Water background gradient
    for y in range(height):
        r = int(7 + (y / height) * 12)
        g = int(25 + (y / height) * 60)
        b = int(45 + (y / height) * 90)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Water ripples & sediment currents
    for i in range(15):
        cy = 200 + i * 30
        cx = 600 + math.sin(i * 0.5) * 200
        rx = 400 + i * 20
        ry = 60 + i * 8
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(0, 180, 216, 40), width=3)

    # Floating Hydro-Sousveillance Sensor Nodes
    nodes = [(350, 320), (600, 280), (850, 350), (480, 480), (720, 460)]
    for idx, (nx, ny) in enumerate(nodes):
        # Glow ring
        draw.ellipse([nx - 40, ny - 40, nx + 40, ny + 40], outline=(56, 189, 248, 120), width=4)
        draw.ellipse([nx - 20, ny - 20, nx + 20, ny + 20], fill=(14, 165, 233), outline=(255, 255, 255), width=2)
        # Antenna stem
        draw.line([(nx, ny), (nx, ny - 35)], fill=(255, 255, 255), width=3)
        draw.ellipse([nx - 6, ny - 41, nx + 6, ny - 29], fill=(236, 72, 153))

    # Glassmorphic Header Card
    draw.rectangle([60, 50, 1140, 150], fill=(15, 23, 42, 200), outline=(56, 189, 248, 100), width=2)
    
    # Text overlay
    try:
        font_lg = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    draw.text((90, 70), "PETER STREET BASIN - SPADINA QUAY", fill=(255, 255, 255), font=font_lg)
    draw.text((90, 112), "Hydro-Sousveillance Real-Time Sensing Mesh & Eco-Restoration Outfall", fill=(56, 189, 248), font=font_sm)

    img.save(filename, "JPEG", quality=95)
    print(f"✓ Saved {filename}")

def create_cap_image(filename):
    width, height = 1200, 700
    img = Image.new('RGB', (width, height), color='#050b14')
    draw = ImageDraw.Draw(img)

    # Concentric Lens Rings (Phenomenological AR Camera Lens & Cap)
    cx, cy = 600, 350
    radii = [300, 260, 220, 180, 140, 100, 60]
    colors = [(30, 41, 59), (15, 23, 42), (2, 132, 199), (14, 165, 233), (56, 189, 248), (236, 72, 153), (255, 255, 255)]

    for r, col in zip(radii, colors):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(255, 255, 255, 100), width=2)

    # Lens reflections & aperture blades
    for a in range(8):
        angle = a * (math.pi / 4)
        x1 = cx + math.cos(angle) * 70
        y1 = cy + math.sin(angle) * 70
        x2 = cx + math.cos(angle + 0.4) * 210
        y2 = cy + math.sin(angle + 0.4) * 210
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 180), width=2)

    # Text overlay
    try:
        font_lg = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    draw.rectangle([60, 50, 1140, 140], fill=(15, 23, 42, 210), outline=(16, 185, 129, 120), width=2)
    draw.text((90, 68), "STEVE MANN PHENOMENOLOGICAL AR (PAR) LENS CAP", fill=(255, 255, 255), font=font_lg)
    draw.text((90, 108), "Sub-Millimeter Optical Water Sensing & Sequential Veillance Spectrometer", fill=(16, 185, 129), font=font_sm)

    img.save(filename, "JPEG", quality=95)
    print(f"✓ Saved {filename}")

if __name__ == '__main__':
    os.makedirs('/Users/cartiksharma/Downloads/neuromorph-main-10/hydrostar/static', exist_ok=True)
    os.makedirs('/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity-2/static', exist_ok=True)
    
    create_psb_image('/Users/cartiksharma/Downloads/neuromorph-main-10/hydrostar/static/psb.jpg')
    create_cap_image('/Users/cartiksharma/Downloads/neuromorph-main-10/hydrostar/static/cap.jpg')
    
    create_psb_image('/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity-2/static/psb.jpg')
    create_cap_image('/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity-2/static/cap.jpg')
