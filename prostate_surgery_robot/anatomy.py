
import numpy as np

class ProstatePhantom:
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        self.image = self.generate_anatomy()
        
    def generate_anatomy(self):
        # Generate Synthetic Transverse T2 Prostate MRI
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        
        # 1. Background (Pelvic Fat/Muscle) - Dark Grey
        img = np.random.normal(0.1, 0.02, (self.width, self.height))
        
        # 2. Prostate Gland (Walnut, Central) - T2 Bright-ish peripheral zone
        # Peripheral Zone (PZ)
        # Shape: Inverted heart/oval
        radius = np.sqrt(xx**2 + (yy * 1.2)**2)
        pz_mask = (radius < 0.4)
        
        # Transition Zone (TZ) - Darker and central
        tz_mask = (radius < 0.25)
        
        # Add PZ texture (Bright)
        img[pz_mask] = 0.6 + np.random.normal(0, 0.05, np.sum(pz_mask))
        
        # Add TZ texture (Darker/Heterogenous)
        img[tz_mask] = 0.35 + np.random.normal(0, 0.08, np.sum(tz_mask))
        
        # 3. Urethra (Dot in center)
        urethra = (xx**2 + yy**2) < 0.02
        img[urethra] = 0.1
        
        # 4. Rectum (Below prostate, dark/air)
        rectum = ((xx)**2 + (yy + 0.6)**2) < 0.2
        img[rectum] = 0.05
        
        # 5. Tumor (Hypointense/Dark spot in PZ)
        # Location: Right Peripheral Zone
        tumor = ((xx - 0.25)**2 + (yy - 0.1)**2) < 0.05
        img[tumor] = 0.2 # T2 Dark lesion
        
        return img
        
    def get_mask(self):
        return (self.image > 0.15).astype(float)
