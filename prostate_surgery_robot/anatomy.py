
import numpy as np
from PIL import Image
import os

class ProstatePhantom:
    def __init__(self, width=128, height=128):
        self.width = width
        self.height = height
        self.image = self.load_anatomy()
        
    def load_anatomy(self):
        # Path to collected MRI slice
        path = "prostate_surgery_robot/assets/mri_slice.png"
        try:
            if os.path.exists(path):
                img = Image.open(path).convert('L') # Grayscale
                img = img.resize((self.width, self.height))
                # Normalize 0-1
                arr = np.array(img).astype(float) / 255.0
                return arr
            else:
                return self.generate_anatomy()
        except Exception as e:
            print(f"Error loading MRI: {e}")
            return self.generate_anatomy()

    def generate_anatomy(self):
        # Fallback: Synthetic Transverse T2 Prostate MRI
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        
        # 1. Background (Pelvic Fat/Muscle) - Dark Grey
        img = np.random.normal(0.1, 0.02, (self.width, self.height))
        
        # 2. Prostate Gland (Walnut, Central) - T2 Bright-ish peripheral zone
        # Peripheral Zone (PZ)
        radius = np.sqrt(xx**2 + (yy * 1.2)**2)
        pz_mask = (radius < 0.4)
        
        # Transition Zone (TZ)
        tz_mask = (radius < 0.25)
        
        # Add PZ texture (Bright)
        img[pz_mask] = 0.6 + np.random.normal(0, 0.05, np.sum(pz_mask))
        
        # Add TZ texture (Darker/Heterogenous)
        img[tz_mask] = 0.35 + np.random.normal(0, 0.08, np.sum(tz_mask))
        
        # 3. Urethra
        urethra = (xx**2 + yy**2) < 0.02
        img[urethra] = 0.1
        
        # 4. Rectum
        rectum = ((xx)**2 + (yy + 0.6)**2) < 0.2
        img[rectum] = 0.05
        
        # 5. Tumor (Hypointense/Dark spot in PZ) - centered at 0.25, 0.1
        tumor = ((xx - 0.25)**2 + (yy - 0.1)**2) < 0.05
        img[tumor] = 0.2
        
        # SAVE MASK
        self.tumor_mask = tumor.astype(float)
        
        return img
        
    def get_mask(self):
        return (self.image > 0.15).astype(float)
        
    def get_tumor_mask(self):
        if hasattr(self, 'tumor_mask'):
            return self.tumor_mask
        else:
            # Fallback for loaded images
            xx, yy = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
            return (((xx - 0.25)**2 + (yy - 0.1)**2) < 0.05).astype(float)
