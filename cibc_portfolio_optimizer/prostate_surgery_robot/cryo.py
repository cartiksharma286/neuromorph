
import numpy as np

class CryoModel:
    def __init__(self, w, h, mask):
        self.w = w
        self.h = h
        self.ice_map = np.zeros((w, h)) # 0=Liquid, 1=Ice
        self.phase_energy = np.zeros((w, h)) # Latent heat tracking
        
    def apply_cooling(self, x, y):
        # Cryo probe logic
        # Just simple tracking of ice formation for visualization if needed
        # Main physics is in thermometry.py for temperature
        pass
        
    def step(self):
        pass
