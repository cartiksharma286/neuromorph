"""
Immunoassay Analysis Engine
- Plate/Spot Detection
- Signal Quantification
- Standard Curve Fitting (4PL)
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from scipy.optimize import curve_fit

class PlateAnalyzer:
    """Analyze microplate images"""
    
    def __init__(self, rows: int = 8, cols: int = 12):
        self.rows = rows
        self.cols = cols
        
    def detect_wells(self, image: np.ndarray) -> List[Dict]:
        """
        Detect wells in the image using grid heuristics
        Assumes a roughly aligned plate image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for circular-ish blobs that could be wells
        potential_wells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Adjust based on resolution
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                potential_wells.append({'x': x, 'y': y, 'r': radius, 'area': area})
        
        # Sort by Y then X to order them (simplified)
        # In a real scenario, we'd fit a grid model (RANSAC or similar)
        potential_wells.sort(key=lambda w: (int(w['y'] // 50), int(w['x'] // 50)))
        
        # If we didn't find enough, generate a synthetic grid based on image size
        if len(potential_wells) < self.rows * self.cols * 0.5:
            return self._generate_synthetic_grid(image.shape)
            
        return potential_wells[:self.rows * self.cols]

    def _generate_synthetic_grid(self, shape: Tuple[int, int]) -> List[Dict]:
        """Generate a perfect grid if detection fails"""
        h, w = shape[:2]
        margin_x = w * 0.1
        margin_y = h * 0.1
        
        step_x = (w - 2 * margin_x) / (self.cols - 1)
        step_y = (h - 2 * margin_y) / (self.rows - 1)
        
        wells = []
        radius = min(step_x, step_y) * 0.35
        
        for r in range(self.rows):
            for c in range(self.cols):
                wells.append({
                    'x': margin_x + c * step_x,
                    'y': margin_y + r * step_y,
                    'r': radius,
                    'row': r,
                    'col': c,
                    'id': f"{chr(65+r)}{c+1}"
                })
        return wells

    def quantify_wells(self, image: np.ndarray, wells: List[Dict]) -> List[Dict]:
        """Calculate intensity for each well"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        results = []
        for well in wells:
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(well['x']), int(well['y'])), int(well['r']), 255, -1)
            
            # Mean intensity
            mean_val = cv2.mean(gray, mask=mask)[0]
            
            results.append({
                **well,
                'intensity': mean_val,
                'absorbance': -np.log10(mean_val / 255.0 + 1e-6)  # Pseudo-absorbance
            })
            
        return results

class CurveFitter:
    """Standard Curve Fitting (4PL)"""
    
    @staticmethod
    def four_pl(x, a, b, c, d):
        """
        4-Parameter Logistic Model
        a: min asymptote (bottom)
        d: max asymptote (top)
        c: inflection point (EC50)
        b: hill slope
        """
        return d + (a - d) / (1.0 + (x / c) ** b)
    
    def fit_standard_curve(self, concentrations: List[float], signals: List[float]):
        """Fit 4PL model to standards"""
        try:
            # Initial guesses
            a_init = min(signals)
            d_init = max(signals)
            c_init = np.median(concentrations)
            b_init = 1.0
            
            p0 = [a_init, b_init, c_init, d_init]
            
            popt, pcov = curve_fit(self.four_pl, concentrations, signals, p0=p0, maxfev=5000)
            
            r_squared = self._calculate_r_squared(concentrations, signals, popt)
            
            return {
                'params': popt.tolist(),
                'r_squared': r_squared,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _calculate_r_squared(self, x, y, params):
        residuals = y - self.four_pl(x, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_concentration(self, signal: float, params: List[float]) -> float:
        """Inverse 4PL to get concentration from signal"""
        a, b, c, d = params
        # y = d + (a - d) / (1 + (x/c)^b)
        # (y - d) / (a - d) = 1 / (1 + (x/c)^b)
        # (a - d) / (y - d) = 1 + (x/c)^b
        # (a - d) / (y - d) - 1 = (x/c)^b
        # x = c * ((a - d) / (y - d) - 1)^(1/b)
        
        try:
            # Clamp signal to asymptotes to avoid domain errors
            if a < d:
                signal = np.clip(signal, a + 1e-6, d - 1e-6)
            else:
                signal = np.clip(signal, d + 1e-6, a - 1e-6)
                
            term = (a - d) / (signal - d) - 1
            if term <= 0: return 0.0
            
            return c * (term ** (1/b))
        except:
            return 0.0
