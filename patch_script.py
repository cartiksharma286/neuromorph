import re
import sys

file_path = "/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/simulator_core.py"
with open(file_path, "r") as f:
    content = f.read()

pattern = r"(    def _generate_synthetic_phantom\(self\):)"
replacement = r"""    def _generate_shoulder_phantom(self):
        \"\"\"Generates anatomically accurate shoulder phantom.\"\"\"
        N = self.resolution
        self.t1_map = np.zeros(self.dims)
        self.t2_map = np.zeros(self.dims)
        self.pd_map = np.zeros(self.dims)
        
        y, x = np.ogrid[:N, :N]
        center = (N//2, N//2)
        
        mask_humerus = ((x - center[1]*1.2)**2 + (y - center[0])**2 < (N//5)**2)
        self.t1_map[mask_humerus] = 365
        self.t2_map[mask_humerus] = 133
        self.pd_map[mask_humerus] = 0.9
        
        mask_glenoid = ((x - center[1]*0.6)**2 / (N//6)**2 + (y - center[0])**2 / (N//3)**2 < 1) & (x < center[1]*0.8)
        self.t1_map[mask_glenoid] = 365
        self.t2_map[mask_glenoid] = 133
        self.pd_map[mask_glenoid] = 0.9
        
        mask_cart = ((x - center[1]*1.2)**2 + (y - center[0])**2 < (N//4.5)**2) & ~mask_humerus & ~mask_glenoid
        self.t1_map[mask_cart] = 1240
        self.t2_map[mask_cart] = 27
        self.pd_map[mask_cart] = 0.7
        
        mask_muscle = ((x - center[1])**2 / (N//1.5)**2 + (y - center[0])**2 / (N//1.2)**2 < 1) & ~mask_humerus & ~mask_glenoid & ~mask_cart
        self.t1_map[mask_muscle] = 900
        self.t2_map[mask_muscle] = 50
        self.pd_map[mask_muscle] = 0.8
        
        self.vol_t1 = np.repeat(self.t1_map[:, :, np.newaxis], N, axis=2)
        self.vol_t2 = np.repeat(self.t2_map[:, :, np.newaxis], N, axis=2)
        self.vol_pd = np.repeat(self.pd_map[:, :, np.newaxis], N, axis=2)

    def _generate_elbow_phantom(self):
        \"\"\"Generates anatomically accurate elbow phantom.\"\"\"
        N = self.resolution
        self.t1_map = np.zeros(self.dims)
        self.t2_map = np.zeros(self.dims)
        self.pd_map = np.zeros(self.dims)
        
        y, x = np.ogrid[:N, :N]
        center = (N//2, N//2)
        
        mask_humerus = ((x - center[1])**2 / (N//6)**2 + (y - center[0] + N//3)**2 / (N//2.5)**2 < 1) & (y < center[0])
        self.t1_map[mask_humerus] = 365
        self.t2_map[mask_humerus] = 133
        self.pd_map[mask_humerus] = 0.9
        
        mask_ulna = ((x - center[1]*1.1)**2 / (N//7)**2 + (y - center[0] - N//3)**2 / (N//2.5)**2 < 1) & (y >= center[0])
        self.t1_map[mask_ulna] = 365
        self.t2_map[mask_ulna] = 133
        self.pd_map[mask_ulna] = 0.9
        
        mask_radius = ((x - center[1]*0.8)**2 / (N//7)**2 + (y - center[0] - N//3)**2 / (N//2.5)**2 < 1) & (y >= center[0])
        self.t1_map[mask_radius] = 365
        self.t2_map[mask_radius] = 133
        self.pd_map[mask_radius] = 0.9
        
        mask_cart = ((x - center[1])**2 / (N//4)**2 + (y - center[0])**2 / (N//8)**2 < 1) & ~mask_humerus & ~mask_ulna & ~mask_radius
        self.t1_map[mask_cart] = 1240
        self.t2_map[mask_cart] = 27
        self.pd_map[mask_cart] = 0.7
        
        mask_bg = ((x - center[1])**2 / (N//1.5)**2 + (y - center[0])**2 / (N//1.2)**2 < 1) & ~mask_humerus & ~mask_ulna & ~mask_radius & ~mask_cart
        self.t1_map[mask_bg] = 900
        self.t2_map[mask_bg] = 50
        self.pd_map[mask_bg] = 0.8
        
        self.vol_t1 = np.repeat(self.t1_map[:, :, np.newaxis], N, axis=2)
        self.vol_t2 = np.repeat(self.t2_map[:, :, np.newaxis], N, axis=2)
        self.vol_pd = np.repeat(self.pd_map[:, :, np.newaxis], N, axis=2)

\g<1>"""

if "def _generate_shoulder_phantom" not in content:
    content = re.sub(pattern, replacement, content)
    with open(file_path, "w") as f:
        f.write(content)
        print("Successfully injected.")
else:
    print("Already injected.")