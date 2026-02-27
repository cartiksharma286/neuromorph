import numpy as np
import scipy.ndimage
from simulator_core import MRIReconstructionSimulator
from advanced_reconstruction import QuantumNoiseSuppressor

def test_black_blob_removal():
    print("Testing Black Blob Removal AND Signal Boosting...")
    
    # 1. Initialize Simulator
    suppressor = QuantumNoiseSuppressor()
    
    # 2. Create Synthetic Image with Blobs
    N = 128
    image = np.zeros((N, N))
    
    # Anatomy (Circle of Willis-like bright structure)
    y, x = np.ogrid[:N, :N]
    mask_anatomy = ((x - N//2)**2 + (y - N//2)**2 <= (N//6)**2) & \
                  ((x - N//2)**2 + (y - N//2)**2 >= (N//8)**2)
    image[mask_anatomy] = 0.8
    
    # Add Black Blobs (Small isolated dark features inside tissue)
    # Give them a value lower than the tissue they sit on
    image[60:64, 60:64] = 0.1  # Black Blob 1
    image[70:72, 70:72] = 0.05 # Black Blob 2
    
    # Add background noise
    image += np.random.normal(0, 0.05, (N, N))
    image = np.clip(image, 0, 1.2)
    
    print(f"Original Black Blob Min (Expect ~0): {np.min(image[60:64, 60:64]):.4f}")
    print(f"Original Anatomy Mean: {np.mean(image[mask_anatomy]):.4f}")
    
    # 3. Apply Quantum Suppression
    cleaned = suppressor.suppress_noise(image)
    
    # 4. Verify
    blob_after = np.min(cleaned[60:64, 60:64])
    anatomy_after = np.mean(cleaned[mask_anatomy])
    
    print(f"Cleaned Black Blob Min (Expect > Original): {blob_after:.4f}")
    print(f"Cleaned Anatomy Mean: {anatomy_after:.4f}")
    
    if blob_after > 0.4 and anatomy_after > 0.5:
        print("SUCCESS: Black Blobs Removed & Anatomy Preserved.")
        return True
    else:
        print("FAILURE: Black artifacts remain or anatomy suppressed.")
        return False

if __name__ == "__main__":
    test_black_blob_removal()
