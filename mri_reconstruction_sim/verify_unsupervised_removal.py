
import numpy as np
import scipy.ndimage
from simulator_core import MRIReconstructionSimulator

def test_unsupervised_blob_removal():
    print("Verifying Unsupervised Blob Removal...")
    
    # 1. Initialize Simulator
    sim = MRIReconstructionSimulator(resolution=128)
    
    # 2. Create Synthetic Image with Blobs
    N = 128
    image = np.zeros((N, N))
    
    # Anatomy (Circle of Willis-like bright structure)
    y, x = np.ogrid[:N, :N]
    mask_anatomy = ((x - N//2)**2 + (y - N//2)**2 <= (N//6)**2) & \
                  ((x - N//2)**2 + (y - N//2)**2 >= (N//8)**2)
    image[mask_anatomy] = 0.8
    
    # Add White Blobs (Small isolated features)
    image[20:23, 20:23] = 1.0  # Blob 1
    image[100:102, 100:102] = 0.95 # Blob 2
    
    # Add background noise
    image += np.random.normal(0, 0.05, (N, N))
    image = np.clip(image, 0, 1.2)
    
    print(f"Original Blob Area Max: {np.max(image[20:23, 20:23]):.4f}")
    print(f"Original Anatomy Mean: {np.mean(image[mask_anatomy]):.4f}")
    
    # 3. Apply Unsupervised Removal
    cleaned = sim._remove_quantum_artifacts(image)
    
    # 4. Verify
    blob_after = np.max(cleaned[20:23, 20:23])
    anatomy_after = np.mean(cleaned[mask_anatomy])
    
    print(f"Cleaned Blob Area Max: {blob_after:.4f}")
    print(f"Cleaned Anatomy Mean: {anatomy_after:.4f}")
    
    # Success Criteria:
    # 1. Blob intensity significantly reduced
    # 2. Anatomy intensity preserved (relatively)
    if blob_after < 0.3 and anatomy_after > 0.5:
        print("SUCCESS: Unsupervised removal cleaned blobs and preserved anatomy.")
        return True
    else:
        print("FAILURE: Artifacts remain or anatomy suppressed.")
        return False

if __name__ == "__main__":
    test_unsupervised_blob_removal()
