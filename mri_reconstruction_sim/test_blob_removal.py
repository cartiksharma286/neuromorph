
import numpy as np
import scipy.ndimage
from advanced_reconstruction import Gemini3SignalEnhancer, QuantumManifoldSignalBooster

def test_blob_removal_and_boost():
    print("Testing Artifact Removal AND Signal Boosting...")
    
    # 1. Create Synthetic Image
    N = 128
    image = np.zeros((N, N))
    
    # Anatomy (Center Oval)
    y, x = np.ogrid[:N, :N]
    mask_anatomy = ((x - N//2)**2 + (y - N//2)**2 <= (N//4)**2)
    # Weak signal that needs boosting
    image[mask_anatomy] = 0.4 
    
    # Add "White Blobs"
    image[10:15, 10:15] = 0.95
    
    # Add noise
    noise = np.random.normal(0, 0.05, (N, N))
    image_noisy = image + noise
    image_noisy = np.clip(image_noisy, 0, 1)
    
    print(f"Original Anatomy Mean (Weak): {np.mean(image_noisy[mask_anatomy]):.4f}")
    
    # 2. Apply Boosting
    booster = QuantumManifoldSignalBooster()
    boosted_image = booster.boost_signal(image_noisy)
    
    boosted_anatomy = np.mean(boosted_image[mask_anatomy])
    print(f"Boosted Anatomy Mean (Expect > 0.4): {boosted_anatomy:.4f}")
    
    # 3. Apply Cleaning
    enhancer = Gemini3SignalEnhancer()
    cleaned_image = enhancer.enhance_signal(boosted_image)
    
    # 4. Verify
    blob_max = np.max(cleaned_image[10:15, 10:15])
    final_anatomy = np.mean(cleaned_image[mask_anatomy])
    
    print(f"Cleaned Blob Max (Expect ~0): {blob_max:.4f}")
    print(f"Final Anatomy Mean: {final_anatomy:.4f}")
    
    if blob_max < 0.1 and final_anatomy > 0.5:
        print("SUCCESS: Signal Boosted & Blobs Removed.")
    else:
        print("FAILURE.")

if __name__ == "__main__":
    test_blob_removal_and_boost()
