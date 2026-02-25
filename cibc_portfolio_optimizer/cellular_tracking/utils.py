"""
Utility Functions for Cell Tracking System
"""

import numpy as np
import cv2
from typing import List, Tuple
import base64


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Preprocess image for analysis"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if target_size is not None:
        gray = cv2.resize(gray, target_size)
    
    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """Data augmentation for training"""
    augmented = [image]
    
    # Rotation
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented.append(rotated)
    
    # Flipping
    augmented.append(cv2.flip(image, 0))  # Vertical
    augmented.append(cv2.flip(image, 1))  # Horizontal
    
    return augmented


def calculate_tracking_metrics(ground_truth: List, predictions: List) -> dict:
    """Calculate tracking accuracy metrics"""
    # Simplified metrics - in practice would use CLEAR MOT metrics
    if len(ground_truth) == 0 or len(predictions) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Placeholder implementation
    accuracy = min(len(predictions), len(ground_truth)) / max(len(predictions), len(ground_truth))
    
    return {
        'accuracy': accuracy,
        'precision': accuracy,
        'recall': accuracy,
        'n_ground_truth': len(ground_truth),
        'n_predictions': len(predictions)
    }


def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string for web display"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    _, buffer = cv2.imencode('.png', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize feature vectors"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized


def generate_sample_data(n_frames: int = 20, n_cells: int = 10) -> List[np.ndarray]:
    """Generate synthetic cell images for demonstration"""
    images = []
    
    for frame_idx in range(n_frames):
        # Create blank image
        image = np.zeros((512, 512), dtype=np.uint8)
        
        # Add cells
        for cell_idx in range(n_cells):
            # Random position with some motion
            x = int(256 + 100 * np.sin(frame_idx * 0.1 + cell_idx))
            y = int(256 + 100 * np.cos(frame_idx * 0.1 + cell_idx * 0.5))
            
            # Random size
            radius = np.random.randint(15, 30)
            
            # Draw cell
            cv2.circle(image, (x, y), radius, 200, -1)
            
            # Add some texture
            noise = np.random.randint(-30, 30, (radius*2, radius*2))
            y1, y2 = max(0, y-radius), min(512, y+radius)
            x1, x2 = max(0, x-radius), min(512, x+radius)
            
            if y2 > y1 and x2 > x1:
                noise_crop = noise[:y2-y1, :x2-x1]
                image[y1:y2, x1:x2] = np.clip(
                    image[y1:y2, x1:x2].astype(int) + noise_crop, 0, 255
                ).astype(np.uint8)
        
        # Add background noise
        background_noise = np.random.randint(0, 20, (512, 512))
        image = np.clip(image.astype(int) + background_noise, 0, 255).astype(np.uint8)
        
        # Blur slightly
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        images.append(image)
    
    return images
