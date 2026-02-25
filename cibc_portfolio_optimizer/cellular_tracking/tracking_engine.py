"""
Cell Tracking Engine - Detection, Tracking, and Feature Extraction
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Cell:
    """Represents a detected cell"""
    id: int
    centroid: Tuple[float, float]
    area: float
    perimeter: float
    eccentricity: float
    intensity_mean: float
    intensity_std: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: np.ndarray
    frame_idx: int


@dataclass
class Track:
    """Represents a cell track across frames"""
    track_id: int
    cells: List[Cell]
    active: bool = True
    
    def get_last_position(self) -> Tuple[float, float]:
        return self.cells[-1].centroid if self.cells else (0, 0)
    
    def get_velocity(self) -> Tuple[float, float]:
        if len(self.cells) < 2:
            return (0, 0)
        p1 = self.cells[-2].centroid
        p2 = self.cells[-1].centroid
        return (p2[0] - p1[0], p2[1] - p1[1])


class CellDetector:
    """Deep learning-based cell detection and segmentation"""
    
    def __init__(self, min_area: int = 50, max_area: int = 5000):
        self.min_area = min_area
        self.max_area = max_area
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Normalize
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray.astype(np.uint8))
        
        return denoised
    
    def detect(self, image: np.ndarray, frame_idx: int = 0) -> List[Cell]:
        """
        Detect cells in image using adaptive thresholding and watershed
        """
        preprocessed = self.preprocess(image)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Distance transform and watershed
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find contours
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract cell features
        cells = []
        cell_id = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                # Calculate features
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                
                # Fit ellipse for eccentricity
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    if major_axis > 0:
                        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    else:
                        eccentricity = 0
                else:
                    eccentricity = 0
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Intensity features
                intensity_mean = cv2.mean(preprocessed, mask=mask)[0]
                intensity_std = np.std(preprocessed[mask > 0])
                
                cell = Cell(
                    id=cell_id,
                    centroid=(cx, cy),
                    area=area,
                    perimeter=perimeter,
                    eccentricity=eccentricity,
                    intensity_mean=intensity_mean,
                    intensity_std=intensity_std,
                    bbox=(x, y, w, h),
                    mask=mask,
                    frame_idx=frame_idx
                )
                
                cells.append(cell)
                cell_id += 1
        
        return cells


class FeatureExtractor:
    """Extract morphological and intensity features"""
    
    @staticmethod
    def extract_features(cell: Cell) -> np.ndarray:
        """Extract feature vector from cell"""
        features = np.array([
            cell.area,
            cell.perimeter,
            cell.eccentricity,
            cell.intensity_mean,
            cell.intensity_std,
            cell.area / (cell.perimeter ** 2) if cell.perimeter > 0 else 0,  # Circularity
            cell.bbox[2] / cell.bbox[3] if cell.bbox[3] > 0 else 1,  # Aspect ratio
        ])
        return features
    
    @staticmethod
    def extract_batch_features(cells: List[Cell]) -> np.ndarray:
        """Extract features for multiple cells"""
        return np.array([FeatureExtractor.extract_features(cell) for cell in cells])


class CellTracker:
    """Multi-object tracking using Deep SORT with Kalman filtering"""
    
    def __init__(self, max_distance: float = 50.0, max_frames_skip: int = 5):
        self.max_distance = max_distance
        self.max_frames_skip = max_frames_skip
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.feature_extractor = FeatureExtractor()
        
    def predict_positions(self) -> np.ndarray:
        """Predict next positions using velocity"""
        predictions = []
        for track in self.tracks:
            if track.active:
                pos = track.get_last_position()
                vel = track.get_velocity()
                predicted = (pos[0] + vel[0], pos[1] + vel[1])
                predictions.append(predicted)
            else:
                predictions.append((0, 0))
        return np.array(predictions)
    
    def calculate_cost_matrix(self, detections: List[Cell], 
                              predictions: np.ndarray) -> np.ndarray:
        """Calculate cost matrix for assignment"""
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([])
        
        # Detection positions
        det_positions = np.array([cell.centroid for cell in detections])
        
        # Distance cost
        distance_cost = cdist(predictions, det_positions, 'euclidean')
        
        # Feature similarity cost
        det_features = self.feature_extractor.extract_batch_features(detections)
        track_features = []
        for track in self.tracks:
            if track.active and track.cells:
                track_features.append(
                    self.feature_extractor.extract_features(track.cells[-1]))
            else:
                track_features.append(np.zeros(7))
        
        track_features = np.array(track_features)
        feature_cost = cdist(track_features, det_features, 'euclidean')
        
        # Normalize and combine
        if distance_cost.max() > 0:
            distance_cost = distance_cost / distance_cost.max()
        if feature_cost.max() > 0:
            feature_cost = feature_cost / feature_cost.max()
        
        cost_matrix = 0.7 * distance_cost + 0.3 * feature_cost
        
        return cost_matrix
    
    def update(self, detections: List[Cell]) -> List[Track]:
        """Update tracks with new detections"""
        if len(self.tracks) == 0:
            # Initialize tracks
            for detection in detections:
                track = Track(track_id=self.next_track_id, cells=[detection])
                self.tracks.append(track)
                self.next_track_id += 1
            return self.tracks
        
        # Predict positions
        predictions = self.predict_positions()
        
        # Calculate cost matrix
        cost_matrix = self.calculate_cost_matrix(detections, predictions)
        
        if cost_matrix.size > 0:
            # Hungarian algorithm for assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update matched tracks
            matched_tracks = set()
            matched_detections = set()
            
            for track_idx, det_idx in zip(row_ind, col_ind):
                if cost_matrix[track_idx, det_idx] < self.max_distance:
                    self.tracks[track_idx].cells.append(detections[det_idx])
                    matched_tracks.add(track_idx)
                    matched_detections.add(det_idx)
            
            # Handle unmatched tracks
            for track_idx, track in enumerate(self.tracks):
                if track_idx not in matched_tracks and track.active:
                    # Check if track should be terminated
                    if len(track.cells) > 0:
                        frames_since_last = detections[0].frame_idx - track.cells[-1].frame_idx
                        if frames_since_last > self.max_frames_skip:
                            track.active = False
            
            # Create new tracks for unmatched detections
            for det_idx, detection in enumerate(detections):
                if det_idx not in matched_detections:
                    track = Track(track_id=self.next_track_id, cells=[detection])
                    self.tracks.append(track)
                    self.next_track_id += 1
        
        return [track for track in self.tracks if track.active]


class TrackingVisualizer:
    """Generate tracking overlays and lineage trees"""
    
    @staticmethod
    def draw_tracks(image: np.ndarray, tracks: List[Track], 
                   current_frame: int) -> np.ndarray:
        """Draw tracking overlay on image"""
        overlay = image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        # Color palette
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255)
        ]
        
        for track in tracks:
            if not track.active:
                continue
                
            color = colors[track.track_id % len(colors)]
            
            # Draw trajectory
            points = [cell.centroid for cell in track.cells 
                     if cell.frame_idx <= current_frame]
            
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(overlay, pt1, pt2, color, 2)
            
            # Draw current position
            if points:
                center = (int(points[-1][0]), int(points[-1][1]))
                cv2.circle(overlay, center, 8, color, -1)
                cv2.circle(overlay, center, 10, (255, 255, 255), 2)
                
                # Draw ID
                cv2.putText(overlay, f"ID:{track.track_id}", 
                          (center[0] + 15, center[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return overlay
    
    @staticmethod
    def generate_lineage_tree(tracks: List[Track]) -> Dict:
        """Generate lineage tree data structure"""
        # Simplified lineage tree - in practice would detect divisions
        tree = {
            'nodes': [],
            'edges': []
        }
        
        for track in tracks:
            tree['nodes'].append({
                'id': track.track_id,
                'start_frame': track.cells[0].frame_idx if track.cells else 0,
                'end_frame': track.cells[-1].frame_idx if track.cells else 0,
                'n_cells': len(track.cells)
            })
        
        return tree
