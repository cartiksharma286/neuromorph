
import numpy as np
import json
import os

class IGSNavigator:
    """
    Image Guided Surgery (IGS) Navigator System for Dental Surgery.
    Handles coordinate transformations, registration, and surgical tracking.
    """
    def __init__(self):
        self.patient_to_image_matrix = np.eye(4)
        self.tracker_to_patient_matrix = np.eye(4)
        self.is_registered = False

    def load_dicom_metadata(self, patient_id):
        """
        Simulate loading patient DICOM metadata to establish image space.
        """
        # Mock metadata
        return {
            "patient_id": patient_id,
            "voxel_spacing": [0.5, 0.5, 0.5],
            "origin": [0, 0, 0],
            "dimensions": [512, 512, 200]
        }

    def register_fiducials(self, image_points, tracker_points):
        """
        Performs rigid body registration using SVD to align tracker space with image space.
        """
        if len(image_points) < 3 or len(tracker_points) < 3:
            raise ValueError("Need at least 3 points for registration.")
            
        P = np.array(image_points)
        Q = np.array(tracker_points)
        
        # Centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        # Centered coordinates
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Covariance matrix
        H = np.dot(P_centered.T, Q_centered)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Det Check
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        t = centroid_Q - np.dot(R, centroid_P)
        
        # Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        self.patient_to_image_matrix = np.linalg.inv(T) # Storing Image -> Patient or vice versa
        self.is_registered = True
        
        return np.mean(np.linalg.norm(Q - (np.dot(P, R.T) + t), axis=1)) # FRE (Fiducial Registration Error)

    def track_instrument(self, instrument_tracker_coords):
        """
        Transforms instrument coordinates from tracker space to image space.
        """
        if not self.is_registered:
            raise RuntimeError("System not registered.")
            
        coords_homog = np.append(instrument_tracker_coords, 1)
        image_coords = np.dot(self.patient_to_image_matrix, coords_homog)
        
        return image_coords[:3]

    def export_navigation_plan(self, output_dir, session_id, implant_location):
        """
        Writes the surgical navigation plan.
        """
        plan = {
            "session_id": session_id,
            "registration_status": "active" if self.is_registered else "pending",
            "target_implant_location": implant_location,
            "trajectory": {
                "entry_point": list(np.array(implant_location) + [0, 0, 10]), # Simple offset
                "target_point": implant_location
            }
        }
        
        filepath = os.path.join(output_dir, f"igs_plan_{session_id}.json")
        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=4)
        
        return filepath
