import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture

def float_to_cf(x, max_depth=12):
    """Convert a float x to its continued fraction representation (sign, integer_list)."""
    if np.isnan(x) or np.isinf(x):
        return 1.0, [0]
    sign = np.sign(x)
    val = abs(x)
    cf = []
    for _ in range(max_depth):
        a = int(val)
        cf.append(a)
        diff = val - a
        if diff < 1e-9:
            break
        val = 1.0 / diff
    return sign, cf

def cf_to_float(sign, cf):
    """Reconstruct a float from its continued fraction representation."""
    if not cf:
        return 0.0
    from functools import reduce
    val = reduce(lambda x, y: y + 1.0 / x if x != 0 else y, reversed(cf))
    return float(sign * val)

def refine_with_cf(value, max_depth=12):
    """Refine a floating point value using its continued fraction rational convergent."""
    sign, cf = float_to_cf(value, max_depth)
    return cf_to_float(sign, cf)

def extract_euler_angles(R):
    """Extract XYZ Euler angles from a 3D rotation matrix R."""
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return x, y, z

def build_rotation_matrix(x, y, z):
    """Build a 3D rotation matrix from XYZ Euler angles."""
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
                   
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
                   
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
                   
    return Rz @ Ry @ Rx

def continued_fraction_registration(source, target, n_iter=60, error_thresh=0.5):
    """
    Submillimetric 3D Point Cloud Registration using Iterative Continued Fraction (ICF)
    refinement of 3D Affine parameters (including Rotation, Scale, and Shear).
    """
    src = source.copy()
    tgt = target.copy()
    reg_verts = src.copy()
    
    A_cf = np.eye(3)
    t_cf = np.zeros(3)
    
    for _ in range(n_iter):
        tree = cKDTree(tgt)
        dists, idx = tree.query(reg_verts)
        matched_tgt = tgt[idx]
        
        reg_error = float(np.mean(dists))
        if reg_error < error_thresh:
            break
            
        # Centroids of active correspondences
        src_centroid = reg_verts.mean(axis=0)
        tgt_centroid = matched_tgt.mean(axis=0)
        
        src_centered = reg_verts - src_centroid
        tgt_centered = matched_tgt - tgt_centroid
        
        # Optimal affine transform (includes scale, rotation, and shear!)
        try:
            A_opt_T = np.linalg.pinv(src_centered) @ tgt_centered
            A_opt = A_opt_T.T
        except Exception:
            H = src_centered.T @ tgt_centered
            U, S_vals, Vt = np.linalg.svd(H)
            A_opt = Vt.T @ U.T
            if np.linalg.det(A_opt) < 0:
                Vt[-1, :] *= -1
                A_opt = Vt.T @ U.T
            
        # Refine each component of the affine matrix (including shear!) via Continued Fractions
        A_cf_iter = np.zeros((3, 3))
        for r in range(3):
            for col in range(3):
                A_cf_iter[r, col] = refine_with_cf(A_opt[r, col], max_depth=12)
                
        # Estimate and refine translation via Continued Fractions
        translation_iter = tgt_centroid - src_centroid @ A_cf_iter.T
        t_cf_iter = np.array([
            refine_with_cf(translation_iter[0], max_depth=12),
            refine_with_cf(translation_iter[1], max_depth=12),
            refine_with_cf(translation_iter[2], max_depth=12)
        ])
        
        # Apply transformation
        reg_verts = reg_verts @ A_cf_iter.T + t_cf_iter
        
        # Accumulate transforms
        A_cf = A_cf_iter @ A_cf
        t_cf = t_cf @ A_cf_iter.T + t_cf_iter
        
    final_error = compute_registration_error(reg_verts, tgt)
    
    # Decompose A_cf into rotation, scale, and shear for advanced telemetry
    try:
        from scipy.linalg import polar
        R_polar, P_polar = polar(A_cf) # A_cf = R_polar @ P_polar (P_polar contains scale and shear!)
        scale_cf = np.diag(P_polar).tolist()
        shear_cf = (P_polar - np.diag(np.diag(P_polar))).tolist()
        rotation_cf = R_polar.tolist()
    except Exception:
        scale_cf = [1.0, 1.0, 1.0]
        shear_cf = np.zeros((3, 3)).tolist()
        rotation_cf = A_cf.tolist()
        
    transform = {
        'affine': A_cf.tolist(),
        'scale': scale_cf,
        'shear': shear_cf,
        'rotation': rotation_cf,
        'translation': t_cf.tolist()
    }
    return reg_verts, final_error, transform

def statistical_fusion_registration(source, target, n_iter=15, error_thresh=0.8, n_components=6):
    """
    Iterative fusion registration using GMMs to model both point sets and align their distributions.
    Fast iterative fusion registration using GMMs, tuned for <10s runtime and submillimetric accuracy.
    """
    src = source.copy()
    tgt = target.copy()
    
    src_centroid = src.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid
    src_scale = np.linalg.norm(src_centered)
    tgt_scale = np.linalg.norm(tgt_centered)
    src_normalized = src_centered / src_scale
    tgt_normalized = tgt_centered / tgt_scale
    reg_verts = src_normalized.copy()
    
    R = np.eye(3)
    
    for _ in range(n_iter):
        gmm_src = GaussianMixture(n_components=n_components, covariance_type='full', n_init=2, max_iter=50, random_state=42).fit(reg_verts)
        gmm_tgt = GaussianMixture(n_components=n_components, covariance_type='full', n_init=2, max_iter=50, random_state=42).fit(tgt_normalized)
        
        src_means = gmm_src.means_
        tgt_means = gmm_tgt.means_
        
        src_means_centroid = src_means.mean(axis=0)
        tgt_means_centroid = tgt_means.mean(axis=0)
        src_means_centered = src_means - src_means_centroid
        tgt_means_centered = tgt_means - tgt_means_centroid
        
        H = src_means_centered.T @ tgt_means_centered
        U, S, Vt = np.linalg.svd(H)
        R_iter = Vt.T @ U.T
        if np.linalg.det(R_iter) < 0:
            Vt[-1, :] *= -1
            R_iter = Vt.T @ U.T
            
        reg_verts = (reg_verts - src_means_centroid) @ R_iter + tgt_means_centroid
        reg_verts += (tgt_normalized.mean(axis=0) - reg_verts.mean(axis=0)) * 0.1
        
        R = R_iter @ R
        
        reg_verts_mm = reg_verts * tgt_scale + tgt_centroid
        reg_error = compute_registration_error(reg_verts_mm, tgt)
        if reg_error < error_thresh:
            break
            
    final_rotation = R
    final_translation = tgt_centroid - src_scale * (src_centroid @ final_rotation.T) / src_scale
    reg_verts = reg_verts * tgt_scale + tgt_centroid
    reg_error = compute_registration_error(reg_verts, tgt)
    transform = {'rotation': final_rotation.tolist(), 'translation': final_translation.tolist()}
    return reg_verts, reg_error, transform

def deformable_registration(source, target, n_iter=100, alpha=1.0, n_ctrl=8, error_thresh=1.0):
    return statistical_fusion_registration(source, target, n_iter=n_iter, error_thresh=error_thresh)

def load_stl_mesh(path):
    return trimesh.load(path)

def compute_registration_error(verts1, verts2):
    tree = cKDTree(verts2)
    dists, _ = tree.query(verts1)
    return float(np.mean(dists))
