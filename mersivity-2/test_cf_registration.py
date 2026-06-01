import os
import numpy as np
import pydicom
from scipy.spatial import cKDTree
from skimage import measure
from scipy.special import sph_harm, legendre
import sys

# Set DICOM DIR
DICOM_DIR = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/mri/DICOM/00000001/00000004'

def load_dicom_stack():
    files = [os.path.join(DICOM_DIR, f) for f in os.listdir(DICOM_DIR) if f.endswith('.dcm')]
    if not files:
        raise RuntimeError('No DICOM files found in the selected directory.')
    def get_instance_number(f):
        try:
            return int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
        except Exception:
            return 0
    files.sort(key=get_instance_number)
    first = pydicom.dcmread(files[0])
    img_shape = list(first.pixel_array.shape)
    img_shape.append(len(files))
    img3d = np.zeros(img_shape, dtype=first.pixel_array.dtype)
    img3d[:, :, 0] = first.pixel_array
    for i, f in enumerate(files[1:], 1):
        img3d[:, :, i] = pydicom.dcmread(f).pixel_array
    return img3d

def main():
    print("Loading DICOM stack...")
    mri_data = load_dicom_stack()
    max_dim = 48
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape]
    mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
    
    level = float(np.percentile(mri_data_ds, 60))
    verts, faces, _, _ = measure.marching_cubes(mri_data_ds, level=level, step_size=1)
    center = verts.mean(axis=0)
    xyz = verts - center
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:,2] / r, -1, 1))
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    lmax = 6
    P = np.vstack([legendre(l)(np.cos(theta)) for l in range(lmax+1)]).T
    Y = []
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y.append(sph_harm(m, l, phi, theta).real)
    Y = np.vstack(Y).T
    features = np.hstack([P, Y])
    coeffs, _, _, _ = np.linalg.lstsq(features, r, rcond=None)
    r_smooth = features @ coeffs
    xyz_smooth = np.zeros_like(xyz)
    xyz_smooth[:,0] = r_smooth * np.sin(theta) * np.cos(phi)
    xyz_smooth[:,1] = r_smooth * np.sin(theta) * np.sin(phi)
    xyz_smooth[:,2] = r_smooth * np.cos(theta)
    verts_smooth = xyz_smooth + center
    
    # Load STL
    import trimesh
    stl_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_Triplane_Setup_Ness_02.stl'
    print("Loading STL...")
    stl_mesh = trimesh.load(stl_path)
    stl_verts = np.array(stl_mesh.vertices)
    
    def stratified_sample(points, n):
        if len(points) <= n:
            return points
        idx = np.linspace(0, len(points)-1, n, dtype=int)
        return points[idx]

    target_n = min(len(stl_verts), len(verts_smooth), 2048)
    stl_verts_ds = stratified_sample(stl_verts, target_n)
    verts_smooth_ds = stratified_sample(verts_smooth, target_n)
    min_n = min(len(stl_verts_ds), len(verts_smooth_ds))
    stl_verts_ds = stl_verts_ds[:min_n]
    verts_smooth_ds = verts_smooth_ds[:min_n]
    
    print(f"Dataset sizes: source={len(verts_smooth_ds)}, target={len(stl_verts_ds)}")
    
    # Test existing GMM registration
    sys.path.append('/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity')
    from registration_utils import deformable_registration, continued_fraction_registration
    
    print("\nRunning current deformable (GMM) registration...")
    _, gmm_error, _ = deformable_registration(verts_smooth_ds, stl_verts_ds, n_iter=60, error_thresh=0.2)
    print(f"GMM registration error: {gmm_error:.4f} mm")
    
    print("\nRunning current continued fraction registration...")
    _, cf_error, _ = continued_fraction_registration(verts_smooth_ds, stl_verts_ds, n_iter=30, error_thresh=5.0)
    print(f"Continued fraction registration error: {cf_error:.4f} mm")

if __name__ == '__main__':
    main()
