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
    print("-----------------------------------------------------------------")
    print("Starting Instant-Loading Submillimetric CF Registration Verification")
    print("-----------------------------------------------------------------")
    
    print("1. Loading DICOM stack...")
    mri_data = load_dicom_stack()
    max_dim = 48
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape]
    mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
    
    print("2. Generating marching cubes cortical surface...")
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
    
    # Load fast cached target
    print("3. Loading target surgical mesh from NumPy cache (instant!)...")
    npy_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_vertices_ds.npy'
    stl_verts = np.load(npy_path)
    
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
    
    print(f"Dataset sizes for registration: source={len(verts_smooth_ds)}, target={len(stl_verts_ds)}")
    
    # Run registration
    sys.path.append('/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity')
    from registration_utils import continued_fraction_registration
    
    print("4. Executing Continued Fraction SVD-based 6-DOF Registration...")
    reg_verts, cf_error, transform = continued_fraction_registration(
        verts_smooth_ds, stl_verts_ds, n_iter=60, error_thresh=0.5
    )
    
    print("\n-----------------------------------------------------------------")
    print("REGISTRATION RESULTS:")
    print(f"Final Continued Fraction Registration Error: {cf_error:.6f} mm")
    print(f"Estimated scale factor: {transform['scale']:.6f}")
    print(f"Estimated translation vector: {transform['translation']}")
    
    if cf_error < 1.0:
        print("VERIFICATION STATUS: SUCCESS! (Submillimetric accuracy achieved)")
    else:
        print("VERIFICATION STATUS: FAILED! (Error is not submillimetric)")
    print("-----------------------------------------------------------------")

if __name__ == '__main__':
    main()
