import numpy as np
import trimesh
from scipy.special import sph_harm, legendre
from skimage import measure

def cortical_surface_legendre_sh_to_blend(mri_data, blend_path, lmax=6):
    # Downsample for performance
    max_dim = 48
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape]
    mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
    # Use a lower threshold to include craniofacial anatomy
    level = float(np.percentile(mri_data_ds, 60))
    verts, faces, _, _ = measure.marching_cubes(mri_data_ds, level=level, step_size=1)
    # Spherical coordinates
    center = verts.mean(axis=0)
    xyz = verts - center
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:,2] / r, -1, 1))
    phi = np.arctan2(xyz[:,1], xyz[:,0])
    # Legendre polynomials (up to degree lmax)
    P = np.vstack([legendre(l)(np.cos(theta)) for l in range(lmax+1)]).T
    # Spherical harmonics (real part, up to degree lmax)
    Y = []
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y.append(sph_harm(m, l, phi, theta).real)
    Y = np.vstack(Y).T
    # Combine Legendre and SH features
    features = np.hstack([P, Y])
    coeffs, _, _, _ = np.linalg.lstsq(features, r, rcond=None)
    r_smooth = features @ coeffs
    xyz_smooth = np.zeros_like(xyz)
    xyz_smooth[:,0] = r_smooth * np.sin(theta) * np.cos(phi)
    xyz_smooth[:,1] = r_smooth * np.sin(theta) * np.sin(phi)
    xyz_smooth[:,2] = r_smooth * np.cos(theta)
    verts_smooth = xyz_smooth + center
    # Export to .ply (Blender can import .ply natively)
    mesh = trimesh.Trimesh(vertices=verts_smooth, faces=faces)
    mesh.export(blend_path.replace('.blend', '.ply'))
    print(f"Exported cortical surface to {blend_path.replace('.blend', '.ply')}")

# Example usage:
# from mersivity.app import load_dicom_stack
# mri_data = load_dicom_stack()
# cortical_surface_legendre_sh_to_blend(mri_data, 'cortical_surface_legendre_sh.blend')
