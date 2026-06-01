import numpy as np
import trimesh
from scipy.special import sph_harm, legendre
from skimage import measure
import sys
import os

# Import load_dicom_stack from app.py in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import load_dicom_stack

def write_cortical_surface_files():
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
    mesh = trimesh.Trimesh(vertices=verts_smooth, faces=faces)
    # Export to STL
    mesh.export('cortical_surface_legendre_sh.stl')
    print('Exported cortical_surface_legendre_sh.stl')
    # Try to export to .blend using Blender's Python API if available
    try:
        import bpy
        import tempfile
        ply_path = tempfile.mktemp(suffix='.ply')
        mesh.export(ply_path)
        bpy.ops.import_mesh.ply(filepath=ply_path)
        bpy.ops.wm.save_as_mainfile(filepath='cortical_surface_legendre_sh.blend')
        print('Exported cortical_surface_legendre_sh.blend using Blender API')
    except ImportError:
        # Fallback: export to .ply for Blender import
        mesh.export('cortical_surface_legendre_sh.ply')
        print('Blender API not found, exported cortical_surface_legendre_sh.ply for Blender import')

if __name__ == '__main__':
    write_cortical_surface_files()




import numpy as np
import trimesh
from scipy.special import sph_harm, legendre
from skimage import measure
import sys
import os

# Import load_dicom_stack from app.py in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import load_dicom_stack

def write_cortical_surface_files():
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
    mesh = trimesh.Trimesh(vertices=verts_smooth, faces=faces)
    # Export to STL
    mesh.export('cortical_surface_legendre_sh.stl')
    print('Exported cortical_surface_legendre_sh.stl')
    # Try to export to .blend using Blender's Python API if available
    try:
        import bpy
        import tempfile
        ply_path = tempfile.mktemp(suffix='.ply')
        mesh.export(ply_path)
        bpy.ops.import_mesh.ply(filepath=ply_path)
        bpy.ops.wm.save_as_mainfile(filepath='cortical_surface_legendre_sh.blend')
        print('Exported cortical_surface_legendre_sh.blend using Blender API')
    except ImportError:
        # Fallback: export to .ply for Blender import
        mesh.export('cortical_surface_legendre_sh.ply')
        print('Blender API not found, exported cortical_surface_legendre_sh.ply for Blender import')

if __name__ == '__main__':
    write_cortical_surface_files()
