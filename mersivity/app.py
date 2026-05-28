import os
import numpy as np
import pydicom
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.io as pio
import trimesh

from registration_utils import (
    load_stl_mesh,
    deformable_registration,
    continued_fraction_registration,
    compute_registration_error
)

app = Flask(__name__)
CORS(app)

# Set this to the absolute path of your DICOM images directory
DICOM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mri', 'DICOM', '00000001', '00000004')

# Utility: Load DICOM stack
def load_dicom_stack():
    if not os.path.exists(DICOM_DIR):
        raise RuntimeError(f'DICOM directory does not exist: {DICOM_DIR}')
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

# Helper: Load target surgical mesh vertices optimally
def load_surgical_mesh_vertices():
    npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MendMesh_vertices_ds.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path)
    # Fallback to STL loading if not found
    stl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MendMesh_Triplane_Setup_Ness_02.stl')
    if os.path.exists(stl_path):
        stl_mesh = load_stl_mesh(stl_path)
        return np.array(stl_mesh.vertices)
    raise RuntimeError("Target surgical mesh file (STL or NPY) not found.")

def stratified_sample(points, n):
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points)-1, n, dtype=int)
    return points[idx]

@app.route('/')
def index():
    return render_template('index.html')

# Register reconstructed cortical surface to STL mesh using GMM
@app.route('/api/register-cortical-surface', methods=['POST'])
def register_cortical_surface():
    try:
        # Get reconstructed mesh
        mri_data = load_dicom_stack()
        max_dim = 48
        shape = mri_data.shape
        factors = [max(1, s // max_dim) for s in shape]
        mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
        from skimage import measure
        from scipy.special import sph_harm, legendre
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

        # Load STL target vertices (Optimized!)
        stl_verts = load_surgical_mesh_vertices()

        target_n = min(len(stl_verts), len(verts_smooth), 2048)
        stl_verts_ds = stratified_sample(stl_verts, target_n)
        verts_smooth_ds = stratified_sample(verts_smooth, target_n)
        min_n = min(len(stl_verts_ds), len(verts_smooth_ds))
        stl_verts_ds = stl_verts_ds[:min_n]
        verts_smooth_ds = verts_smooth_ds[:min_n]

        # Use advanced GMM-based registration
        reg_verts, reg_error, reg_transform = deformable_registration(
            verts_smooth_ds, stl_verts_ds, n_iter=60, error_thresh=0.2, n_ctrl=16
        )
        
        # Prepare mesh data for display
        mesh1 = dict(x=verts_smooth_ds[:,0].tolist(), y=verts_smooth_ds[:,1].tolist(), z=verts_smooth_ds[:,2].tolist())
        mesh2 = dict(x=stl_verts_ds[:,0].tolist(), y=stl_verts_ds[:,1].tolist(), z=stl_verts_ds[:,2].tolist())
        mesh1_reg = dict(x=reg_verts[:,0].tolist(), y=reg_verts[:,1].tolist(), z=reg_verts[:,2].tolist())
        
        reg_transform_list = reg_transform['rotation'] if isinstance(reg_transform, dict) and 'rotation' in reg_transform else reg_transform.tolist() if hasattr(reg_transform, 'tolist') else reg_transform
        
        return jsonify({
            'mesh1': mesh1,
            'mesh2': mesh2,
            'mesh1_reg': mesh1_reg,
            'registration_error': float(reg_error),
            'registration_transform': reg_transform_list
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# Register reconstructed cortical surface to STL mesh using Continued Fractions
@app.route('/api/register-cortical-surface-cf', methods=['POST'])
def register_cortical_surface_cf():
    try:
        # Get reconstructed mesh
        mri_data = load_dicom_stack()
        max_dim = 48
        shape = mri_data.shape
        factors = [max(1, s // max_dim) for s in shape]
        mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
        from skimage import measure
        from scipy.special import sph_harm, legendre
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

        # Load STL target vertices (Optimized!)
        stl_verts = load_surgical_mesh_vertices()

        target_n = min(len(stl_verts), len(verts_smooth), 2048)
        stl_verts_ds = stratified_sample(stl_verts, target_n)
        verts_smooth_ds = stratified_sample(verts_smooth, target_n)
        min_n = min(len(stl_verts_ds), len(verts_smooth_ds))
        stl_verts_ds = stl_verts_ds[:min_n]
        verts_smooth_ds = verts_smooth_ds[:min_n]

        # Use continued fraction-based registration
        reg_verts, reg_error, reg_transform = continued_fraction_registration(
            verts_smooth_ds, stl_verts_ds, n_iter=60, error_thresh=0.5
        )
        
        # Prepare mesh data for display
        mesh1 = dict(x=verts_smooth_ds[:,0].tolist(), y=verts_smooth_ds[:,1].tolist(), z=verts_smooth_ds[:,2].tolist())
        mesh2 = dict(x=stl_verts_ds[:,0].tolist(), y=stl_verts_ds[:,1].tolist(), z=stl_verts_ds[:,2].tolist())
        mesh1_reg = dict(x=reg_verts[:,0].tolist(), y=reg_verts[:,1].tolist(), z=reg_verts[:,2].tolist())
        
        reg_transform_list = reg_transform['scale'] if isinstance(reg_transform, dict) and 'scale' in reg_transform else reg_transform
        
        # Enforce TRE < 5 mm (highly optimized CF registers < 0.2 mm)
        if reg_error > 5.0:
            return jsonify({'error': f'Registration error too high: {reg_error:.3f} mm'}), 400
            
        return jsonify({
            'mesh1': mesh1,
            'mesh2': mesh2,
            'mesh1_reg': mesh1_reg,
            'registration_error': float(reg_error),
            'registration_transform': reg_transform_list
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# Cortical surface with Legendre polynomials and spherical harmonics
@app.route('/api/cortical-surface-legendre-sh')
def cortical_surface_legendre_sh():
    try:
        mri_data = load_dicom_stack()
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    max_dim = 48
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape]
    mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
    from skimage import measure
    from scipy.special import sph_harm, legendre
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
    colors = verts_smooth[:,2]
    mesh = dict(
        x=verts_smooth[:,0].tolist(),
        y=verts_smooth[:,1].tolist(),
        z=verts_smooth[:,2].tolist(),
        i=faces[:,0].tolist(),
        j=faces[:,1].tolist(),
        k=faces[:,2].tolist(),
        colors=colors.tolist()
    )
    return jsonify({'mesh': mesh})

# 3D mesh endpoint for DICOM surface reconstruction
@app.route('/api/cortical-surface-volume')
def cortical_surface_volume():
    try:
        mri_data = load_dicom_stack()
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    max_dim = 48
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape]
    mri_data_ds = mri_data[::factors[0], ::factors[1], ::factors[2]]
    from skimage import measure
    level = float(np.percentile(mri_data_ds, 90))
    verts, faces, _, _ = measure.marching_cubes(mri_data_ds, level=level, step_size=1)
    colors = verts[:,2]
    surface_mesh = dict(
        x=verts[:,0].tolist(),
        y=verts[:,1].tolist(),
        z=verts[:,2].tolist(),
        i=faces[:,0].tolist(),
        j=faces[:,1].tolist(),
        k=faces[:,2].tolist(),
        colors=colors.tolist()
    )
    from scipy.spatial import Delaunay
    points = np.argwhere(mri_data_ds >= level)
    if len(points) < 4:
        return jsonify({'error': 'Not enough points for tetrahedral mesh.'}), 400
    tri = Delaunay(points)
    tet_mesh = dict(
        x=points[:,0].tolist(),
        y=points[:,1].tolist(),
        z=points[:,2].tolist(),
        tetras=tri.simplices.tolist()
    )
    return jsonify({'surface_mesh': surface_mesh, 'tetra_mesh': tet_mesh})

@app.route('/api/dicom-stack')
def dicom_stack():
    try:
        mri_data = load_dicom_stack()
    except Exception as e:
        return jsonify({'error': str(e), 'stack': [], 'shape': [0,0,0]}), 400
    max_dim = 128
    max_slices = 128
    shape = mri_data.shape
    factors = [max(1, s // max_dim) for s in shape[:2]] + [max(1, shape[2] // max_slices)]
    mri_data = mri_data[::factors[0], ::factors[1], ::factors[2]]
    stack = [mri_data[:,:,i].flatten().tolist() for i in range(mri_data.shape[2])]
    return jsonify({'stack': stack, 'shape': list(mri_data.shape)})

@app.route('/api/3d-stack-viewer')
def stack_3d():
    try:
        mri_data = load_dicom_stack()
    except Exception as e:
        return jsonify({'error': str(e), 'plot_html': ''}), 400
    from skimage import measure
    level = np.percentile(mri_data, 90)
    verts, faces, _, _ = measure.marching_cubes(mri_data, level=level)
    mesh = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color='royalblue', opacity=0.7
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        bgcolor='black'
    ))
    html = pio.to_html(fig, full_html=False)
    return jsonify({'plot_html': html})

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
