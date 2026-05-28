import os
import numpy as np

def main():
    stl_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_Triplane_Setup_Ness_02.stl'
    npy_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_vertices_ds.npy'
    
    print(f"Opening {stl_path} in fast binary mode...")
    if not os.path.exists(stl_path):
        print(f"Error: STL file not found at {stl_path}")
        return
        
    try:
        # Binary STL structured dtype
        dtype = np.dtype([
            ('normal', '<f4', (3,)),
            ('vertices', '<f4', (3, 3)),
            ('attr', '<u2')
        ])
        
        print("Reading binary STL directly into numpy array (this takes < 5 seconds)...")
        data = np.fromfile(stl_path, dtype=dtype, offset=84)
        
        print("Extracting vertices...")
        # Shape: (N, 3, 3) -> reshape to (N*3, 3)
        verts = data['vertices'].reshape(-1, 3)
        print(f"Loaded {len(verts)} vertices successfully!")
        
        # Take unique vertices to remove duplicates and downsample
        print("Removing duplicates and downsampling...")
        # Since unique can be slow on millions of points, we can just downsample the raw vertices directly:
        target_n = min(len(verts), 16384)
        print(f"Selecting {target_n} stratified vertices...")
        idx = np.linspace(0, len(verts) - 1, target_n, dtype=int)
        verts_ds = verts[idx]
        
        # Save to numpy file
        np.save(npy_path, verts_ds)
        print(f"Successfully saved downsampled vertices to {npy_path}!")
        print(f"Numpy file size: {os.path.getsize(npy_path) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error in fast binary loading: {str(e)}")
        print("Falling back to standard trimesh loading...")
        import trimesh
        mesh = trimesh.load(stl_path)
        verts = np.array(mesh.vertices)
        target_n = min(len(verts), 16384)
        idx = np.linspace(0, len(verts) - 1, target_n, dtype=int)
        verts_ds = verts[idx]
        np.save(npy_path, verts_ds)
        print("Saved via trimesh fallback.")

if __name__ == '__main__':
    main()
