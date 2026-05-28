import os
import numpy as np
import trimesh

def main():
    stl_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_Triplane_Setup_Ness_02.stl'
    npy_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/mersivity/MendMesh_vertices_ds.npy'
    
    print("Checking if downsampled numpy file already exists...")
    if os.path.exists(npy_path):
        print(f"Downsampled file already exists at {npy_path}. Exiting.")
        return
        
    print(f"Loading large STL file from {stl_path} (727 MB)...")
    if not os.path.exists(stl_path):
        print(f"Error: STL file not found at {stl_path}")
        return
        
    mesh = trimesh.load(stl_path)
    verts = np.array(mesh.vertices)
    print(f"Successfully loaded STL. Total vertices: {len(verts)}")
    
    # Downsample to 16,384 vertices to maintain a very high-quality representation 
    # while allowing instant loading and fast registration.
    target_n = min(len(verts), 16384)
    print(f"Downsampling to {target_n} vertices...")
    idx = np.linspace(0, len(verts) - 1, target_n, dtype=int)
    verts_ds = verts[idx]
    
    np.save(npy_path, verts_ds)
    print(f"Saved downsampled vertices to {npy_path} (size: {os.path.getsize(npy_path) / 1024:.2f} KB)!")

if __name__ == '__main__':
    main()
