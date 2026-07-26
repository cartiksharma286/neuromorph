import bpy
import os
import sys

# Ensure Blender console output is flushed
sys.stdout.flush()

def clean_scene():
    """Remove default Blender objects to start fresh."""
    print("Clearing default scene objects...")
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)
    # Also clean up any orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

def create_collection(name, parent_collection=None):
    """Create a Blender collection and link it."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent_collection:
        parent_collection.children.link(col)
    else:
        bpy.context.scene.collection.children.link(col)
    return col

def create_premium_material(name, diffuse_color, roughness=0.5, metallic=0.0, transmission=0.0, emission_color=None, emission_strength=1.0):
    """Create a modern shader material using the Principled BSDF node."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Nodes layout
    node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    
    links = mat.node_tree.links
    links.new(node_principled.outputs['BSDF'], node_output.inputs['Surface'])
    
    # Configure shader parameters
    node_principled.inputs['Base Color'].default_value = diffuse_color
    node_principled.inputs['Roughness'].default_value = roughness
    node_principled.inputs['Metallic'].default_value = metallic
    
    # Handle transmission differences across Blender versions (3.x vs 4.x)
    if 'Transmission' in node_principled.inputs:
        node_principled.inputs['Transmission'].default_value = transmission
    elif 'Transmission Weight' in node_principled.inputs:
        node_principled.inputs['Transmission Weight'].default_value = transmission
        
    # Handle emission settings
    if emission_color:
        if 'Emission Color' in node_principled.inputs:
            node_principled.inputs['Emission Color'].default_value = emission_color[:3]
        elif 'Emission' in node_principled.inputs:
            node_principled.inputs['Emission'].default_value = emission_color
        
        if 'Emission Strength' in node_principled.inputs:
            node_principled.inputs['Emission Strength'].default_value = emission_strength
            
    return mat

def import_stl(filepath):
    """Import STL file, supporting multiple Blender API versions."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    
    # Select nothing first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Try the newer 4.0+ stl import, fall back to older ops
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
    except AttributeError:
        try:
            bpy.ops.import_mesh.stl(filepath=filepath)
        except Exception as e:
            print(f"Error importing {filepath}: {e}")
            return None
            
    # Imported objects are selected
    imported_objs = [obj for obj in bpy.context.selected_objects]
    if imported_objs:
        return imported_objs[0]
    return None

def setup_lighting_and_camera():
    """Set up studio lighting and a camera pointing to the grid center."""
    print("Setting up camera and lights...")
    # Grid center is roughly at X=240, Y=75, Z=0
    grid_center = (240.0, 75.0, 0.0)
    
    # Camera
    cam_data = bpy.data.cameras.new("StudioCamera")
    cam_obj = bpy.data.objects.new("StudioCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (240.0, -450.0, 300.0)
    # Point camera at center
    cam_obj.rotation_euler = (1.1, 0.0, 0.0) # Angle down towards center
    
    # Sun light
    light_data = bpy.data.lights.new(name="SunLight", type='SUN')
    light_data.energy = 5.0
    light_obj = bpy.data.objects.new("SunLight", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (240.0, 75.0, 400.0)
    light_obj.rotation_euler = (0.2, 0.4, 0.0)
    
    # Warm Fill Light
    fill_data = bpy.data.lights.new(name="WarmFill", type='POINT')
    fill_data.energy = 20000.0
    fill_data.color = (1.0, 0.8, 0.6)
    fill_obj = bpy.data.objects.new("WarmFill", fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (60.0, -100.0, 150.0)

    # Cool Rim Light
    rim_data = bpy.data.lights.new(name="CoolRim", type='POINT')
    rim_data.energy = 30000.0
    rim_data.color = (0.6, 0.8, 1.0)
    rim_obj = bpy.data.objects.new("CoolRim", rim_data)
    bpy.context.scene.collection.objects.link(rim_obj)
    rim_obj.location = (420.0, 250.0, 200.0)

def main():
    clean_scene()
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collections setup
    col_reconstruction = create_collection("Reconstruction")
    col_registration = create_collection("Registration")
    col_eeg_cap = create_collection("EEG_Cap")
    
    # Setup premium materials (RGBA colors: Red, Green, Blue, Alpha)
    materials = {
        'marching_cubes': create_premium_material('Mat_MarchingCubes', (0.7, 0.75, 0.8, 1.0), roughness=0.3, metallic=0.1),
        'legendre_sh': create_premium_material('Mat_LegendreSH', (0.1, 0.6, 0.8, 0.6), roughness=0.1, metallic=0.2, transmission=0.7),
        'tetra_surface': create_premium_material('Mat_TetraSurface', (0.05, 0.8, 0.5, 0.5), roughness=0.2, metallic=0.0, transmission=0.5),
        'tetra_volume': create_premium_material('Mat_TetraVolume', (1.0, 0.4, 0.1, 1.0), roughness=0.5, metallic=0.0, emission_color=(1.0, 0.4, 0.1, 1.0), emission_strength=5.0),
        'reg_gmm': create_premium_material('Mat_RegGMM', (0.5, 0.2, 0.8, 1.0), roughness=0.2, metallic=0.8),
        'reg_cf': create_premium_material('Mat_RegCF', (0.1, 0.8, 0.3, 1.0), roughness=0.3, metallic=0.7),
        'reg_qml': create_premium_material('Mat_RegQML', (0.0, 0.8, 1.0, 1.0), roughness=0.2, metallic=0.9, emission_color=(0.0, 0.8, 1.0, 1.0), emission_strength=2.0),
        'reg_qlora': create_premium_material('Mat_RegQLoRA', (1.0, 0.2, 0.6, 1.0), roughness=0.2, metallic=0.8),
        'reg_feynman': create_premium_material('Mat_RegFeynman', (1.0, 0.7, 0.0, 1.0), roughness=0.1, metallic=0.9, emission_color=(1.0, 0.7, 0.0, 1.0), emission_strength=1.5),
        'reg_superimposed': create_premium_material('Mat_RegSuperimposed', (0.9, 0.9, 0.9, 1.0), roughness=0.1, metallic=0.95),
        'eeg_cap': create_premium_material('Mat_EEGScubaCap', (0.2, 0.2, 0.25, 0.3), roughness=0.05, metallic=0.1, transmission=0.85)
    }

    # Configuration for STL files: file name, collection, material, and coordinate offset in Blender grid
    models_config = [
        # --- RECONSTRUCTION LAYER ---
        ('marching_cubes_interpolated.stl', col_reconstruction, materials['marching_cubes'], (0.0, 0.0, 0.0)),
        ('cortical_surface_legendre_sh.stl', col_reconstruction, materials['legendre_sh'], (120.0, 0.0, 0.0)),
        ('tetrahedral_mesh_surface.stl', col_reconstruction, materials['tetra_surface'], (240.0, 0.0, 0.0)),
        ('tetrahedral_mesh_volume.stl', col_reconstruction, materials['tetra_volume'], (360.0, 0.0, 0.0)),
        
        # --- REGISTRATION OUTPUTS ---
        ('registered_surface.stl', col_registration, materials['reg_gmm'], (0.0, 150.0, 0.0)),
        ('registered_surface_cf.stl', col_registration, materials['reg_cf'], (120.0, 150.0, 0.0)),
        ('registered_surface_qml.stl', col_registration, materials['reg_qml'], (240.0, 150.0, 0.0)),
        ('registered_surface_qlora.stl', col_registration, materials['reg_qlora'], (360.0, 150.0, 0.0)),
        ('registered_surface_feynman.stl', col_registration, materials['reg_feynman'], (480.0, 150.0, 0.0)),
        ('registered_superimposed.stl', col_registration, materials['reg_superimposed'], (240.0, 300.0, 0.0)),
        
        # --- SCUBA EEG HEAD CAP ---
        ('scuba_eeg_head_cap.stl', col_eeg_cap, materials['eeg_cap'], (240.0, -150.0, 0.0))
    ]

    print("Beginning import of meshes to Blender scene...")
    for filename, collection, material, location in models_config:
        filepath = os.path.join(base_dir, filename)
        print(f"Importing {filename}...")
        
        obj = import_stl(filepath)
        if obj:
            # Move to target collection
            old_collections = list(obj.users_collection)
            collection.objects.link(obj)
            for c in old_collections:
                c.objects.unlink(obj)
                
            # Set material
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)
                
            # Apply layout position offset
            obj.location = location
            print(f"Successfully positioned {filename} at {location} in collection '{collection.name}'")
        else:
            print(f"Error: Failed to import {filename}")

    # Set up lighting & camera
    setup_lighting_and_camera()
    
    # Save the scene
    blend_out_path = os.path.join(base_dir, 'mersivity_scene.blend')
    bpy.ops.wm.save_as_mainfile(filepath=blend_out_path)
    print(f"Native Blender scene successfully generated and saved to: {blend_out_path}")

if __name__ == '__main__':
    main()
