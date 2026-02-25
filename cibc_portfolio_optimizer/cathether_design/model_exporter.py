"""
3D Model Export for Manufacturing

Exports catheter designs to STL, STEP, and manufacturing specifications
"""

import numpy as np
import trimesh
from typing import Optional, Dict
import json
from pathlib import Path
from catheter_geometry import CatheterMesh, CompleteCatheterGenerator
from quantum_catheter_optimizer import DesignParameters, PatientConstraints


class STLExporter:
    """Export to STL format for 3D printing"""
    
    @staticmethod
    def export(
        catheter_mesh: CatheterMesh,
        filename: str,
        binary: bool = True
    ):
        """
        Export mesh to STL file
        
        Args:
            catheter_mesh: CatheterMesh object
            filename: Output filename (.stl)
            binary: Use binary STL format (smaller file size)
        """
        # Ensure filename has .stl extension
        if not filename.endswith('.stl'):
            filename += '.stl'
        
        # Export using trimesh
        catheter_mesh.mesh.export(filename, file_type='stl')
        
        print(f"Exported STL to: {filename}")
        print(f"  File size: {Path(filename).stat().st_size / 1024:.2f} KB")
        print(f"  Vertices: {len(catheter_mesh.vertices)}")
        print(f"  Faces: {len(catheter_mesh.faces)}")
    
    @staticmethod
    def export_multi_resolution(
        catheter_mesh: CatheterMesh,
        base_filename: str,
        resolutions: list = [1.0, 0.5, 0.25]
    ):
        """
        Export multiple resolution versions
        
        Args:
            catheter_mesh: CatheterMesh object
            base_filename: Base name for output files
            resolutions: List of resolution factors (1.0 = original)
        """
        for res in resolutions:
            mesh = catheter_mesh.mesh.copy()
            
            if res < 1.0:
                # Simplify mesh
                target_faces = int(len(mesh.faces) * res)
                simplified = mesh.simplify_quadric_decimation(target_faces)
            else:
                simplified = mesh
            
            # Generate filename
            res_str = f"{int(res*100):03d}"
            filename = f"{base_filename}_res{res_str}.stl"
            
            # Export
            simplified.export(filename, file_type='stl')
            
            print(f"Exported {res*100:.0f}% resolution to: {filename}")
            print(f"  Faces: {len(simplified.faces)}")


class STEPExporter:
    """Export to STEP format for CAD software"""
    
    @staticmethod
    def export(
        catheter_mesh: CatheterMesh,
        filename: str
    ):
        """
        Export mesh to STEP file
        
        Note: Requires pythonocc or similar CAD kernel.
        This is a placeholder implementation.
        """
        # Ensure filename has .step extension
        if not filename.lower().endswith(('.step', '.stp')):
            filename += '.step'
        
        # For now, export as OBJ which can be imported into CAD
        obj_filename = filename.replace('.step', '.obj').replace('.stp', '.obj')
        catheter_mesh.mesh.export(obj_filename, file_type='obj')
        
        print(f"Note: Full STEP export requires pythonocc library")
        print(f"Exported OBJ (CAD-compatible) to: {obj_filename}")
        
        # Would implement STEP export here with pythonocc
        # from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        # etc.


class ManufacturingSpecGenerator:
    """Generate manufacturing specifications and documentation"""
    
    @staticmethod
    def generate_spec_sheet(
        design: DesignParameters,
        constraints: PatientConstraints,
        catheter_mesh: CatheterMesh,
        output_filename: str
    ):
        """
        Generate comprehensive manufacturing specification
        """
        spec = {
            "document_type": "Catheter Manufacturing Specification",
            "version": "1.0",
            
            "design_parameters": {
                "outer_diameter_mm": round(design.outer_diameter, 3),
                "inner_diameter_mm": round(design.inner_diameter, 3),
                "wall_thickness_mm": round(design.wall_thickness, 3),
                "tip_angle_degrees": round(design.tip_angle, 1),
                "flexibility_index": round(design.flexibility_index, 3),
                "total_length_mm": round(constraints.required_length, 1)
            },
            
            "material_composition": design.material_composition,
            
            "side_holes": {
                "count": len(design.side_holes),
                "positions_mm": [round(pos, 1) for pos in design.side_holes],
                "typical_diameter_mm": round(design.inner_diameter * 0.3, 2)
            },
            
            "patient_specifications": {
                "target_vessel_diameter_mm": round(constraints.vessel_diameter, 1),
                "vessel_curvature_1_per_mm": round(constraints.vessel_curvature, 4),
                "target_flow_rate_ml_per_min": round(constraints.flow_rate, 1)
            },
            
            "geometric_properties": {
                "total_volume_mm3": round(catheter_mesh.mesh.volume, 2),
                "surface_area_mm2": round(catheter_mesh.mesh.area, 2),
                "is_watertight": bool(catheter_mesh.mesh.is_watertight)
            },
            
            "tolerances": {
                "outer_diameter_tolerance_mm": 0.05,
                "inner_diameter_tolerance_mm": 0.05,
                "wall_thickness_tolerance_mm": 0.02,
                "length_tolerance_mm": 2.0,
                "tip_angle_tolerance_degrees": 2.0
            },
            
            "quality_control": {
                "pressure_test_psi": 300,
                "flow_test_ml_per_min": constraints.flow_rate * 1.5,
                "kink_test_radius_mm": 10,
                "biocompatibility_standard": "ISO 10993-1"
            },
            
            "printing_parameters": {
                "recommended_layer_height_mm": 0.1,
                "recommended_infill_percent": 100,
                "recommended_wall_line_count": 4,
                "print_temperature_celsius": ManufacturingSpecGenerator._get_print_temp(design),
                "bed_temperature_celsius": 60,
                "print_speed_mm_per_s": 40
            },
            
            "post_processing": [
                "Remove support structures carefully",
                "Smooth internal lumen with mandrel",
                "Polish external surface",
                "Sterilize per ISO 11137",
                "Quality inspection per specifications"
            ]
        }
        
        # Save to JSON
        with open(output_filename, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Manufacturing specification saved to: {output_filename}")
        
        return spec
    
    @staticmethod
    def _get_print_temp(design: DesignParameters) -> int:
        """Estimate print temperature based on material composition"""
        temps = {
            'polyurethane': 220,
            'silicone': 200,
            'ptfe': 380,
            'pebax': 210,
            'nylon': 250
        }
        
        weighted_temp = sum(
            temps.get(material, 220) * ratio
            for material, ratio in design.material_composition.items()
        )
        
        return int(weighted_temp)
    
    @staticmethod
    def generate_assembly_instructions(
        design: DesignParameters,
        output_filename: str
    ):
        """Generate assembly and handling instructions"""
        instructions = {
            "document_type": "Catheter Assembly Instructions",
            "version": "1.0",
            
            "components": [
                "Main catheter body",
                "Tip assembly",
                f"{len(design.side_holes)} side drainage holes",
                "Luer lock connector (if applicable)"
            ],
            
            "assembly_steps": [
                {
                    "step": 1,
                    "description": "Inspect all components for defects",
                    "verification": "Visual inspection, no cracks or deformities"
                },
                {
                    "step": 2,
                    "description": "Clean internal lumen with sterile saline",
                    "verification": "Clear fluid flow through catheter"
                },
                {
                    "step": 3,
                    "description": "Verify side hole patency",
                    "verification": "Fluid drainage through all holes"
                },
                {
                    "step": 4,
                    "description": "Attach connector (if applicable)",
                    "verification": "Secure connection, no leaks"
                },
                {
                    "step": 5,
                    "description": "Final sterile packaging",
                    "verification": "Sealed package, sterility indicators"
                }
            ],
            
            "handling_precautions": [
                "Store in sterile packaging until use",
                "Avoid kinking during handling",
                "Do not exceed maximum flow rate",
                "Single use only - do not resterilize",
                f"Maximum bend radius: {round(1/constraints.vessel_curvature, 1)} mm"
            ],
            
            "sterilization": {
                "method": "Ethylene oxide (EtO) or Gamma radiation",
                "cycle_parameters": "Per ISO 11135 or ISO 11137",
                "validation": "Biological indicators required"
            }
        }
        
        with open(output_filename, 'w') as f:
            json.dump(instructions, f, indent=2)
        
        print(f"Assembly instructions saved to: {output_filename}")
        
        return instructions


class ExportManager:
    """High-level export management"""
    
    def __init__(self, output_directory: str = "./exports"):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def export_complete_package(
        self,
        design: DesignParameters,
        constraints: PatientConstraints,
        catheter_mesh: CatheterMesh,
        project_name: str = "catheter_design"
    ) -> Dict[str, str]:
        """
        Export complete manufacturing package
        
        Includes:
        - STL files (multiple resolutions)
        - STEP file
        - Manufacturing specifications
        - Assembly instructions
        """
        print(f"Exporting complete package: {project_name}")
        print("=" * 60)
        
        exports = {}
        
        # Create project directory
        project_dir = self.output_dir / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Export STL (multiple resolutions)
        stl_dir = project_dir / "stl"
        stl_dir.mkdir(exist_ok=True)
        
        base_stl = str(stl_dir / project_name)
        STLExporter.export_multi_resolution(
            catheter_mesh,
            base_stl,
            resolutions=[1.0, 0.5, 0.25, 0.1]
        )
        exports['stl_high'] = f"{base_stl}_res100.stl"
        exports['stl_medium'] = f"{base_stl}_res050.stl"
        exports['stl_low'] = f"{base_stl}_res025.stl"
        exports['stl_preview'] = f"{base_stl}_res010.stl"
        
        # Export STEP
        step_file = str(project_dir / f"{project_name}.step")
        STEPExporter.export(catheter_mesh, step_file)
        exports['step'] = step_file
        
        # Generate specifications
        spec_file = str(project_dir / f"{project_name}_specifications.json")
        ManufacturingSpecGenerator.generate_spec_sheet(
            design,
            constraints,
            catheter_mesh,
            spec_file
        )
        exports['specifications'] = spec_file
        
        # Generate assembly instructions
        assembly_file = str(project_dir / f"{project_name}_assembly.json")
        ManufacturingSpecGenerator.generate_assembly_instructions(
            design,
            assembly_file
        )
        exports['assembly_instructions'] = assembly_file
        
        # Generate README
        readme_file = str(project_dir / "README.txt")
        self._generate_readme(readme_file, project_name, exports)
        exports['readme'] = readme_file
        
        print("\n" + "=" * 60)
        print(f"Complete package exported to: {project_dir}")
        print(f"Total files: {len(exports)}")
        
        return exports
    
    def _generate_readme(
        self,
        filename: str,
        project_name: str,
        exports: Dict[str, str]
    ):
        """Generate README file for export package"""
        content = f"""
Quantum-Optimized Catheter Design Package
==========================================

Project: {project_name}
Generated by: NVQLink Quantum ML Catheter Designer

Contents:
---------

STL Files (3D Printing):
  - High resolution (100%): For final manufacturing
  - Medium resolution (50%): For verification
  - Low resolution (25%): For quick testing
  - Preview resolution (10%): For visualization

CAD Files:
  - STEP file: For import into CAD software

Documentation:
  - Manufacturing specifications: Detailed dimensions and tolerances
  - Assembly instructions: Step-by-step assembly guide
  - This README file

Usage:
------

1. Review manufacturing specifications for requirements
2. Import high-resolution STL into 3D printer software
3. Configure printer settings per specifications
4. Print catheter components
5. Follow assembly instructions for post-processing
6. Perform quality control checks per specifications

Important Notes:
----------------

- This is a research/prototype design
- Not FDA approved for clinical use
- Requires biocompatibility testing
- Follow all applicable medical device regulations
- Sterilization required before any use

For questions or support, refer to project documentation.
"""
        
        with open(filename, 'w') as f:
            f.write(content.strip())


if __name__ == '__main__':
    from quantum_catheter_optimizer import PatientConstraints, DesignParameters
    from catheter_geometry import CompleteCatheterGenerator
    
    print("3D Model Exporter - Manufacturing Package")
    print("=" * 60)
    
    # Example design
    design = DesignParameters(
        outer_diameter=4.5,
        inner_diameter=3.8,
        wall_thickness=0.35,
        tip_angle=30.0,
        flexibility_index=0.75,
        side_holes=[200, 400, 600, 800],
        material_composition={'polyurethane': 0.7, 'silicone': 0.2, 'ptfe': 0.1}
    )
    
    constraints = PatientConstraints(
        vessel_diameter=5.5,
        vessel_curvature=0.15,
        required_length=1000.0,
        flow_rate=8.0
    )
    
    # Generate mesh
    print("\nGenerating catheter mesh...")
    generator = CompleteCatheterGenerator(design, constraints)
    catheter = generator.generate()
    
    # Export complete package
    print("\n" + "=" * 60)
    exporter = ExportManager(output_directory="./catheter_exports")
    
    exports = exporter.export_complete_package(
        design=design,
        constraints=constraints,
        catheter_mesh=catheter,
        project_name="quantum_optimized_catheter_v1"
    )
    
    print("\nExported files:")
    for file_type, filepath in exports.items():
        print(f"  {file_type}: {filepath}")
