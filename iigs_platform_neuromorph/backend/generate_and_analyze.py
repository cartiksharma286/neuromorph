
import os
import sys
import uuid
import datetime
import json

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nvqlink import NVQLink
from quantum_classifier import QuantumStatisticalClassifier
from fea_engine import FEAEngine
from igs_system import IGSNavigator

def main():
    print("Initializing NVQLink Dental Platform...")
    
    # 1. Setup
    classifier = QuantumStatisticalClassifier()
    nvqlink = NVQLink(resolution=50) # Coarse resolution for demo
    fea = FEAEngine(output_dir=os.path.join(os.path.dirname(__file__), "data"))
    igs = IGSNavigator()
    
    # 2. Simulate Patient Data
    patient_data = {
        "patient_id": "P-90210",
        "bone_density": 0.75, # Good density
        "gap_size": 4.2       # mm
    }
    print(f"Processing for Patient {patient_data['patient_id']}...")
    
    # 3. Classify and Optimize
    classifier.fit(patient_data)
    params = classifier.predict_optimal_parameters()
    print(f"Optimal Parameters Derived: {params}")
    
    # 4. Generate Geometry
    implant_geo = nvqlink.generate_implant_geometry(params)
    print("Implant geometry generated via NVQLink.")
    
    abutment_geo = nvqlink.generate_abutment_geometry(params)
    print("Abutment geometry generated via NVQLink.")
    
    # 5. Run FEA
    print("Running Finite Element Analysis on Implant...")
    implant_fea, implant_stress = fea.run_analysis(implant_geo, material_props={'modulus': 210}) # Titanium
    print(f"Implant FEA Completed. Safety Factor: {implant_fea['safety_factor']:.2f}")
    
    print("Running Finite Element Analysis on Abutment...")
    abutment_fea, abutment_stress = fea.run_analysis(abutment_geo, material_props={'modulus': 110}) # Zirconia often used for abutments, lower modulus than Ti
    print(f"Abutment FEA Completed. Safety Factor: {abutment_fea['safety_factor']:.2f}")
    
    # 6. IGS Planning & Registration
    print("Initializing IGS Registration...")
    # Mock fiducials
    image_points = [[10, 10, 10], [50, 10, 10], [10, 50, 10], [10, 10, 50]]
    # Tracker points (rotated/translated measurements of the same physical points)
    # Applying a simple translation of (5, 5, 5) for simulation
    tracker_points = [[15, 15, 15], [55, 15, 15], [15, 55, 15], [15, 15, 55]]
    
    fre = igs.register_fiducials(image_points, tracker_points)
    print(f"IGS Registration Completed. FRE: {fre:.4f} mm")
    
    # 7. Save Results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())
    
    # Save Implant Design
    implant_filename = f"implant_{patient_data['patient_id']}_{timestamp}.json"
    fea.export_implant_design(implant_filename, implant_geo)
    
    # Save Abutment Design
    abutment_filename = f"abutment_{patient_data['patient_id']}_{timestamp}.json"
    fea.export_implant_design(abutment_filename, abutment_geo)
    
    # Save ANSYS files
    ansys_implant = f"analysis_implant_{patient_data['patient_id']}_{timestamp}.ans"
    fea.export_ansys(ansys_implant, implant_geo, implant_stress)
    
    ansys_abutment = f"analysis_abutment_{patient_data['patient_id']}_{timestamp}.ans"
    fea.export_ansys(ansys_abutment, abutment_geo, abutment_stress)
    
    # Generate and Save Material LUTs
    print("Generating Stress-Strain Lookup Tables...")
    ti_lut = fea.generate_stress_strain_lut("Titanium")
    ti_lut_file = f"lut_titanium_{timestamp}.json"
    fea.export_lut(ti_lut_file, ti_lut)
    
    zr_lut = fea.generate_stress_strain_lut("Zirconia")
    zr_lut_file = f"lut_zirconia_{timestamp}.json"
    fea.export_lut(zr_lut_file, zr_lut)
    
    # Save IGS Plan
    # Target location based on implant geometry center (approx)
    target_loc = [0.0, 0.0, 0.0] 
    igs_plan_file = igs.export_navigation_plan(os.path.join(os.path.dirname(__file__), "data", "sessions"), session_id, target_loc)
    igs_plan_filename = os.path.basename(igs_plan_file)
    print(f"IGS Plan saved to {igs_plan_filename}")
    
    # Save Session
    session_data = {
        "session_id": session_id,
        "timestamp": timestamp,
        "patient": patient_data,
        "optimization_params": params,
        "results": {
            "implant_fea": implant_fea,
            "abutment_fea": abutment_fea,
            "igs_registration_fre": fre
        },
        "files": {
            "implant_design": implant_filename,
            "abutment_design": abutment_filename,
            "ansys_implant": ansys_implant,
            "ansys_abutment": ansys_abutment,
            "lut_titanium": ti_lut_file,
            "lut_zirconia": zr_lut_file,
            "igs_plan": igs_plan_filename
        }
    }
    session_filename = f"session_{session_id}.json"
    fea.export_design_session(session_filename, session_data)
    
    print("All tasks completed successfully. Files generated:")
    print(json.dumps(session_data['files'], indent=2))

if __name__ == "__main__":
    main()
