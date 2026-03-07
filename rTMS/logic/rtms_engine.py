import numpy as np
import time

def simulate_fea(resolution=20):
    """
    Simulates Finite Element Analysis (FEA) of cortical manifolds for E-field distribution.
    Returns a mock 2D grid of electromagnetic field intensities.
    """
    # Create a 2D grid representing cortical surface E-field
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a focal hotspot (e.g., motor cortex or DLPFC)
    hotspot_x, hotspot_y = np.random.uniform(-0.5, 0.5, 2)
    Z = np.exp(-((X - hotspot_x)**2 + (Y - hotspot_y)**2) / 0.1)
    
    # Add some structural noise (sulci/gyri geometry effects)
    noise = np.sin(X * 10) * np.cos(Y * 10) * 0.1
    Z = Z + noise
    
    return Z.tolist()

def simulate_bem(nodes=50):
    """
    Simulates Boundary Element Method (BEM) for head tissue boundaries to account
    for skin, skull, CSF, and gray matter conductivity differences.
    """
    # Return a 3D boundary geometry array (x,y,z) with scalar potentials
    theta = np.linspace(0, 2 * np.pi, nodes)
    phi = np.linspace(0, np.pi, nodes // 2)
    THETA, PHI = np.meshgrid(theta, phi)
    
    R = 1.0 + 0.05 * np.sin(4 * THETA) * np.cos(4 * PHI)  # Simple deformed sphere model
    
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    
    # Flatten it to just a list of vertices and a mock potential value
    vertices = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            potential = np.exp(-Z[i][j]**2) * 10.0 # simulated BEM potential
            vertices.append({
                "x": round(float(X[i][j]), 3),
                "y": round(float(Y[i][j]), 3),
                "z": round(float(Z[i][j]), 3),
                "c": round(float(potential), 3)
            })
            
    # Sample down to keep payload manageable
    np.random.shuffle(vertices)
    return vertices[:500]

def optimize_protocol(condition="stroke"):
    """
    Uses statistical optimization techniques to dynamically adjust rTMS frequency,
    intensity, and pulse train duration for optimal neuromodulation based on condition.
    """
    # Mocking convergence over several iterations
    iterations = []
    current_freq = 1.0 # Hz
    current_intensity = 40.0 # % MSO
    
    target_freq = 10.0 if condition == "stroke" else 20.0
    target_intensity = 80.0 if condition == "stroke" else 100.0 # higher for dementia deep brain targets
    
    for i in range(20):
        # Statistical gradient descent step
        current_freq += (target_freq - current_freq) * 0.2 + np.random.normal(0, 0.5)
        current_intensity += (target_intensity - current_intensity) * 0.15 + np.random.normal(0, 1.0)
        
        # Calculate a mock fitness/cost function score (closer to 1.0 is better)
        fitness = 1.0 - (abs(target_freq - current_freq) / target_freq + abs(target_intensity - current_intensity) / target_intensity) / 2
        
        iterations.append({
            "iteration": i + 1,
            "frequency_hz": round(max(1.0, current_freq), 2),
            "intensity_mso": round(max(10.0, current_intensity), 2),
            "fitness": round(max(0.0, float(fitness)), 4)
        })
        
    return {
        "final_parameters": iterations[-1],
        "convergence_trajectory": iterations,
        "protocol_summary": f"Optimized {condition.capitalize()} Protocol: {iterations[-1]['frequency_hz']} Hz at {iterations[-1]['intensity_mso']}% MSO"
    }

def run_full_simulation(condition="stroke"):
    """
    Simulates Google Cloud Server execution of complex BEM/FEA and Statistical Optimization.
    """
    return {
        "condition": condition,
        "timestamp": time.time(),
        "cloud_node": "gcp-us-central1-c-tensor-node",
        "optimization": optimize_protocol(condition),
        "fea_grid": simulate_fea(),
        "bem_mesh": simulate_bem()
    }


def get_equipment_list():
    """
    Returns a comprehensive list of rTMS equipment with clinical operating characteristics.
    """
    equipment = [
        {
            "id": "EQ-001",
            "name": "MagVenture MagPro X100",
            "category": "Stimulator Unit",
            "description": "High-performance biphasic/monophasic TMS stimulator with cTBS and TBS capabilities.",
            "specs": {
                "Max Output (% MSO)": "100%",
                "Peak E-Field (V/m)": "220",
                "Pulse Width (µs)": "280",
                "Frequency Range (Hz)": "0.1 – 100",
                "Max Continuous Duty Cycle": "50%",
                "Cooling System": "Active liquid cooling",
                "Power Supply Voltage (V)": "200 – 240 VAC",
                "Weight (kg)": "32"
            },
            "operating_characteristics": {
                "op_temp_c": 25,
                "max_temp_c": 40,
                "efficiency_pct": 92,
                "heat_dissipation_w": 180,
                "emi_shielding_db": 45
            }
        },
        {
            "id": "EQ-002",
            "name": "Magstim Horizon 3.0",
            "category": "Stimulator Unit",
            "description": "Next-gen triple-pulse TMS system with integrated neuronavigation readiness.",
            "specs": {
                "Max Output (% MSO)": "100%",
                "Peak E-Field (V/m)": "200",
                "Pulse Width (µs)": "290",
                "Frequency Range (Hz)": "0.1 – 50",
                "Max Continuous Duty Cycle": "40%",
                "Cooling System": "Forced air + heat sink",
                "Power Supply Voltage (V)": "110 – 240 VAC",
                "Weight (kg)": "28"
            },
            "operating_characteristics": {
                "op_temp_c": 22,
                "max_temp_c": 38,
                "efficiency_pct": 88,
                "heat_dissipation_w": 150,
                "emi_shielding_db": 42
            }
        },
        {
            "id": "EQ-003",
            "name": "Figure-8 Coil (70mm Air-Cooled)",
            "category": "Stimulation Coil",
            "description": "Standard focal coil for precise cortical targeting in stroke motor rehab therapy.",
            "specs": {
                "Coil Diameter (mm)": "70",
                "Inductance (µH)": "16.4",
                "Resistance (mΩ)": "105",
                "Max Surface Temperature (°C)": "41",
                "Focal Depth (mm)": "20 – 35",
                "Max Repetition Rate (Hz)": "30",
                "Cooling System": "Natural convection",
                "Weight (g)": "320"
            },
            "operating_characteristics": {
                "op_temp_c": 36,
                "max_temp_c": 41,
                "efficiency_pct": 85,
                "heat_dissipation_w": 40,
                "emi_shielding_db": 30
            }
        },
        {
            "id": "EQ-004",
            "name": "H7 Deep TMS Coil (Brainsway)",
            "category": "Stimulation Coil",
            "description": "H-coil geometry enabling bilateral deep prefrontal cortex activation — ideal for dementia and depression.",
            "specs": {
                "Coil Geometry": "H-shaped (bilateral)",
                "Focal Depth (mm)": "50 – 70",
                "Max Surface Temperature (°C)": "43",
                "Inductance (µH)": "22.1",
                "Resistance (mΩ)": "130",
                "Max Repetition Rate (Hz)": "20",
                "Cooling System": "Fluid-cooled helmet insert",
                "Weight (g)": "850"
            },
            "operating_characteristics": {
                "op_temp_c": 38,
                "max_temp_c": 43,
                "efficiency_pct": 78,
                "heat_dissipation_w": 70,
                "emi_shielding_db": 28
            }
        },
        {
            "id": "EQ-005",
            "name": "Localite TMS Navigator 4.0",
            "category": "Neuronavigation System",
            "description": "Real-time optical tracking neuronavigation system synced with patient MRI for precise coil targeting.",
            "specs": {
                "Tracking Technology": "Infrared Optical",
                "Spatial Accuracy (mm)": "< 1.5",
                "Update Rate (Hz)": "60",
                "MRI Compatibility": "T1 / T2 / FLAIR",
                "Coil Interfaces": "Universal (Magstim, MagVenture, Deymed)",
                "Display Resolution": "4K",
                "OS": "Windows 11 Embedded"
            },
            "operating_characteristics": {
                "op_temp_c": 23,
                "max_temp_c": 35,
                "efficiency_pct": 97,
                "heat_dissipation_w": 25,
                "emi_shielding_db": 55
            }
        },
        {
            "id": "EQ-006",
            "name": "64-Channel EEG Amplifier (BrainProducts)",
            "category": "EEG Monitoring",
            "description": "High-resolution TMS-compatible EEG system for real-time cortical excitability monitoring.",
            "specs": {
                "Channels": "64 + 8 AUX",
                "Sampling Rate (kHz)": "25",
                "Input Impedance (MΩ)": "> 1000",
                "CMRR (dB)": "130",
                "ADC Resolution (bit)": "24",
                "TMS Artifact Recovery (ms)": "< 5",
                "Bandwidth (Hz)": "DC – 5000"
            },
            "operating_characteristics": {
                "op_temp_c": 22,
                "max_temp_c": 35,
                "efficiency_pct": 99,
                "heat_dissipation_w": 12,
                "emi_shielding_db": 60
            }
        },
        {
            "id": "EQ-007",
            "name": "Robotic Coil Positioning Arm (RPA-3)",
            "category": "Positioning System",
            "description": "6 DOF robotic arm for automated, reproducible coil placement with < 1mm repeatability.",
            "specs": {
                "Degrees of Freedom": "6",
                "Repeatability (mm)": "0.8",
                "Max Payload (kg)": "5",
                "Reach (mm)": "900",
                "Control Interface": "USB 3.0 / LAN",
                "Force Sensor": "6-axis, 0.01 N resolution",
                "Safety Standard": "ISO 13849 PLd"
            },
            "operating_characteristics": {
                "op_temp_c": 21,
                "max_temp_c": 40,
                "efficiency_pct": 95,
                "heat_dissipation_w": 55,
                "emi_shielding_db": 35
            }
        },
        {
            "id": "EQ-008",
            "name": "GCP rTMS Cloud Processing Node",
            "category": "Cloud Infrastructure",
            "description": "Tensor Processing Unit (TPU) cluster for real-time FEA/BEM statistical optimization bursts.",
            "specs": {
                "Node Type": "n2-highmem-32",
                "vCPUs": "32",
                "RAM (GB)": "256",
                "TPU Version": "v4",
                "Network Bandwidth (Gbps)": "100",
                "Region": "us-central1-c",
                "Latency (ms)": "< 12",
                "SLA Uptime": "99.99%"
            },
            "operating_characteristics": {
                "op_temp_c": 20,
                "max_temp_c": 28,
                "efficiency_pct": 98,
                "heat_dissipation_w": 850,
                "emi_shielding_db": 70
            }
        }
    ]
    return equipment
