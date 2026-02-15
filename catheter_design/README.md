# Quantum ML Catheter Designer

A professional-grade platform for designing patient-specific 3D printed catheter parts using quantum machine learning optimization with NVIDIA's NVQLink (CUDA-Q) framework.

## Features

### 🧬 Quantum ML Optimization
- **Variational Quantum Circuits (VQC)**: Parameterized quantum circuits explore high-dimensional catheter design space
- **Quantum Feature Mapping**: Encodes patient-specific anatomical constraints into quantum states
- **Multi-Objective Optimization**: Simultaneously optimizes fluid dynamics, structural integrity, and anatomical fit
- **Quantum-Classical Hybrid**: Leverages quantum gradients with classical parameter updates

### 🎯 Advanced Design System
- **Parametric Geometry Generation**: Creates optimized catheter body, tip, and drainage holes
- **Computational Fluid Dynamics**: Real-time flow analysis with quantum-enhanced linear solvers
- **Material Optimization**: Biocompatible polymer selection and composite design
- **Patient-Specific Customization**: Tailored to vessel diameter, curvature, and flow requirements

### 🖥️ Premium Web Interface
- **3D Visualization**: Interactive Three.js-based catheter preview with real-time updates
- **Quantum Circuit Display**: Visual representation of variational quantum circuit execution
- **Convergence Tracking**: Real-time optimization progress monitoring
- **Dark Theme Glassmorphism**: Modern, professional medical device aesthetic

### 📦 Manufacturing Export
- **Multi-Format 3D Export**: STL (multiple resolutions), STEP, OBJ formats
- **Manufacturing Specifications**: Detailed dimensions, tolerances, and quality control parameters
- **Assembly Instructions**: Step-by-step post-processing guidelines
- **Biocompatibility Documentation**: Material safety and sterilization protocols

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (for quantum simulation)
- NVIDIA CUDA Toolkit 11.8+ (for cuQuantum)

### Setup

1. **Clone or create project directory:**
```bash
cd catheter-nvqlink
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install CUDA-Q (NVQLink):**
```bash
# Follow NVIDIA's installation guide for your platform
# https://nvidia.github.io/cuda-quantum/latest/install.html
```

## Usage

### Quick Start

1. **Start the backend server:**
```bash
python server.py
```

2. **Open web interface:**
Navigate to `http://localhost:8765` in your browser

3. **Design a catheter:**
   - Enter patient constraints (vessel diameter, curvature, flow rate)
   - Configure quantum settings (qubits, layers, iterations)
   - Click "Start Optimization"
   - View 3D preview and performance metrics
   - Export to STL for 3D printing

### Python API

```python
from quantum_catheter_optimizer import QuantumCatheterOptimizer, PatientConstraints
from catheter_geometry import CompleteCatheterGenerator
from model_exporter import ExportManager

# Define patient constraints
patient = PatientConstraints(
    vessel_diameter=5.5,  # mm
    vessel_curvature=0.15,  # 1/mm
    required_length=1000.0,  # mm
    flow_rate=8.0  # ml/min
)

# Create optimizer
optimizer = QuantumCatheterOptimizer(
    n_qubits=8,
    n_layers=3,
    max_iterations=100
)

# Run optimization
design = optimizer.optimize(patient)

# Generate 3D geometry
generator = CompleteCatheterGenerator(design, patient)
catheter_mesh = generator.generate()

# Export for manufacturing
exporter = ExportManager()
exports = exporter.export_complete_package(
    design, patient, catheter_mesh, "my_catheter_design"
)
```

## Project Structure

```
catheter-nvqlink/
├── quantum_catheter_optimizer.py   # Quantum ML optimization engine
├── catheter_geometry.py            # Parametric 3D geometry generation
├── fluid_dynamics_sim.py           # CFD simulation and analysis
├── material_properties.py          # Biomaterial database and selection
├── model_exporter.py               # 3D export and manufacturing specs
├── server.py                       # FastAPI backend server
├── requirements.txt                # Python dependencies
├── web/
│   ├── index.html                  # Main web application
│   ├── styles.css                  # Premium dark theme UI
│   ├── app.js                      # 3D visualization and controls
│   └── quantum_viz.js              # Quantum circuit visualization
└── exports/                        # Generated STL and spec files
```

## Technical Details

### Quantum Circuit Architecture

The variational quantum circuit uses the following structure:

1. **Feature Encoding Layer**: Patient constraints → quantum states (angle encoding)
2. **Parameterized Layers**: Ry, Rz rotations for design exploration
3. **Entangling Layers**: Circular CNOT gates for parameter correlations
4. **Measurement**: Computational basis measurement → design parameters

### Design Parameter Mapping

Quantum measurement states decode to:
- **Outer Diameter**: 80-95% of vessel diameter
- **Wall Thickness**: 0.1-0.3 mm
- **Tip Angle**: 20-45 degrees
- **Flexibility Index**: 0.5-0.9
- **Side Hole Placement**: Based on vessel curvature

### Performance Characteristics

- **Optimization Time**: ~5-15 seconds (depends on iterations and qubits)
- **Mesh Generation**: <1 second
- **CFD Simulation**: <3 seconds (analytical), ~10 seconds (numerical)
- **Export Time**: <2 seconds

## Safety and Regulatory Notes

⚠️ **IMPORTANT**: This is a research and prototyping tool. 

- **Not FDA Approved**: Not cleared for clinical use
- **Requires Validation**: All designs must undergo biocompatibility testing
- **Regulatory Compliance**: Follow ISO 10993, ISO 11137, and applicable medical device regulations
- **Manufacturing**: Use certified medical-grade materials only
- **Sterilization**: Validate sterilization process per ISO standards

## Contributing

This project demonstrates quantum ML applications in medical device design. Contributions welcome for:

- Advanced CFD solvers
- Additional catheter types (neurovascular, urinary, etc.)
- Quantum algorithm improvements
- Clinical validation studies
- Manufacturing process optimization

## License

Research and educational use. For commercial applications, ensure compliance with medical device regulations and obtain necessary certifications.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{quantum_catheter_designer,
  title={Quantum ML Catheter Designer},
  author={NVQLink Development Team},
  year={2025},
  description={Patient-specific catheter design using quantum machine learning}
}
```

## Support

For questions, issues, or collaboration inquiries:
- Open an issue on the project repository
- Refer to CUDA-Q documentation: https://nvidia.github.io/cuda-quantum/

---

**Powered by NVIDIA NVQLink (CUDA-Q) Quantum Computing Framework**
