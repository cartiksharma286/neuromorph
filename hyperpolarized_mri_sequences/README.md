# Hyperpolarized Pulse Sequence Generator

A professional web-based tool for designing, optimizing, and exporting pulse sequences for hyperpolarized MRI imaging, developed for **Sunnybrook Research Institute**.

![Hyperpolarized Sequence Generator](https://img.shields.io/badge/Version-1.0.0-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

### Sequence Types
- **Variable Flip Angle (VFA)**: Constant signal and max SNR optimization strategies
- **Spiral Trajectories**: Uniform and variable density Archimedean spirals
- **EPI**: Single-shot and multi-shot echo planar imaging
- **Dynamic Time-Resolved**: Kinetic imaging with temporal optimization
- **Metabolic Imaging**: Spectral-spatial excitation for multi-metabolite imaging

### Supported Nuclei
- **¹³C**: Pyruvate, Lactate, Bicarbonate, Alanine
- **¹²⁹Xe**: Gas phase and dissolved phase
- **³He**: Hyperpolarized helium-3
- Custom nuclei configuration

### Visualization
- Real-time sequence timing diagrams
- k-space trajectory visualization
- Signal evolution prediction
- Flip angle schedule plots
- SNR and efficiency calculations

### Export Capabilities
- **PyPulseq**: Vendor-neutral Python sequences
- **Siemens IDEA**: Parameter sheets
- **GE EPIC**: Control variable definitions
- **Philips**: Protocol parameter files
- **JSON**: Standardized sequence definitions

## 🚀 Getting Started

### Web Interface

Simply open `index.html` in a modern web browser:

```bash
# Navigate to the directory
cd hyperpolarized_sequences

# Open in browser (or double-click index.html)
open index.html  # macOS
start index.html  # Windows
xdg-open index.html  # Linux
```

### Python Backend (Optional)

For advanced optimization and PyPulseq generation:

```bash
# Install dependencies
pip install -r requirements.txt

# Run VFA optimizer
python sequence_optimizer.py

# Generate PyPulseq sequences
python pypulseq_generator.py
```

## 📖 Usage Guide

### 1. Select Nucleus
Choose your hyperpolarized nucleus from the sidebar (e.g., ¹³C Pyruvate, ¹²⁹Xe). The tool automatically loads appropriate T1, T2, and chemical shift parameters.

### 2. Design Sequence
Navigate to your desired sequence type:

#### Variable Flip Angle
1. Set number of frames and TR
2. Choose optimization strategy (Constant Signal or Max SNR)
3. Click "Calculate VFA"
4. View flip angle schedule and predicted signal evolution

#### Spiral Trajectory
1. Configure FOV, resolution, and interleaves
2. Set gradient limits (max gradient, max slew rate)
3. Choose uniform or variable density
4. Click "Design Spiral"
5. Visualize k-space trajectory and gradient waveforms

#### EPI Sequence
1. Set matrix size, FOV, and number of shots
2. Configure partial Fourier and trajectory type (blipped/flyback)
3. Click "Design EPI"
4. View k-space coverage and timing metrics

#### Dynamic Imaging
1. Set temporal resolution and number of frames
2. Select readout type (spiral, EPI, radial, Cartesian)
3. Enable VFA integration if desired
4. Configure kinetic model parameters
5. Click "Design Dynamic Sequence"

#### Metabolic Imaging
1. Select target metabolites (pyruvate, lactate, etc.)
2. Set B0 field strength and slice thickness
3. Choose pulse design type (spectral-spatial, multiband, EPSI)
4. Click "Design Pulse"
5. View spectral and spatial profiles

### 3. Export Sequence
1. Click "Export Sequence" button
2. Choose export format:
   - **PyPulseq** for vendor-neutral sequences
   - **Vendor formats** (Siemens/GE/Philips) for scanner implementation
   - **JSON** for archiving and sharing
3. Copy to clipboard or download file

## 🔬 Scientific Background

### Variable Flip Angle Strategies

**Constant Signal Approach (CSA)**:
- Maintains uniform signal across all acquisitions
- Formula: θₙ = arctan(1/√(N-n))
- Reference: Nagashima K. MRM 2008

**Maximum SNR Approach**:
- Maximizes total SNR over all frames
- Formula: cos(θₙ) = √((N-n)/(N-n+1)) × e^(-TR/T1)
- Reference: Larson PEZ et al. MRM 2013

### Spiral Trajectories
- Archimedean spiral with gradient/slew constraints
- Variable density for compressed sensing
- Multi-interleave acquisition for SNR

### Spectral-Spatial Pulses
- Simultaneous spatial and spectral selectivity
- Chemical shift selective excitation
- Multi-metabolite imaging support

## 🏥 Clinical Applications

### ¹³C Pyruvate Imaging
- Metabolic imaging of lactate conversion
- Oncology: tumor metabolism assessment
- Cardiology: cardiac metabolism

### ¹²⁹Xe Lung Imaging
- Ventilation mapping
- Gas exchange measurements
- Pulmonary function assessment

### Multi-Metabolite Studies
- Pyruvate → Lactate, Bicarbonate, Alanine
- Kinetic modeling
- Therapeutic response monitoring

## 📊 Python Tools

### VFA Optimization
```python
from sequence_optimizer import VFAOptimizer

# Create optimizer
optimizer = VFAOptimizer(num_frames=20, t1=43, tr=100)

# Calculate optimal flip angles
flip_angles = optimizer.constant_signal_vfa()

# Compare strategies
results = optimizer.compare_strategies()
```

### PyPulseq Generation
```python
from pypulseq_generator import HyperpolarizedSequence

# Create sequence generator
seq_gen = HyperpolarizedSequence()

# Generate VFA sequence
seq = seq_gen.create_vfa_sequence(
    flip_angles=flip_angles,
    tr=0.1,  # 100 ms
    fov=0.24,  # 24 cm
    slice_thickness=0.01  # 10 mm
)

# Export
seq_gen.write_sequence('hyperpolarized.seq')
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📝 Citation

If you use this tool in your research, please cite:

```
Hyperpolarized Pulse Sequence Generator
Sunnybrook Research Institute
Version 1.0.0 (2025)
```

## 📧 Contact

**Sunnybrook Research Institute**
- Website: [sunnybrook.ca](https://sunnybrook.ca)
- Email: research@sunnybrook.ca

## 🙏 Acknowledgments

- PyPulseq community for open-source MRI pulse programming
- Hyperpolarized MRI research community
- Sunnybrook Research Institute

## 📄 License

MIT License - see LICENSE file for details

---

**Built with precision for hyperpolarized imaging research** 🔬✨
