# Deep Brain Stimulation for PTSD Treatment with Generative AI

A comprehensive research and educational system for designing and optimizing deep brain stimulation (DBS) protocols for PTSD treatment using generative AI.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH AND EDUCATIONAL TOOL ONLY**

This system is designed for research, education, and conceptual exploration. It is NOT intended for clinical use. Any clinical application of deep brain stimulation requires:
- Extensive regulatory approval (FDA, CE marking)
- Clinical trials and validation
- Medical oversight and expertise
- Proper safety testing and certification

## Features

### 🔌 Circuit Schematic Generator
- Professional-grade DBS device schematics
- Electrode array design with multi-contact configurations
- Programmable pulse generator circuits
- Power management systems with wireless charging
- Comprehensive safety monitoring systems
- Signal processing for closed-loop control

### 🤖 Generative AI Engine
- **VAE (Variational Autoencoder)**: Learns latent representations of effective stimulation patterns
- **GAN (Generative Adversarial Network)**: Generates optimized waveforms
- **RL (Reinforcement Learning)**: Adaptive parameter optimization using Deep Q-Networks

### 🧠 PTSD Neural Modeling
- Computational models of fear conditioning and extinction circuits
- Target brain regions: Amygdala, Hippocampus, vmPFC, Hypothalamus
- Symptom tracking: Hyperarousal, Re-experiencing, Avoidance, Negative Cognition
- Treatment response prediction over 12-week periods
- Biomarker monitoring (HRV, cortisol, skin conductance, sleep quality)

### 🛡️ Safety Validation
- Charge density limits (Shannon 1992 safe stimulation guidelines)
- Current density monitoring
- Thermal safety calculations
- Biocompatibility checks
- Regulatory compliance validation (IEC 60601, ISO 14708-3, FDA 21 CFR 820)

### 💻 Premium Web Interface
- Interactive circuit designer with component specifications
- 3D brain visualization with target region highlighting
- Real-time waveform generator with safety validation
- AI-driven parameter optimization interface
- Clinical monitoring dashboard with symptom tracking

## Installation

### Prerequisites
- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd c:\Users\User\.gemini\antigravity\scratch\quantum-hebbian-tbi
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the backend server:**
```bash
python server.py
```

The server will start on `http://localhost:5000`

4. **Open the web interface:**
Open `index.html` in your web browser, or use a local server:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000`

## Usage

### 1. Circuit Designer
- Click on different circuit components to view detailed schematics
- Explore electrode arrays, pulse generators, power systems, safety circuits
- View the complete SVG circuit diagram

### 2. 3D Brain Model
- Interact with the 3D brain visualization
- Click on target regions to highlight them
- Rotate and zoom using mouse controls

### 3. Waveform Generator
- Adjust stimulation parameters (amplitude, frequency, pulse width)
- View real-time biphasic waveform visualization
- Check safety metrics and validation status
- Simulate stimulation effects on neural model

### 4. AI Optimizer
- Train VAE and GAN models (50 epochs recommended)
- Generate optimized stimulation parameters
- Run reinforcement learning optimization
- Visualize training progress and results

### 5. Clinical Dashboard
- Monitor PTSD symptom severity in real-time
- Track biomarkers (HRV, cortisol, etc.)
- Predict long-term treatment response
- Visualize neural activity across brain regions

## System Architecture

```
Backend (Python/Flask)
├── server.py                    # REST API server
├── dbs_circuit_generator.py     # Circuit schematic generation
├── generative_ai_engine.py      # VAE, GAN, RL models
├── ptsd_neural_model.py         # Neural pathway simulation
└── safety_validator.py          # Safety and compliance checks

Frontend (HTML/CSS/JavaScript)
├── index.html                   # Main application structure
├── styles.css                   # Premium dark theme styling
├── app.js                       # Application controller
├── circuit_designer.js          # Circuit visualization
├── brain_visualizer.js          # 3D brain model (Three.js)
├── waveform_generator.js        # Waveform design and validation
├── ai_optimizer.js              # AI model interface
└── clinical_dashboard.js        # Clinical monitoring
```

## API Endpoints

### Circuit Generation
- `GET /api/circuit/electrode-array` - Electrode array schematic
- `GET /api/circuit/pulse-generator` - Pulse generator circuit
- `GET /api/circuit/complete` - Complete system schematic
- `GET /api/circuit/svg` - SVG circuit diagram

### AI Engine
- `POST /api/ai/train` - Train VAE and GAN models
- `POST /api/ai/generate/vae` - Generate parameters with VAE
- `POST /api/ai/generate/gan` - Generate parameters with GAN
- `POST /api/ai/optimize/rl` - Optimize with RL agent

### Neural Model
- `GET /api/neural/state` - Get current neural state
- `POST /api/neural/simulate` - Simulate DBS stimulation
- `POST /api/neural/predict` - Predict treatment response
- `POST /api/neural/optimize` - Optimize parameters

### Safety Validation
- `POST /api/safety/validate` - Validate stimulation parameters
- `POST /api/safety/charge-balance` - Check charge balance
- `POST /api/safety/thermal` - Thermal safety check
- `POST /api/safety/compliance` - Regulatory compliance

## Scientific Background

### Target Brain Regions for PTSD
- **Amygdala (BLA)**: Fear processing and emotional memory consolidation
- **Hippocampus**: Contextual memory and fear extinction
- **vmPFC**: Fear extinction learning and emotional regulation
- **Hypothalamus**: Stress response via HPA axis

### DBS Parameters
- **Amplitude**: 0.5-8.0 mA (typical: 3.0 mA)
- **Frequency**: 20-185 Hz (common: 60, 130, 185 Hz)
- **Pulse Width**: 60-210 μs (typical: 90 μs)
- **Waveform**: Biphasic (charge-balanced)

### Safety Limits
- **Charge Density**: <30 μC/cm² (Shannon limit)
- **Current Density**: <2.0 mA/cm²
- **Temperature**: <38°C
- **Power**: <100 mW

## References

1. Shannon, R. V. (1992). A model of safe levels for electrical stimulation. IEEE Transactions on Biomedical Engineering.
2. Langevin, J. P., et al. (2016). Deep brain stimulation for PTSD. Biological Psychiatry.
3. Goodman, W. K., & Alterman, R. L. (2012). Deep brain stimulation for intractable psychiatric disorders. Annual Review of Medicine.

## License

This project is for educational and research purposes only. Not licensed for clinical use.

## Contributing

This is a research prototype. For questions or collaboration opportunities, please contact the development team.

## Acknowledgments

- Circuit designs based on medical device industry standards
- Neural models inspired by computational neuroscience literature
- Safety guidelines from IEC, ISO, and FDA regulations
- AI architectures based on state-of-the-art deep learning research

---

**Remember**: This system is a powerful educational tool for understanding DBS technology and AI-driven medical device optimization. Always prioritize patient safety and regulatory compliance in any real-world applications.
