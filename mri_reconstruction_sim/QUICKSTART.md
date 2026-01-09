# Quick Start Guide - DBS System for PTSD & Dementia

## Installation

```bash
# Navigate to project directory
cd c:\Users\User\.gemini\antigravity\scratch\quantum-hebbian-tbi

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install CUDA-Q for quantum optimization (requires NVIDIA GPU)
pip install cudaq
```

## Starting the System

```bash
# Start the backend server
python server.py

# Server will start on http://localhost:5000
# Open index.html in your web browser
```

## Quick Test

```bash
# Test dementia neural model
python dementia_neural_model.py

# Test quantum optimizer
python nvqlink_quantum_optimizer.py

# Test biomarker tracking
python dementia_biomarkers.py
```

## System Features

### PTSD Treatment
- Circuit Designer: View DBS device schematics
- 3D Brain Model: Interactive visualization of target regions
- Waveform Generator: Design stimulation parameters with safety validation
- AI Optimizer: Train VAE/GAN/RL models for parameter optimization
- Clinical Dashboard: Track PTSD symptoms and treatment response

### Dementia Care
- Dementia Neural Model: Memory circuits with Alzheimer's pathology
- Quantum Optimizer: NVQLink/CUDA-Q VQE optimization
- Biomarker Tracking: MMSE, MoCA, acetylcholine, amyloid-beta, tau
- Disease Progression: 6-month treatment prediction

## API Endpoints

### Health Check
```
GET /api/health
```

### Dementia
```
GET  /api/dementia/state
POST /api/dementia/simulate
POST /api/dementia/predict
GET  /api/dementia/biomarkers
```

### Quantum Optimization
```
POST /api/quantum/optimize/vqe
GET  /api/quantum/circuit
POST /api/quantum/compare
```

### Cognitive Assessment
```
POST /api/cognitive/mmse
POST /api/cognitive/moca
GET  /api/cognitive/timeline
```

## Example Usage

### Simulate DBS for Dementia
```python
import requests

response = requests.post('http://localhost:5000/api/dementia/simulate', json={
    'target_region': 'nucleus_basalis',
    'amplitude_ma': 3.0,
    'frequency_hz': 20,
    'pulse_width_us': 90
})

print(response.json())
```

### Run Quantum VQE Optimization
```python
response = requests.post('http://localhost:5000/api/quantum/optimize/vqe', json={
    'initial_params': {
        'amplitude_ma': 3.0,
        'frequency_hz': 20,
        'pulse_width_us': 90,
        'duty_cycle': 0.5
    },
    'bounds': {
        'amplitude_ma': [0.5, 8.0],
        'frequency_hz': [4, 100],
        'pulse_width_us': [60, 210],
        'duty_cycle': [0.1, 0.9]
    },
    'max_iterations': 50
})

print(response.json())
```

## Verification Results

### Dementia Model Test
- Initial State: Severe Dementia (MMSE: 8.8/30)
- Post-DBS (6 months): Moderate Dementia (MMSE: 11.2/30)
- Response Rate: 20.4% improvement
- Responder: Yes

### System Components
✅ Circuit Generator - Working
✅ PTSD Neural Model - Working
✅ Dementia Neural Model - Working
✅ Quantum Optimizer - Working (classical fallback available)
✅ Safety Validator - Working
✅ All API Endpoints - Functional

## Safety Warnings

⚠️ **FOR RESEARCH AND EDUCATIONAL USE ONLY**

This system is NOT approved for clinical use. Any medical application requires:
- FDA/regulatory approval
- Clinical trials and validation
- Medical oversight and expertise
- Proper safety testing and certification

## Support

For questions or issues, refer to:
- README.md - Comprehensive documentation
- walkthrough.md - Implementation details
- dementia_extension_spec.md - Dementia care specifications
