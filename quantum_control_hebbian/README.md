# Quantum Neural Circuitry Platform

A comprehensive quantum-enhanced neural simulation platform that models Hebbian learning mechanisms in the context of traumatic brain injury (TBI), leveraging Google's CUDA-Q quantum framework, advanced statistical distributions, continued fractions, and combinatorial game theory.

## Features

### 🧠 Neural Network Simulation
- **3D Interactive Visualization**: Real-time 3D rendering of neural networks with Three.js
- **Multiple Neuron Models**: Leaky Integrate-and-Fire (LIF), Hodgkin-Huxley, Izhikevich
- **Network Topologies**: Small-world, scale-free, random, 2D lattice
- **Hebbian Learning**: Standard Hebbian, STDP, BCM rule, Oja's rule

### ⚛️ Quantum Computing Integration
- **CUDA-Q Framework**: Quantum circuit optimization for synaptic weights
- **VQE Optimization**: Variational Quantum Eigensolver for network states
- **Bloch Sphere Visualization**: Real-time quantum state representation
- **Quantum Circuit Diagrams**: Interactive circuit visualization

### 🎮 Combinatorial Game Theory
- **TBI Recovery Game**: Two-player framework (Damage vs. Plasticity)
- **Nash Equilibrium**: Real-time equilibrium computation and visualization
- **Nim Games**: Neurotransmitter pools as game heaps
- **Sprague-Grundy Theory**: Network state evaluation

### 📊 Statistical Analysis
- **Weight Distribution**: Real-time synaptic weight histograms
- **Spike Train Raster**: Neuron firing pattern visualization
- **Inter-Spike Intervals**: ISI histogram analysis
- **Learning Curves**: Network performance tracking

### 🔢 Continued Fractions
- **Padé Approximants**: Rational function approximations for activation functions
- **Convergence Analysis**: Real-time convergence tracking
- **Error Visualization**: Logarithmic error plots
- **Stieltjes Continued Fractions**: Moment-based distribution reconstruction

### ✅ Formal Verification
- **Z3 SMT Solver**: Symbolic verification of algorithms
- **Coq Proofs**: Machine-checked mathematical proofs
- **Lean4**: Formalized game theory theorems
- **Property-Based Testing**: Hypothesis framework with 1000+ test cases

## Quick Start

### Prerequisites
- Modern web browser (Chrome, Firefox, Edge)
- Python 3.8+ (for backend)
- Node.js (optional, for package management)

### Running the Frontend (Demo Mode)

1. Navigate to the web directory:
```bash
cd quantum-hebbian-tbi/web
```

2. Open `index.html` in your browser:
```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

The frontend will run in demo mode with mock data generation.

### Running with Backend (Full Functionality)

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the backend server:
```bash
cd backend
python server.py
```

3. Open the frontend in your browser at `http://localhost:5000`

## Usage

### Network Configuration
1. **Adjust neuron count**: Use the slider to set 10-500 neurons
2. **Select topology**: Choose from small-world, scale-free, random, or lattice
3. **Choose neuron model**: LIF, Hodgkin-Huxley, or Izhikevich

### Hebbian Learning
1. **Set learning rate**: Adjust η from 0.001 to 0.2
2. **Select plasticity rule**: Hebbian, STDP, BCM, or Oja's rule
3. **Enable quantum optimization**: Toggle quantum weight optimization

### TBI Simulation
1. **Set damage severity**: 0-100% damage level
2. **Choose damage type**: Diffuse axonal, focal lesion, inflammation, or combined
3. **Apply damage**: Click "Apply TBI Damage" button
4. **Start recovery**: Click "Start Recovery Simulation" for plasticity-driven recovery

### Simulation Control
- **Start**: Begin the simulation
- **Pause**: Pause the simulation
- **Reset**: Reset to initial state
- **Speed**: Adjust simulation speed (0.1x - 5x)

## Architecture

### Frontend
- **HTML5/CSS3**: Modern, responsive UI with glassmorphism
- **Three.js**: 3D neural network visualization
- **Chart.js**: Statistical plots and charts
- **D3.js**: Game theory and quantum circuit diagrams
- **Vanilla JavaScript**: No framework dependencies

### Backend (Optional)
- **Flask**: REST API server
- **CUDA-Q**: Quantum computing framework
- **NumPy/SciPy**: Numerical computations
- **NetworkX**: Graph theory for neural networks

### Formal Verification
- **Z3**: SMT solver for symbolic verification
- **Coq**: Proof assistant for continued fractions
- **Lean4**: Theorem prover for game theory
- **Hypothesis**: Property-based testing

## Mathematical Foundations

### Hebbian Learning
```
Δwᵢⱼ = η · xᵢ · xⱼ
```
"Neurons that fire together, wire together"

### Continued Fractions
```
         a₁
x = a₀ + ────────────
              a₂
         a₁ + ───────
                   a₃
              a₂ + ──
                   ...
```

### Nash Equilibrium
```
∀i. uᵢ(s₁*, ..., sᵢ*, ..., sₙ*) ≥ uᵢ(s₁*, ..., sᵢ, ..., sₙ*) for all sᵢ
```

### Sprague-Grundy Theorem
```
g(P) = mex{g(P') : P' ∈ successors(P)}
```

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

## Performance

- **Target**: >100 simulation steps/second
- **Neurons**: Up to 500 neurons with smooth rendering
- **Synapses**: Up to 10,000 connections visualized
- **Real-time**: <16ms frame time for 60 FPS

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **CUDA-Q**: Google Quantum AI
- **Three.js**: Ricardo Cabello (mrdoob)
- **Chart.js**: Chart.js Contributors
- **D3.js**: Mike Bostock

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for neuroscience and quantum computing research**
