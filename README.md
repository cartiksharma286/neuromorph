# RF Coil Designer with Generative AI

A comprehensive Python package for designing RF coils using generative AI algorithms, parametric modeling, and optimization techniques.

## Features

### 🧬 Generative Design
- **Evolutionary Algorithms**: Genetic algorithm-based coil optimization
- **Parametric Generation**: Create designs from physical constraints
- **Multi-objective Optimization**: Balance frequency, Q-factor, size

### 🔧 Coil Types Supported
- **Solenoid Coils**: Air-core cylindrical coils
- **Planar Spiral Coils**: PCB-compatible flat spirals
- **Helmholtz Pairs**: Dual-coil configurations

### 📊 Analysis & Calculation
- Inductance calculation (Wheeler's formula)
- Resonant frequency determination
- Quality factor (Q) estimation
- Skin effect compensation
- Impedance matching

### 🎨 Visualization
- 2D cross-sectional views
- 3D helix rendering
- Circuit schematics with component values
- Comparison plots

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from coil_designer import RFCoilDesigner, CoilParameters

# Create designer
designer = RFCoilDesigner()

# Define coil
params = CoilParameters(
    coil_type='solenoid',
    wire_diameter=1.0,  # mm
    turns=15,
    diameter=30,  # mm
    length=40  # mm
)

# Calculate properties
L = designer.calculate_inductance(params)
f_res = designer.calculate_resonant_frequency(L, capacitance=100e-12)
Q = designer.calculate_quality_factor(params, f_res)

print(f"Inductance: {L*1e6:.2f} µH")
print(f"Resonant Frequency: {f_res/1e6:.2f} MHz")
print(f"Quality Factor: {Q:.1f}")
```

### Generative Design for Target Frequency

```python
# Design coil for specific frequency (e.g., 13.56 MHz ISM band)
target_freq = 13.56e6

constraints = {
    'max_diameter': 50,
    'capacitance': 150e-12
}

optimized = designer.design_for_frequency(
    target_freq,
    coil_type='solenoid',
    constraints=constraints
)
```

### Evolutionary Optimization

```python
from coil_designer import GenerativeCoilDesigner

gen_designer = GenerativeCoilDesigner(designer)

# Evolve optimal design
evolved = gen_designer.evolve_design(
    target_freq=27.12e6,
    population_size=50,
    generations=20,
    constraints=constraints
)
```

### Visualization

```python
from coil_visualizer import CoilVisualizer

visualizer = CoilVisualizer(designer)

# 3D geometry
visualizer.plot_coil_geometry(params, save_path='coil_3d.png')

# Circuit schematic
visualizer.generate_schematic(params, capacitance=100e-12, 
                             save_path='schematic.png')
```

## Running the Demo

```bash
cd C:\Users\User\.gemini\antigravity\scratch\rf_coil_designer
python demo.py
```

The demo includes:
1. **Parametric Design**: Manual coil specification and analysis
2. **Target Frequency Design**: Optimize for 13.56 MHz ISM band
3. **Evolutionary Design**: GA-based optimization for 27.12 MHz
4. **Planar Spiral**: PCB coil design
5. **Coil Comparison**: Side-by-side comparison of different types

## Theory

### Inductance Calculation

**Solenoid (Wheeler's Formula):**
```
L (µH) = N² × r² / (9r + 10l)
```
where N = turns, r = radius (mm), l = length (mm)

**Planar Spiral:**
```
L = μ₀N²d_avg² / (8d_avg + 11(d_outer - d_inner))
```

### Resonant Frequency
```
f = 1 / (2π√(LC))
```

### Quality Factor
```
Q = ωL / R = 2πfL / R
```
where R accounts for DC resistance and skin effect.

## API Reference

### `RFCoilDesigner`

Main class for coil design and analysis.

**Methods:**
- `calculate_inductance(params)` - Calculate coil inductance
- `calculate_resonant_frequency(L, C)` - Find resonant frequency
- `calculate_quality_factor(params, freq)` - Estimate Q factor
- `design_for_frequency(target_freq, ...)` - Optimize for target frequency

### `GenerativeCoilDesigner`

Generative AI-based design system.

**Methods:**
- `evolve_design(target_freq, ...)` - Evolutionary optimization
- `generate_design_variations(params, n)` - Create design variants
- `evaluate_fitness(params, target)` - Score design quality

### `CoilVisualizer`

Visualization and schematic generation.

**Methods:**
- `plot_2d_solenoid(params)` - 2D side view
- `plot_2d_planar_spiral(params)` - 2D top view
- `plot_3d_solenoid(params)` - 3D helix
- `generate_schematic(params, ...)` - Circuit schematic
- `plot_coil_geometry(params)` - Complete visualization

### `CoilParameters`

Dataclass for coil specifications.

**Fields:**
- `coil_type`: 'solenoid', 'planar_spiral', or 'helmholtz'
- `wire_diameter`: Wire diameter in mm
- `turns`: Number of turns
- `diameter`: Outer diameter in mm
- `length`: Length (solenoid) or spacing (planar)
- `substrate_thickness`: PCB thickness (optional)

## Applications

- **RF Communications**: Antenna tuning circuits
- **NFC/RFID**: Reader coils (13.56 MHz)
- **Wireless Power**: Inductive charging coils
- **Medical Imaging**: MRI RF coils
- **Plasma Generators**: Induction heating
- **Scientific Instruments**: NMR probes
- **Amateur Radio**: Impedance matching networks

## Common ISM Bands

Pre-optimized designs available for:
- 6.78 MHz (Industrial)
- 13.56 MHz (NFC, RFID)
- 27.12 MHz (Industrial)
- 40.68 MHz (Medical)

## Optimization Algorithms

### Differential Evolution
- Global optimization for continuous parameters
- Used in `design_for_frequency()`
- Fast convergence (typically < 100 iterations)

### Genetic Algorithm
- Population-based evolutionary search
- Used in `evolve_design()`
- Good for multi-objective optimization
- Crossover + mutation + selection

## Performance

Typical optimization times (on modern CPU):
- Differential Evolution: 1-3 seconds
- Genetic Algorithm (50 pop, 20 gen): 3-5 seconds
- Single inductance calculation: < 1 ms

## Design Guidelines

### Solenoid Coils
- Length/Diameter ratio: 1-3 for optimal Q
- Wire spacing: 1-2× wire diameter
- Typical Q: 50-200 at HF/VHF

### Planar Spirals
- Track spacing: ≥ track width
- Inner diameter: ≥ 30% outer diameter
- Typical Q: 20-80 (lower than solenoid)

### Quality Factor Optimization
- Use larger wire diameter
- Minimize length (fewer turns at larger diameter)
- Use Litz wire for high frequency

## Future Enhancements

- [ ] Multi-layer coil support
- [ ] Mutual inductance calculations
- [ ] Transformer design
- [ ] CAD file export (DXF, STEP)
- [ ] FEA integration
- [ ] Temperature coefficient analysis
- [ ] Cost optimization

## References

1. Wheeler, H. A. (1928). "Simple Inductance Formulas for Radio Coils"
2. Terman, F. E. (1943). "Radio Engineers' Handbook"
3. Grover, F. W. (2009). "Inductance Calculations"

## License

MIT License - Free for educational and commercial use

---

**Developed with Generative AI for RF Engineering Applications**
