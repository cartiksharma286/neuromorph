# Quantum MR Pulse Generator

This project provides tools for generating, optimizing, and visualizing MRI Pulse Sequences using simulated Quantum Surface Integral methods.

## Standalone Applications

Two standalone Python applications have been added to generate `.seq` files (Pulseq format) directly from your desktop.

### 1. Command Line Interface (CLI)
Use `seq_app.py` for batch processing or terminal-based generation.

**Usage:**
```bash
# List available sequence types
python seq_app.py list

# Generate a GRE sequence
python seq_app.py generate --type GRE --te 15 --tr 100 --flip_angle 45 --output my_sequence.seq

# Generate a Spin Echo sequence with optimization
python seq_app.py generate --type SE --te 30 --tr 500 --optimize --output optimized_se.seq
```

### 2. Graphical User Interface (GUI)
Use `gui_app.py` for an interactive experience.

**Usage:**
```bash
python gui_app.py
```
This launches a window where you can:
- Select Sequence Type (GRE / SE)
- Configure Parameters (TE, TR, Flip Angle)
- Enable Quantum Optimization
- Click "GENERATE SEQUENCE" to save to a file.

## Core Files
- `pulse_generator.py`: Core logic for sequence design.
- `quantum_integrals.py`: Quantum math calculations.
- `reconstruction.py`: Simple image reconstruction simulator.
- `server.py`: Backend for the web interface.
