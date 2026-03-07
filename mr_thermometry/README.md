# MR Guided Thermometry App

This is a Python application that uses VTK to simulate and visualize Magnetic Resonance Guided Focused Ultrasound (MRgFUS) or similar thermal ablation procedures.

## Features

- **Synthetic Data Generation**: Creates a placeholder structural MR magnitude image and simulates baseline phase changes.
- **Proton Resonance Frequency (PRF) Shift calculation**: Converts phase changes into simulated temperature maps.
- **VTK rendering**: Visualizes the anatomical image with a superimposed color-coded temperature map overlay.
- **Interactive UI**: Includes a slider to interactively increase the simulated heating power, dynamically recalculating the temperature profile and instantly updating the display.

## Installation / Prerequisites

Ensure you have VTK and numpy installed:
```bash
pip install vtk numpy
```

## Running the App

```bash
python app.py
```

## Usage

- The slider at the bottom controls the simulated MRgFUS heating power.
- Dragging the slider increases the temperature.
- Temperatures above normal body baseline (37°C) will progressively appear as a colored overlay moving from blue to red scaling up to thermal ablation ranges.
