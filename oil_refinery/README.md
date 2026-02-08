
# Oil Refinery Dashboard Application

This is a Flask-based web application for the Oil Refinery project.
It integrates engineering drawings, CFD simulations, pipeline network visualization, and a trading desk simulation.

## Features
- **Engineering Drawings**: Detailed SVG blueprints of Heat Exchangers, Reactors, Valves, and Plant Layouts.
- **CFD Simulation**: Live Computational Fluid Dynamics simulation of oil flow through a valve (using Lattice Boltzmann Method).
- **Pipeline Network**: Visualization of the Jamnagar and AB-BC pipeline networks using NetworkX.
- **Trading Desk**: Real-time simulated market data ticker for Crude Oil prices.

## Prerequisites
- Python 3.x
- Flask
- Matplotlib
- NetworkX
- Numpy

## Installation
```bash
pip install flask matplotlib networkx numpy
```

## Running the App
```bash
python3 app.py
```
Access the dashboard at: http://127.0.0.1:5000

## Architecture
- `app.py`: Main Flask application entry point.
- `refinery_design.py`: Generates SVG engineering drawings.
- `cfd_simulation.py`: Runs fluid dynamics simulations and generates plots.
- `pipeline_network.py`: Manages and visualizes pipeline graph networks.
- `trading_engine.py`: Simulates oil market data.
- `templates/index.html`: Main dashboard frontend.
