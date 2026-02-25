from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys
import os
import shutil
import uuid
import numpy as np

# Add parent directory to path to import local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from reactor_engine import ReactorDesigner
from visualizer import ReactorVisualizer

from fea_quantum import QuantumFEASolver

app = FastAPI(title="Statistical Congruence Reactor Designer")

# Setup Directories
OUTPUT_DIR = os.path.join(BASE_DIR, "vessel_results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

class DesignRequest(BaseModel):
    volume: float = 7500.0 # Liters
    pressure: float = 3.5 # Bar (Diketene slightly pressurized)
    temp: float = 120.0 # Reaction temp C
    
class FEAEquest(BaseModel):
    diameter: float
    height: float
    grid_resolution: int = 50
    partitions: int = 4

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.post("/generate")
def generate_vessel(request: DesignRequest):
    design_id = str(uuid.uuid4())
    output_base = os.path.join(OUTPUT_DIR, design_id)
    
    # Volume convert L -> m3
    vol_m3 = request.volume / 1000.0
    
    # 1. Design Engine
    designer = ReactorDesigner(vol_m3, request.pressure, request.temp)
    designer.calculate_dimensions()
    designer.optimize_jacket()
    
    # 2. Visualize
    vis_filename = f"{output_base}.png"
    visualizer = ReactorVisualizer(designer)
    visualizer.plot_vessel(vis_filename)
    
    # 3. Specs
    specs = designer.get_specs()
    # Add design_id to specs for easy retrieval later
    specs["parameters"] = {"D": specs['geometry']['diameter'], "H": specs['geometry']['tan_tan_height']}
    
    response = {
        "design_id": design_id,
        "specs": specs,
        "image_url": f"/results/{design_id}.png"
    }
    
    return response

@app.post("/analyze_fea")
def analyze_fea(request: FEAEquest):
    design_id = str(uuid.uuid4())
    output_base = os.path.join(OUTPUT_DIR, design_id)
    
    # Run Quantum FEA
    fea_solver = QuantumFEASolver(request.diameter, request.height, request.grid_resolution)
    fea_solver.generate_mesh()
    fea_solver.assemble_matrix_quantum_partition(num_partitions=request.partitions)
    fea_solver.solve_field()
    fea_plots = fea_solver.render_plots(output_base)
    
    return {
        "design_id": design_id,
        "partition_plot": f"/results/{design_id}_partition.png",
        "solution_plot": f"/results/{design_id}_solution.png",
        "nodes": fea_solver.num_nodes,
        "partitions": request.partitions
    }

if __name__ == "__main__":
    import uvicorn
    # Kill any existing server on 8085
    os.system("lsof -ti:8085 | xargs kill -9 >/dev/null 2>&1")
    uvicorn.run(app, host="0.0.0.0", port=8085)
