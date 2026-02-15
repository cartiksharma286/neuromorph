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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stent_generator import StentGenerator
from recommendation_engine import StentRecommender
from visualizer import StentVisualizer

app = FastAPI(title="Parametric Stent Designer")


# Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "stent_results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

class DesignRequest(BaseModel):
    diameter: float = 3.0
    length: float = 20.0
    thickness: float = 0.09

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.post("/design")
def design_stent(request: DesignRequest):
    design_id = str(uuid.uuid4())
    output_base = os.path.join(OUTPUT_DIR, design_id)
    
    # 1. Analyze & Optimize (Continued Fraction Logic)
    recommender = StentRecommender(request.diameter, request.thickness)
    recommender.analyze()
    report = recommender.get_report()
    
    opt_data = report.get("optimization_data", {})
    crowns = opt_data.get("Optimal_Crowns", 6)
    
    # 2. Generate
    generator = StentGenerator(request.length, request.diameter, request.thickness)
    generator.generate_sine_rings(crowns=crowns)
    generator.add_connectors()
    
    json_path = f"{output_base}.json"
    generator.export_to_json(json_path)
    
    # 3. Visualize
    vis_path = f"{output_base}.png"
    visualizer = StentVisualizer(generator)
    visualizer.plot_unrolled_pattern(vis_path)
    
    # 4. Return Data
    response = {
        "design_id": design_id,
        "parameters": {
            "diameter": request.diameter,
            "length": request.length,
            "thickness": request.thickness,
            "crowns": crowns,
            "reynolds": opt_data.get("Re", 0),
            "target_prime": opt_data.get("Target_Prime", 0),
            "congruence": opt_data.get("Congruence", 0)
        },
        "report": report,
        "files": {
            "image": f"/results/{design_id}.png",
            "json": f"/results/{design_id}.json"
        }
    }
    return response

if __name__ == "__main__":
    import uvicorn
    # Kill any existing server on 8000
    os.system("lsof -ti:8000 | xargs kill -9 >/dev/null 2>&1")
    uvicorn.run(app, host="0.0.0.0", port=8000)
