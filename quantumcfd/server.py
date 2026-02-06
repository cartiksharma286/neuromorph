from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from solver import QuantumHyperFluidSolver
from geometry_loader import generate_naca_airfoil

app = FastAPI(title="Quantum CFD Turbine Server")

OUTPUT_DIR = "simulation_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount simulation results so they are accessible via URL
app.mount("/simulation_results", StaticFiles(directory=OUTPUT_DIR), name="simulation_results")

class SimulationConfig(BaseModel):
    naca_code: str = "0012" # e.g. 0012, 2412
    angle_of_attack: float = 0.0
    steps: int = 10
    reynolds: float = 1000.0
    grid_size: int = 32 # assumes cubic nx=ny=nz
    forcing: bool = False
    forcing_intensity: float = 0.0


# Global status tracker
SIM_STATUS = {}

def run_simulation_task(sim_id: str, config: SimulationConfig):
    sim_dir = os.path.join(OUTPUT_DIR, sim_id)
    os.makedirs(sim_dir, exist_ok=True)
    
    SIM_STATUS[sim_id] = {"status": "running", "progress": 0, "current_step": 0, "total_steps": config.steps}
    
    print(f"[{sim_id}] Starting Turbine Simulation (NACA {config.naca_code})")
    
    # 1. Generate Geometry
    nw, nz, ny, nx = 3, config.grid_size, config.grid_size, config.grid_size
    shape = (nw, nz, ny, nx)
    
    print(f"[{sim_id}] Generating Airfoil...")
    obstacle_mask = generate_naca_airfoil(
        shape, 
        code=config.naca_code, 
        angle_of_attack=config.angle_of_attack,
        chord_length=0.4
    )
    
    # 2. Initialize Solver
    lid_vel = 1.0
    nu = (lid_vel * 1.0) / config.reynolds
    
    solver = QuantumHyperFluidSolver(
        nx=nx, ny=ny, nz=nz, nw=nw,
        nu=nu,
        lid_velocity=lid_vel,
        obstacles=obstacle_mask, 
        forcing_intensity=config.forcing_intensity if config.forcing else 0.0
    )
    
    # 3. Time Loop
    lift_history = []
    drag_history = []
    frames = []
    
    import visualize 
    
    print(f"[{sim_id}] Running {config.steps} steps...")
    for n in range(config.steps):
        solver.step(n)
        
        # Compute Forces
        forces = solver.compute_quantum_surface_integral()
        lift_history.append(forces['lift'])
        drag_history.append(forces['drag'])
        
        # Real-time updates
        SIM_STATUS[sim_id]["progress"] = int((n / config.steps) * 100)
        SIM_STATUS[sim_id]["current_step"] = n
        
        if n % 2 == 0:
            # Capture frame data
            mid_w = nw // 2
            mid_z = nz // 2
            u_slice = solver.u[mid_w, mid_z, :, :]
            v_slice = solver.v[mid_w, mid_z, :, :]
            mag = np.sqrt(u_slice**2 + v_slice**2)
            frames.append(mag.copy())
            
            # Save latest frame for real-time viewing
            # We use p for pressure field visualization in plot_frame, let's grab it
            p_slice = solver.p[mid_w, mid_z, :, :]
            
            # Save "latest.png"
            visualize.plot_frame(n, u_slice, v_slice, p_slice, sim_dir, save_static=False, filename="latest.png")
            
    # 4. Save Results
    plt.figure()
    plt.plot(lift_history, label='Lift')
    plt.plot(drag_history, label='Drag')
    plt.legend()
    plt.title(f"Aerodynamic Forces (NACA {config.naca_code}, Re={config.reynolds})")
    plt.savefig(os.path.join(sim_dir, "forces.png"))
    plt.close()
    
    # Animation
    if len(frames) > 0:
        all_frames = np.array(frames)
        v_min, v_max = np.min(all_frames), np.max(all_frames)
        if v_max == v_min: v_max = v_min + 1e-5

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(frames[0], animated=True, origin='lower', cmap='plasma', vmin=v_min, vmax=v_max)
        ax.set_title(f"Velocity Field (NACA {config.naca_code})")
        fig.colorbar(im, ax=ax, label="Magnitude")
        
        def update(frame):
            im.set_array(frame)
            return [im]
            
        anim = FuncAnimation(fig, update, frames=frames, blit=True)
        anim.save(os.path.join(sim_dir, "flow.gif"), writer='pillow', fps=15)
        plt.close()
        
    SIM_STATUS[sim_id]["status"] = "complete"
    SIM_STATUS[sim_id]["progress"] = 100
    print(f"[{sim_id}] Simulation Complete.")

@app.get("/status/{sim_id}")
async def get_status(sim_id: str):
    if sim_id in SIM_STATUS:
        return SIM_STATUS[sim_id]
    return {"status": "unknown"}


@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/simulate")
async def start_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    sim_id = str(uuid.uuid4())
    background_tasks.add_task(run_simulation_task, sim_id, config)
    return {"simulation_id": sim_id, "status": "queued"}

@app.get("/results/{sim_id}")
async def get_results(sim_id: str):
    sim_dir = os.path.join(OUTPUT_DIR, sim_id)
    if not os.path.exists(sim_dir):
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    files = os.listdir(sim_dir)
    return {"simulation_id": sim_id, "files": files, "base_path": os.path.abspath(sim_dir)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
