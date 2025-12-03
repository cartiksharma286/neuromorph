"""
Backend Server for Quantum Catheter Designer

FastAPI server with WebSocket support for real-time optimization updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from pathlib import Path
import uvicorn

from quantum_catheter_optimizer import (
    QuantumCatheterOptimizer,
    PatientConstraints,
    DesignParameters
)
from catheter_geometry import CompleteCatheterGenerator
from fluid_dynamics_sim import CatheterFlowAnalyzer
from model_exporter import ExportManager


app = FastAPI(title="Quantum Catheter Designer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

# Active WebSocket connections
active_connections = set()

# Export manager
exporter = ExportManager(output_directory="./exports")


@app.get("/")
async def root():
    """Serve main application"""
    index_path = web_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Quantum Catheter Designer API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time optimization updates"""
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "optimize":
                # Run optimization in background
                await run_optimization(websocket, message["data"])
                
            elif message["type"] == "export":
                # Export design
                await export_design(websocket, message)
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        active_connections.remove(websocket)


async def run_optimization(websocket: WebSocket, params: dict):
    """
    Run quantum optimization and send updates to client
    """
    try:
        # Parse parameters
        pc = params["patient_constraints"]
        qs = params["quantum_settings"]
        
        patient_constraints = PatientConstraints(
            vessel_diameter=pc["vessel_diameter"],
            vessel_curvature=pc["vessel_curvature"],
            bifurcation_angle=pc.get("bifurcation_angle"),
            required_length=pc["required_length"],
            flow_rate=pc["flow_rate"]
        )
        
        # Create optimizer
        optimizer = QuantumCatheterOptimizer(
            n_qubits=qs["n_qubits"],
            n_layers=qs["n_layers"],
            max_iterations=qs["max_iterations"]
        )
        
        # Send initial status
        await websocket.send_json({
            "type": "optimization_started",
            "message": "Quantum optimization started..."
        })
        
        # Custom optimization loop with progress updates
        design = await optimize_with_progress(
            optimizer,
            patient_constraints,
            websocket
        )
        
        # Generate geometry
        generator = CompleteCatheterGenerator(design, patient_constraints)
        catheter = generator.generate()
        
        # Run flow analysis
        analyzer = CatheterFlowAnalyzer(design, patient_constraints)
        flow_results = analyzer.run_quick_analysis()
        
        # Send completion message
        await websocket.send_json({
            "type": "optimization_complete",
            "design": {
                "outer_diameter": design.outer_diameter,
                "inner_diameter": design.inner_diameter,
                "wall_thickness": design.wall_thickness,
                "tip_angle": design.tip_angle,
                "flexibility_index": design.flexibility_index,
                "side_holes": design.side_holes,
                "material_composition": design.material_composition
            },
            "metrics": {
                "pressure_drop": flow_results.pressure_drop,
                "reynolds_number": flow_results.reynolds_number,
                "average_velocity": flow_results.average_velocity,
                "max_velocity": flow_results.max_velocity,
                "flow_regime": flow_results.flow_regime,
                "flexibility_index": design.flexibility_index
            }
        })
        
    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback
        traceback.print_exc()
        
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })


async def optimize_with_progress(
    optimizer: QuantumCatheterOptimizer,
    constraints: PatientConstraints,
    websocket: WebSocket
) -> DesignParameters:
    """
    Run optimization with progress updates
    """
    # This is a simplified version - actual implementation would
    # modify the optimizer to yield progress
    
    # For now, run in executor to not block
    loop = asyncio.get_event_loop()
    
    # Simulate progress updates
    async def send_progress():
        for i in range(optimizer.max_iterations):
            await asyncio.sleep(0.1)
            
            # Send progress
            if i % 10 == 0:
                await websocket.send_json({
                    "type": "optimization_progress",
                    "iteration": i,
                    "cost": 1.0 - (i / optimizer.max_iterations) * 0.8
                })
    
    # Start progress updates
    progress_task = asyncio.create_task(send_progress())
    
    # Run actual optimization
    design = await loop.run_in_executor(
        None,
        optimizer.optimize,
        constraints
    )
    
    # Cancel progress updates
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    
    return design


async def export_design(websocket: WebSocket, message: dict):
    """Export design to file"""
    try:
        design_data = message.get("design")
        export_format = message.get("format", "stl")
        
        # Convert to DesignParameters
        design = DesignParameters(
            outer_diameter=design_data["outer_diameter"],
            inner_diameter=design_data["inner_diameter"],
            wall_thickness=design_data["wall_thickness"],
            tip_angle=design_data["tip_angle"],
            flexibility_index=design_data["flexibility_index"],
            side_holes=design_data["side_holes"],
            material_composition=design_data["material_composition"]
        )
        
        # Generate mesh
        # This would use actual patient constraints from session
        constraints = PatientConstraints(
            vessel_diameter=5.5,
            vessel_curvature=0.15,
            required_length=1000.0,
            flow_rate=8.0
        )
        
        generator = CompleteCatheterGenerator(design, constraints)
        catheter = generator.generate()
        
        # Export
        exports = exporter.export_complete_package(
            design,
            constraints,
            catheter,
            f"catheter_design_{int(asyncio.get_event_loop().time())}"
        )
        
        await websocket.send_json({
            "type": "export_complete",
            "files": exports
        })
        
    except Exception as e:
        print(f"Export error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })


@app.post("/api/optimize")
async def optimize_api(request: dict):
    """REST API endpoint for optimization (alternative to WebSocket)"""
    try:
        pc = request["patient_constraints"]
        qs = request.get("quantum_settings", {})
        
        patient_constraints = PatientConstraints(
            vessel_diameter=pc["vessel_diameter"],
            vessel_curvature=pc["vessel_curvature"],
            bifurcation_angle=pc.get("bifurcation_angle"),
            required_length=pc["required_length"],
            flow_rate=pc["flow_rate"]
        )
        
        optimizer = QuantumCatheterOptimizer(
            n_qubits=qs.get("n_qubits", 8),
            n_layers=qs.get("n_layers", 3),
            max_iterations=qs.get("max_iterations", 100)
        )
        
        design = optimizer.optimize(patient_constraints)
        
        return JSONResponse({
            "success": True,
            "design": {
                "outer_diameter": design.outer_diameter,
                "inner_diameter": design.inner_diameter,
                "wall_thickness": design.wall_thickness,
                "tip_angle": design.tip_angle,
                "flexibility_index": design.flexibility_index,
                "side_holes": design.side_holes,
                "material_composition": design.material_composition
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Catheter Designer - Backend Server")
    print("=" * 60)
    print("\nStarting server...")
    print("WebSocket: ws://localhost:8765/ws")
    print("Web UI: http://localhost:8765")
    print("API: http://localhost:8765/api")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info"
    )
