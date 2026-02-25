from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from pulse_generator import PulseGenerator
from quantum_integrals import QuantumIntegrals
from nvqlink_verifier import NVQLinkVerifier
from geodesics import PrimeGeodesicSolver
from reconstruction import ReconstructionSimulator

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class GenerateParams(BaseModel):
    sequence_type: str
    te: float
    tr: float
    flip_angle: float
    fov: float
    matrix_size: int
    optimize: bool = False

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/api/generate")
async def generate_sequence(params: GenerateParams):
    pg = PulseGenerator()
    if params.sequence_type == 'GRE':
        data = pg.generate_gre(params.te, params.tr, params.flip_angle, params.fov, params.matrix_size, optimize=params.optimize)
    elif params.sequence_type == 'SE':
        data = pg.generate_se(params.te, params.tr, params.fov, params.matrix_size, optimize=params.optimize)
    else:
        # Default fallback
        data = pg.generate_gre(params.te, params.tr, params.flip_angle, params.fov, params.matrix_size, optimize=params.optimize)
        
    return data

@app.post("/api/analyze")
async def analyze_integrals(data: dict = Body(...)):
    qi = QuantumIntegrals()
    results = qi.calculate_surface_integral(data)
    return results

@app.post("/api/analyze_geodesics")
async def analyze_geodesics(data: dict = Body(...)):
    solver = PrimeGeodesicSolver()
    results = solver.calculate_geodesics(data)
    return results

@app.post("/api/export_seq")
async def export_seq(data: dict = Body(...)):
    pg = PulseGenerator()
    seq_content = pg.export_to_seq(data)
    return {"content": seq_content, "filename": "quantum_pulse.seq"}

@app.post("/api/reconstruct")
async def reconstruct_image(data: dict = Body(...)):
    recon = ReconstructionSimulator()
    results = recon.reconstruct(data)
    return results

@app.post("/api/verify")
async def verify_nvqlink(payload: dict = Body(...)):
    pulse_data = payload.get('pulse_data')
    params = payload.get('params')
    verifier = NVQLinkVerifier()
    results = verifier.verify(pulse_data, params)
    return results

if __name__ == '__main__':
    import socket
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    host_ip = get_ip()
    print(f"\nServer running! Access from other devices at: http://{host_ip}:8001\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
