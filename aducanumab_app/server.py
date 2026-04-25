from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import kinetics

app = FastAPI(title="Aducanumab Dementia Kinetic Model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimRequest(BaseModel):
    days: int = 180
    start_plaque: float = 100.0
    dose_mg: float = 10.0
    affinity: float = 0.05
    clearance: float = 0.02

@app.post("/api/simulate")
def run_simulation(req: SimRequest):
    result = kinetics.simulate_treatment(
        days=req.days,
        start_plaque=req.start_plaque,
        dose_mg=req.dose_mg,
        affinity=req.affinity,
        clearance=req.clearance
    )
    return {"status": "success", "data": result}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
