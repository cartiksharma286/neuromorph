from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from risk_engine import RiskEngine, StatisticalClassifier
from models import StentData, RiskItem, MRPulseSeqData, Device, IEC13485Clause, QMSDocument
from standards_data import IEC_13485_CLAUSES

app = FastAPI(title="Medical Device Documentation System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/api/analyze-stent")
def analyze_stent(data: StentData):
    classifier = StatisticalClassifier()
    # Stent Fatigue Analysis (Limit > 380M cycles)
    fatigue_risk = classifier.classify_component_risk(
        data.fatigue_cycles, limit_upper=float('inf'), limit_lower=380e6
    )
    # Radial Force Analysis (Limit > 1.2 N)
    force_risk = classifier.classify_component_risk(
        data.radial_force_n, limit_upper=float('inf'), limit_lower=1.2
    )
    
    return {
        "fatigue_risk_prob": fatigue_risk,
        "force_risk_prob": force_risk,
        "overall_status": "PASS" if fatigue_risk in ["Improbable", "Remote"] else "REVIEW"
    }

@app.post("/api/analyze-mr-pulse")
def analyze_mr_pulse(data: MRPulseSeqData):
    classifier = StatisticalClassifier()
    # SAR Analysis (Limit < 3.2 W/kg usually, but let's say 4.0 for Critical)
    sar_risk = classifier.classify_component_risk(
        data.sar_levels, limit_upper=4.0
    )
    # dB/dt analysis (PNS stimulation risk)
    dbdt_risk = classifier.classify_component_risk(
        data.db_dt, limit_upper=200.0 # T/s
    )
    
    return {
        "sar_risk_prob": sar_risk,
        "dbdt_risk_prob": dbdt_risk,
        "overall_status": "PASS" if sar_risk in ["Improbable", "Remote", "Occasional"] else "REVIEW"
    }

@app.post("/api/calculate-risk")
def calculate_risk(item: RiskItem):
    engine = RiskEngine()
    score = engine.calculate_risk_score(item.severity, item.probability)
    level = engine.get_risk_level(score)
    return {"score": score, "level": level}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/standards/iec13485", response_model=List[IEC13485Clause])
def get_iec13485_clauses():
    return IEC_13485_CLAUSES

@app.post("/api/check-compliance")
def check_compliance(device: Device):
    # Simple mock compliance check
    covered_clauses = set()
    for doc in device.qms_documents:
        covered_clauses.update(doc.related_clauses)
    
    for risk in device.risks:
        covered_clauses.update(risk.related_clauses)
        
    total_clauses = len(IEC_13485_CLAUSES)
    covered_count = len(covered_clauses)
    
    compliance_score = (covered_count / total_clauses) * 100 if total_clauses > 0 else 0.0
    
    return {
        "device_id": device.id,
        "compliance_score": compliance_score,
        "covered_clauses": list(covered_clauses),
        "missing_clauses": [c.id for c in IEC_13485_CLAUSES if c.id not in covered_clauses],
        "status": "COMPLIANT" if compliance_score == 100 else "PARTIAL"
    }

@app.post("/api/documents/create")
def create_document(doc: QMSDocument):
    # In a real app, save to DB. Mocking response.
    return {"status": "success", "doc_id": doc.id, "message": "Document created in DRAFT"}

@app.post("/api/documents/{doc_id}/approve")
def approve_document(doc_id: str, approver: str):
    # Mock approval logic
    return {
        "doc_id": doc_id,
        "status": "APPROVED",
        "approved_by": approver,
        "timestamp": "2025-12-06T10:30:00Z"
    }

@app.post("/api/documents/{doc_id}/reject")
def reject_document(doc_id: str, reason: str):
    return {
        "doc_id": doc_id,
        "status": "DRAFT", # Revert to draft
        "note": f"Rejected: {reason}"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
