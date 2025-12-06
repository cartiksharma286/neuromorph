from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from risk_engine import RiskEngine, StatisticalClassifier
from models import StentData, RiskItem, MRPulseSeqData, Device, IEC13485Clause, QMSDocument, IEC60601Clause, VAndVPlan
from standards_data import IEC_13485_CLAUSES, IEC_60601_CLAUSES
import database
from datetime import datetime
import crud
from quantum_optimizer import QuantumPulseOptimizer

app = FastAPI(title="Medical Device Documentation System", version="1.0.0")

@app.on_event("startup")
def startup_event():
    database.init_db()

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

@app.get("/api/documents", response_model=List[QMSDocument])
def get_documents():
    return crud.get_all_documents()

@app.post("/api/documents/create")
def create_document(doc: QMSDocument):
    success = crud.create_document(doc)
    if success:
        return {"status": "success", "doc_id": doc.id, "message": "Document created"}
    else:
        raise HTTPException(status_code=500, detail="Failed to create document")

@app.post("/api/documents/{doc_id}/approve")
def approve_document(doc_id: str, approver: str):
    result = crud.approve_document(doc_id, approver)
    return result

@app.post("/api/documents/{doc_id}/reject")
def reject_document(doc_id: str, reason: str):
    result = crud.reject_document(doc_id, reason)
    return result

class PulseOptRequest(BaseModel):
    target_flip_angle: float
    duration_ms: float = 5.0

@app.post("/api/optimize-pulse")
def optimize_pulse(req: PulseOptRequest):
    optimizer = QuantumPulseOptimizer(
        target_flip_angle=req.target_flip_angle, 
        duration_ms=req.duration_ms
    )
    result = optimizer.run_optimization()
    return result

@app.get("/api/standards/iec60601", response_model=List[IEC60601Clause])
def get_iec60601_clauses():
    return IEC_60601_CLAUSES

@app.post("/api/generate-doc")
def generate_document(device_id: str, doc_type: str):
    # Mock GenAI Response (would be an LLM call in real life)
    content = ""
    title = ""
    doc_id = ""
    
    if doc_type == "verification_plan":
        doc_id = f"VPLAN-{device_id}-{int(datetime.utcnow().timestamp())}"
        title = f"V&V Plan for {device_id}"
        content = """
        VERIFICATION AND VALIDATION PLAN
        
        1. Scope
        This document details the verification strategy for the Quantum Pulse Optimizer.
        
        2. Verification Activities
        - Unit Testing: Verify quantum_optimizer.py cost function convergence.
        - Integration Testing: API endpoints /api/optimize-pulse.
        - System Testing: End-to-end pulse generation and SAR analysis.
        
        3. Validation Activities
        - Usability Testing: Clinician review of the optimization UI.
        - Clinical Evaluation: Comparison of optimized sequences vs standard of care.
        
        4. Acceptance Criteria
        - SAR reduction > 15%
        - Flip angle error < 5%
        """
    else:
        raise HTTPException(status_code=400, detail="Unsupported doc type")

    # Persist to DB
    new_doc = QMSDocument(
        id=doc_id,
        title=title,
        version="0.1",
        author="AI_Generator",
        approval_status="DRAFT",
        last_updated=datetime.utcnow().isoformat(),
        content=content
    )
    
    if crud.create_document(new_doc):
        return {"status": "success", "doc_id": doc_id, "message": "Plan generated and saved."}
    else:
        raise HTTPException(status_code=500, detail="Failed to save generated document")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
