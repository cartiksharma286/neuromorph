from pydantic import BaseModel
from typing import List, Optional

class IEC13485Clause(BaseModel):
    id: str
    clause_number: str
    title: str
    description: str
    parent_id: Optional[str] = None
    sub_clauses: List['IEC13485Clause'] = []

class IEC60601Clause(BaseModel):
    id: str
    clause_number: str
    title: str
    description: str
    category: str # e.g., "Electrical Safety", "Mechanical Safety"

class VAndVPlan(BaseModel):
    id: str
    device_id: str
    generated_content: str
    verification_steps: List[str]
    validation_metrics: List[str]
    created_at: str
    status: str = "DRAFT"

class RiskItem(BaseModel):
    id: str
    hazard: str
    cause: str
    harm: str
    severity: str
    probability: str
    risk_score: Optional[int] = None
    risk_level: Optional[str] = None
    mitigation: Optional[str] = None
    related_clauses: List[str] = [] # List of Clause IDs

class DocumentVersion(BaseModel):
    version: str
    change_description: str
    changed_by: str
    timestamp: str

class QMSDocument(BaseModel):
    id: str
    title: str
    version: str
    author: str
    approval_status: str  # DRAFT, REVIEW, APPROVED
    last_updated: str
    related_clauses: List[str] = [] # List of Clause IDs
    content: str = ""
    version_history: List[DocumentVersion] = []
    approved_by: Optional[str] = None
    approval_date: Optional[str] = None

class Device(BaseModel):
    id: str
    name: str
    type: str  # Class I, II, III
    description: str
    risks: List[RiskItem] = []
    qms_documents: List[QMSDocument] = []
    compliance_standard: str = "IEC 13485:2016"
    compliance_report: Optional[str] = None

class StentData(BaseModel):
    radial_force_n: List[float]
    fatigue_cycles: List[float]
    recoil_percent: List[float]

class MRPulseSeqData(BaseModel):
    sar_levels: List[float] # W/kg
    db_dt: List[float] # T/s
