# Traceability Matrix

**Project**: Medical Device Documentation System

| Requirement ID | Requirement Description | Design Component (SDS) | Verification Test (V&V) | Status |
|----------------|-------------------------|------------------------|-------------------------|--------|
| **FR-RISK-001** | Input Severity/Probability | `models.RiskItem`, `frontend/RiskView` | TC-001 | Implemented |
| **FR-RISK-002** | Calculate Risk Score | `risk_engine.RiskEngine` | TC-001 | Implemented |
| **FR-RISK-003** | Visualize Risk Matrix | `frontend/app.js:initRiskMatrix` | VAL-01 | Implemented |
| **FR-RISK-004** | Statistical Risk Classification | `risk_engine.StatisticalClassifier` | TC-006 | Implemented |
| **FR-DOC-001** | Create/Approve/Reject Docs | `backend/crud.py`, `backend/main.py` | TC-002, TC-003 | Implemented |
| **FR-DOC-002** | Document Version History | `database.document_versions` | TC-002 | Implemented |
| **FR-DOC-003** | IEC 13485 Compliance | `models.IEC13485Clause`, `frontend/standards` | VAL-01 | Implemented |
| **FR-OPT-001** | Simulate MRI Pulse | `quantum_optimizer.QuantumPulseOptimizer` | TC-004 | Implemented |
| **FR-OPT-002** | Optimize for SAR/Flip Angle | `quantum_optimizer.cost_function` | TC-005 | Implemented |
| **FR-OPT-003** | Visualize Waveform | `frontend/app.js:Chart.js` | VAL-02 | Implemented |
| **FR-ANA-001** | Analyze Stent Fatigue | `main.analyze_stent` | TC-006 | Implemented |
| **NFR-PERF-001**| Optimization < 5s | `quantum_optimizer.run_optimization` | TC-PERF-01 | Pending Verification |
| **NFR-SEC-001** | Input Validation | `Pydantic Models` | TC-SEC-01 | Implemented |

## Coverage Analysis
- **Requirements Covered**: 100%
- **Design Traced**: 100%
- **Tests Defined**: 100%
