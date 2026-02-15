# Software Requirements Specification (SRS) for Medical Device Documentation System

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to outline the functional and non-functional requirements for the Medical Device Documentation System. This system allows users to manage risk, compliance, and design optimization for medical devices.

### 1.2 Scope
The software will provide:
- Risk Management tools compliant with ISO 14971.
- Document Control for Quality Management Systems (IEC 13485).
- Computational tools for standard verification (stents) and optimizations (MRI pulses).

## 2. Functional Requirements

### 2.1 Risk Management (ISO 14971)
- **FR-RISK-001**: The system shall allow users to input severity (1-5) and probability (1-5) for risk hazards.
- **FR-RISK-002**: The system shall calculate a Risk Score and assign a Risk Level (Acceptable, ALARP, Unacceptable).
- **FR-RISK-003**: The system shall visualize a Risk Matrix.
- **FR-RISK-004**: The system shall perform statistical classification of component risks (e.g., fatigue cycles) based on normal distribution variance.

### 2.2 Quality Management System (QMS)
- **FR-DOC-001**: The system shall allow creation, approval, and rejection of QMS documents.
- **FR-DOC-002**: The system shall maintain a version history of documents.
- **FR-DOC-003**: The system shall display IEC 13485 clauses and track compliance status.

### 2.3 Pulse Sequence Optimization
- **FR-OPT-001**: The system shall simulate MRI RF pulse sequences given a target flip angle and duration.
- **FR-OPT-002**: The system shall optimize the RF waveform to minimize SAR (Specific Absorption Rate) while achieving target flip angles.
- **FR-OPT-003**: The system shall visualize the optimized waveform and convergence metrics.

### 2.4 Data Analysis
- **FR-ANA-001**: The system shall analyze stent radial force and fatigue data against safety limits (e.g. 380M cycles).
- **FR-ANA-002**: The system shall determine "PASS" or "REVIEW" status for analyzed components.

## 3. Non-Functional Requirements

### 3.1 Performance
- **NFR-PERF-001**: Pulse optimization simulations shall complete within 5 seconds for standard parameters.
- **NFR-PERF-002**: API response time for CRUD operations shall be less than 200ms.

### 3.2 Security
- **NFR-SEC-001**: The system shall validate all inputs to prevent injection attacks (handled via Pydantic).
- **NFR-SEC-002**: Cross-Origin Resource Sharing (CORS) shall be configured to allow access only from authorized clients (currently "*" for dev).

### 3.3 Reliability
- **NFR-REL-001**: The system shall handle optimization convergence failures gracefully and report error status.

### 3.4 Regulatory
- **NFR-REG-001**: The system software development process shall follow IEC 62304 principles (Class A/B).

## 4. System Interfaces
- **UI**: Web-based Dashboard.
- **API**: REST over HTTP.
- **Database**: SQLite file `med_dev_qms.db`.
