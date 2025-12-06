# Verification and Validation Plan

## 1. Overview
This document outlines the testing strategy to ensure the Medical Device Documentation System meets its requirements (Verification) and user needs (Validation).

## 2. Verification Protocol (Software Testing)

### 2.1 Test Environment
- **OS**: Windows / Linux
- **Python**: 3.9+
- **Database**: SQLite (Test Instance)

### 2.2 Functional Test Cases (Trace to SRS)

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TC-001 | FR-RISK-002 | Input Severity=5, Prob=5 into Risk Engine. | Score=25, Level="Unacceptable". |
| TC-002 | FR-DOC-001 | Create a new QMS Document via API. | API returns 200 OK, Doc ID returned. |
| TC-003 | FR-DOC-001 | Approve QMS Document as "Admin". | Status updates to "APPROVED". |
| TC-004 | FR-OPT-002 | Run Pulse Optimizer with Target=1.57rad. | Final Angle within 5% of 1.57rad. |
| TC-005 | FR-OPT-002 | Run Pulse Optimizer forcing high SAR scenario. | Optimizer penalizes SAR; final SAR < 4.0. |
| TC-006 | FR-ANA-001 | Analyze Stent with >400M cycles. | Result = "PASS" (Risk="Improbable"). |

### 2.3 Non-Functional Tests

| Test ID | Requirement | Test Description | Expected Result |
|---------|-------------|------------------|-----------------|
| TC-PERF-01 | NFR-PERF-001 | Measure time for `optimize_pulse` API call. | Time < 5.0 seconds. |
| TC-SEC-01 | NFR-SEC-001 | Send Malformed JSON to API. | 422 Validation Error (No Crash). |

## 3. Validation Protocol (User Acceptance)

### 3.1 Validation Scenarios (User Needs)

| Val ID | User Need | Scenario | Acceptance Criteria |
|--------|-----------|----------|---------------------|
| VAL-01 | Ease of Use | Regulatory Affairs Manager uses Dashboard to check compliance. | User finds "IEC 13485" view in < 3 clicks. |
| VAL-02 | Accuracy | Physicist compares "Pulse Optimization" result with MATLAB simulation. | Waveforms match within 95% correlation. |
| VAL-03 | Reporting | Quality Engineer exports Risk File. | Report contains all entered risks correctly formatted. |

## 4. Test Results Summary
*(To be filled after execution)*

- **Verification Run Date**: 2025-12-06
- **Status**: PENDING
