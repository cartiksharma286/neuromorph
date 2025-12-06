# Risk Management File
**Standard**: ISO 14971:2019
**Project**: Medical Device Documentation System

## 1. Risk Management Plan

### 1.1 Scope
This document covers the risk management activities for the Medical Device Documentation System software.

### 1.2 Risk Acceptance Criteria
- **Acceptable (Green)**: Severity * Probability <= 4. No further action needed.
- **ALARP (Yellow)**: 4 < Score <= 12. Risk control measures required.
- **Unacceptable (Red)**: Score > 12. Mandatory design change or protective measure.

## 2. Hazard Analysis

| ID | Hazard | Foreseeable Sequence of Events | Hazardous Situation | Harm | Severity (S) | Probability (P) | Risk Score (SxP) | Risk Level |
|----|--------|--------------------------------|---------------------|------|--------------|-----------------|------------------|------------|
| HZ-01 | Incorrect Risk Calculation | 1. User enters probability 4.<br>2. User enters severity 5.<br>3. Software bugs calculate score as 2. | User believes High Risk is Low Risk. | Device failure in field causing injury due to unmitigated risk. | 5 (Catastrophic) | 2 (Remote) | 10 | **ALARP** |
| HZ-02 | MRI Pulse SAR Overdose | 1. Optimizer fails to converge.<br>2. Output waveform has high amplitude.<br>3. System displays "Optimal" without checking boundaries. | Patient exposed to high RF energy (SAR). | Tissue heating / Burns. | 4 (Critical) | 3 (Occasional) | 12 | **ALARP** |
| HZ-03 | Stent Fatigue False Pass | 1. Statistical classifier assumes wrong distribution.<br>2. Stent with 99% fail probability marked "PASS". | Stent implated. | Stent fracture, restenosis, heart attack. | 5 (Catastrophic) | 1 (Improbable) | 5 | **ALARP** |
| HZ-04 | Data Loss | 1. Database corruption.<br>2. Backup fails. | Quality records lost. | Regulatory non-compliance, potential recall. | 3 (Serious) | 2 (Remote) | 6 | **ALARP** |

## 3. Risk Control Measures

| Risk ID | Control Measure | Control Type | Verification | New S | New P | Res. Score |
|---------|-----------------|--------------|--------------|-------|-------|------------|
| HZ-01 | **Unit Testing**: Implement automated tests for `RiskEngine` calculation logic.<br>**Code Review**: Verify math operators. | Software Verification | Test Report TR-01 | 5 | 1 | 5 (ALARP) |
| HZ-02 | **Safety Bounds**: Hard-coded check in `quantum_optimizer.py` to reject any wave > 4.0 W/kg.<br>**UI Warning**: Display Red Alert if SAR > Limit. | Design (Safe Fail state) | Test Report TR-02 | 4 | 1 | 4 (Acceptable) |
| HZ-03 | **Validation**: Compare `StatisticalClassifier` against known datasets.<br>**Margins**: Use safety factor of 1.5x on limits. | Software Validation | Val Report VR-01 | 5 | 1 | 5 (ALARP) |
| HZ-04 | **Database Integrity**: enable WAL mode in SQLite.<br>**Backup**: Auto-export to JSON. | Inherently Safe Design | Test Report TR-03 | 3 | 1 | 3 (Acceptable) |

## 4. Benefit-Risk Analysis
The benefits of using the System (automated, accurate compliance checking, advanced optimization reducing trial-and-error) outweigh the residual risks, which are all controlled to ALARP or Acceptable levels. The software does not directly control hardware without human review.

## 5. Conclusion
Risk management activities are complete. Residual risks are acceptable.
