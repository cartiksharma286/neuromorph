from crud import create_document
from models import QMSDocument
from datetime import datetime

def populate_fda_docs():
    print("Generating FDA Documentation...")
    
    # 1. 510(k) Summary
    k_summary = QMSDocument(
        id="K250001",
        title="510(k) Summary - Q-Pulse Optimizer",
        version="1.0",
        author="Regulatory Affairs",
        approval_status="DRAFT",
        last_updated=datetime.utcnow().isoformat(),
        content="""
        510(k) SUMMARY
        
        Submitter: NeuroMorph Medical Inc.
        Device Name: Q-Pulse Optimizer
        Common Name: MRI Pulse Sequence Optimization Software
        Classification: Class II
        Regulation: 21 CFR 892.1000
        
        Predicate Device: Standard Gradient Optimizer (K200000)
        
        Device Description:
        The Q-Pulse Optimizer is a software module that utilizes Variational Quantum Eigensolver (VQE) algorithms to calculate optimal gradient waveforms for MRI pulse sequences. It allows for the reduction of Specific Absorption Rate (SAR) while maintaining image contrast.
        
        Substantial Equivalence:
        The subject device has the same intended use as the predicate. Technological characteristics differ in the method of optimization (Quantum ML vs Classical Convex Optimization), but performance data demonstrates that the subject device is as safe and effective as the predicate.
        
        Non-Clinical Testing:
        - IEC 62304 Software Verification
        - Algorithm accuracy validation against phantom data (NIST traceable).
        - SAR mitigation efficiency comparison (Subject vs Predicate).
        
        Conclusion:
        The Q-Pulse Optimizer is substantially equivalent to the predicate device.
        """
    )
    
    # 2. Software Requirements Specification (SRS)
    srs_doc = QMSDocument(
        id="SW-REQ-001",
        title="SRS - Quantum Pulse Engine",
        version="1.0",
        author="Systems Engineering",
        approval_status="DRAFT",
        last_updated=datetime.utcnow().isoformat(),
        content="""
        SOFTWARE REQUIREMENTS SPECIFICATION
        Project: Q-Pulse Optimizer
        Standard: IEC 62304
        
        1. FUNCTIONAL REQUIREMENTS
        
        REQ-01: SAR Minimization
        The system shall use VQE (Variational Quantum Eigensolver) to minimize Global SAR for a given k-space trajectory.
        
        REQ-02: Constraint Handling
        The system shall strictly enforce hardware constraints:
        - Max Gradient Amplitude: < 80 mT/m
        - Max Slew Rate: < 200 T/m/s
        
        REQ-03: Quantum Noise Mitigation
        The system shall apply Quantum Error Correction (QEC) or a specific noise filtering layer ('Sanity Check') before determining final waveform parameters.
        
        REQ-04: User override
        The clinician shall be able to override the AI-suggested parameter if it exceeds physiological comfort thresholds (PNS).
        
        2. INTERFACE REQUIREMENTS
        
        REQ-05: DICOM Export
        The system shall export optimized sequences in DICOM MR object format.
        """
    )
    
    # 3. Risk Management File
    risk_doc = QMSDocument(
        id="RISK-QML-01",
        title="Risk Analysis - Quantum Hallucinations",
        version="1.0",
        author="Risk Manager",
        approval_status="DRAFT",
        last_updated=datetime.utcnow().isoformat(),
        content="""
        RISK MANAGEMENT FILE
        Standard: ISO 14971:2019
        Device: Q-Pulse Optimizer
        
        Hazard Analysis #1: Quantum Optimization Hallucination
        
        Hazard: Quantum state decoherence leads to mathematically valid but physiologically dangerous gradient waveform.
        Cause: QPU thermal noise or gate error.
        Harm: Patient nerve stimulation (PNS) or tissue heating (Burns).
        
        Initial Risk:
        - Severity: Critical (4)
        - Probability: Occasional (3)
        - Risk Score: 12 (ALARP)
        
        Mitigation Measure (Software):
        - Control: Classical 'Sanity Check' Layer. A deterministic classical algorithm verifies the output waveform against IEC 60601-2-33 limits before release.
        - Safety Factor: 2.0x buffer on nerve stimulation thresholds.
        
        Residual Risk:
        - Severity: Critical (4)
        - Probability: Remote (2)
        - Risk Score: 8 (Acceptable with justification)
        """
    )

    docs = [k_summary, srs_doc, risk_doc]
    
    for doc in docs:
        if create_document(doc):
            print(f"Success: Created {doc.id}")
        else:
            print(f"Error: Failed to create {doc.id}")

if __name__ == "__main__":
    populate_fda_docs()
