import os
from datetime import datetime
from typing import List, Dict

class DocumentGenerator:
    def __init__(self, output_dir: str = "generated_docs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, filename: str, content: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def generate_sds(self, project_name: str) -> str:
        content = f"""# Software Design Specification
**Project**: {project_name}
**Date**: {datetime.now().strftime("%Y-%m-%d")}

## 1. Architecture
The system uses a Microservices-ready architecture with FastAPI backend and Vanilla JS frontend.

## 2. Components
- **RiskEngine**: Calculates risk scores.
- **QuantumOptimizer**: Simulates pulse sequences.
"""
        return self._save("Software_Design_Specification.md", content)

    def generate_srs(self, requirements: List[Dict[str, str]]) -> str:
        req_rows = "\n".join([f"| {r['id']} | {r['desc']} |" for r in requirements])
        content = f"""# Software Requirements Specification
**Date**: {datetime.now().strftime("%Y-%m-%d")}

## Functional Requirements
| ID | Description |
|----|-------------|
{req_rows}
"""
        return self._save("Software_Requirements_Specification.md", content)

    def generate_risk_file(self, risks: List[Dict[str, str]]) -> str:
        rows = "\n".join([f"| {r['hazard']} | {r['severity']} | {r['probability']} | {int(r['severity'])*int(r['probability'])} |" for r in risks])
        content = f"""# Risk Management File (ISO 14971)

## Hazard Analysis
| Hazard | Severity | Probability | Risk Score |
|--------|----------|-------------|------------|
{rows}
"""
        return self._save("Risk_Management_File.md", content)

    def generate_all(self):
        # Mock data for demonstration
        paths = []
        paths.append(self.generate_sds("NeuroMorph MedSystem"))
        
        reqs = [
            {"id": "FR-01", "desc": "Calculate Risk Score"},
            {"id": "FR-02", "desc": "Optimize Pulse Sequence"}
        ]
        paths.append(self.generate_srs(reqs))
        
        risks = [
            {"hazard": "Overheating", "severity": "4", "probability": "2"},
            {"hazard": "Data Corruption", "severity": "3", "probability": "2"}
        ]
        paths.append(self.generate_risk_file(risks))
        
        return paths
