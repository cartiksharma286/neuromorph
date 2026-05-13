import re

with open("server.py", "r") as f:
    content = f.read()

import_statement = "from quantum_continued_fractions import QuantumContinuedFractions\n"
if "QuantumContinuedFractions" not in content:
    content = content.replace("from ane_simulation", import_statement + "from ane_simulation")

api_endpoint = """
qcf_model = QuantumContinuedFractions(num_qubits=24)

@app.get("/api/quantum_cf/apply")
def apply_quantum_cf():
    try:
        res = qcf_model.apply_quantum_statistical_distribution()
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

if "/api/quantum_cf/apply" not in content:
    content = content.replace("# Mount static files", api_endpoint + "\n# Mount static files")

with open("server.py", "w") as f:
    f.write(content)
