from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def generate_docs():
    print("Triggering V&V Plan Generation...")
    response = client.post("/api/generate-doc?device_id=QuantumPulseOpt&doc_type=verification_plan")
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['message']} (ID: {data['doc_id']})")
    else:
        print(f"Failed: {response.text}")

if __name__ == "__main__":
    generate_docs()
