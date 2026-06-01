import requests
import json

def test_registration(endpoint, payload=None):
    url = f"http://127.0.0.1:5055{endpoint}"
    response = requests.post(url, json=payload) if payload else requests.post(url)
    try:
        data = response.json()
    except Exception:
        data = response.text
    with open(f"test_result_{endpoint.strip('/').replace('/', '_')}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Tested {endpoint}, status: {response.status_code}, result written to disk.")
    return data

if __name__ == "__main__":
    # Test GMM registration endpoint
    test_registration("/api/register-cortical-surface")
    # Test Continued Fractions registration endpoint
    test_registration("/api/register-cortical-surface-cf")
    # Test Quantum ML registration endpoint
    test_registration("/api/register-cortical-surface-qml")
