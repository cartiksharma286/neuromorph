
import requests
import json

def test_genai():
    url = "http://127.0.0.1:8000/api/dementia/treat"
    payload = {
        "treatment_type": "generative_ai",
        "intensity": 0.8
    }
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_genai()
