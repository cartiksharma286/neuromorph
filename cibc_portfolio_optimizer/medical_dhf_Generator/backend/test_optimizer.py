from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_optimize_pulse():
    payload = {
        "target_flip_angle": 1.57, # Pi/2
        "duration_ms": 5.0
    }
    response = client.post("/api/optimize-pulse", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "optimized_waveform" in data
    assert "final_flip_angle" in data
    assert "final_sar" in data
    
    # Check if flip angle is reasonably close (heuristic check, simulator is simple)
    # The initial guess is non-optimized, so result should be better or at least valid
    print(f"Goal: 1.57, Actual: {data['final_flip_angle']}")
    assert abs(data["final_flip_angle"] - 1.57) < 0.5 # Allow some slop for mock optimizer

if __name__ == "__main__":
    test_optimize_pulse()
    print("Optimization API Test Passed")
