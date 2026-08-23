import sys
import os
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

def test_peter_street_basin_endpoints():
    client = app.test_client()
    
    print("Testing GET /peter-street-basin page...")
    res_page = client.get('/peter-street-basin')
    assert res_page.status_code == 200
    assert b"PETER STREET BASIN" in res_page.data
    assert b"STEVE MANN" in res_page.data
    print("✓ /peter-street-basin HTML page rendered successfully!")

    print("Testing GET /api/peter-street-basin-ntu-predict...")
    res_api = client.get('/api/peter-street-basin-ntu-predict?rainfall_mm_hr=35.0&flow_rate_m3_s=8.0&sediment_mg_l=200&baffle_eff_pct=80&sensor_nodes=32')
    assert res_api.status_code == 200
    data = json.loads(res_api.data)
    assert 'predictions' in data
    assert 'predicted_ntu' in data['predictions']
    assert 'water_clarity_pct' in data['predictions']
    assert data['predictions']['predicted_ntu'] > 0
    print(f"✓ NTU Prediction API Success! Predicted NTU: {data['predictions']['predicted_ntu']}, Clarity: {data['predictions']['water_clarity_pct']}%")

if __name__ == '__main__':
    test_peter_street_basin_endpoints()
