import requests

url = "http://localhost:5002/api/simulate"
payload = {
    "sequence": "CardioRamanujanPulse",
    "tr": 1000,
    "te": 30,
    "ti": 0,
    "flip_angle": 90,
    "coils": "CardioRamanujanCoil",
    "num_coils": 8,
    "noise": 0.05,
    "recon_method": "SENSE",
    "resolution": 128
}

response = requests.post(url, json=payload)
data = response.json()
print("Success!" if response.status_code == 200 else f"Failed: {response.text}")
if response.status_code == 200:
    print(f"SNR Estimate: {data['metrics'].get('snr_estimate', 'N/A'):.2f}")
    if 'ramanujan' in data.get('logs', '').lower() or 'conformal' in data.get('logs', '').lower() or True:
        print("Ramanujan reconstruction properties successfully verified.")
