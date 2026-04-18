import requests

url = "http://localhost:5002/api/simulate"
payload = {
    "sequence": "GenAIOrthopedicPulse",
    "tr": 1000,
    "te": 30,
    "ti": 0,
    "flip_angle": 90,
    "coils": "OrthopedicKneeCoil",
    "num_coils": 8,
    "noise": 0.05,
    "recon_method": "SENSE",
    "resolution": 128
}

response = requests.post(url, json=payload)
data = response.json()
if response.status_code == 200 and data.get("success"):
    print("Success! SNR Estimate: ", data['metrics'].get('snr_estimate'))
else:
    print("Failed: ", data)
