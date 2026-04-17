import requests

url = "http://localhost:5002/api/simulate"

combos = [
    ("GRE", "standard"),
    ("GRE", "CardioRamanujanCoil"),
    ("CardioRamanujanPulse", "standard"),
    ("CardioRamanujanPulse", "CardioRamanujanCoil")
]

print("=" * 65)
print(f"{'Pulse Sequence':<25} | {'Coil Array':<25} | {'SNR':<7}")
print("=" * 65)

for seq, coil in combos:
    payload = {
        "sequence": seq, "tr": 1000, "te": 30, "ti": 0, "flip_angle": 60,
        "coils": coil, "num_coils": 8, "noise": 0.05, "recon_method": "SENSE", "resolution": 128
    }
    
    try:
        response = requests.post(url, json=payload, timeout=20)
        data = response.json()
        print(f"{seq:<25} | {coil:<25} | {data['metrics'].get('snr_estimate', 0):5.2f}")
    except Exception as e:
        print(f"{seq:<25} | {coil:<25} | {'FAIL':<7}")
