import requests, json, sys

def test_simulate():
    url = 'http://127.0.0.1:5050/api/simulate'
    payload = {
        "resolution": 128,
        "sequence": "SE",
        "tr": 2000,
        "te": 100,
        "ti": 500,
        "flip_angle": 30,
        "coils": "standard",
        "num_coils": 8,
        "noise": 0.01,
        "recon_method": "SoS",
        "shimming": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        print('Status:', resp.status_code)
        data = resp.json()
        print('Success:', data.get('success'))
        if data.get('success'):
            print('Metrics keys:', list(data.get('metrics', {}).keys()))
        else:
            print('Error:', data.get('error'))
    except Exception as e:
        print('Request failed:', e)
        sys.exit(1)

if __name__ == '__main__':
    test_simulate()
