import requests
import base64
import json

url = 'http://127.0.0.1:5002/api/signal_reconstruction/coil_geometry'
payload = {
    'sequence': 'GRE',
    'coils': ['Head']
}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    if data.get('success'):
        for i, result in enumerate(data['results']):
            img_data = base64.b64decode(result['plot'])
            with open(f'/Users/cartiksharma/.gemini/antigravity/brain/4f513cc6-175b-4fd6-b733-8b4703e472ef/recon_display_{i}.png', 'wb') as f:
                f.write(img_data)
            print(f"Saved recon_display_{i}.png")
    else:
        print("API returned error:", data)
except Exception as e:
    print("Error:", e)
