import requests
import json

try:
    r = requests.get('http://localhost:5000/api/workflow/postop')
    data = r.json()
    print("Keys:", data['data'].keys())
    print("Outcomes:", data['data']['outcomes'])
except Exception as e:
    print(e)
