import requests
import sys
import time

BASE_URL = "http://localhost:5000"

def check_endpoint(method, endpoint, payload=None, description="", validator=None):
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == 'GET':
            response = requests.get(url)
        else:
            response = requests.post(url, json=payload)
            
        print(f"[*] Checking {description} ({endpoint}) ... ", end="")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                if validator:
                    if validator(data.get('data')):
                        print("OK")
                        return True
                    else:
                        print("FAIL (Validation Error)")
                        return False
                print("OK")
                return True
            else:
                print(f"FAIL (Status: {data.get('status')})")
                return False
        else:
            print(f"FAIL (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

print("=============================================")
print("   NVQLink Orthopedic System Verification")
print("=============================================")

# Allow server to startup
time.sleep(2)

failures = 0

# 1. Health Economics
if not check_endpoint('GET', '/api/economics?type=knee', description="Economics (Knee)"): failures += 1
if not check_endpoint('GET', '/api/economics?type=hip', description="Economics (Hip)"): failures += 1
if not check_endpoint('GET', '/api/economics/cost?size=1.2&complexity=genai', description="Implant Cost Cost"): failures += 1

# 2. Geometry
if not check_endpoint('GET', '/api/geometry/nvqlink', description="Geometry (Knee NVQLink)"): failures += 1
if not check_endpoint('GET', '/api/geometry/genai', description="Geometry (Knee GenAI)"): failures += 1
if not check_endpoint('GET', '/api/geometry/hip/nvqlink', description="Geometry (Hip NVQLink)"): failures += 1
if not check_endpoint('GET', '/api/geometry/hip/genai', description="Geometry (Hip GenAI)"): failures += 1

# 3. Workflow
if not check_endpoint('POST', '/api/workflow/resection', description="Workflow (Resection)"): failures += 1
if not check_endpoint('POST', '/api/workflow/balancing', description="Workflow (Balancing)"): failures += 1

def validate_postop(data):
    try:
        required_keys = ['rom', 'laxity', 'recovery', 'outcomes']
        for k in required_keys:
            if k not in data:
                print(f"[Missing Key: {k}]", end=" ")
                return False
        
        # specific check
        if 'koos_score' not in data['outcomes']:
             print("[Missing KOOS]", end=" ")
             return False
             
        return True
    except:
        return False

if not check_endpoint('GET', '/api/workflow/postop', description="Workflow (Post-Op Data)", validator=validate_postop): failures += 1

print("=============================================")
if failures == 0:
    print("ALL SYSTEMS OPERATIONAL. Verification PASSED.")
    sys.exit(0)
else:
    print(f"{failures} MODULES FAILED. Verification FAILED.")
    sys.exit(1)
