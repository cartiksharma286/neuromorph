import requests
import json
import time

endpoints = [
    ("/api/register-cortical-surface", "POST"),
    ("/api/register-cortical-surface-cf", "POST"),
    ("/api/cortical-surface-legendre-sh", "GET"),
    ("/api/cortical-surface-volume", "GET"),
    ("/api/register-cortical-surface-qml", "POST"),
    ("/api/geodesic-superposition", "POST"),
    ("/api/register-cortical-surface-qlora", "POST"),
    ("/api/register-cortical-surface-feynman", "POST")
]

def run_triggers():
    for endpoint, method in endpoints:
        url = f"http://127.0.0.1:5055{endpoint}"
        print(f"Triggering {method} {url}...")
        try:
            start_time = time.time()
            if method == "POST":
                res = requests.post(url, json={})
            else:
                res = requests.get(url)
            elapsed = time.time() - start_time
            print(f"Status: {res.status_code}, Time: {elapsed:.2f}s")
            if res.status_code != 200:
                print(f"Error Response: {res.text[:300]}")
        except Exception as e:
            print(f"Exception triggering {url}: {e}")

if __name__ == "__main__":
    run_triggers()
