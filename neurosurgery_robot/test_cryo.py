import requests
import time
import json

BASE_URL = "http://127.0.0.1:5000"

def set_cryo(enabled):
    print(f"[*] Setting Cryo Enabled: {enabled}")
    try:
        requests.post(f"{BASE_URL}/api/control", json={"cryo": enabled})
    except Exception as e:
        print(f"Error: {e}")

def monitor_temp(duration_sec):
    print(f"[*] Monitoring temperature for {duration_sec} seconds...")
    for _ in range(duration_sec * 2): # 2Hz poll
        try:
            res = requests.get(f"{BASE_URL}/api/telemetry")
            data = res.json()
            cryo_map = data.get('cryo_map')
            
            if cryo_map:
                # Find min temp in the grid
                # Flatten the 2D list
                flat = [item for sublist in cryo_map for item in sublist]
                min_t = min(flat)
                print(f"   -> Min Temp: {min_t:.2f} C")
        except Exception as e:
            print(f"Telemtry Error: {e}")
        time.sleep(0.5)

if __name__ == "__main__":
    print("=== Cryo-Ablation Test Script ===")
    
    # 1. Start Cryo
    set_cryo(True)
    
    # 2. Watch it freeze
    monitor_temp(5)
    
    # 3. Stop Cryo
    set_cryo(False)
    
    print("=== Test Complete ===")
