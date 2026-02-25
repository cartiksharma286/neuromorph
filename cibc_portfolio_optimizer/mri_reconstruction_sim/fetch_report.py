import urllib.request
import json
import shutil

def run():
    # 1. Trigger Simulation
    url_sim = "http://127.0.0.1:5050/api/simulate"
    payload = {
        "sequence": "SE",
        "coils": "quantum_vascular",
        "nvqlink": True,
        "tr": 2000,
        "te": 100
    }
    
    req = urllib.request.Request(url_sim)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(payload).encode('utf-8')
    req.add_header('Content-Length', len(jsondata))
    
    print("Triggering simulation...")
    try:
        response = urllib.request.urlopen(req, jsondata)
        data = json.loads(response.read())
        if data.get("success"):
            print("Simulation successful.")
        else:
            print("Simulation failed:", data)
            return
    except Exception as e:
        print(f"Error reaching simulator: {e}")
        return

    # 2. Get Report
    url_report = "http://127.0.0.1:5050/api/report"
    print("Downloading report...")
    try:
        with urllib.request.urlopen(url_report) as response, open("NeuroPulse_Detailed_Report.pdf", 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Report saved to NeuroPulse_Detailed_Report.pdf")
    except Exception as e:
        print(f"Error downloading report: {e}")

if __name__ == "__main__":
    run()
