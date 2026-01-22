
import os
import sys
import socket
import time
import requests
import subprocess
import signal
import webbrowser

def get_ip_address():
    try:
        # Connect to an external server (Google DNS) to determine the routing IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def kill_process_on_port(port):
    try:
        # Find PID using lsof
        cmd = f"lsof -ti :{port}"
        pid_output = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid_output:
            pids = pid_output.replace('\n', ' ')
            print(f"Killing process(es) {pids} on port {port}")
            subprocess.run(f"kill -9 {pids}", shell=True)
            time.sleep(1)
    except subprocess.CalledProcessError:
        pass # No process found

def main():
    port = 5002
    ip = get_ip_address()
    
    print(f"Configuration: IP={ip}, Port={port}")
    
    # 1. Cleanup
    kill_process_on_port(port)
    
    # 2. Start Server
    print("Starting server...")
    log_file = open("server.log", "w")
    # Using nohup/setsid-like behavior by not waiting
    server_process = subprocess.Popen(
        ["python3", "server.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    print(f"Server started (PID: {server_process.pid}). Waiting for initialization...")
    
    # 3. Test Connection
    max_retries = 60
    url = f"http://{ip}:{port}/api/health"
    app_url = f"http://{ip}:{port}"
    
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print("\n✅ Connection Verified!")
                print(f"Data: {r.json()}")
                print(f"\n🚀 Launching Browser: {app_url}")
                webbrowser.open(app_url)
                return
        except requests.exceptions.RequestException:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)
            
    print("\n❌ Failed to connect to server after 30 seconds.")
    # Fallback to localhost check
    try:
        r = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
        if r.status_code == 200:
            print("But localhost is working. Opening localhost...")
            webbrowser.open(f"http://127.0.0.1:{port}")
    except:
        print("Localhost also failed.")

if __name__ == "__main__":
    main()
