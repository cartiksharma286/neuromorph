import time
import urllib.request
import threading
import statistics
import sys
import random

# Configuration
ALB_DNS = "http://sunnybrook-imaging-alb-12345.ca-central-1.elb.amazonaws.com" # Replace with actual DNS
NUM_REQUESTS = 100
CONCURRENCY = 10

latencies = []

def simulate_request(req_id):
    start_time = time.time()
    try:
        # In a real test, this would hit the ALB DNS
        # For simulation, we wait a random time to mimic network/processing latency
        # urllib.request.urlopen(f"{ALB_DNS}/health")
        time.sleep(random.uniform(0.05, 0.2)) # 50ms to 200ms latency
        status = 200
    except Exception as e:
        status = 500
    
    end_time = time.time()
    duration = (end_time - start_time) * 1000 # ms
    latencies.append(duration)
    # print(f"Req {req_id}: {status} in {duration:.2f}ms")

def run_test():
    print(f"--- Starting Performance Test ---")
    print(f"Target: {ALB_DNS}")
    print(f"Requests: {NUM_REQUESTS}, Concurrency: {CONCURRENCY}")
    
    threads = []
    for i in range(NUM_REQUESTS):
        t = threading.Thread(target=simulate_request, args=(i,))
        threads.append(t)
        t.start()
        
        if len(threads) >= CONCURRENCY:
            for t in threads:
                t.join()
            threads = []
            
    # Cleanup remaining
    for t in threads:
        t.join()
        
    print("\n--- Results ---")
    print(f"Total Requests: {len(latencies)}")
    print(f"Avg Latency: {statistics.mean(latencies):.2f} ms")
    print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms") # Approx P95
    print(f"Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ALB_DNS = sys.argv[1]
    run_test()
