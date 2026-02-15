
import sys
import time
from combinatorial_manifold_neurogenesis import PTSDDementiaRepairModel

print("Starting debug init...")
start = time.time()
try:
    model = PTSDDementiaRepairModel(num_neurons=100, pathology_type='dementia')
    print(f"Model created in {time.time() - start:.2f}s")
    
    start = time.time()
    analysis = model.analyze_topology()
    print(f"Analysis done in {time.time() - start:.2f}s")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
