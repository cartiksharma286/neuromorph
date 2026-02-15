try:
    print("Testing imports...")
    import numpy as np
    import torch
    from flask import Flask
    print("Core imports OK")
    import dbs_circuit_generator
    print("dbs_circuit_generator OK")
    import generative_ai_engine
    print("generative_ai_engine OK")
    import ptsd_neural_model
    print("ptsd_neural_model OK")
    import safety_validator
    print("safety_validator OK")
    import fea_simulator
    print("fea_simulator OK")
    import treatment_optimizer
    print("treatment_optimizer OK")
    import depression_neural_model
    print("depression_neural_model OK")
    print("All imports SUCCESSFUL")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
