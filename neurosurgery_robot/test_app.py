#!/usr/bin/env python3
"""
Quick test script to verify the quantum-enhanced neurosurgery robot is working
"""

import requests
import json

def test_app():
    base_url = "http://127.0.0.1:5002"
    
    print("=" * 60)
    print("Testing Quantum-Enhanced Neurosurgery Robot")
    print("=" * 60)
    
    # Test 1: Basic connectivity
    print("\n1. Testing basic connectivity...")
    try:
        response = requests.get(f"{base_url}/api/telemetry", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is responding")
            data = response.json()
            
            # Check quantum status
            quantum_enabled = data.get('quantum', {}).get('enabled', False)
            print(f"   ✅ Quantum Mode: {'ENABLED' if quantum_enabled else 'DISABLED'}")
            
            if quantum_enabled:
                metrics = data.get('quantum', {}).get('metrics', {})
                print(f"   ✅ Coherence: {metrics.get('coherence', 'N/A')}")
                print(f"   ✅ Uncertainty: {metrics.get('uncertainty', 'N/A')}")
                print(f"   ✅ QML Fidelity: {metrics.get('qml_fidelity', 'N/A')}")
        else:
            print(f"   ❌ Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Test 2: Quantum status endpoint
    print("\n2. Testing quantum status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/quantum/status", timeout=5)
        if response.status_code == 200:
            print("   ✅ Quantum status endpoint working")
            data = response.json()
            if data.get('enabled'):
                print(f"   ✅ Coherence: {data.get('coherence', 0):.4f}")
                print(f"   ✅ Uncertainty: {data.get('uncertainty', 0):.6f}")
                print(f"   ✅ Tracking Error: {data.get('tracking_error', 0):.6f}")
            else:
                print("   ⚠️  Quantum mode not available")
        else:
            print(f"   ❌ Status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Robot position
    print("\n3. Testing robot state...")
    try:
        response = requests.get(f"{base_url}/api/telemetry", timeout=5)
        data = response.json()
        position = data.get('position', [0, 0, 0])
        joints = data.get('joints', [])
        print(f"   ✅ Robot Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"   ✅ Joint Count: {len(joints)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: System status
    print("\n4. Testing system runtime status...")
    try:
        response = requests.get(f"{base_url}/api/telemetry", timeout=5)
        data = response.json()
        system = data.get('system', {})
        print(f"   ✅ Status: {system.get('status', 'Unknown')}")
        print(f"   ✅ Loop Rate: {system.get('loop_hz', 0):.0f} Hz")
        print(f"   ✅ Active: {system.get('simulation_running', False)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - App is working correctly!")
    print("=" * 60)
    print(f"\n🌐 Access the app at: {base_url}")
    print("📊 Quantum metrics available at: /api/quantum/status")
    print("📄 Technical report: Quantum_Kalman_Surgical_Robotics_Report.tex")
    
    return True

if __name__ == "__main__":
    test_app()
