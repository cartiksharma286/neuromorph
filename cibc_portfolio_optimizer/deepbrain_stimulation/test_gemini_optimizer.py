#!/usr/bin/env python3
"""
Test script to verify Gemini optimizer integration
"""

import requests
import json

BASE_URL = "http://localhost:5002"

def test_health_check():
    """Test server health check"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/api/health")
    if response.status_code == 200:
        print("✓ Health check passed")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False

def test_optimizer_info():
    """Test Gemini optimizer info endpoint"""
    print("\nTesting Gemini optimizer info...")
    response = requests.get(f"{BASE_URL}/api/quantum/info")
    if response.status_code == 200:
        data = response.json()
        print("✓ Optimizer info retrieved")
        print(f"  Optimizer: {data.get('optimizer')}")
        print(f"  Gemini Available: {data.get('gemini_available')}")
        print(f"  Model: {data.get('model')}")
        print(f"  Capabilities: {', '.join(data.get('capabilities', []))}")
        return True
    else:
        print(f"✗ Optimizer info failed: {response.status_code}")
        return False

def test_vqe_optimization():
    """Test VQE optimization with Gemini"""
    print("\nTesting VQE optimization...")
    
    payload = {
        "initial_params": {
            "amplitude_ma": 2.0,
            "frequency_hz": 130,
            "pulse_width_us": 90
        },
        "bounds": {
            "amplitude_ma": [0.5, 8.0],
            "frequency_hz": [20, 185],
            "pulse_width_us": [60, 210]
        },
        "max_iterations": 20
    }
    
    response = requests.post(f"{BASE_URL}/api/quantum/optimize/vqe", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("✓ VQE optimization successful")
            print(f"  Method: {data.get('method')}")
            print(f"  Iterations: {data.get('iterations')}")
            print(f"  Energy: {data.get('energy'):.6f}")
            print(f"  Confidence Score: {data.get('confidence_score', 0):.2%}")
            print(f"  Optimal Parameters:")
            for param, value in data.get('optimal_parameters', {}).items():
                print(f"    {param}: {value:.4f}")
            if data.get('gemini_insights'):
                print(f"  Gemini Insights: {data.get('gemini_insights')[:100]}...")
            return True
        else:
            print(f"✗ Optimization failed: {data.get('error')}")
            return False
    else:
        print(f"✗ Request failed: {response.status_code}")
        return False

def main():
    print("="*60)
    print("Gemini 3.0 Optimizer Integration Test")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Optimizer Info", test_optimizer_info),
        ("VQE Optimization", test_vqe_optimization)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} error: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Gemini optimizer is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the server logs.")

if __name__ == "__main__":
    main()
