#!/usr/bin/env python3
"""
Quantum Noise Reduction Integration Verification
================================================

This script verifies that the quantum noise reduction system is properly
integrated with the Flask application and all endpoints are operational.
"""

import json
import base64
import numpy as np
import requests
from PIL import Image
import io

# Configuration
BASE_URL = "http://localhost:5050"
QUANTUM_ENDPOINTS = {
    "info": f"{BASE_URL}/quantum/info",
    "reconstruct": f"{BASE_URL}/quantum/reconstruct",
    "wiener": f"{BASE_URL}/quantum/wiener",
    "qml": f"{BASE_URL}/quantum/qml",
    "compare": f"{BASE_URL}/quantum/compare"
}

def test_info_endpoint():
    """Test the /quantum/info endpoint"""
    print("\n✓ Testing /quantum/info endpoint...")
    try:
        response = requests.get(QUANTUM_ENDPOINTS["info"], timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Name: {data['name']}")
            print(f"  Version: {data['version']}")
            print(f"  Methods: {', '.join(data['methods'].keys())}")
            return True
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def create_synthetic_image(size=128):
    """Create a synthetic cardiac image with noise"""
    # Cardiac phantom
    y, x = np.ogrid[:size, :size]
    cx, cy = size/2, size/2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Cardiac structure
    phantom = np.zeros((size, size))
    phantom[r < size/3] = 0.8  # LV
    phantom[(r >= size/3) & (r < size/2.5)] = 0.5  # Myocardium
    phantom[(r >= size/2.5) & (r < size/2)] = 0.3  # Epicardium
    
    # Add noise
    phantom += np.random.normal(0, 0.05, phantom.shape)
    phantom = np.clip(phantom, 0, 1)
    phantom = (phantom * 255).astype(np.uint8)
    
    return Image.fromarray(phantom)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_wiener_endpoint(image_b64):
    """Test the /quantum/wiener endpoint"""
    print("\n✓ Testing /quantum/wiener endpoint...")
    try:
        response = requests.post(
            QUANTUM_ENDPOINTS["wiener"],
            json={"image_base64": image_b64},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Method: {data.get('method', 'N/A')}")
            snr = data.get('snr_db', 'N/A')
            if isinstance(snr, (int, float)):
                print(f"  SNR: {snr:.2f} dB")
            else:
                print(f"  SNR: {snr} dB")
            return True
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_qml_endpoint(image_b64):
    """Test the /quantum/qml endpoint"""
    print("\n✓ Testing /quantum/qml endpoint...")
    try:
        response = requests.post(
            QUANTUM_ENDPOINTS["qml"],
            json={"image_base64": image_b64},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Artifacts detected: {data.get('artifacts_detected', 'N/A')}")
            artifact_pct = data.get('artifact_percentage', 'N/A')
            if isinstance(artifact_pct, (int, float)):
                print(f"  Artifact %: {artifact_pct:.1f}%")
            else:
                print(f"  Artifact %: {artifact_pct}%")
            return True
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_compare_endpoint(image_b64):
    """Test the /quantum/compare endpoint"""
    print("\n✓ Testing /quantum/compare endpoint...")
    try:
        response = requests.post(
            QUANTUM_ENDPOINTS["compare"],
            json={"image_base64": image_b64},
            timeout=45
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  Methods compared: {len(data.get('results', {}))}")
            print(f"  Best method: {data.get('best_method', 'N/A')}")
            for method, result in data.get('results', {}).items():
                snr = result.get('snr_db', result.get('SNR_dB', 'N/A'))
                if isinstance(snr, (int, float)):
                    print(f"    - {method}: SNR={snr:.2f}dB")
                else:
                    print(f"    - {method}: SNR={snr}dB")
            return True
        else:
            print(f"  ✗ Error: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("QUANTUM NOISE REDUCTION INTEGRATION TEST")
    print("="*60)
    
    # Test endpoints without image
    success_count = 0
    total_tests = 5
    
    if test_info_endpoint():
        success_count += 1
    
    # Create test image
    print("\n✓ Creating synthetic cardiac image...")
    test_image = create_synthetic_image(128)
    image_b64 = image_to_base64(test_image)
    print("  Image size: 128x128, converted to base64")
    
    # Test endpoints with image
    if test_wiener_endpoint(image_b64):
        success_count += 1
    
    if test_qml_endpoint(image_b64):
        success_count += 1
    
    if test_compare_endpoint(image_b64):
        success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"INTEGRATION TEST RESULT: {success_count}/{total_tests} endpoints working")
    print("="*60)
    
    if success_count == total_tests:
        print("✅ All quantum noise reduction endpoints verified!")
        print("\nEndpoints available at:")
        print(f"  • {QUANTUM_ENDPOINTS['info']}")
        print(f"  • {QUANTUM_ENDPOINTS['reconstruct']} (POST)")
        print(f"  • {QUANTUM_ENDPOINTS['wiener']} (POST)")
        print(f"  • {QUANTUM_ENDPOINTS['qml']} (POST)")
        print(f"  • {QUANTUM_ENDPOINTS['compare']} (POST)")
    else:
        print(f"⚠️  {total_tests - success_count} endpoint(s) failed")
    
    print()

if __name__ == "__main__":
    main()
