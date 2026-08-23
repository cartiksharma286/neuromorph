import requests
import json

def test_qlora():
    url = "http://127.0.0.1:5055/api/register-cortical-surface-qlora"
    print("Sending POST request to register-cortical-surface-qlora...")
    response = requests.post(url)
    if response.status_code == 200:
        data = response.json()
        print("\n--- qLoRA Registration Results ---")
        print(f"Status Code: {response.status_code}")
        print(f"Physical Registration Error (TRE): {data['registration_error']:.6f} mm")
        print(f"Number of Epochs Run: {len(data['qlora_history'])}")
        print(f"Initial Epoch Error (TRE): {data['qlora_history'][0]:.6f} mm")
        print(f"Final Epoch Error (TRE): {data['qlora_history'][-1]:.6f} mm")
        
        transform = data['registration_transform']
        print(f"Base Projection Matrix W0 Quantized Size: {len(transform['W0_quant'])}x{len(transform['W0_quant'][0])}")
        print(f"Active Adapter A Size: {len(transform['lora_A'])}x{len(transform['lora_A'][0])}")
        print(f"Active Adapter B Size: {len(transform['lora_B'])}x{len(transform['lora_B'][0])}")
        
        # Save output JSON for completeness
        with open("test_result_api_register-cortical-surface-qlora.json", "w") as f:
            json.dump(data, f, indent=2)
        print("\nResults successfully saved to test_result_api_register-cortical-surface-qlora.json")
    else:
        print(f"Failed! Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_qlora()
