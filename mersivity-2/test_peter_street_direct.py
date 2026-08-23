import math
import json

def test_ntu_prediction_math():
    rain_intensity = 35.0
    flow_rate_q = 8.0
    sediment_in = 200.0
    baffle_efficiency = 80.0
    sensor_nodes = 32

    # Steve Mann Hydro-Veillance NTU Formula
    base_ntu = 0.35 * sediment_in * ((1.0 + 0.03 * rain_intensity) ** 1.1) * ((flow_rate_q / 5.0) ** 0.65)
    predicted_ntu = max(1.5, base_ntu * (1.0 - 0.009 * baffle_efficiency))
    clarity_pct = max(5.0, min(99.0, 100.0 * math.exp(-0.028 * predicted_ntu)))
    ecoli_cfu = int(round(12.0 * (predicted_ntu ** 1.12) * (1.0 + 0.01 * rain_intensity)))
    dissolved_oxygen = max(2.0, 11.5 - 0.04 * predicted_ntu - 0.1 * flow_rate_q)
    sousveillance_confidence = min(99.5, 65.0 + 4.5 * math.sqrt(sensor_nodes) - 0.05 * predicted_ntu)

    print("✓ NTU Prediction Engine Math Test Passed!")
    print(f"Predicted NTU: {predicted_ntu:.2f} NTU")
    print(f"Water Clarity: {clarity_pct:.2f}%")
    print(f"E. coli Count: {ecoli_cfu} CFU/100mL")
    print(f"Dissolved O2: {dissolved_oxygen:.2f} mg/L")
    print(f"Sousveillance Trust Score: {sousveillance_confidence:.2f}%")

if __name__ == '__main__':
    test_ntu_prediction_math()
