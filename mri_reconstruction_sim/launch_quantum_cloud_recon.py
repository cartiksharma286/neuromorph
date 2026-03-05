import time
import numpy as np
from simulator_core import MRIReconstructionSimulator
from statistical_adaptive_pulse import ADAPTIVE_SEQUENCES
from quantum_vascular_coils import QUANTUM_VASCULAR_COIL_LIBRARY

def run_quantum_cloud_recon_test():
    print("=" * 80)
    print("INITIATING EXPEDITED QUANTUM CLOUD RECONSTRUCTION LAUNCH")
    print("Optimization: Tensor-based signal preservation bypass.")
    print("=" * 80)

    # Core sequences to test, including complex/advanced ones
    sequences_to_test = [
        'SE', 'GRE', 'SSFP', 'InversionRecovery', 'EPI',
        'QuantumRBMThermometry', 'StatisticalBayesianThermometry',
        'RoboticsFMRI', 'QuantumGeometry'
    ]

    # Core coil modalities to test
    coils_to_test = [
        'standard',
        'quantum_vascular',
        'head_coil_50_turn',
        'cardiothoracic_array',
        'knee_vascular_array',
        'neurovascular_prism',
        'fractional_geodesic_array'
    ]
    
    # Try a few specific quantum library coils as well to ensure total coverage
    library_coils = list(QUANTUM_VASCULAR_COIL_LIBRARY.keys())
    if len(library_coils) > 5:
        coils_to_test.extend(library_coils[:5])

    results = []
    total_time = 0.0

    print(f"Testing {len(sequences_to_test)} sequences across {len(coils_to_test)} coil configurations.\n")

    for seq in sequences_to_test:
        for coil in coils_to_test:
            try:
                # 1. Initialize Simulator
                sim = MRIReconstructionSimulator(resolution=128)
                
                # Assign phantom
                phantom_type = 'brain'
                if coil in ['cardiothoracic_array', 'cardiovascular_coil']:
                    phantom_type = 'cardiac'
                elif coil in ['knee_vascular_array']:
                    phantom_type = 'knee'

                sim.setup_phantom(use_real_data=True, phantom_type=phantom_type)
                
                # Check 50-turn
                if coil == 'head_coil_50_turn':
                    sim.head_coil_50_turn['enabled'] = True
                    
                # Setup coils
                sim.generate_coil_sensitivities(num_coils=8, coil_type=coil, optimal_shimming=False)
                
                # 2. Acquire Signal
                kspace, M_ref = sim.acquire_signal(
                    sequence_type=seq, 
                    TR=500, TE=20, TI=1500, 
                    flip_angle=90, noise_level=0.01
                )
                
                # 3. Expedited Quantum Cloud Reconstruction
                start_time = time.time()
                recon_img, _ = sim.reconstruct_image(
                    kspace, 
                    method='SoS',
                    expedited=True,
                    quantum_cloud=True
                )
                recon_time = time.time() - start_time
                total_time += recon_time

                # 4. Validation Metrics
                metrics = sim.compute_metrics(recon_img, M_ref)
                snr = metrics.get('snr_estimate', 0.0)
                
                status = "PASS" if snr > 5.0 else "FAIL"
                
                record = {
                    'sequence': seq,
                    'coil': coil,
                    'snr': snr,
                    'time_ms': recon_time * 1000,
                    'status': status
                }
                results.append(record)
                
                # Print single-line summary
                print(f"[{status}] Seq: {seq:<25} | Coil: {coil:<25} | SNR: {snr:>7.2f} | Time: {recon_time*1000:>6.2f} ms")

            except Exception as e:
                print(f"[ERROR] Seq: {seq:<25} | Coil: {coil:<25} | {str(e)}")
                results.append({
                    'sequence': seq,
                    'coil': coil,
                    'snr': 0.0,
                    'time_ms': 0.0,
                    'status': 'ERROR'
                })

    print("\n" + "=" * 80)
    print("TESTING COMPLETE - SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"Total Combinations: {len(results)}")
    print(f"Passed (SNR > 5):   {passed}")
    print(f"Failed (Low SNR):   {failed}")
    print(f"Errors:             {errors}")
    print(f"Total Recon Time:   {total_time:.3f} s  (Avg: {total_time/max(1, len(results))*1000:.2f} ms/recon)")

if __name__ == "__main__":
    run_quantum_cloud_recon_test()
