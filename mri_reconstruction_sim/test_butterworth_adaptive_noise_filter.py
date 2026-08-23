"""
Test Suite for Butterworth Filters & Adaptive Signal Processing Noise Speckle Filtering
======================================================================================
"""

import sys
import numpy as np
import unittest

sys.path.insert(0, '.')

from simulator_core import MRIReconstructionSimulator
from ellipsoidal_artifact_removal import EllipsoidalArtifactRemover
from app_enhanced import app


class TestButterworthAndAdaptiveNoiseFilter(unittest.TestCase):

    def setUp(self):
        self.sim = MRIReconstructionSimulator(resolution=64)
        self.sim.setup_phantom(use_real_data=False, phantom_type='brain')
        self.sim.generate_coil_sensitivities(num_coils=4)

    def test_butterworth_filter_execution(self):
        img = np.random.rand(64, 64)
        # Add artificial speckles
        img[10, 10] = 5.0
        img[30, 40] = 4.5

        filtered = self.sim.apply_butterworth_filter(img, cutoff=0.25, order=2)
        self.assertEqual(filtered.shape, (64, 64))
        self.assertFalse(np.isnan(filtered).any())
        self.assertTrue(filtered[10, 10] < 5.0, "Butterworth filter should suppress bright speckle peaks")

    def test_adaptive_signal_processing_filter_execution(self):
        img = np.random.rand(64, 64)
        img[15, 15] = 6.0
        img[50, 50] = 5.5

        filtered = self.sim.apply_adaptive_signal_processing_filter(img, window_size=5)
        self.assertEqual(filtered.shape, (64, 64))
        self.assertFalse(np.isnan(filtered).any())
        self.assertTrue(filtered[15, 15] < 6.0, "Adaptive filter should smooth out speckle peak in homogeneous region")

    def test_reconstruct_image_with_butterworth(self):
        kspace, M_ref = self.sim.acquire_signal(sequence_type='SE', TR=2000, TE=100, noise_level=0.05)
        recon, _ = self.sim.reconstruct_image(kspace, method='SoS', noise_filter='Butterworth', ellipsoidal_mask=True)
        self.assertEqual(recon.shape, (64, 64))
        self.assertFalse(np.isnan(recon).any())

    def test_reconstruct_image_with_adaptive_signal_processing(self):
        kspace, M_ref = self.sim.acquire_signal(sequence_type='SE', TR=2000, TE=100, noise_level=0.05)
        recon, _ = self.sim.reconstruct_image(kspace, method='SoS', noise_filter='Adaptive Signal Processing', ellipsoidal_mask=True)
        self.assertEqual(recon.shape, (64, 64))
        self.assertFalse(np.isnan(recon).any())

    def test_ellipsoidal_artifact_remover_with_new_filters(self):
        remover = EllipsoidalArtifactRemover(phantom_type='brain', resolution=64)
        img = np.random.rand(64, 64)
        
        cleaned_bw, stats_bw = remover.remove_artifacts(img, filter_type='butterworth')
        self.assertEqual(cleaned_bw.shape, (64, 64))
        self.assertEqual(stats_bw['filter_type'], 'butterworth')

        cleaned_ad, stats_ad = remover.remove_artifacts(img, filter_type='adaptive')
        self.assertEqual(cleaned_ad.shape, (64, 64))
        self.assertEqual(stats_ad['filter_type'], 'adaptive')

    def test_api_simulate_with_butterworth_and_adaptive(self):
        client = app.test_client()

        # Test Butterworth API call
        res_bw = client.post('/api/simulate', json={
            'resolution': 64,
            'sequence': 'SE',
            'noise_filter': 'Butterworth',
            'noise': 0.05,
            'ellipsoidal_mask': True
        })
        self.assertEqual(res_bw.status_code, 200)
        data_bw = res_bw.get_json()
        self.assertTrue(data_bw.get('success', False))

        # Test Adaptive Signal Processing API call
        res_ad = client.post('/api/simulate', json={
            'resolution': 64,
            'sequence': 'SE',
            'noise_filter': 'Adaptive Signal Processing',
            'noise': 0.05,
            'ellipsoidal_mask': True
        })
        self.assertEqual(res_ad.status_code, 200)
        data_ad = res_ad.get_json()
        self.assertTrue(data_ad.get('success', False))


if __name__ == '__main__':
    unittest.main()
