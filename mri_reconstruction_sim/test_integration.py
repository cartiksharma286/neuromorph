import unittest
import json
from app import app

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
        
    def test_simulation_endpoint(self):
        payload = {
            "sequence": "SE",
            "coils": "standard",
            "tr": 2000,
            "te": 100,
            "ti": 0,
            "flip_angle": 90,
            "resolution": 64,
            "noise": 0.0,
            "recon_method": "SoS",
            "shimming": False,
            "slice_orientation": "axial",
            "slice_pos": 0.5
        }
        res = self.client.post('/api/simulate', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        
        # Verify that neuro_prism and cardio_conformal reconstructed images are returned unconditionally
        self.assertIn('neuro_prism', data['plots'])
        self.assertIn('cardio_conformal', data['plots'])
        self.assertIn('recon', data['plots'])
        
    def test_neuro_pulse_ca_endpoint(self):
        res = self.client.post('/api/neuro_pulse_ca/generate', data=json.dumps({}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['count'], 5)
        
    def test_render_cortical_endpoint(self):
        res = self.client.post('/api/render_cortical', data=json.dumps({}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])

if __name__ == '__main__':
    unittest.main()
