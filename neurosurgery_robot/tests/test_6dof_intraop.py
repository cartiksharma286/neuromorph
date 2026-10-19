import unittest
import numpy as np
from app import app
from intraoperative_robot_6dof import Intraoperative6DOFRobotEngine

class Test6DOFIntraoperativeRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_robot_engine_fk(self):
        engine = Intraoperative6DOFRobotEngine()
        fk = engine.forward_kinematics()
        self.assertIn('end_effector_position_mm', fk)
        self.assertIn('manipulability_index', fk)
        self.assertIn('condition_number', fk)
        self.assertGreater(fk['manipulability_index'], 1e5)
        self.assertGreater(fk['condition_number'], 1.0)
        print(f"[Pass] 6DOF FK & Manipulability Verified: {fk['manipulability_index']:.2e}")

    def test_02_robot_engine_ablation(self):
        engine = Intraoperative6DOFRobotEngine()
        engine.reset_ablation()
        step = engine.step_ablation(power_w=12.0, dt_s=5.0)
        self.assertGreater(step['peak_core_temperature_c'], 37.0)
        self.assertGreater(step['necrosis_radius_mm'], 0.0)
        self.assertTrue(step['safety_margin_intact'])
        print(f"[Pass] 6DOF Bioheat Ablation Verified: Temp={step['peak_core_temperature_c']:.1f}C, Radius={step['necrosis_radius_mm']:.1f}mm")

    def test_03_api_endpoints(self):
        # GET status
        res = self.client.get('/api/robot/6dof/status')
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data['status'], 'ok')
        self.assertIn('kinematics', data)
        self.assertIn('ablation', data)

        # POST kinematics
        res_k = self.client.post('/api/robot/6dof/kinematics', json={
            'joint_angles': [10.0, 30.0, -20.0, 0.0, 15.0, 0.0]
        })
        self.assertEqual(res_k.status_code, 200)
        self.assertIn('kinematics', res_k.json)

        # POST ablate
        res_ab = self.client.post('/api/robot/6dof/ablate', json={
            'action': 'step',
            'power_w': 14.0,
            'dt_s': 2.0
        })
        self.assertEqual(res_ab.status_code, 200)
        self.assertIn('ablation', res_ab.json)

        # Preprint PDF endpoint
        res_pdf = self.client.get('/api/preprint/6dof_ablation/view')
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, 'application/pdf')
        print("[Pass] 6DOF API & Preprint PDF Routes Validated")

if __name__ == '__main__':
    unittest.main()
