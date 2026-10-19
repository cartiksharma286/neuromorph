import json
import math
import unittest
from dataclasses import asdict

import numpy as np

from app import app
from backend.underwater_vision_engine import (
    UnderwaterVisionEngine,
    VisionInputs,
    complete_elliptic_integral_k,
    elliptic_modular_invariants,
    jaffe_mcglamery_optics,
    parse_vision_inputs,
    simulate_qml_vision_classifier,
    statistical_distributions,
)
from backend.vision_preprint import render_vision_preprint


class VisionSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = UnderwaterVisionEngine()
        cls.default_inputs = VisionInputs()
        cls.report = cls.engine.simulate(cls.default_inputs)

    def test_complete_elliptic_integral_agm(self):
        # K(0) = pi / 2
        k0 = complete_elliptic_integral_k(0.0)
        self.assertAlmostEqual(k0, math.pi / 2.0, places=7)

        # K(k) increases with k
        k_half = complete_elliptic_integral_k(0.5)
        self.assertGreater(k_half, k0)

        # Complementary integral and modular parameter tau
        invariants = elliptic_modular_invariants(0.6)
        self.assertGreater(invariants["tau_imag"], 0.0)
        self.assertGreater(invariants["j_invariant"], 0.0)
        self.assertGreater(invariants["K_k"], 0.0)

    def test_jaffe_mcglamery_optics_attenuation(self):
        opt = jaffe_mcglamery_optics(target_distance_m=5.0, turbidity_ntu=4.0, ambient_lux=10.0)
        self.assertGreater(opt["absorption_coeff_m_minus_1"], 0.0)
        self.assertGreater(opt["scatter_coeff_m_minus_1"], 0.0)
        self.assertAlmostEqual(
            opt["attenuation_coeff_m_minus_1"],
            opt["absorption_coeff_m_minus_1"] + opt["scatter_coeff_m_minus_1"],
            places=5
        )
        self.assertTrue(0.0 <= opt["optical_transmittance"] <= 1.0)
        self.assertTrue(0.0 <= opt["contrast_transmission"] <= 1.0)

        # Higher turbidity must reduce transmittance
        opt_turbid = jaffe_mcglamery_optics(target_distance_m=5.0, turbidity_ntu=20.0, ambient_lux=10.0)
        self.assertLess(opt_turbid["optical_transmittance"], opt["optical_transmittance"])

    def test_statistical_distributions_shapes(self):
        stats = statistical_distributions(turbidity_ntu=5.0, seed=42)
        self.assertEqual(len(stats["weibull_intensity_x"]), len(stats["weibull_pdf_y"]))
        self.assertEqual(len(stats["timeline_sec"]), 61)
        self.assertEqual(len(stats["event_anomaly_score"]), 61)
        self.assertTrue(all(p >= 0.0 for p in stats["weibull_pdf_y"]))

    def test_qml_vision_classifier_convergence(self):
        qml = simulate_qml_vision_classifier(k_modulus=0.75, j_log=3.2, contrast=0.6, layers=3, seed=42)
        self.assertEqual(qml["num_qubits"], 4)
        self.assertTrue(0.0 <= qml["detection_probability"] <= 1.0)
        self.assertTrue(0.0 <= qml["quantum_state_fidelity"] <= 1.0)
        self.assertEqual(len(qml["iterations"]), 25)
        self.assertEqual(len(qml["variational_loss"]), 25)
        self.assertGreater(qml["held_out_accuracy"], 0.85)

    def test_input_validation(self):
        valid = parse_vision_inputs({"turbidity_ntu": 8.5, "chosen_scene": "hydrothermal_vent"})
        self.assertEqual(valid.turbidity_ntu, 8.5)
        self.assertEqual(valid.chosen_scene, "hydrothermal_vent")

        invalid_cases = [
            [],
            "string",
            {"turbidity_ntu": -2.0},
            {"depth_m": 600.0},
            {"chosen_scene": "invalid_scene_name"},
            {"pqc_layers": 10},
            {"pqc_layers": 2.5},
            {"unknown_field": 123},
        ]
        for item in invalid_cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_vision_inputs(item)

    def test_api_endpoints_and_pdf(self):
        client = app.test_client()

        # Defaults
        res_defaults = client.get("/api/vision/defaults")
        self.assertEqual(res_defaults.status_code, 200)
        self.assertIn("inputs", res_defaults.json)
        self.assertIn("scenes", res_defaults.json)

        # Scenes
        res_scenes = client.get("/api/vision/scenes")
        self.assertEqual(res_scenes.status_code, 200)
        self.assertIn("pipeline_infrastructure", res_scenes.json["scenes"])

        # Simulate
        res_sim = client.post("/api/vision/simulate", json={"turbidity_ntu": 6.0})
        self.assertEqual(res_sim.status_code, 200)
        report = res_sim.json["report"]
        self.assertIn("optics", report)
        self.assertIn("elliptic", report)
        self.assertIn("qml", report)
        self.assertIn("conservation", report)
        self.assertIn("detected_objects", report)

        con = report["conservation"]
        self.assertIn("gsd_mm_per_px", con)
        self.assertIn("geo_extents", con)
        self.assertIn("rho_quartz_sand", con)
        self.assertIn("rho_diatom_silt", con)
        self.assertIn("rho_deep_moss", con)

        # Invalid simulation input returns 400
        res_bad = client.post("/api/vision/simulate", json={"turbidity_ntu": 999.0})
        self.assertEqual(res_bad.status_code, 400)

        # Preprint PDF download
        res_pdf = client.post("/api/vision/preprint", json={"turbidity_ntu": 6.0})
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, "application/pdf")
        self.assertTrue(res_pdf.data.startswith(b"%PDF-"))
        self.assertGreater(len(res_pdf.data), 12000)

        # Inline preprint view
        res_view = client.get("/api/vision/preprint/view")
        self.assertEqual(res_view.status_code, 200)
        self.assertTrue(res_view.data.startswith(b"%PDF-"))

        # HTML page includes vision tab
        res_html = client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn(b'id="tab-vision"', res_html.data)
        self.assertIn(b'/static/vision.js', res_html.data)


if __name__ == "__main__":
    unittest.main()
