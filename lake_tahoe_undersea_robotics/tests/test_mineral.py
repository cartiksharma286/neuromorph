import json
import unittest
from dataclasses import asdict

from app import app
from backend.mineral_excavation_engine import (
    COMMODITY_PRICES,
    DEPOSIT_PRESETS,
    MineralExcavationEngine,
    MineralInputs,
    compute_10yr_mineral_revenue,
    parse_mineral_inputs,
    simulate_fermionic_parity_qml,
    simulate_ore_spectroscopy,
)
from backend.mineral_preprint import render_mineral_preprint


class MineralExcavationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = MineralExcavationEngine()
        cls.default_inputs = MineralInputs()
        cls.report = cls.engine.simulate(cls.default_inputs)

    def test_fermionic_parity_properties(self):
        preset = DEPOSIT_PRESETS["abyssal_nodule_field"]
        parity_res = simulate_fermionic_parity_qml(deposit_preset=preset, num_qubits=5, depth=3, seed=42)

        self.assertEqual(parity_res["num_qubits"], 5)
        self.assertTrue(-1.0 <= parity_res["parity_expectation"] <= 1.0)
        self.assertAlmostEqual(
            parity_res["even_parity_probability"] + parity_res["odd_parity_probability"],
            1.0,
            places=5
        )
        self.assertTrue(0.0 <= parity_res["qml_confidence"] <= 1.0)
        self.assertEqual(len(parity_res["depth_axis"]), len(parity_res["depth_parity_sweep"]))

    def test_hyperspectral_reflectance_shapes(self):
        preset = DEPOSIT_PRESETS["hydrothermal_vein"]
        spec = simulate_ore_spectroscopy(deposit_preset=preset, turbidity_ntu=5.0, seed=42)

        self.assertEqual(len(spec["wavelengths_nm"]), 61)
        self.assertEqual([spec["wavelengths_nm"][0], spec["wavelengths_nm"][-1]], [400.0, 1000.0])
        self.assertEqual(len(spec["iron_spectrum"]), 61)
        self.assertEqual(len(spec["uranium_spectrum"]), 61)
        self.assertEqual(len(spec["ree_spectrum"]), 61)
        self.assertEqual(len(spec["silver_spectrum"]), 61)
        self.assertEqual(len(spec["cobalt_spectrum"]), 61)
        self.assertEqual(len(spec["composite_ore_spectrum"]), 61)
        self.assertTrue(all(0.0 <= val <= 1.0 for val in spec["composite_ore_spectrum"]))

    def test_10yr_discounted_cash_flow(self):
        preset = DEPOSIT_PRESETS["abyssal_nodule_field"]
        inputs = MineralInputs(excavation_rate_tpd=1000.0, operating_days_year=300.0)
        econ = compute_10yr_mineral_revenue(inputs, preset)

        self.assertEqual(len(econ["trajectory"]), 11)  # Years 0 to 10
        self.assertEqual(econ["trajectory"][0]["year"], 0)
        self.assertLess(econ["trajectory"][0]["cumulative_discounted_npv_usd"], 0)
        self.assertGreater(econ["annual_gross_revenue_usd"], 0)
        self.assertGreater(econ["annual_metal_production"]["iron_tonnes"], 0)
        self.assertGreater(econ["annual_metal_production"]["uranium_kg"], 0)
        self.assertGreater(econ["annual_metal_production"]["ree_kg"], 0)
        self.assertGreater(econ["annual_metal_production"]["silver_kg"], 0)
        self.assertGreater(econ["annual_metal_production"]["cobalt_tonnes"], 0)
        self.assertGreater(econ["ten_year_discounted_npv_usd"], 0)

    def test_input_validation(self):
        valid = parse_mineral_inputs({"excavation_rate_tpd": 1200.0, "chosen_deposit": "granitic_placer"})
        self.assertEqual(valid.excavation_rate_tpd, 1200.0)
        self.assertEqual(valid.chosen_deposit, "granitic_placer")

        invalid_cases = [
            [],
            {"excavation_rate_tpd": -100.0},
            {"capital_capex_usd_m": 5000.0},
            {"discount_rate": 0.95},
            {"chosen_deposit": "nonexistent_mine"},
            {"pqc_depth": 15},
            {"pqc_depth": 2.5},
            {"unknown_key": 99},
        ]
        for item in invalid_cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_mineral_inputs(item)

    def test_endpoints_and_pdf(self):
        client = app.test_client()

        # Defaults
        res_defaults = client.get("/api/mineral/defaults")
        self.assertEqual(res_defaults.status_code, 200)
        self.assertIn("inputs", res_defaults.json)
        self.assertIn("deposits", res_defaults.json)
        self.assertIn("commodities", res_defaults.json)

        # Deposits
        res_dep = client.get("/api/mineral/deposits")
        self.assertEqual(res_dep.status_code, 200)
        self.assertIn("abyssal_nodule_field", res_dep.json["deposits"])

        # Simulate
        res_sim = client.post("/api/mineral/simulate", json={"excavation_rate_tpd": 900.0})
        self.assertEqual(res_sim.status_code, 200)
        report = res_sim.json["report"]
        self.assertIn("spectroscopy", report)
        self.assertIn("qml_parity", report)
        self.assertIn("economics", report)
        self.assertIn("deposit_preset", report)

        # Invalid simulation input returns 400
        res_bad = client.post("/api/mineral/simulate", json={"excavation_rate_tpd": 99999.0})
        self.assertEqual(res_bad.status_code, 400)

        # Preprint PDF download
        res_pdf = client.post("/api/mineral/preprint", json={"excavation_rate_tpd": 900.0})
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.mimetype, "application/pdf")
        self.assertTrue(res_pdf.data.startswith(b"%PDF-"))
        self.assertGreater(len(res_pdf.data), 12000)

        # Inline preprint view
        res_view = client.get("/api/mineral/preprint/view")
        self.assertEqual(res_view.status_code, 200)
        self.assertTrue(res_view.data.startswith(b"%PDF-"))

        # HTML page includes mineral tab
        res_html = client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn(b'id="tab-mineral"', res_html.data)
        self.assertIn(b'/static/mineral.js', res_html.data)


if __name__ == "__main__":
    unittest.main()
