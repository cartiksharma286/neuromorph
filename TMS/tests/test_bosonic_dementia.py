import unittest
from app import app
from logic.bosonic_dementia_engine import (
    BosonicDementiaEngine,
    simulate_bosonic_waveform_qnn,
    simulate_cortical_fem_qml_partitions,
    simulate_dementia_recovery_care_trajectories,
    DEMENTIA_CORTICAL_LOBES,
    GANGOBO_STAGING,
)
from generate_nature_bosonic_dementia_preprint import build_bosonic_dementia_pdf


class TestBosonicDementia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.engine = BosonicDementiaEngine()
        cls.res = cls.engine.simulate()

    def test_01_bosonic_waveform_synthesis(self):
        wave = simulate_bosonic_waveform_qnn(squeezing_r=1.25, displacement_alpha=0.65, kerr_chi=0.08)
        self.assertEqual(len(wave["time_ms"]), len(wave["bosonic_pulse_waveform_t_s"]))
        self.assertEqual(len(wave["time_ms"]), len(wave["induced_e_field_v_m"]))
        self.assertGreater(wave["squeezing_db"], 10.0)
        self.assertGreater(max(wave["induced_e_field_v_m"]), 80.0)
        self.assertEqual(len(wave["wigner_matrix"]), len(wave["wigner_matrix"][0]))
        print(f"[Pass] Bosonic Waveform Synthesis Verified: Squeezing={wave['squeezing_db']} dB, Peak E-Field={max(wave['induced_e_field_v_m']):.1f} V/m")

    def test_02_cortical_fem_qml_partitions(self):
        fem = simulate_cortical_fem_qml_partitions(num_partitions=32, target_lobe_key="precuneus_pcc")
        self.assertEqual(fem["num_partitions"], 32)
        self.assertEqual(len(fem["partitions"]), 32)
        self.assertIn("precuneus_pcc", fem["lobe_summary"])
        self.assertIn("mesial_temporal", fem["lobe_summary"])
        self.assertIn("left_dlpfc", fem["lobe_summary"])
        self.assertGreater(fem["base_e_field_v_m"], 88.0)  # Exceeds pyramidal depolarization threshold
        self.assertEqual(len(fem["depth_attenuation"]["depth_mm"]), 31)
        print(f"[Pass] Cortical FEM QML Partitions Verified: 32 elements, Target E-Field={fem['base_e_field_v_m']} V/m")

    def test_03_dementia_recovery_trajectories(self):
        traj = simulate_dementia_recovery_care_trajectories(baseline_stage="mild_ad", horizon_months=60)
        self.assertEqual(len(traj["months"]), 21)
        self.assertGreater(traj["adas_cog_14"]["improvement_pct"], 35.0)
        self.assertGreater(traj["cdr_sb"]["improvement_pct"], 35.0)
        self.assertGreater(traj["mmse"]["gain_points"], 3.0)
        self.assertGreaterEqual(traj["institutionalization_delay_years"], 3.0)
        print(f"[Pass] 60-Month Dementia Recovery Trajectories Verified: ADAS-Cog -{traj['adas_cog_14']['improvement_pct']}%, Delay={traj['institutionalization_delay_years']} Yrs")

    def test_04_api_simulation_and_pdf(self):
        # Defaults endpoint
        res_def = self.client.get("/api/bosonic-dementia-defaults")
        self.assertEqual(res_def.status_code, 200)
        self.assertIn("lobes_affected", res_def.json)
        self.assertIn("gangobo_staging", res_def.json)

        # Simulation endpoint
        payload = {
            "squeezing_r": 1.35,
            "target_lobe_key": "precuneus_pcc",
            "baseline_stage": "mci"
        }
        res_sim = self.client.post("/api/bosonic-dementia-simulation", json=payload)
        self.assertEqual(res_sim.status_code, 200)
        data = res_sim.json
        self.assertEqual(data["status"], "success")
        self.assertIn("bosonic_waveform", data)
        self.assertIn("cortical_fem_qml", data)
        self.assertIn("clinical_trajectories", data)

        # PDF Compilation
        output_path = "/tmp/test_bosonic_dementia.pdf"
        built_path = build_bosonic_dementia_pdf(output_path, {})
        self.assertTrue(built_path.endswith(".pdf"))
        with open(built_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")
        print(f"[Pass] Nature Preprint PDF Compilation Verified: {built_path}")


if __name__ == '__main__':
    unittest.main()
