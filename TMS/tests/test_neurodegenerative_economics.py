import unittest
from app import app
from logic.neurodegenerative_economics import (
    NeurodegenerativeEconomicsEngine,
    compute_combinatorial_stimulation_invariants,
    compute_quantum_statistical_invariants,
    compute_10yr_neurodegenerative_projections,
    NEURODEGENERATIVE_INDICATIONS
)
from generate_nature_neurodegenerative_economics import build_neurodegenerative_economics_pdf


class TestNeurodegenerativeEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.engine = NeurodegenerativeEconomicsEngine()
        cls.res = cls.engine.simulate()

    def test_01_combinatorial_invariants(self):
        c = compute_combinatorial_stimulation_invariants(num_sites=5)
        self.assertEqual(c["num_target_sites"], 5)
        self.assertEqual(c["bell_number_partitions"], 52)
        self.assertGreater(c["schur_polynomial_invariant"], 0.0)
        self.assertGreater(c["partition_lattice_entropy_nats"], 0.5)
        print(f"[Pass] Combinatorics Verified: B_5={c['bell_number_partitions']} partitions, Schur={c['schur_polynomial_invariant']}")

    def test_02_quantum_statistical_invariants(self):
        q = compute_quantum_statistical_invariants(num_qubits=6)
        self.assertEqual(q["num_qubits"], 6)
        self.assertEqual(len(q["stages"]), 5)
        self.assertGreater(q["final_qfi"], q["quantum_fisher_information"][0])
        self.assertGreater(q["von_neumann_entropy"][0], q["von_neumann_entropy"][-1])
        self.assertGreater(q["final_entropy_reduction_pct"], 50.0)
        print(f"[Pass] Quantum Invariants Verified: QFI={q['final_qfi']}, S(rho) drop={q['final_entropy_reduction_pct']}%")

    def test_03_ten_year_projections(self):
        p = compute_10yr_neurodegenerative_projections(expansion_scenario="enterprise_scale")
        self.assertEqual(len(p["years"]), 10)
        self.assertGreater(p["ten_year_discounted_npv_usd_m"], 10.0)
        self.assertGreater(p["ten_year_cumulative_ebitda_usd_m"], 20.0)
        self.assertGreater(p["total_societal_healthcare_savings_usd_m"], 50.0)
        self.assertLessEqual(p["payback_year"], 4)
        print(f"[Pass] 10-Yr Projections Verified: NPV=${p['ten_year_discounted_npv_usd_m']}M, EBITDA=${p['ten_year_cumulative_ebitda_usd_m']}M, Payback={p['payback_year']} yrs")

    def test_04_api_simulation_and_pdf(self):
        # Simulation API
        res_sim = self.client.post("/api/neurodegenerative-economics-simulation", json={"expansion_scenario": "enterprise_scale"})
        self.assertEqual(res_sim.status_code, 200)
        data = res_sim.json
        self.assertEqual(data["status"], "success")
        self.assertIn("summary_kpis", data)
        self.assertIn("financial_projections", data)

        # PDF Compilation
        output_path = "/tmp/test_neurodegenerative_economics.pdf"
        built_path = build_neurodegenerative_economics_pdf(output_path, {})
        self.assertTrue(built_path.endswith(".pdf"))
        with open(built_path, 'rb') as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-")
        print(f"[Pass] Nature PDF Compilation Verified: {built_path}")


if __name__ == '__main__':
    unittest.main()
