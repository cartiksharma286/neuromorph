import unittest
from logic.addiction_recovery_tms import (
    simulate_addiction_recovery_trajectories,
    simulate_hebbian_rewiring,
    simulate_qml_addiction_policy,
    compute_stieltjes_craving_fraction
)

class TestAddictionRecoveryTMS(unittest.TestCase):
    def test_01_hebbian_rewiring(self):
        res = simulate_hebbian_rewiring(sessions=30, intensity_mso=110.0)
        self.assertIn("hebbian_control_gain", res)
        self.assertGreater(res["final_ltp_potentiation"], 1.0)
        self.assertGreater(res["final_ltd_suppression_pct"], 30.0)
        print(f"[Pass] Hebbian Plasticity Rewiring Verified: LTP={res['final_ltp_potentiation']}x, LTD={res['final_ltd_suppression_pct']}%")

    def test_02_qml_policy_optimization(self):
        res = simulate_qml_addiction_policy(baseline_ftnd_or_audit=8.0, addiction_type='nicotine')
        self.assertIn("quantum_fidelity", res)
        self.assertGreater(res["optimal_fidelity"], 0.85)
        self.assertIn("selected_protocol", res)
        print(f"[Pass] QML Variational Policy Verified: Fidelity={res['optimal_fidelity']}")

    def test_03_full_36mo_trajectories(self):
        res = simulate_addiction_recovery_trajectories(
            addiction_type='nicotine',
            baseline_severity=8.0,
            sessions=30,
            intensity_mso_pct=110.0
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("trajectories", res)
        self.assertIn("summary_kpis", res)
        
        kpi = res["summary_kpis"]
        self.assertGreater(kpi["month12_abstinence_rate_pct"], kpi["standard_month12_abstinence_pct"])
        self.assertGreater(kpi["craving_reduction_pct"], 50.0)
        print(f"[Pass] 36-Month Longitudinal Trajectories Verified: 12-Mo Abstinence={kpi['month12_abstinence_rate_pct']}% vs Sham={kpi['sham_month12_abstinence_pct']}%")

    def test_04_alcohol_use_disorder(self):
        res = simulate_addiction_recovery_trajectories(
            addiction_type='alcohol',
            baseline_severity=26.0,
            sessions=30
        )
        self.assertEqual(res["consumption_unit"], "Standard Drinks / Week")
        self.assertGreater(res["summary_kpis"]["month12_abstinence_rate_pct"], 70.0)
        print(f"[Pass] AUD Trajectories Verified: Unit={res['consumption_unit']}")

    def test_05_quantum_agentic_ai(self):
        res = simulate_addiction_recovery_trajectories(
            addiction_type='nicotine',
            baseline_severity=8.0,
            sessions=30
        )
        self.assertIn("quantum_agentic_ai", res)
        agent = res["quantum_agentic_ai"]
        self.assertIn("von_neumann_entropy", agent)
        self.assertIn("policy_reward", agent)
        self.assertGreater(agent["initial_entropy"], agent["final_entropy"])
        self.assertGreater(agent["entropy_reduction_pct"], 60.0)
        self.assertEqual(len(agent["von_neumann_entropy"]), 30)
        print(f"[Pass] Quantum Agentic AI Verified: S(rho_0)={agent['initial_entropy']} -> S(rho_f)={agent['final_entropy']} (-{agent['entropy_reduction_pct']}%)")

    def test_06_health_economics_revenue_distributions(self):
        res = simulate_addiction_recovery_trajectories(
            addiction_type='nicotine',
            baseline_severity=8.0,
            sessions=30
        )
        self.assertIn("health_economics", res)
        self.assertIn("revenue_distributions", res)
        he = res["health_economics"]
        rev = res["revenue_distributions"]

        # Health Economics checks
        self.assertGreater(he["qaly_gained_superoptimal"], he["qaly_gained_standard"])
        self.assertGreater(he["medical_savings_per_patient_usd"], 5000.0)
        self.assertLess(he["icer_usd_per_qaly"], 0.0) # Dominant cost-saving

        # Statistical Revenue Distribution checks
        self.assertLess(rev["p10_annual_revenue_usd_m"], rev["p50_annual_revenue_usd_m"])
        self.assertLess(rev["p50_annual_revenue_usd_m"], rev["p90_annual_revenue_usd_m"])
        self.assertGreater(rev["benefit_cost_ratio"], 1.5)
        self.assertLess(rev["cumulative_npv_p10_m"][-1], rev["cumulative_npv_p50_m"][-1])
        self.assertLess(rev["cumulative_npv_p50_m"][-1], rev["cumulative_npv_p90_m"][-1])
        print(f"[Pass] Health Economics & Revenue Distributions Verified: QALY={he['qaly_gained_superoptimal']}, ICER={he['icer_usd_per_qaly']}/QALY, P50 Rev=${rev['p50_annual_revenue_usd_m']}M, 5-Yr NPV=${rev['five_year_npv_p50_usd_m']}M")

if __name__ == '__main__':
    unittest.main()
