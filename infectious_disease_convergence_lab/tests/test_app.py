import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app, build_health_economics, build_long_term_horizon, build_simulation


class ConvergenceLabTests(unittest.TestCase):
    def test_simulation_is_normalized_and_finite(self):
        payload = build_simulation({'transmission': 1.7, 'intervention': .55, 'diffusion': .7})
        self.assertEqual(len(payload['trajectory']), 42)
        self.assertEqual(len(payload['convergence']), 42)
        for point in payload['trajectory']:
            self.assertAlmostEqual(sum(point[key] for key in ('susceptible', 'exposed', 'infectious', 'recovered')), 100, delta=.02)
            self.assertTrue(all(math.isfinite(point[key]) for key in ('susceptible', 'exposed', 'infectious', 'recovered')))

    def test_health_economics_projections(self):
        econ = build_health_economics({
            'cohort_size': 300000,
            'horizon_years': 6,
            'intervention_cost': 400.0,
            'hospitalization_cost': 20000.0,
            'learning_regularizer': 0.3,
        })
        self.assertEqual(len(econ['annual_projections']), 6)
        headline = econ['headline']
        self.assertGreater(headline['cumulative_gross_revenue_m'], 0)
        self.assertGreater(headline['net_present_value_m'], 0)
        self.assertGreater(headline['statistical_r2'], 0.8)
        self.assertGreater(headline['icer_per_qaly'], 0)
        for row in econ['annual_projections']:
            self.assertLessEqual(row['gross_lower_m'], row['gross_revenue_m'])
            self.assertGreaterEqual(row['gross_upper_m'], row['gross_revenue_m'])
            self.assertTrue(math.isfinite(row['net_margin_m']))

    def test_long_term_horizon(self):
        horizon = build_long_term_horizon(transmission=1.4, intervention=0.5, repair_capacity=0.6, phase_shift=1.0, horizon_years=5)
        self.assertEqual(len(horizon), 60)
        for pt in horizon:
            self.assertTrue(0 <= pt['susceptible'] <= 100)
            self.assertTrue(0 <= pt['infectious'] <= 100)
            self.assertTrue(math.isfinite(pt['effective_r']))

    def test_simulation_endpoint(self):
        client = app.test_client()
        response = client.post('/api/simulate', json={'intervention': .9, 'cohort_size': 200000})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('finite_math', data)
        self.assertIn('health_economics', data)
        self.assertIn('long_term_horizon', data)

    def test_health_economics_endpoint(self):
        client = app.test_client()
        response = client.post('/api/health-economics', json={'cohort_size': 150000, 'horizon_years': 4})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('statistical_model', data)
        self.assertEqual(len(data['annual_projections']), 4)

    def test_preprint_endpoints(self):
        client = app.test_client()
        res1 = client.get('/preprint.pdf')
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.mimetype, 'application/pdf')
        res2 = client.get('/preprint-longterm.pdf')
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.mimetype, 'application/pdf')


if __name__ == '__main__':
    unittest.main()
