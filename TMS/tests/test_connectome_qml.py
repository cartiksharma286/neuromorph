"""Unit Tests for Human Cortical Connectome, Modular Harmonic Forms, and QML."""

import os
import unittest
from pathlib import Path

from logic.connectome_qml import (
    build_parcellation,
    healthy_connectome,
    dementia_connectome,
    ramanujan_tau,
    modular_forms_at,
    modular_harmonics,
    train_qml_kernel,
    simulate_connectome_qml,
)
from app import app
from generate_nature_connectome_preprint import build_pdf
import numpy as np


class TestConnectomeQml(unittest.TestCase):
    def test_parcellation_and_connectomes(self):
        parc = build_parcellation()
        self.assertEqual(len(parc["names"]), 68)
        self.assertEqual(parc["coords"].shape, (68, 3))

        Wh = healthy_connectome(parc)
        self.assertEqual(Wh.shape, (68, 68))
        self.assertTrue(np.allclose(Wh, Wh.T))
        self.assertEqual(np.diag(Wh).sum(), 0.0)

        Wd, atrophy = dementia_connectome(Wh, parc, severity=0.6)
        self.assertEqual(Wd.shape, (68, 68))
        self.assertTrue(np.all(Wd <= Wh + 1e-12))
        self.assertTrue(np.all(atrophy >= 0.0))
        self.assertGreater(atrophy[parc["braak"] <= 2].mean(), atrophy[parc["braak"] >= 5].mean())

    def test_modular_forms_and_harmonics(self):
        tau_coeffs = ramanujan_tau(68)
        # Verify first known Ramanujan tau values: tau(1)=1, tau(2)=-24, tau(3)=252, tau(4)=-1472
        self.assertEqual(tau_coeffs[1], 1)
        self.assertEqual(tau_coeffs[2], -24)
        self.assertEqual(tau_coeffs[3], 252)
        self.assertEqual(tau_coeffs[4], -1472)

        mf = modular_forms_at(complex(0.25, 0.85))
        self.assertIn("nome_abs", mf)
        self.assertGreater(mf["nome_abs"], 0.0)
        self.assertLess(mf["nome_abs"], 1.0)
        self.assertLess(mf["delta_product_residual"], 1e-10)

        parc = build_parcellation()
        Wh = healthy_connectome(parc)
        deg = Wh.sum(axis=1)
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        L = np.eye(68) - inv_sqrt[:, None] * Wh * inv_sqrt[None, :]

        tau_hat = np.array([0.0] + [t / (k ** 5.5) for k, t in enumerate(tau_coeffs[1:68], start=1)])
        H = modular_harmonics(L, complex(0.25, 0.85), tau_hat)
        self.assertEqual(H["K"].shape, (68, 68))
        self.assertTrue(np.allclose(H["K"], H["K"].T))
        self.assertGreater(H["energy"], 0.0)

    def test_qml_kernel_training(self):
        features = np.random.uniform(0.0, 1.0, (12, 6))
        target = np.random.uniform(0.0, 1.0, (12, 12))
        target = 0.5 * (target + target.T)
        res = train_qml_kernel(features, target, layers=2, iters=15)
        self.assertEqual(res["K"].shape, (12, 12))
        self.assertTrue(np.allclose(res["K"], res["K"].T))
        self.assertGreaterEqual(res["alignment_final"], res["alignment_initial"] - 0.05)

    def test_full_simulation_pipeline(self):
        res = simulate_connectome_qml(severity=0.5, qml_layers=2, qml_iters=20)
        self.assertEqual(res["status"], "success")
        self.assertIn("nodes", res)
        self.assertIn("matrices", res)
        self.assertIn("metrics", res)
        self.assertIn("modular_forms", res)
        self.assertIn("qml_training", res)
        self.assertEqual(len(res["nodes"]), 68)

        # Invariants sanity
        mh = res["metrics"]["healthy"]
        md = res["metrics"]["dementia"]
        self.assertGreaterEqual(mh["edges"], md["edges"])
        self.assertGreaterEqual(mh["mean_strength"], md["mean_strength"])
        self.assertGreaterEqual(mh["global_efficiency"], md["global_efficiency"])

    def test_flask_endpoints(self):
        client = app.test_client()

        # POST /api/connectome-qml
        post_resp = client.post(
            "/api/connectome-qml",
            json={
                "severity": 0.4,
                "tau_x": 0.25,
                "tau_y": 0.85,
                "qml_layers": 2,
                "qml_iters": 20
            }
        )
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.get_json()
        self.assertEqual(post_data["status"], "success")
        self.assertIn("matrices", post_data)

        # GET /api/connectome-nature-preprint
        pdf_resp = client.get("/api/connectome-nature-preprint")
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertEqual(pdf_resp.mimetype, "application/pdf")
        self.assertGreater(len(pdf_resp.data), 40000)

    def test_preprint_pdf_generation(self):
        test_pdf = Path(__file__).parent / "test_connectome_preprint.pdf"
        try:
            out = build_pdf(str(test_pdf), sim_params={"severity": 0.5, "qml_layers": 2, "qml_iters": 20})
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 40000)
        finally:
            if test_pdf.exists():
                test_pdf.unlink()


if __name__ == "__main__":
    unittest.main()
