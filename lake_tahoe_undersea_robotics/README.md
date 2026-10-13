# Underwater robotics scenario dashboard

Run from this directory with the repository's Python environment:

```sh
python -m pip install -r requirements.txt
FLASK_RUN_HOST=127.0.0.1 FLASK_RUN_PORT=5070 python app.py
```

## Cost Economics & Carbon Viability

The ten-year tab exposes editable CAD, kWh and kg CO2e inputs. Defaults are
illustrative, not measured costs or a verified lifecycle assessment. Replace them
with sourced project data. Additional annual benefits must not duplicate savings
from the avoided baseline cost. Baseline capital/embodied assets are assumed sunk.
Transport, replacements, disposal, salvage and inflation are excluded unless
represented in the entered aggregates. QML/cryogenic power is a shared facility
load during the entered operating hours; zero QML energy savings is the default.

The app separates financial NPV, carbon shadow NPV (not carbon-credit revenue),
embodied plus operational footprint, carbon break-even and capital sensitivity.
Joint viability means nonnegative financial NPV and nonnegative avoided ten-year
emissions under the entered assumptions, not demonstrated deployment feasibility.

A three-qubit product-state fidelity kernel is calculated classically and trained
on six synthetic points. Its output does not drive the forecast or demonstrate
quantum advantage. Prime residues are scheduling labels, not optimal geography.

The PDF button generates an independent Nature-style manuscript for **current
inputs**, with finite equations, results, plots and limitations. It is not a
Nature publication. Generate the persistent default-input PDF separately with:

```sh
python -m backend.carbon_preprint
```

## Post-Quantum Report Verification

The application uses `pqcrypto==1.0.0` for real ML-DSA-65 report signatures and
ML-KEM-768 functional tests. It imports only arithmetic helpers from the sibling
`../quantum_privacy_cloak/number_theoretic_invisibility.py`; that repository's
simulated cryptographic/security scores are not trusted as verification evidence.
Keep the sibling directory available.

The tab signs RFC 8785 canonical JSON with a domain prefix and verifies against the
server's process-pinned key. It tests altered report rejection, shared-secret
agreement and altered-ciphertext secret mismatch. JSON envelopes are downloadable.
Numeric formatting is normalized across Python/browser JSON round-trips.
Private keys stay in memory and are never returned. Restarting invalidates old
reports; no durable trust anchor or external identity is supplied.

This is a local functional demonstrator, **not a security audit, FIPS-certified
module, encrypted transport, authentication system or replay defense**. HTTP
remains unencrypted. The KEM test does not encrypt reports and the PDF is not
digitally signed. Do not use this server for sensitive or production telemetry.

## Focused validation

```sh
python -m unittest discover -s tests -v
```
