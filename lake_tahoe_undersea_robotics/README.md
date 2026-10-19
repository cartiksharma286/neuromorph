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

## Advanced NMR Spectroscopy & Robotic SLAM for E. coli Water Analysis

Open the dedicated tab directly or use **🧲 Advanced NMR E. coli Spectroscopy & SLAM**
in the Lake Ontario Qubit & 10-Yr Recovery tab. Editable assumed concentrations
produce high-resolution 1H-NMR chemical shift spectra (0.0 to 10.0 ppm) characterizing
key coliform metabolic byproducts: acetate (1.92 ppm singlet), lactate (1.33 ppm doublet),
succinate (2.40 ppm singlet), formate (8.45 ppm singlet), peptidoglycan envelope fragments
(2.05 ppm broad), and presaturated water solvent suppression notch (4.70 ppm).

In lieu of quantum machine learning (QML), the platform applies physical NMR
relaxation editing and regularized chemometrics:
- **CPMG Spin-Echo T2 Relaxation Editing**: `I(2n tau) = I_0 exp(-2n tau / T_2)`,
  differentiating low-molecular-weight metabolites (`T_2 ~ 220 ms`) from rapidly
  decaying macromolecular background (`T_2 ~ 28 ms`).
- **Diffusion-Ordered Spectroscopy (DOSY)**: Stejskal-Tanner pulsed field gradient equation
  `ln[I(g)/I(0)] = -gamma^2 g^2 delta^2 (Delta - delta/3) D`, separating high-molecular-weight
  cellular aggregates (`D ~ 1.2 * 10^-12 m^2/s`) from freely diffusing metabolites (`D ~ 0.85 * 10^-9 m^2/s`).
- **Tikhonov-Regularized Spectral Deconvolution & Ridge Chemometrics**:
  `c_hat = (Phi^T Phi + lambda I)^-1 Phi^T s`, validated across 180 training and
  60 held-out synthetic water samples.

The 72-hour sample trajectories model in-vitro bacterial kinetics with scenario bounds;
they are **not clinical or patient trajectories**. Concentrations are inputs, not
inferred spectral measurements. The ten-year catchment source-control scenario
models watershed remediation; monitoring does not itself remediate water.
Joint linear Gaussian planar SLAM estimates the robot position and three unknown
landmarks with known associations, synthetic relative sonar observations and
Joseph-form covariance updates.

`GET /api/ontario/ecoli/defaults` supplies inputs and bounds.
`POST /api/ontario/ecoli/analyze` accepts a JSON object of input overrides;
`POST /api/ontario/ecoli/preprint` returns the matching Nature-style NMR research PDF.
The PDF button uses the last successfully analyzed inputs and is disabled after
edits until analysis is rerun. The independent, non-peer-reviewed PDF includes
finite Bloch equations, Free Induction Decay Fourier transform, Lorentzian line shapes,
CPMG/DOSY formulas, chemometric equations, SLAM updates, vector plots, limitations
and method references. Generate the persistent default-input manuscript from this directory with:

```sh
.venv/bin/python -m backend.ecoli_preprint
```

This is a synthetic research demonstrator, **not a validated E. coli assay or a
water-safety assessment**. No patient or field data are used. Confirmation requires
accredited microbiology/chemistry tests (e.g. US EPA Method 1603 or ISO 9308) and
independent external validation.

## Underwater Robot Camera & Vision System: QML & Elliptic Moduli

Open the dedicated tab **👁️ Underwater Vision & QML Elliptic** for real-time
benthic perception, optical radiative transfer modeling, and object/event detection.
Features include:
- **Jaffe-McGlamery Optical Transfer Model**: Decomposes total sensor radiance into direct
  beam attenuation $E_d = E_0 e^{-c r} \rho(x,y)$, forward scatter point spread convolution
  $E_{fs} = E_d * g_{fs}$, and veiling backscatter $E_{bs} = B_\infty (1 - e^{-c r})$.
- **Statistical Scattering & Anomaly Distributions**:
  - Two-parameter Weibull speckle intensity distribution $f(I; k_w, \lambda_w)$ capturing particulate marine snow.
  - Generalized Extreme Value (GEV) distribution for optical anomaly burst detection (e.g. bubble plumes, laser strobe pings, bioluminescent bursts).
- **Elliptic Moduli Conformal Invariant Theory**:
  - Closed object boundary contours $\partial \Omega$ mapped to Riemannian 2-tori $\mathbb{C} / (\mathbb{Z} + \tau \mathbb{Z})$.
  - Complete elliptic integrals of the first kind $K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - k^2 \sin^2\theta}}$ and complementary $K'(k)$.
  - Complex modular parameter $\tau = i \frac{K'(k)}{K(k)} \in \mathbb{H}$.
  - Klein $j$-invariant $j(k) = 256 \frac{(1 - k^2 + k^4)^3}{k^4 (1 - k^2)^2}$, invariant under scale, rotation, uniform turbidity contrast degradation, and affine perspective distortion.
- **Variational Quantum Machine Learning (QML)**:
  - 4-qubit Parameterized Quantum Circuit (PQC) Ansatz optimized via Quantum Natural Gradient (QNG) using the Fubini-Study metric tensor $g_{ij}$.
  - Real-time classification confidence and quantum state fidelity computation.
- **Interactive Simulated Viewfinder**:
  - 640x360 HTML5 canvas with realistic depth-dependent water color grading, turbidity-dependent veiling haze, marine snow speckle, targeted object bounding reticles, and real-time HUD telemetry.
- **Nature Vision Preprint**:
  - Generates [Nature_Underwater_Robot_Quantum_Vision_Elliptic_Preprint.pdf](/Users/cartiksharma/Downloads/neuromorph-main-10/lake_tahoe_undersea_robotics/Nature_Underwater_Robot_Quantum_Vision_Elliptic_Preprint.pdf) with finite math derivations, figures, and limitations:
  ```sh
  .venv/bin/python -m backend.vision_preprint
  ```

## Benthic Mineral Ore Excavation: QML Fermionic Parity & Revenue Projections

Open the dedicated tab **⛏️ Benthic Mineral Ore & QML Parity** for autonomous perception,
elemental ore grade assessment, and 10-year discounted cash flow modeling across five critical
sub-lacustrine commodities:
- **Strategic Mineral Targets**: Iron (Fe, magnetite/hematite nodules), Uranium (U, uraninite/reducing sapropel),
  Rare Earth Elements (REE, Nd/Dy/Eu monazite/xenotime placers), Silver (Ag, electrum/argentite veins),
  and Cobalt (Co, ferromanganese crusts).
- **Hyperspectral Optical Reflectance Spectroscopy (400 - 1000 nm)**:
  - Resolves diagnostic crystal field and charge-transfer absorption bands (Fe: 890 nm dip, U: 530 nm uranyl vibronic,
    REE: 520/580/740/800 nm 4f-4f multiplets, Ag: specular metallic reflectance rise, Co: 590 nm d-d ligand field).
- **Quantum Machine Learning with Jordan-Wigner Fermionic Parity**:
  - 5-qubit quantum register encoding valence electron configurations through Jordan-Wigner transformation
    $c_j^\dagger = (\bigotimes_{k<j} Z_k) \frac{X_j - i Y_j}{2}$.
  - Parity operator $P = \bigotimes_{j=1}^5 Z_j = Z_1 Z_2 Z_3 Z_4 Z_5$.
  - Particle-conserving Givens rotations maintain fermionic parity invariance, yielding expectation $\langle P \rangle \in [-1, +1]$
    as an intrinsic signature of transition metal and actinide oxidation states.
- **Ten-Year Discounted Cash Flow (DCF) & Multi-Commodity Economics**:
  - Annual recovered metal production $M_m = R \cdot D \cdot g_m \cdot \eta_m$.
  - Market benchmark pricing (Fe $125/t, U $185/kg, REE $95/kg, Ag $1.05/g, Co $34,500/t).
  - Operating expenses, mandatory 8% environmental remediation endowment levy, 10-year NPV, IRR, and payback period.
- **Nature Mineral Preprint**:
  - Generates [Nature_Benthic_Mineral_Excavation_QML_Parity_Preprint.pdf](/Users/cartiksharma/Downloads/neuromorph-main-10/lake_tahoe_undersea_robotics/Nature_Benthic_Mineral_Excavation_QML_Parity_Preprint.pdf):
  ```sh
  .venv/bin/python -m backend.mineral_preprint
  ```

## Focused validation

```sh
python -m unittest discover -s tests -v
```
