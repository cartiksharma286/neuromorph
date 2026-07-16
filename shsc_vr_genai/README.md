# Sunnybrook Health Sciences Centre - VR GenAI Platform Simulation (shsc_vr_genai)

A highly immersive, unified Virtual Reality Surgical Simulation suite styled for the Sunnybrook Health Sciences Centre. Developed in Flask, powered by hybrid statistical neural model frameworks, and combined with custom real-time animations, Plotly analytics, and Visual-Language reasoning engines.

---

## 🔬 Operational Modules

### 1. 🧠 Neurosurgery VR Classroom
Emulates subcortical neurosurgical targeting pathways, including Deep Brain Stimulation (DBS) target entry and ventricular catheter localization:
- ** Tremor Wiener Model**: High haptic tremor and physical grip drift modeled dynamically via a Wiener drift stochastic equation:
  $$dx_t = -\gamma (x_t - x_{\text{target}}) dt + \sigma dW_t$$
  where $\gamma$ is targeting skill density and $\sigma$ is physiological hand tremors.
- **Biomechanical Force Metrics**: Calculations of brain parenchyma shear stress ($\text{kPa}$) and torque moments based on angle deviations.

### 2. 🦴 Orthopedic Surgery VR Suite
Emulates transpedicular corridor drilling for L4-L5 screws and femoral nailing:
- **Bone Penetration Density Profile**: Torque resistant curves mapped over dense outer cortical layers, transition portals, and cancellous/osteoporotic segments.
- **Accident Prevention Flags**: Live breach alarms if the angulation exceeds bottlenecks, warning of medial spinal canal breaches or lateral cortical fissures.

### 3. 👁️ VLM Visual Reasoning & Attention Maps
Models a high-dimensional Vision-Language-Action Multi-modal reasoner:
- **Attention Overlays**: Generates coordinates-focused statistical attention concentric circular bounds mimicking attention heads inside VLM convolutional layers.
- **Multimodal Feedback**: Dynamic clinical prescriptions assessing anatomical boundaries, safety margins, and structural risk zones.

### 4. 📈 Educational Analytics Dashboard
A comprehensive cognitive evaluation interface tracking cohort progress:
- **Learning Convergence Profile**: Mapped via the Power Law of Practice with exponential trial gains:
  $$SK(t) = SK_0 + (SK_{\infty} - SK_0)(1 - e^{-\alpha \cdot t})$$
- **Retention Curve Decay**: Incorporates Ebbinghaus Ejection memory loss model during periods of non-retention.
- **Cognitive IMM Flow state modeling**: Transition states of mental load simulated as a 3-State Hidden Markov Process:
  $$X_t = [P_{\text{nominal}}, P_{\text{focused}}, P_{\text{overload}}]$$
- **Wald SPRT Decision Matrix (Stopped Cohorts)**: Statistical confirmation boundary modeling for certifying when surgical residents meet qualifications:
  $$S_k \ge \ln \left(\frac{1-\beta}{\alpha}\right)$$

---

## ⚡ Setup & Launch Instructions

Start the server using the workspace virtual environment:
```bash
./.venv/bin/python shsc_vr_genai/app.py
```
By default, the server launches on port `8200`. Access the interactive suite via your local web browser at `http://127.0.0.1:8200/`.
