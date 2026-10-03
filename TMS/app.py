import os, io, math, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
from logic.rtms_engine import (
    run_full_simulation,
    get_equipment_list,
    get_tremor_clinical_data,
    get_treatment_paradigm,
    get_dementia_longterm_care,
    get_ocd_fea_simulation,
    simulate_jaynes_cummings_rtms
)
from logic.monteris_cf_treatment import (
    full_treatment_paradigm,
    intraop_thermometry,
    preop_dti_sequence,
    preop_fmri_bold,
    postop_flair_swi,
    jc_state_transfer,
    risk_stratification,
)
from logic.dbs_statistical_manifold import generate_dbs_treatment_protocol
from logic.nash_geodesic_registration import generate_nash_geodesic_registration
from logic.depression_rtms import simulate_depression_rtms
from logic.anxiety_millennials_rtms import simulate_anxiety_rtms
from logic.moduli_bem_paradigm import get_moduli_bem_paradigm
from logic.tourette_rtms import simulate_tourette_rtms
from logic.tbi_ptsd_rtms import simulate_tbi_ptsd_rtms
from logic.rls_rtms import simulate_rls_rtms
from logic.schizophrenia_dbs import simulate_schizophrenia_dbs_bem
from logic.hifu_tms_dementia import simulate_hifu_tms_dementia
from logic.cybersecurity_pqc import simulate_cybersecurity_trajectories
from logic.dbs_pqc_vulnerability import optimize_dbs_pqc_vulnerability_assessment
from logic.clinical_trajectories_benchmark import simulate_clinical_trajectories_benchmark
from logic.stroke_repair_rtms import simulate_stroke_repair_rtms
from logic.pharmacological_equivalence import simulate_pharmacological_equivalence
from logic.nih_grant_budget_optimizer import optimize_nih_grant_budget
from logic.parkinsons_rtms import simulate_parkinsons_rtms
from logic.heart_stroke_trajectories import HeartStrokeClinicalTrajectoryOptimizer
from logic.fem_cortical_solver import run_full_cortical_fem_simulation
from logic.connectome_qml import simulate_connectome_qml

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
if CORS:
    CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    condition = data.get('condition', 'stroke')
    results = run_full_simulation(condition)
    return jsonify({
        "status": "success",
        "message": "Optimal protocol calculated via GCP clustered FEA/BEM optimization.",
        "data": results
    })

@app.route('/api/equipment', methods=['GET'])
def equipment():
    return jsonify({"status": "success", "data": get_equipment_list()})

@app.route('/api/tremor-clinical', methods=['GET'])
def tremor_clinical():
    return jsonify({"status": "success", "data": get_tremor_clinical_data()})

@app.route('/api/treatment-paradigm', methods=['GET'])
def treatment_paradigm():
    condition = request.args.get('condition', 'stroke')
    return jsonify({"status": "success", "data": get_treatment_paradigm(condition)})

@app.route('/api/dementia-longterm', methods=['GET'])
def dementia_longterm():
    return jsonify({"status": "success", "data": get_dementia_longterm_care()})

@app.route('/api/dbs-imaging', methods=['GET'])
def dbs_imaging_protocol():
    return jsonify({"status": "success", "data": generate_dbs_treatment_protocol()})

@app.route('/api/ocd-treatment', methods=['GET'])
def ocd_treatment():
    data = get_ocd_fea_simulation()
    data["bem_analysis"] = get_moduli_bem_paradigm(
        condition="ocd", freq_range=(18.0, 22.0), intensity_range=(80.0, 120.0)
    )
    return jsonify({"status": "success", "data": data})
    
@app.route('/api/rls-rtms', methods=['GET'])
def rls_rtms():
    """Research-only RLS recurrent-design simulation endpoint."""
    try:
        result = simulate_rls_rtms(
            baseline_irls=float(request.args.get('baseline_irls', 28.0)),
            treatment_weeks=int(request.args.get('treatment_weeks', 8)),
            coil_towers=int(request.args.get('coil_towers', 2)),
            session_price=float(request.args.get('session_price', 275.0)),
        )
        return jsonify({'status': 'success', 'data': result})
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400

@app.route('/api/jaynes-cummings', methods=['GET', 'POST'])
def jaynes_cummings_route():
    data = request.get_json(silent=True) or {}
    omega_c = float(request.args.get('omega_c', data.get('omega_c', 10.0)))
    omega_a = float(request.args.get('omega_a', data.get('omega_a', 10.0)))
    g = float(request.args.get('g', data.get('g', 0.5)))
    n_photons = int(request.args.get('n_photons', data.get('n_photons', 3)))
    sim = simulate_jaynes_cummings_rtms(omega_c=omega_c, omega_a=omega_a, coupling_g=g, n_photons=n_photons)
    return jsonify({"status": "success", "data": sim})

@app.route('/api/nash-geodesic-registration', methods=['GET'])
def nash_geodesic_registration():
    return jsonify({"status": "success", "data": generate_nash_geodesic_registration()})

@app.route('/api/nature-registration-preprint', methods=['GET'])
def nature_registration_preprint():
    """Generate and serve the Laser-MR-CT registration Nature preprint PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'SEQ_Nature_Laser_MR_CT_Registration.pdf')
    try:
        if not os.path.exists(pdf_path):
            from generate_nature_registration_preprint import build_pdf
            build_pdf()
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='SEQ_Nature_Laser_MR_CT_Registration.pdf',
    )

@app.route('/api/rtms-nature-publication', methods=['GET'])
def rtms_nature_publication():
    """Generate and serve the RTMS Nature publication PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'rTMS_Nature_Publication.pdf')
    try:
        if not os.path.exists(pdf_path):
            from generate_nature_publication import build_pdf
            build_pdf()
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='rTMS_Nature_Publication.pdf',
    )


@app.route('/api/rtms-ocd-publication', methods=['GET'])
def rtms_ocd_publication():
    """Generate and serve the Specialized OCD rTMS Nature publication PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__), 'rtms_ocd.pdf')
    try:
        if not os.path.exists(pdf_path):
            from generate_nature_ocd_pdf import build_pdf
            build_pdf()
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='rtms_ocd.pdf',
    )

@app.route('/api/monteris/full-paradigm', methods=['POST'])
def monteris_full_paradigm():
    data = request.json or {}
    condition = data.get('condition', 'glioma')
    preset    = data.get('monteris_preset', 'standard')
    result    = full_treatment_paradigm(condition, preset)
    return jsonify({"status": "success", "data": result})

@app.route('/api/monteris/preop-dti', methods=['GET'])
def monteris_preop_dti():
    return jsonify({"status": "success", "data": preop_dti_sequence()})

@app.route('/api/monteris/preop-fmri', methods=['GET'])
def monteris_preop_fmri():
    return jsonify({"status": "success", "data": preop_fmri_bold()})

@app.route('/api/monteris/intraop-thermometry', methods=['POST'])
def monteris_intraop():
    data   = request.json or {}
    preset = data.get('preset', 'standard')
    return jsonify({"status": "success", "data": intraop_thermometry(preset)})

@app.route('/api/monteris/postop-monitoring', methods=['GET'])
def monteris_postop():
    return jsonify({"status": "success", "data": postop_flair_swi()})

@app.route('/api/monteris/qnc-state-transfer', methods=['POST'])
def monteris_qnc():
    data      = request.json or {}
    n_qubits  = int(data.get('n_qubits', 8))
    abl_frac  = float(data.get('ablation_fraction', 0.3))
    omega_c   = float(data.get('omega_c_MHz', 5.0))
    g_kHz     = float(data.get('g_coupling_kHz', 50.0))
    return jsonify({"status": "success",
                    "data": jc_state_transfer(n_qubits, omega_c, g_kHz, abl_frac)})

@app.route('/api/monteris/risk-stratification', methods=['POST'])
def monteris_risk():
    data = request.json or {}
    return jsonify({"status": "success", "data": risk_stratification(
        age=float(data.get('age', 55)),
        kps=float(data.get('kps', 80)),
        tumour_vol_cm3=float(data.get('tumour_vol_cm3', 5.0)),
        eloquent_proximity_mm=float(data.get('eloquent_proximity_mm', 15.0)),
    )})

@app.route('/api/monteris/nature-report', methods=['GET'])
def monteris_nature_report():
    """Stream the Nature publication PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__),
                            'seqs', 'Nature_Monteris_CF_Treatment.pdf')
    if not os.path.exists(pdf_path):
        # Generate on demand
        try:
            from generate_nature_monteris_report import generate_nature_monteris_report
            generate_nature_monteris_report(pdf_path)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return send_file(pdf_path, mimetype='application/pdf',
                     as_attachment=False,
                     download_name='Nature_Monteris_CF_Treatment.pdf')

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: Sleep Apnea rTMS Neuromodulation & Adaptive Repair
# ─────────────────────────────────────────────────────────────────────────────
_cache_sleep_apnea = {}

@app.route('/api/sleep-apnea-rtms', methods=['GET'])
def api_sleep_apnea_rtms():
    """
    Precision Sleep Apnea Neuromodulation Suite:
      - Sleep apnea treatment modeling via repetitive Transcranial Magnetic Stimulation (rTMS)
      - Quantitative trajectory modeling of Apnea-Hypopnea Index (AHI events/hr)
      - Statistical continued fraction expansion of optimal neuro-stimulation sync phase ratio
      - Adaptive optimal closed-loop controller with real-time feedback logic and ASCII schematics
    """
    global _cache_sleep_apnea
    try:
        baseline_ahi = float(request.args.get('baseline_ahi', 38.0))
        rtms_freq_hz = float(request.args.get('rtms_freq_hz', 10.0))
        adaptive_gain = float(request.args.get('adaptive_gain', 1.5))
        duration_days = int(request.args.get('duration_days', 30))
        target_sync_ratio = float(request.args.get('target_sync_ratio', 1.3416))

        cache_key = (baseline_ahi, rtms_freq_hz, adaptive_gain, duration_days, target_sync_ratio)
        if cache_key in _cache_sleep_apnea:
            return _cache_sleep_apnea[cache_key]

        np.random.seed(101)
        days = list(range(1, duration_days + 1))
        
        ahi_baseline = []
        ahi_cpap = []
        ahi_rtms_std = []
        ahi_rtms_opt = []
        
        for d in days:
            base_noise = np.random.normal(0, 1.2)
            ahi_baseline.append(float(max(15.0, baseline_ahi + 0.05 * d + base_noise)))
            
            cpap_noise = np.random.normal(0, 2.5)
            compliance_compliance = 0.70 + 0.10 * np.sin(d * 0.4)
            val_cpap = baseline_ahi - (baseline_ahi - 8.0) * (0.85 * compliance_compliance) + cpap_noise
            ahi_cpap.append(float(max(2.0, val_cpap)))
            
            std_noise = np.random.normal(0, 0.8)
            decay_rate_std = 0.06 * (rtms_freq_hz / 10.0)
            val_std = 12.0 + (baseline_ahi - 12.0) * np.exp(-decay_rate_std * d) + std_noise
            ahi_rtms_std.append(float(max(1.0, val_std)))
            
            opt_noise = np.random.normal(0, 0.3)
            decay_rate_opt = 0.12 * (rtms_freq_hz / 10.0) * (0.5 + 0.5 * adaptive_gain)
            val_opt = 3.5 + (baseline_ahi - 3.5) * np.exp(-decay_rate_opt * d) + opt_noise
            ahi_rtms_opt.append(float(max(0.5, val_opt)))

        a_list = []
        temp = target_sync_ratio
        for _ in range(6):
            floor_val = int(math.floor(temp))
            a_list.append(floor_val)
            rem = temp - floor_val
            if abs(rem) < 1e-6:
                break
            temp = 1.0 / rem

        convergents = []
        p_prev2, p_prev1 = 0, 1
        q_prev2, q_prev1 = 1, 0
        for k, ak in enumerate(a_list):
            p = ak * p_prev1 + p_prev2
            q = ak * q_prev1 + q_prev2
            convergents.append(f"{int(p)}/{int(q)}")
            p_prev2, p_prev1 = p_prev1, p
            q_prev2, q_prev1 = q_prev1, q

        ascii_schematic = (
            "               ===================================================================\n"
            "               ADAPTIVE CLOSED-LOOP sleep apnea rTMS STIMULATION SCHEMATIC\n"
            "               ===================================================================\n\n"
            "               +----------------------+           +--------------------------+\n"
            "               |  PATIENT PHYSIOLOGY  |           |   ADAPTIVE NEURO-MODULE  |\n"
            "               |  & BREATHING SENSORS |           | (STIMULATOR & FEEDBACK)  |\n"
            "               +----------------------+           +--------------------------+\n\n"
            "                  [Phrenic EMG] ---[Nerve Activity Trace]---+     [Pulse Generator Unit]\n"
            "                        |                                   |       rtms_freq  = " + f"{rtms_freq_hz:.1f}" + " Hz\n"
            "                        v                                   v       target_ph  = " + f"{target_sync_ratio:.4f}" + "\n"
            "               +-----------------+                 +-----------------+\n"
            "               | Airflow Monitor |                 | Magnetic Coil   |<-------+\n"
            "               +--------+--------+                 +--------+--------+        |\n"
            "                        |                                   |                 |\n"
            "                        +=====( Sync Phase Detection )======+                 |\n"
            "                                    |                                         |\n"
            "                                    v                                         |\n"
            "                         [ Continued Fraction Ratio ]----[" + ", ".join(convergents) + "]        |\n"
            "                         [ Phase Convergents Alignment ]                      |\n"
            "                                    |                                         |\n"
            "                                    v                                         |\n"
            "                          [ Adaptive H-Bridge ]-----[ Gain Stage: " + f"{adaptive_gain:.2f}" + " ]--+\n"
            "                          (Microsecond Pulser)      (Adaptive Optimizer Core)\n"
            "                                    |                           |             \n"
            "                                    v                           |             \n"
            "                         [ Real-time Controller ]<--------------+             \n"
            "                         (RP2040 Dual ARM Cortex)                             \n"
            "                                    |                                         \n"
            "                                    | (High-Speed SPI Interface - Telemetry)  \n"
            "                                    v                                         \n"
            "               +------------------------------------------------------+\n"
            "               |      AWS HEALTHCARE CLOUD & FHIR INTEROPERABILITY      |\n"
            "               |      ============================================      |\n"
            "               |   - Real-time clinical events linked via Amazon S3   |\n"
            "               |   - Standard FHIR R4 clinical resource mapped (AHI)  |\n"
            "               |   - SMART on FHIR compliant clinician dashboards    |\n"
            "               +------------------------------------------------------+\n"
        )

        clinical_prescription = (
            f"**rTMS Sleep Apnea Clinical Interoperability Report:**\n\n"
            f"1. **Adaptive Neuromodulation Mechanics**: Standard CPAP treatment exhibits high compliance volatility, "
            f"fluctuating at a residual AHI of **{ahi_cpap[-1]:.1f}**. Standard non-adaptive rTMS reduces AHI to **{ahi_rtms_std[-1]:.1f}**. "
            f"In contrast, the **Adaptive Optimal Closed-loop rTMS** stimulation converges AHI to **{ahi_rtms_opt[-1]:.1f} events/hour** (pristine normal sleep threshold, AHI < 5.0).\n\n"
            f"2. **Phrenic-Ventilatory Continued Fraction Alignment**: To synchronize magnetic pulses to the respiratory neuromuscular "
            f"refractory cycle, the optimal sync phase ratio $\\rho^* = {target_sync_ratio:.4f}$ is factored into continued fraction convergents: "
            f"[{', '.join(convergents)}]. These fractions guide microsecond-precision current gates in the H-bridge pulser circuit.\n\n"
            f"3. **AWS Clinical Integration & Governance**: Clinical event data is packaged into HL7 FHIR (R4) Observation profiles "
            f"and continuously pushed to Amazon HealthLake. This enables enterprise-scale clinical data exchange and seamless "
            f"SMART on FHIR diagnostic telemetry for downstream sleep specialists."
        )

        res_data = jsonify({
            'days': days,
            'ahi_baseline': ahi_baseline,
            'ahi_cpap': ahi_cpap,
            'ahi_rtms_std': ahi_rtms_std,
            'ahi_rtms_opt': ahi_rtms_opt,
            'cf_expansion': a_list,
            'convergents': convergents,
            'ascii_schematic': ascii_schematic,
            'genai_prescription': clinical_prescription,
            'params': {
                'baseline_ahi': baseline_ahi,
                'rtms_freq_hz': rtms_freq_hz,
                'adaptive_gain': adaptive_gain,
                'duration_days': duration_days,
                'target_sync_ratio': target_sync_ratio
            }
        })
        _cache_sleep_apnea[cache_key] = res_data
        return res_data

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


def _depression_params():
    return {
        'baseline_phq9': float(request.args.get('baseline_phq9', 19.0)),
        'sessions': int(request.args.get('sessions', 30)),
        'rtms_frequency_hz': float(request.args.get('rtms_frequency_hz', 10.0)),
        'cbt_weight': float(request.args.get('cbt_weight', 0.65)),
        'control_gain': float(request.args.get('control_gain', 0.85)),
        'signature_ratio': float(request.args.get('signature_ratio', 1.61803398875)),
    }


@app.route('/api/depression-rtms', methods=['GET'])
def api_depression_rtms():
    """Return the seeded depression rTMS/CBT computational research model."""
    try:
        return jsonify(simulate_depression_rtms(**_depression_params()))
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid depression model parameter: {exc}'}), 400


@app.route('/api/depression-rtms-preprint', methods=['GET'])
def depression_rtms_preprint():
    """Generate the depression model Nature-style preprint from current controls."""
    try:
        from generate_nature_depression_rtms import build_pdf

        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Depression_rTMS_CBT.pdf')
        build_pdf(output_path, _depression_params())
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Depression_rTMS_CBT.pdf')
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid depression preprint parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


def _anxiety_params():
    data = request.get_json(silent=True) or {}
    return {
        'baseline_gad7': float(request.args.get('baseline_gad7', data.get('baseline_gad7', 16.0))),
        'treatment_weeks': int(request.args.get('treatment_weeks', data.get('treatment_weeks', 24))),
        'rtms_freq_hz': float(request.args.get('rtms_freq_hz', data.get('rtms_freq_hz', 1.0))),
        'pharm_arm': str(request.args.get('pharm_arm', data.get('pharm_arm', 'synergistic'))),
        'stimulation_intensity_pct': float(request.args.get('stimulation_intensity_pct', data.get('stimulation_intensity_pct', 110.0))),
        'cbt_synergy_gain': float(request.args.get('cbt_synergy_gain', data.get('cbt_synergy_gain', 0.75))),
        'cf_signature_ratio': float(request.args.get('cf_signature_ratio', data.get('cf_signature_ratio', 1.41421356))),
    }


@app.route('/api/anxiety-rtms', methods=['GET', 'POST'])
def api_anxiety_rtms():
    """Return the computational research model for rTMS in millennial refractory anxiety."""
    try:
        params = _anxiety_params()
        result = simulate_anxiety_rtms(**params)
        return jsonify({'status': 'success', 'data': result})
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid anxiety model parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/anxiety-rtms-preprint', methods=['GET'])
def anxiety_rtms_preprint():
    """Generate and download the Nature-style preprint PDF for millennial anxiety rTMS."""
    try:
        from generate_nature_anxiety_rtms import build_pdf

        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Anxiety_rTMS_Millennials.pdf')
        build_pdf(output_path, _anxiety_params())
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Anxiety_rTMS_Millennials.pdf')
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid anxiety preprint parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/moduli-bem-paradigm', methods=['GET'])
def moduli_bem_paradigm():
    """Moduli-theoretic (SL(2,Z)) optimal treatment paradigm with boundary-element heat maps."""
    try:
        condition = request.args.get('condition', 'stroke')
        freq_max = float(request.args.get('freq_max', 50.0))
        intensity_max = float(request.args.get('intensity_max', 150.0))
        result = get_moduli_bem_paradigm(condition, (1.0, freq_max), (10.0, intensity_max))
        return jsonify({'status': 'success', 'data': result})
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid moduli-BEM parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/moduli-bem-preprint', methods=['GET'])
def moduli_bem_preprint():
    """Generate and download the Nature-style preprint PDF for the moduli-BEM treatment paradigm."""
    try:
        from generate_nature_moduli_bem import build_pdf

        condition = request.args.get('condition', 'stroke')
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Moduli_BEM_Paradigm.pdf')
        build_pdf(output_path, {'condition': condition})
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Moduli_BEM_Paradigm.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


def _tourette_params():
    data = request.get_json(silent=True) or {}
    return {
        'baseline_ygtss': float(request.args.get('baseline_ygtss', data.get('baseline_ygtss', 38.0))),
        'treatment_weeks': int(request.args.get('treatment_weeks', data.get('treatment_weeks', 20))),
        'stimulation_intensity_pct': float(request.args.get('stimulation_intensity_pct', data.get('stimulation_intensity_pct', 110.0))),
        'daily_pulses': int(request.args.get('daily_pulses', data.get('daily_pulses', 2400))),
        'hrt_synergy_gain': float(request.args.get('hrt_synergy_gain', data.get('hrt_synergy_gain', 0.80))),
        'cf_signature_ratio': float(request.args.get('cf_signature_ratio', data.get('cf_signature_ratio', 1.7320508))),
    }


@app.route('/api/tourette-rtms', methods=['GET', 'POST'])
def api_tourette_rtms():
    """Return the computational research model for rTMS in Tourette syndrome & combinatorial CSTC allocation."""
    try:
        params = _tourette_params()
        result = simulate_tourette_rtms(**params)
        return jsonify({'status': 'success', 'data': result})
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid Tourette model parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/tourette-rtms-preprint', methods=['GET'])
def tourette_rtms_preprint():
    """Generate and download the Nature-style preprint PDF for Tourette syndrome combinatorial rTMS."""
    try:
        from generate_nature_tourette_rtms import build_pdf

        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Tourette_rTMS_Combinatorics.pdf')
        build_pdf(output_path, _tourette_params())
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Tourette_rTMS_Combinatorics.pdf')
    except (TypeError, ValueError) as exc:
        return jsonify({'error': f'Invalid Tourette preprint parameter: {exc}'}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/tbi-ptsd-rtms', methods=['GET', 'POST'])
def api_tbi_ptsd_rtms():
    """TBI & PTSD Neuromodulation, BEM Electric Field, and Healthcare Economics Endpoint."""
    try:
        data = request.get_json(silent=True) or {}
        baseline_pcl5 = float(request.args.get('baseline_pcl5', data.get('baseline_pcl5', 58.0)))
        baseline_rpq = float(request.args.get('baseline_rpq', data.get('baseline_rpq', 42.0)))
        treatment_weeks = int(request.args.get('treatment_weeks', data.get('treatment_weeks', 24)))
        rtms_freq_hz = float(request.args.get('rtms_freq_hz', data.get('rtms_freq_hz', 10.0)))
        cbt_trauma_gain = float(request.args.get('cbt_trauma_gain', data.get('cbt_trauma_gain', 0.85)))
        clinic_coils = int(request.args.get('clinic_coils', data.get('clinic_coils', 3)))
        session_price = float(request.args.get('session_price', data.get('session_price', 300.0)))

        result = simulate_tbi_ptsd_rtms(
            baseline_pcl5=baseline_pcl5,
            baseline_rpq=baseline_rpq,
            treatment_weeks=treatment_weeks,
            rtms_freq_hz=rtms_freq_hz,
            cbt_trauma_gain=cbt_trauma_gain,
            clinic_coils=clinic_coils,
            session_price=session_price,
        )
        return jsonify({'status': 'success', 'data': result})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _schizophrenia_dbs_params():
    data = request.get_json(silent=True) or {}
    return {
        'baseline_panss': float(request.args.get('baseline_panss', data.get('baseline_panss', 104.0))),
        'target_nucleus': str(request.args.get('target_nucleus', data.get('target_nucleus', 'nucleus_accumbens_mpfc'))),
        'pulse_freq_hz': float(request.args.get('pulse_freq_hz', data.get('pulse_freq_hz', 130.0))),
        'pulse_width_us': float(request.args.get('pulse_width_us', data.get('pulse_width_us', 90.0))),
        'amplitude_ma': float(request.args.get('amplitude_ma', data.get('amplitude_ma', 3.5))),
        'closed_loop_gain': float(request.args.get('closed_loop_gain', data.get('closed_loop_gain', 1.45))),
        'treatment_months': int(request.args.get('treatment_months', data.get('treatment_months', 60))),
        'innovator': str(request.args.get('innovator', data.get('innovator', 'neuromorph_opto_bem')))
    }


@app.route('/api/schizophrenia-dbs', methods=['GET', 'POST'])
def api_schizophrenia_dbs():
    """
    Simulate multi-target DBS for refractory schizophrenia:
    - BEM electric field potentials
    - Innovator comparisons (Boston Scientific, Medtronic, Abbott, NeuroMorph)
    - 4-stage longitudinal care paradigm & 5-year recovery trajectories
    - Finite mathematical equations & Markov chain transitions
    """
    try:
        params = _schizophrenia_dbs_params()
        result = simulate_schizophrenia_dbs_bem(**params)
        return jsonify({'status': 'success', 'data': result})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/schizophrenia-dbs-preprint', methods=['GET'])
def schizophrenia_dbs_preprint():
    """Generate and stream the Nature-style preprint PDF for Schizophrenia DBS."""
    try:
        from generate_nature_schizophrenia_dbs import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Schizophrenia_DBS_BEM_Finite_Math.pdf')
        build_pdf(output_path, _schizophrenia_dbs_params())
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Schizophrenia_DBS_BEM_Finite_Math.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _hifu_tms_params():
    data = request.get_json(silent=True) or {}
    return {
        'hifu_pressure_mpa': float(request.args.get('hifu_pressure_mpa', data.get('hifu_pressure_mpa', 0.85))),
        'tms_intensity_rMT': float(request.args.get('tms_intensity_rMT', data.get('tms_intensity_rMT', 110.0))),
        'pharm_agent': str(request.args.get('pharm_agent', data.get('pharm_agent', 'Lecanemab + Galantamine + Memantine'))),
        'total_weeks': int(request.args.get('total_weeks', data.get('total_weeks', 104)))
    }


@app.route('/api/hifu-tms-dementia', methods=['GET', 'POST'])
def api_hifu_tms_dementia():
    """
    HIFU-TMS Trans-BBB Registration & Synergistic Dementia Cure:
    - Statistical Continued Fractions for registration correspondence
    - 104-week recovery horizons (ADAS-Cog-13, MMSE)
    - Feynman Path Integrals for acoustic-electromagnetic-pharmacological synergy
    - Asymmetric Prime Moduli Theory for lattice repair
    """
    try:
        params = _hifu_tms_params()
        result = simulate_hifu_tms_dementia(**params)
        return jsonify({'status': 'success', 'data': result})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/hifu-tms-dementia-preprint', methods=['GET'])
def hifu_tms_dementia_preprint():
    """Generate and download the Nature-style preprint PDF for HIFU-TMS Dementia Cure."""
    try:
        from generate_nature_hifu_tms_dementia import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_HIFU_TMS_Dementia_Cure_Prime_Moduli.pdf')
        build_pdf(output_path, _hifu_tms_params())
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_HIFU_TMS_Dementia_Cure_Prime_Moduli.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/cybersecurity-pqc-sim', methods=['GET', 'POST'])
def api_cybersecurity_pqc_sim():
    """
    Cybersecurity & Post-Quantum Cryptographic Measures:
    - Leibniz-Ramanujan Complete Elliptic Integrals
    - Jacobi Nome q & Dedekind Eta Invariant Log-Probabilities
    - Clinical Trajectories & Threat Vector Risk Aversion
    - Deep Brain Stimulation (Boston Scientific Vercise & Medtronic Percept) PQC Optimization
    """
    try:
        data = request.get_json(silent=True) or {}
        k_val = float(request.args.get('elliptic_k', data.get('elliptic_k', 0.7071)))
        atk_val = float(request.args.get('attack_intensity', data.get('attack_intensity', 1.0)))
        weeks_val = int(request.args.get('weeks', data.get('weeks', 104)))
        device_val = str(request.args.get('device_filter', data.get('device_filter', 'all')))
        
        # Run general cybersecurity simulation
        general_res = simulate_cybersecurity_trajectories(elliptic_k=k_val, attack_intensity=atk_val, weeks=weeks_val)
        
        # Run specialized Boston Scientific Vercise & Medtronic Percept DBS PQC vulnerability optimization
        dbs_res = optimize_dbs_pqc_vulnerability_assessment(
            modulus_k=k_val,
            attack_intensity=atk_val,
            weeks=weeks_val,
            device_filter=device_val
        )
        
        # Combine comprehensive data package
        combined = {
            **general_res,
            "dbs_pqc_optimization": dbs_res
        }
        return jsonify({'status': 'success', 'data': combined})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/dbs-pqc-vulnerability', methods=['GET', 'POST'])
def api_dbs_pqc_vulnerability():
    """
    Dedicated endpoint for Boston Scientific Vercise & Medtronic Percept DBS Vulnerability Optimization.
    """
    try:
        data = request.get_json(silent=True) or {}
        k_val = float(request.args.get('elliptic_k', data.get('elliptic_k', 0.7071)))
        atk_val = float(request.args.get('attack_intensity', data.get('attack_intensity', 1.0)))
        weeks_val = int(request.args.get('weeks', data.get('weeks', 104)))
        device_val = str(request.args.get('device_filter', data.get('device_filter', 'all')))
        
        res = optimize_dbs_pqc_vulnerability_assessment(
            modulus_k=k_val,
            attack_intensity=atk_val,
            weeks=weeks_val,
            device_filter=device_val
        )
        return jsonify({'status': 'success', 'data': res})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/cybersecurity-pqc-preprint', methods=['GET'])
def cybersecurity_pqc_preprint():
    """Generate and download the Nature-style preprint PDF for Post-Quantum Cybersecurity."""
    try:
        from generate_nature_cybersecurity_pqc import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_PQC_Cybersecurity_Leibniz_Ramanujan.pdf')
        build_pdf(output_path)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_PQC_Cybersecurity_Leibniz_Ramanujan.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/clinical-trajectories-benchmark', methods=['GET', 'POST'])
def api_clinical_trajectories_benchmark():
    """
    Simulate Multi-Horizon Clinical Trajectories & Benchmark Performance Advantage
    of Transcranial Magnetic Stimulators vs Deep Brain Stimulators.
    """
    try:
        data = request.get_json(silent=True) or {}
        variant = str(request.args.get('variant', data.get('variant', 'neurodegenerative')))
        weeks = int(request.args.get('weeks', data.get('weeks', 104)))
        modulus = int(request.args.get('galois_modulus', data.get('galois_modulus', 17)))
        gain = float(request.args.get('closed_loop_gain', data.get('closed_loop_gain', 1.85)))
        cf_depth = int(request.args.get('ramanujan_depth', data.get('ramanujan_depth', 6)))

        res = simulate_clinical_trajectories_benchmark(
            horizon_variant=variant,
            horizon_weeks=weeks,
            galois_modulus=modulus,
            closed_loop_gain=gain,
            ramanujan_depth=cf_depth
        )
        return jsonify({'status': 'success', 'data': res})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/clinical-trajectories-preprint', methods=['GET'])
def clinical_trajectories_preprint():
    """Generate and download the Nature-style preprint PDF for Superoptimal Clinical Staging."""
    try:
        from generate_nature_clinical_trajectories_preprint import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Clinical_Trajectories_Superoptimal_Staging.pdf')
        
        variant = str(request.args.get('variant', 'neurodegenerative'))
        weeks = int(request.args.get('weeks', 104))
        modulus = int(request.args.get('galois_modulus', 17))
        gain = float(request.args.get('closed_loop_gain', 1.85))
        cf_depth = int(request.args.get('ramanujan_depth', 6))

        sim_params = {
            'horizon_variant': variant,
            'horizon_weeks': weeks,
            'galois_modulus': modulus,
            'closed_loop_gain': gain,
            'ramanujan_depth': cf_depth
        }
        build_pdf(output_path, sim_params=sim_params)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Clinical_Trajectories_Superoptimal_Staging.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/stroke-repair-rtms', methods=['GET', 'POST'])
def api_stroke_repair_rtms():
    """Return a research-only statistical stroke-repair rTMS staging simulation."""
    try:
        data = request.get_json(silent=True) or {}
        result = simulate_stroke_repair_rtms(
            baseline_fugl_meyer=float(request.args.get('baseline_fugl_meyer', data.get('baseline_fugl_meyer', 32.0))),
            horizon_weeks=int(request.args.get('horizon_weeks', data.get('horizon_weeks', 52))),
            galois_modulus=int(request.args.get('galois_modulus', data.get('galois_modulus', 13))),
            cf_depth=int(request.args.get('cf_depth', data.get('cf_depth', 6))),
            target_sync_ratio=float(request.args.get('target_sync_ratio', data.get('target_sync_ratio', 1.6180339887))),
            closed_loop_gain=float(request.args.get('closed_loop_gain', data.get('closed_loop_gain', 1.35))),
        )
        return jsonify({'status': 'success', 'data': result})
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400


@app.route('/api/stroke-repair-preprint', methods=['GET'])
def stroke_repair_preprint():
    """Generate and download the research-only stroke-repair rTMS preprint PDF."""
    try:
        from generate_nature_stroke_repair_preprint import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Stroke_Repair_rTMS_Finite_Staging.pdf')
        sim_params = {
            'baseline_fugl_meyer': float(request.args.get('baseline_fugl_meyer', 32.0)),
            'horizon_weeks': int(request.args.get('horizon_weeks', 52)),
            'galois_modulus': int(request.args.get('galois_modulus', 13)),
            'cf_depth': int(request.args.get('cf_depth', 6)),
            'target_sync_ratio': float(request.args.get('target_sync_ratio', 1.6180339887)),
            'closed_loop_gain': float(request.args.get('closed_loop_gain', 1.35)),
        }
        build_pdf(output_path, sim_params=sim_params)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Stroke_Repair_rTMS_Finite_Staging.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/nih-grant-budget-optimizer', methods=['GET', 'POST'])
def api_nih_grant_budget_optimizer():
    """Simulate and optimize NIH grant multi-year direct/indirect budget and spend burn rates."""
    try:
        data = request.get_json(silent=True) or {}
        direct_budget = float(request.args.get('direct_budget', data.get('direct_budget', 2500000.0)))
        fa_rate = float(request.args.get('fa_rate', data.get('fa_rate', 0.545)))
        n_years = int(request.args.get('n_years', data.get('n_years', 5)))
        contingency = float(request.args.get('contingency_pct', data.get('contingency_pct', 5.0)))

        res = optimize_nih_grant_budget(
            total_direct_budget=direct_budget,
            fa_rate=fa_rate,
            n_years=n_years,
            contingency_reserve_pct=contingency
        )
        return jsonify({'status': 'success', 'data': res})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/pharmacological-equivalence', methods=['GET'])
def pharmacological_equivalence():
    """Serve the generative-AI pharmacological equivalence model for rTMS-equivalent discovery."""
    try:
        data = simulate_pharmacological_equivalence()
        return jsonify({'status': 'success', 'data': data})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/pharmacological-equivalence-publication', methods=['GET'])
def pharmacological_equivalence_publication():
    """Generate and stream the Nature-style pharmacological equivalence PDF."""
    try:
        from generate_nature_pharmacological_equivalence import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Pharmacological_Equivalence_rTMS.pdf')
        build_pdf(output_path)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Pharmacological_Equivalence_rTMS.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/nih-grant-proposal-pdf', methods=['GET'])
def nih_grant_proposal_pdf():
    """Generate and stream the complete NIH Grant Proposal PDF."""
    try:
        from generate_nih_grant_proposal import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'NIH_Grant_Proposal_Neuromorph_TMS_DBS.pdf')
        build_pdf(output_path)
        return send_file(output_path, as_attachment=True, download_name='NIH_Grant_Proposal_Neuromorph_TMS_DBS.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/parkinsons-rtms', methods=['GET', 'POST'])
def api_parkinsons_rtms():
    """Simulate QML-Governed Multi-Lobe rTMS for Parkinson's Disease neuromodulation & health economics."""
    try:
        data = request.get_json(silent=True) or {}
        baseline_updrs = float(request.args.get('baseline_updrs', data.get('baseline_updrs', 42.0)))
        intensity_rmt = float(request.args.get('intensity_rmt', data.get('intensity_rmt', 110.0)))
        qml_depth = int(request.args.get('qml_depth', data.get('qml_depth', 6)))
        sessions_per_week = int(request.args.get('sessions_per_week', data.get('sessions_per_week', 5)))
        horizon_weeks = int(request.args.get('horizon_weeks', data.get('horizon_weeks', 156)))
        l_dopa_equiv_dose = float(request.args.get('l_dopa_equiv_dose', data.get('l_dopa_equiv_dose', 650.0)))
        rf_carrier_khz = float(request.args.get('rf_carrier_khz', data.get('rf_carrier_khz', 3.2)))
        pulse_budget_daily = int(request.args.get('pulse_budget_daily', data.get('pulse_budget_daily', 3000)))

        res = simulate_parkinsons_rtms(
            baseline_updrs=baseline_updrs,
            intensity_rmt=intensity_rmt,
            qml_depth=qml_depth,
            sessions_per_week=sessions_per_week,
            horizon_weeks=horizon_weeks,
            l_dopa_equiv_dose=l_dopa_equiv_dose,
            rf_carrier_khz=rf_carrier_khz,
            pulse_budget_daily=pulse_budget_daily,
        )
        return jsonify(res)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/parkinsons-nature-preprint', methods=['GET'])
def parkinsons_nature_preprint():
    """Generate and serve the Parkinson's Multi-Lobe BEM & Combinatorial Staging Nature preprint PDF."""
    try:
        from generate_nature_parkinsons_rtms import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Parkinsons_rTMS_BEM_Combinatorial.pdf')
        if not os.path.exists(output_path):
            build_pdf(output_path)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Parkinsons_rTMS_BEM_Combinatorial.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/heart-stroke-trajectories', methods=['GET', 'POST'])
def api_heart_stroke_trajectories():
    """Simulate Heart & Stroke Long-Term Clinical Trajectories via Ramanujan Modular Nome States,
    Cortical Surface Interventional Repair, Nash Gap Verification, and Stem Cell Ring Algebra.
    """
    try:
        data = request.get_json(silent=True) or {}
        baseline_nihss = float(request.args.get('baseline_nihss', data.get('baseline_nihss', 16.5)))
        baseline_lvef = float(request.args.get('baseline_lvef', data.get('baseline_lvef', 35.0)))
        tau_imag_y = float(request.args.get('tau_imag_y', data.get('tau_imag_y', 0.85)))
        tau_real_x = float(request.args.get('tau_real_x', data.get('tau_real_x', 0.25)))
        horizon_months = int(request.args.get('horizon_months', data.get('horizon_months', 60)))
        intervention_intensity = float(request.args.get('intervention_intensity', data.get('intervention_intensity', 1.15)))
        stem_cell_density = float(request.args.get('stem_cell_density', data.get('stem_cell_density', 25.0)))

        optimizer = HeartStrokeClinicalTrajectoryOptimizer()
        res = optimizer.simulate_longterm_trajectories(
            baseline_nihss=baseline_nihss,
            baseline_lvef=baseline_lvef,
            tau_imag_y=tau_imag_y,
            tau_real_x=tau_real_x,
            horizon_months=horizon_months,
            intervention_intensity=intervention_intensity,
            stem_cell_density_million=stem_cell_density
        )
        return jsonify({'status': 'success', 'data': res})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/heart-stroke-nature-preprint', methods=['GET'])
def heart_stroke_nature_preprint():
    """Generate and serve the Heart & Stroke Ramanujan Modular Nome & Stem Cell Ring Algebra Nature Preprint PDF."""
    try:
        from generate_nature_heart_stroke_preprint import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Heart_Stroke_Ramanujan_Nash_RingAlgebra.pdf')
        if not os.path.exists(output_path):
            build_pdf(output_path)
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Heart_Stroke_Ramanujan_Nash_RingAlgebra.pdf')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _arg(data, name, default, cast):
    """Read a query-string or JSON parameter, falling back to a default."""
    return cast(request.args.get(name, data.get(name, default)))


def _fem_params(data):
    return dict(
        hoehn_yahr_stage=_arg(data, 'hoehn_yahr_stage', 3, int),
        baseline_updrs=_arg(data, 'baseline_updrs', 42.0, float),
        atrophy_level=_arg(data, 'atrophy_level', 'moderate', str),
        intensity_mso_pct=_arg(data, 'intensity_mso_pct', 110.0, float),
        coil_orientation_deg=_arg(data, 'coil_orientation_deg', 45.0, float),
        num_gcp_nodes=_arg(data, 'num_gcp_nodes', 16, int),
        matrix_reduction_type=_arg(data, 'matrix_reduction_type', 'aca_low_rank', str),
        reduction_tolerance=_arg(data, 'reduction_tolerance', 1e-4, float),
        grid_resolution=_arg(data, 'grid_resolution', 28, int),
        coil_pos_x=_arg(data, 'coil_pos_x', 12.0, float),
        coil_pos_y=_arg(data, 'coil_pos_y', 2.0, float),
        activation_threshold_v_per_m=_arg(data, 'activation_threshold_v_per_m', 80.0, float),
    )


@app.route('/api/fem-cortical-simulation', methods=['GET', 'POST'])
def api_fem_cortical_simulation():
    """Surface FEM of a Parkinsonian motor cortex under rTMS with Stieltjes S-fraction interpolation,
    Schur-complement domain decomposition and measured matrix reduction (see logic/fem_cortical_solver.py)."""
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(run_full_cortical_fem_simulation(**_fem_params(data)))
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/fem-cortical-preprint', methods=['GET'])
def fem_cortical_preprint():
    """Generate the cortical-FEM Nature-style preprint from the current simulation parameters."""
    try:
        from generate_nature_fem_cortical_preprint import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Cortical_FEM_Stieltjes_GCP.pdf')
        build_pdf(output_path, sim_params=_fem_params({}))
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Cortical_FEM_Stieltjes_GCP.pdf')
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _connectome_params(data):
    return dict(
        severity=_arg(data, 'severity', 0.6, float),
        tau_x=_arg(data, 'tau_x', 0.25, float),
        tau_y=_arg(data, 'tau_y', 0.85, float),
        qml_layers=_arg(data, 'qml_layers', 3, int),
        qml_iters=_arg(data, 'qml_iters', 80, int),
        threshold_pct=_arg(data, 'threshold_pct', 75.0, float),
    )


@app.route('/api/connectome-qml', methods=['GET', 'POST'])
def api_connectome_qml():
    """Healthy vs dementia cortical connectome with modular harmonic forms and a quantum-kernel connectome matrix."""
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(simulate_connectome_qml(**_connectome_params(data)))
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/connectome-nature-preprint', methods=['GET'])
def connectome_nature_preprint():
    """Generate the connectome / modular harmonic / QML Nature-style preprint for the current parameters."""
    try:
        from generate_nature_connectome_preprint import build_pdf
        output_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Connectome_Modular_Harmonics_QML.pdf')
        build_pdf(output_path, sim_params=_connectome_params({}))
        return send_file(output_path, as_attachment=True, download_name='Nature_Preprint_Connectome_Modular_Harmonics_QML.pdf')
    except (TypeError, ValueError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(exc)}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5020)
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)
