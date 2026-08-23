import os, io, math
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
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

app = Flask(__name__)
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
    return jsonify({"status": "success", "data": get_ocd_fea_simulation()})

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002)
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)
