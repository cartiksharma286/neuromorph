from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from simulator_core import MRIReconstructionSimulator
from statistical_adaptive_pulse import create_adaptive_sequence, ADAPTIVE_SEQUENCES
from quantum_vascular_coils import get_coil_summary, QUANTUM_VASCULAR_COIL_LIBRARY
from circuit_schematic_generator import CircuitSchematicGenerator
from neuro_pulse_ca import generate_all_canadian_sequences, create_canadian_sequence, CANADIAN_NEURO_SEQUENCES
import os
import generate_pdf
import generate_report_images
import numpy as np
import json

def sanitize_for_json(obj):
    """Recursively replace NaNs and Infs with None."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, np.generic): # Handle numpy scalars
        return sanitize_for_json(obj.item())
    return obj


app = Flask(__name__)
CORS(app)
LATEST_CONTEXT = {}
# NVQLink option removed per user request

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/human_readable_status')
def status():
    return "MRI Simulator Online - Enhanced with Quantum Vascular Coils"

# Global Cache for Simulator Instances
SIMULATOR_CACHE = {}

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json or {}
        res = int(data.get('resolution', 128))
        seq_type = data.get('sequence', 'SE')
        tr = float(data.get('tr', 2000))
        te = float(data.get('te', 100))
        ti = float(data.get('ti', 500))
        flip_angle = float(data.get('flip_angle', 30))
        coil_mode = data.get('coils', 'standard')
        num_coils = int(data.get('num_coils', 8))
        noise = float(data.get('noise', 0.0))
        recon_method = data.get('recon_method', 'SoS')
        use_shimming = data.get('shimming', False)
        
        # Cache Key: (Resolution, CoilMode, NumCoils, Shimming)
        # We only re-init simulator if these structural parameters change.
        # TR/TE/Seq changes do not require re-generating phantom/coils.
        cache_key = (res, coil_mode, num_coils, use_shimming)
        
        if cache_key in SIMULATOR_CACHE:
            sim = SIMULATOR_CACHE[cache_key]
        else:
            sim = MRIReconstructionSimulator(resolution=res)
            
            # NVQLink disabled (removed per user request)
            
            phantom_type = 'brain'
            if coil_mode == 'cardiothoracic_array' or coil_mode == 'cardiovascular_coil':
                phantom_type = 'cardiac'
            elif coil_mode == 'knee_vascular_array':
                phantom_type = 'knee'
    
            sim.setup_phantom(use_real_data=True, phantom_type=phantom_type)
            sim.generate_coil_sensitivities(num_coils=num_coils, coil_type=coil_mode, optimal_shimming=use_shimming)
            
            # Cache the initialized simulator
            SIMULATOR_CACHE[cache_key] = sim
        
        # Always run these optimization steps if needed, or if cached they are already done.
        # But wait, shim report is generated during generation.
        # We might miss shim_report if cached.
        # However, for real-time responsiveness, we skip re-shimming on every TR change.
        
        # Set View (fast)
        
        shim_report = None
        if use_shimming:
            w_opt, shim_info = sim.optimize_shimming_b_field(sim.coils)
            shim_report = sim.generate_shim_report_data(w_opt, shim_info)
            if coil_mode != 'n25_array':
                # Apply optimized weights AND localized 25% SNR boost
                sim.apply_localized_shimming(w_opt)
        
        slice_orientation = data.get('slice_orientation', 'axial')
        slice_pos = float(data.get('slice_pos', 0.5))
        sim.set_view(slice_orientation, slice_pos)
        
        kspace, M_ref = sim.acquire_signal(sequence_type=seq_type, TR=tr, TE=te, TI=ti, flip_angle=flip_angle, noise_level=noise)
        
        if recon_method == 'DeepLearning':
            recon_img = sim.deep_learning_reconstruct(kspace)
            coil_imgs = []
        else:
            recon_img, coil_imgs = sim.reconstruct_image(kspace, method=recon_method)
        
        metrics = sim.compute_metrics(recon_img, M_ref)
        stat_metrics = sim.classifier.analyze_image(recon_img)
        metrics.update(stat_metrics)
        
        # Add quantum/50-turn coil info to metrics
        metrics['quantum_vascular_enabled'] = sim.quantum_vascular_enabled
        metrics['head_coil_50_enabled'] = sim.head_coil_50_turn['enabled']
        metrics['nvqlink_enabled'] = sim.nvqlink_enabled
        
        plots = sim.generate_plots(kspace, recon_img, M_ref)
        
        # QML Thermometry Reasoning (Plots added to dict after global generation)
        if seq_type == 'QuantumMLThermometry':
            from statistical_adaptive_pulse import QMLThermometrySequence
            qml_reasoner = QMLThermometrySequence(nvqlink_enabled=sim.nvqlink_enabled)
            distribution_stats = qml_reasoner.reason_about_distributions(recon_img)
            metrics.update(distribution_stats)
            plots['temperature_map'] = sim.generate_temperature_map_plot(recon_img, distribution_stats['inferred_mean_temp_c'])
            plots['distribution_curve'] = sim.generate_distribution_curve_plot(recon_img, distribution_stats)

        if seq_type == 'QuantumGeometry':
            from quantum_geometry_pulse import QuantumGeometryContinuedFractionSequence
            qg_analyzer = QuantumGeometryContinuedFractionSequence()
            geom_stats = qg_analyzer.compute_geometric_analytics(recon_img)
            metrics.update(geom_stats)
            # Add cf_depth from sequence generation if available
            # In simulation, we re-run generate_sequence to get the same params for reporting
            # but wait, simulate() already does acquire_signal which uses the params.
            # We can just put a placeholder or call generate_sequence with mock stats
            qg_params = qg_analyzer.generate_sequence(stat_metrics)
            metrics['cf_depth'] = qg_params['cf_depth']
            plots['quantum_manifold'] = plots['recon'] # Manifold is the recon with geometric interpretation

        signal_study = sim.generate_signal_study(seq_type)
        
        # Images are returned directly as base64 strings in the JSON response
        # No need to write to disk, avoiding "No space left on device" errors
        aux_maps = sim.get_auxiliary_maps()
        LATEST_CONTEXT.update({
            "coil_mode": coil_mode,
            "seq_type": seq_type,
            "metrics": metrics,
            "signal_study": signal_study,
            "timestamp": "January 14, 2026",
            "shim_report": shim_report,
            "circuit_schematic": plots['circuit'],
            "auxiliary_maps": aux_maps
        })
        
        return jsonify(sanitize_for_json({
            "success": True,
            "metrics": metrics,
            "plots": plots,
            "shim_report": shim_report,
            "auxiliary_maps": aux_maps,
            "signal_study": signal_study
        }))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/render_cortical', methods=['POST'])
def render_cortical():
    try:
        sim = MRIReconstructionSimulator()
        pd_surface = sim.renderCorticalSurface2D()
        
        # We need to capture the 3D metadata if available, otherwise just use 2D info
        if hasattr(sim, 'renderCorticalSurface3D'):
            metadata = sim.renderCorticalSurface3D()
        else:
            metadata = {'geometry': '2D Projected Surface'}
            
        import matplotlib.pyplot as plt
        import io
        import base64
        
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0f172a')
        ax.imshow(pd_surface, cmap='gray')
        ax.set_title("Cortical Surface Reconstruction", color='white')
        ax.axis('off')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"success": True, "plot": plot_b64, "metadata": metadata})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/quantum_coils/list', methods=['GET'])
def list_quantum_coils():
    """Returns list of all 25 quantum vascular coils."""
    try:
        coils_info = []
        for idx, coil_class in QUANTUM_VASCULAR_COIL_LIBRARY.items():
            coil = coil_class()
            coils_info.append({
                'id': idx,
                'name': coil.name,
                'elements': coil.num_elements,
                'frequency_mhz': coil.frequency / 1e6
            })
        return jsonify({'success': True, 'coils': coils_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/head_coil_50/specs', methods=['GET'])
def head_coil_50_specs():
    """Returns specifications for 50-turn head coil."""
    try:
        sim = MRIReconstructionSimulator()
        specs = sim.head_coil_50_turn.copy()
        specs['description'] = "Ultra-high resolution 50-turn head coil for neuroimaging"
        specs['applications'] = [
            "Neurovasculature imaging at 300 micron resolution",
            "Capillary-level blood flow visualization",
            "Ultra-high field (7T+) neuroimaging",
            "Cortical layer differentiation"
        ]
        return jsonify({'success': True, 'specs': specs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/adaptive_sequence/generate', methods=['POST'])
def generate_adaptive_sequence():
    """Generates statistical adaptive pulse sequence."""
    try:
        data = request.json or {}
        sequence_type = data.get('type', 'adaptive_se')
        nvqlink_enabled = False  # NVQLink removed
        
        # Create adaptive sequence
        adaptive_seq = create_adaptive_sequence(sequence_type, nvqlink_enabled)
        
        # Simulate tissue statistics
        mock_kspace = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)
        tissue_stats = adaptive_seq.estimate_tissue_statistics(mock_kspace)
        
        # Generate optimized parameters
        sequence_params = adaptive_seq.generate_sequence(tissue_stats)
        
        return jsonify({
            'success': True,
            'sequence': sequence_params,
            'tissue_stats': tissue_stats,
            'adaptation_history': adaptive_seq.adaptation_history
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/neuro_pulse_ca/generate', methods=['POST'])
def generate_neuro_pulse_ca():
    """
    Generates all 5 Canadian neuroimaging pulse sequences with
    statistical distribution plots and Bayesian inference results.

    Optional POST body:
      { "sequence_key": "canadian_mprage" }  — single sequence
      or omit for all 5.
    """
    try:
        data = request.json or {}
        specific_key = data.get('sequence_key', None)

        if specific_key and specific_key in CANADIAN_NEURO_SEQUENCES:
            rng = np.random.default_rng(7)
            mock_kspace = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
            seq = create_canadian_sequence(specific_key)
            tissue_stats = seq.estimate_tissue_statistics(mock_kspace)
            seq_params = seq.generate_sequence(tissue_stats)
            dist_info = seq.compute_distribution_stats()
            return jsonify(sanitize_for_json({
                'success': True,
                'sequences': [{
                    'key': specific_key,
                    'params': seq_params,
                    'tissue_stats': {k: v for k, v in tissue_stats.items()
                                     if not isinstance(v, dict)},
                    'distribution_plot': dist_info['plot'],
                    'distribution_name': dist_info['distribution'],
                    'distribution_stats': dist_info['stats'],
                    'institution': dist_info['institution'],
                }],
                'count': 1
            }))
        else:
            result = generate_all_canadian_sequences()
            return jsonify(sanitize_for_json({'success': True, **result}))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/nvqlink/status', methods=['GET'])
def nvqlink_status():
    """Returns NVQLink status."""
# NVQLink status endpoint removed

@app.route('/api/nvqlink/toggle', methods=['POST'])
def nvqlink_toggle():
    """Toggles NVQLink on/off."""
# NVQLink toggle endpoint removed

@app.route('/api/signal_reconstruction/coil_geometry', methods=['POST'])
def signal_reconstruction_coil_geometry():
    """Performs signal reconstruction analysis vis-a-vis coil geometry."""
    try:
        data = request.json or {}
        coil_types = data.get('coil_types', ['standard', 'quantum_vascular', 'head_coil_50_turn'])
        sequence = data.get('sequence', 'GRE')
        
        results = []
        
        for coil_type in coil_types:
            sim = MRIReconstructionSimulator(resolution=128)
            sim.setup_phantom(use_real_data=True, phantom_type='brain')
            
            if data.get('nvqlink', False):
                sim.nvqlink_enabled = True
            
            sim.generate_coil_sensitivities(num_coils=8, coil_type=coil_type)
            kspace, M_ref = sim.acquire_signal(sequence_type=sequence, TR=150, TE=10)
            recon_img, _ = sim.reconstruct_image(kspace, method='SoS')
            
            metrics = sim.compute_metrics(recon_img, M_ref)
            stat_metrics = sim.classifier.analyze_image(recon_img)
            
            import matplotlib.pyplot as plt
            import io
            import base64
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('#0f172a')
            
            if len(sim.coils) > 0:
                axes[0].imshow(np.abs(sim.coils[0]), cmap='gray')
                axes[0].set_title(f'{coil_type} - Sensitivity', color='white')
                axes[0].axis('off')
            
            axes[1].imshow(np.log(np.abs(kspace[0]) + 1), cmap='gray')
            axes[1].set_title('K-Space', color='white')
            axes[1].axis('off')
            
            axes[2].imshow(recon_img, cmap='gray')
            axes[2].set_title(f'Reconstruction (SNR: {stat_metrics.get("snr_estimate", 0):.1f})', color='white')
            axes[2].axis('off')
            
            for ax in axes:
                ax.set_facecolor('#0f172a')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            results.append({
                'coil_type': coil_type,
                'metrics': {**metrics, **stat_metrics},
                'plot': plot_b64,
                'num_coils': len(sim.coils),
                'quantum_enabled': sim.quantum_vascular_enabled,
                'head_coil_50_enabled': sim.head_coil_50_turn['enabled']
            })
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/neurovasculature/render', methods=['POST'])
def render_neurovasculature():
    """Renders ultra-high resolution neurovasculature."""
    try:
        data = request.json or {}
        enable_50_turn = data.get('enable_50_turn', True)
        
        sim = MRIReconstructionSimulator(resolution=128)
        
        if enable_50_turn:
            sim.head_coil_50_turn['enabled'] = True
        
        sim.setup_phantom(use_real_data=True, phantom_type='brain')
        
        import matplotlib.pyplot as plt
        import io
        import base64
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.patch.set_facecolor('#0f172a')
        
        orientations = ['axial', 'coronal', 'sagittal']
        for idx, orientation in enumerate(orientations[:3]):
            ax = axes.flatten()[idx]
            sim.set_view(orientation, 0.5)
            
            ax.imshow(sim.pd_map, cmap='gray', vmin=0, vmax=1.5)
            ax.set_title(f'{orientation.title()} - Neurovasculature', color='white')
            ax.axis('off')
            ax.set_facecolor('#0f172a')
        
        ax = axes.flatten()[3]
        ax.axis('off')
        ax.set_facecolor('#0f172a')
        
        specs_text = "50-Turn Head Coil Specs:\\n\\n"
        specs_text += f"Turns: {sim.head_coil_50_turn['turns']}\\n"
        specs_text += f"SNR Enhancement: {sim.head_coil_50_turn['snr_enhancement']}x\\n"
        specs_text += f"Resolution: {sim.head_coil_50_turn['spatial_resolution_mm']} mm\\n"
        specs_text += f"Status: {'ENABLED' if sim.head_coil_50_turn['enabled'] else 'DISABLED'}"
        
        ax.text(0.1, 0.5, specs_text, color='#38bdf8', fontsize=12, 
                verticalalignment='center', family='monospace')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)
        plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'plot': plot_b64,
            'head_coil_enabled': sim.head_coil_50_turn['enabled'],
            'resolution_mm': sim.head_coil_50_turn['spatial_resolution_mm']
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/report')
def get_report():
    try:
        if not LATEST_CONTEXT:
            return jsonify({"success": False, "error": "No simulation run yet."}), 400
        c = LATEST_CONTEXT
        coil_name = c["coil_mode"].replace('_', ' ').title()
        
        physics_desc = "Standard MRI Acquisition Loop."
        math_block = r"$$ S(k) = \int M(r) e^{-i k r} dr $$"
        
        cmd = c['coil_mode']
        if 'quantum_vascular' in cmd:
            physics_desc = "Quantum Vascular Coil with elliptic integral coupling for enhanced SNR."
            math_block = r"$$ M = \mu_0 \sqrt{ab}[(2-k^2)K(k) - 2E(k)]/k $$"
        elif 'head_coil_50' in cmd:
            physics_desc = "50-Turn Head Coil providing 3.2x SNR enhancement and 300 micron resolution."
            math_block = r"$$ L = \mu_0 n^2 A, \quad SNR \propto \sqrt{L} \propto n $$"
        elif 'quantum' in cmd:
            physics_desc = "This coil topology exploits the Berry Phase of adiabatic spin transport."
            math_block = r"$$ \gamma_n(C) = i \oint_C \langle \psi_n | \nabla | \psi_n \rangle \cdot d\mathbf{R} $$"
        
        content = f"""
# NeuroPulse Clinical Physics Report
**Date:** {c['timestamp']}
**Simulation ID:** {c['seq_type']}-{c['coil_mode']}

---

## 1. Executive Summary
This report details the simulation results for the **{coil_name}** operating with **{c['seq_type']}**.

## 2. Physics & Circuit Topology
{physics_desc}

### Circuit Schematic
![Schematic](current_schematic.png)

### Coil Derivation
{math_block}

---

## 3. Metrics
* **Contrast:** {c['metrics'].get('contrast', 0):.4f}
* **SNR Estimate:** {c['metrics'].get('snr_estimate', 0):.2f}
* **Quantum Vascular Enabled:** {c['metrics'].get('quantum_vascular_enabled', False)}
* **50-Turn Head Coil Enabled:** {c['metrics'].get('head_coil_50_enabled', False)}
* **NVQLink Enabled:** {c['metrics'].get('nvqlink_enabled', False)}

## 4. Finite Math Calculations
The simulation employs discrete finite mathematical operators for signal reconstruction.

### Discrete Fourier Transform (Finite Summation)
The image space $M(x,y)$ is recovered from the discretized k-space $S(u,v)$ via the Inverse Discrete Fourier Transform (IDFT):

$$ M(x,y) = \\frac{{1}}{{N^2}} \\sum_{{u=0}}^{{N-1}} \\sum_{{v=0}}^{{N-1}} S(u,v) \\cdot e^{{i 2\\pi (\\frac{{ux}}{{N}} + \\frac{{vy}}{{N}})}} $$

### Finite Difference Gradient (Edge Detection)
To assess sharpness, we compute the discrete gradient magnitude $|\\nabla M|$ using central finite differences:

$$ \\frac{{\\partial M}}{{\\partial x}} \\approx \\frac{{M_{{i+1,j}} - M_{{i-1,j}}}}{{2\\Delta x}}, \\quad \\frac{{\\partial M}}{{\\partial y}} \\approx \\frac{{M_{{i,j+1}} - M_{{i,j-1}}}}{{2\\Delta y}} $$

$$ |\\nabla M|_{{i,j}} = \\sqrt{{ \\left(\\frac{{M_{{i+1,j}} - M_{{i-1,j}}}}{{2}}\\right)^2 + \\left(\\frac{{M_{{i,j+1}} - M_{{i,j-1}}}}{{2}}\\right)^2 }} $$

### Quantum Finite Element (For Vascular Coils)
For the quantum vascular coils, the magnetic flux $\\Phi$ is discretized over the loop elements $E_k$:

$$ \\Phi \\approx \\sum_{{k=1}}^{{N_{{elem}}}} \\mathbf{{B}}_k \\cdot \\mathbf{{n}}_k A_k $$

"""
        
        import base64
        if 'circuit_schematic' in c:
            with open(os.path.join(os.getcwd(), 'current_schematic.png'), 'wb') as f:
                f.write(base64.b64decode(c['circuit_schematic']))
        
        md_path = os.path.join(os.getcwd(), 'detailed_coil_report.md')
        with open(md_path, 'w') as f:
            f.write(content)
            
        pdf_path = os.path.join(os.getcwd(), 'NeuroPulse_Coil_Report.pdf')
        generate_pdf.md_to_pdf(md_path, pdf_path)
        
        return send_file(pdf_path, as_attachment=True, download_name='NeuroPulse_Detailed_Report.pdf')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/schematics/generate', methods=['GET'])
def generate_schematics():
    """Generates circuit schematics for all coil types."""
    try:
        gen = CircuitSchematicGenerator()
        schematics = gen.generate_all()
        return jsonify({'success': True, 'schematics': schematics})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("MRI RECONSTRUCTION SIMULATOR - ENHANCED")
    print("=" * 80)
    print("Features:")
    print("  ✓ Quantum Vascular Coils (25 designs)")
    print("  ✓ 50-Turn Head Coil (3.2x SNR, 300μm resolution)")
    print("  ✓ Statistical Adaptive Pulse Sequences")

    print("  ✓ Ultra-High Resolution Neurovasculature")
    print("=" * 80)
    print("Server running on http://0.0.0.0:5002")
    print("=" * 80)
    app.run(host='0.0.0.0', port=5002, debug=True)
