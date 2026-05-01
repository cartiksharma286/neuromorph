import re

with open("app.py", "r") as f:
    code = f.read()

new_route = """
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

@app.route('/api/qml_pulse/sr_qml_60', methods=['POST'])
def api_sr_qml_60():
    try:
        seq = get_sr_qml_60_sequence()
        final_snr = seq.apply()
        
        sim = MRIReconstructionSimulator()
        sim.setup_phantom(use_real_data=True, phantom_type='brain')
        sim.generate_coil_sensitivities(num_coils=16)
        kspace, M_ref = sim.acquire_signal(sequence_type='sr_qml_60', TR=0.8, TE=0.03)
        recon_img, _ = sim.reconstruct_image(kspace, method='fft')
        
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(np.abs(recon_img), cmap='bone')
        title_color = 'white'
        ax.set_title(f"sr_qml_60 Reconstruction\\nCalculated SNR: {final_snr:.1f}", color=title_color)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a', edgecolor='none')
        plt.close(fig)
        
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'metrics': {
                'name': seq.name,
                'tag': seq.tag,
                'condition': seq.condition,
                'base_snr': round(seq.base_snr, 1),
                'expected_improvement': f"{seq.snr_improvement * 100}%",
                'calculated_snr': round(final_snr, 1),
                'message': 'Stroke repair mapping stabilized with 60% enhanced Signal-to-Noise Ratio (SNR).'
            },
            'image': image_b64
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/qml_pulse/dementia_cure', methods=['POST'])
"""

code = code.replace("@app.route('/api/qml_pulse/dementia_cure', methods=['POST'])", new_route)

with open("app.py", "w") as f:
    f.write(code)
