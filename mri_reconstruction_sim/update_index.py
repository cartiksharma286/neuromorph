import re

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/templates/index.html', 'r') as f:
    content = f.read()

# Add sequences
seq_addition = """                    <option value="positron_knocking">⚛️ Positron Knocking Thermometry (Optimal SNR)</option>
                    <option value="qml_photonic_therm_v1">🔮 QML Photonic Qubit Thermometry v1</option>
                    <option value="qml_photonic_therm_v2">⚛️ QML Photonic Qubit Thermometry v2</option>"""

content = content.replace('<option value="positron_knocking">⚛️ Positron Knocking Thermometry (Optimal SNR)</option>', seq_addition)

# Add coil
coil_addition = """                    <option value="gnn_coupling">🕸️ GNN Coupling-Aware Array (30ch)</option>
                    <option value="qubit_photonic_coil">💎 Qubit Photonic Circuitry Array</option>"""

content = content.replace('<option value="gnn_coupling">🕸️ GNN Coupling-Aware Array (30ch)</option>', coil_addition)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/templates/index.html', 'w') as f:
    f.write(content)
