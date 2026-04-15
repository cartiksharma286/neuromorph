with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/simulator_core.py', 'r') as f:
    content = f.read()

addition = """        elif coil_type == 'qubit_photonic_coil':
            # Qubit Photonic Circuitry Array
            num_elements = 16
            for i in range(num_elements):
                coils_data.append(base_sens * (1.0 + 0.3*np.random.rand(n, n) + 0.2*np.sin(i*x/n)))

        elif coil_type == 'geodesic_chassis':"""

content = content.replace("        elif coil_type == 'geodesic_chassis':", addition)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/simulator_core.py', 'w') as f:
    f.write(content)
