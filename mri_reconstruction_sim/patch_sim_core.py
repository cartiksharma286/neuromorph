with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/simulator_core.py', 'r') as f:
    content = f.read()

addition = """        elif sequence_type == 'qml_photonic_therm_v1':
            from qml_photonic_thermometry import run_qml_thermometry_sim
            qml_sim = run_qml_thermometry_sim(sequence_type, getattr(self, 'last_coil_config', [self.latest_sequence_type])[0])
            t1 = T1_safe
            t2 = T2_safe
            # Enhance contrast based on QML factors
            M = self.pd_map * (1 - np.exp(-TR / t1)) * np.exp(-TE / t2) * (qml_sim['snr']/100.0)
            
        elif sequence_type == 'qml_photonic_therm_v2':
            from qml_photonic_thermometry import run_qml_thermometry_sim
            qml_sim = run_qml_thermometry_sim(sequence_type, getattr(self, 'last_coil_config', [self.latest_sequence_type])[0])
            t1 = T1_safe
            t2 = T2_safe
            # Deep photonic geometry contrast
            M = self.pd_map * (1 - np.exp(-TR / t1)) * np.exp(-TE / t2) * (qml_sim['snr']/80.0)

        elif sequence_type == 'GRE':"""

content = content.replace("        elif sequence_type == 'GRE':", addition)

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/simulator_core.py', 'w') as f:
    f.write(content)
