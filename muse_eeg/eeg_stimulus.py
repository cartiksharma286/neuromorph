# Placeholder for EEG stimulus-response experiments

import numpy as np

def run_stimulus_response_experiment(eeg_data, stimulus):
    # Simulate a response by adding stimulus to EEG data (placeholder)
    eeg = np.array(eeg_data)
    stim = np.array(stimulus)
    if stim.shape[0] != eeg.shape[1]:
        return "Stimulus must match number of sensors (40)"
    response = eeg + stim
    return response
