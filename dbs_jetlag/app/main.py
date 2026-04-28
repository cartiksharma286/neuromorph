import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

st.title("Hebbian Amplification for Jet Lag Mitigation")
st.write("""
This app simulates the effects of Hebbian amplification to mitigate jet lag, using deep brain stimulation (DBS) as the representational energy transfer mechanism and congruential prime repair for neuronal pruning.
""")

# Parameters
time_steps = st.slider("Time Steps", 50, 500, 200)
neurons = st.slider("Number of Neurons", 10, 200, 50)
hebbian_strength = st.slider("Hebbian Amplification Strength", 0.01, 1.0, 0.1)
dbs_intensity = st.slider("DBS Intensity", 0.0, 2.0, 1.0)
prune_prime = st.slider("Congruential Prime for Pruning", 2, 23, 7)

# Initial state
np.random.seed(42)
activity = np.random.rand(neurons)
weights = np.random.rand(neurons, neurons) * 0.1

activity_history = [activity.copy()]

for t in range(time_steps):
    # Hebbian learning
    delta_w = hebbian_strength * np.outer(activity, activity)
    weights += delta_w
    # Deep brain stimulation (energy transfer)
    activity += dbs_intensity * np.dot(weights, activity)
    # Congruential prime repair (pruning)
    prune_indices = np.arange(neurons)[np.arange(neurons) % prune_prime == 0]
    activity[prune_indices] *= 0.5  # prune by halving activity
    # Normalize activity
    activity = np.clip(activity, 0, 1)
    activity_history.append(activity.copy())

activity_history = np.array(activity_history)

# Visualization
st.subheader("Neuronal Activity Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(activity_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xlabel("Time Step")
ax.set_ylabel("Neuron Index")
ax.set_title("Activity Heatmap")
st.pyplot(fig)

st.write("""
- **Hebbian Amplification** strengthens connections based on co-activity.
- **DBS** boosts overall energy transfer.
- **Congruential Prime Repair** prunes neurons at prime intervals to optimize recovery.
""")
