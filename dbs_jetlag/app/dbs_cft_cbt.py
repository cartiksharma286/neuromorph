import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

st.title("DBS & Continued Fractions for Jet Lag Recovery")
st.write("""
This app simulates deep brain stimulation (DBS) paradigms using continued fractions for optimal treatment scheduling, and integrates cognitive behavioral therapy (CBT) modules for holistic jet lag recovery.
""")

# Parameters
time_steps = st.slider("Time Steps", 50, 500, 200)
neurons = st.slider("Number of Neurons", 10, 200, 50)
dbs_intensity = st.slider("DBS Intensity", 0.0, 2.0, 1.0)
continued_fraction_depth = st.slider("Continued Fraction Depth", 1, 10, 4)
cbt_sessions = st.slider("CBT Sessions", 1, 20, 5)

# Initial state
np.random.seed(42)
activity = np.random.rand(neurons)
weights = np.random.rand(neurons, neurons) * 0.1
activity_history = [activity.copy()]

# Continued fraction for DBS timing
cf_terms = [2 + i for i in range(continued_fraction_depth)]
cf = Fraction(cf_terms[0])
for t in cf_terms[1:]:
    cf = t + Fraction(1, cf)
dbs_schedule = [(i % int(cf)) == 0 for i in range(time_steps)]

cbt_effect = np.linspace(0.1, 0.5, cbt_sessions)
cbt_timepoints = np.linspace(0, time_steps-1, cbt_sessions, dtype=int)

for t in range(time_steps):
    # DBS applied at continued fraction intervals
    if dbs_schedule[t]:
        activity += dbs_intensity * np.dot(weights, activity)
    # CBT session effect
    if t in cbt_timepoints:
        activity += cbt_effect[list(cbt_timepoints).index(t)]
    # Normalize activity
    activity = np.clip(activity, 0, 1)
    activity_history.append(activity.copy())

activity_history = np.array(activity_history)

# Visualization
st.subheader("Neuronal Activity Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(activity_history.T, aspect='auto', cmap='plasma', interpolation='nearest')
ax.set_xlabel("Time Step")
ax.set_ylabel("Neuron Index")
ax.set_title("Activity Heatmap")
st.pyplot(fig)

st.write("""
- **DBS** is scheduled using continued fractions for optimal periodicity.
- **CBT** sessions are interleaved to support cognitive recovery.
- Adjust parameters to explore different paradigms for jet lag repair.
""")
