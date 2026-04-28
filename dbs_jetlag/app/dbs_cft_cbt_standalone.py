import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# Parameters
TIME_STEPS = 200
NEURONS = 50
DBS_INTENSITY = 1.0
CONTINUED_FRACTION_DEPTH = 4
CBT_SESSIONS = 5

np.random.seed(42)
activity = np.random.rand(NEURONS)
weights = np.random.rand(NEURONS, NEURONS) * 0.1
activity_history = [activity.copy()]

# Continued fraction for DBS timing
cf_terms = [2 + i for i in range(CONTINUED_FRACTION_DEPTH)]
cf = Fraction(cf_terms[0])
for t in cf_terms[1:]:
    cf = t + Fraction(1, cf)
dbs_schedule = [(i % int(cf)) == 0 for i in range(TIME_STEPS)]

cbt_effect = np.linspace(0.1, 0.5, CBT_SESSIONS)
cbt_timepoints = np.linspace(0, TIME_STEPS-1, CBT_SESSIONS, dtype=int)

for t in range(TIME_STEPS):
    # DBS applied at continued fraction intervals
    if dbs_schedule[t]:
        activity += DBS_INTENSITY * np.dot(weights, activity)
    # CBT session effect
    if t in cbt_timepoints:
        activity += cbt_effect[list(cbt_timepoints).index(t)]
    # Normalize activity
    activity = np.clip(activity, 0, 1)
    activity_history.append(activity.copy())

activity_history = np.array(activity_history)

# Visualization
plt.figure(figsize=(10, 4))
plt.imshow(activity_history.T, aspect='auto', cmap='plasma', interpolation='nearest')
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")
plt.title("Neuronal Activity Over Time (DBS + CBT)")
plt.colorbar(label="Activity Level")
plt.tight_layout()
plt.show()

print("\nSimulation complete. The heatmap shows neuronal activity over time with DBS and CBT interventions for jet lag recovery.")
