import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipkinc

# 1. Efficacy vs. Time (Sigmoid)
t = np.linspace(0, 48, 100)
efficacy = 20 + 75 / (1 + np.exp(-0.16 * (t - 16)))
plt.figure(figsize=(6,4))
plt.plot(t, efficacy, label='DBS Efficacy (Sigmoid)')
plt.xlabel('Time (months)')
plt.ylabel('Efficacy')
plt.title('DBS Efficacy vs. Time')
plt.legend()
plt.tight_layout()
plt.savefig('dbs/plots/plot_efficacy.png')
plt.close()

# 2. Response vs. Time (Elliptic Integral)
k = 0.7
omega = 0.2
response = [ellipkinc(omega*tt, k) for tt in t]
plt.figure(figsize=(6,4))
plt.plot(t, response, label='DBS Response (Elliptic Integral)')
plt.xlabel('Time (months)')
plt.ylabel('Response')
plt.title('DBS Response vs. Time')
plt.legend()
plt.tight_layout()
plt.savefig('dbs/plots/plot_response.png')
plt.close()

# 3. Combined Protocol
alpha, beta = 0.6, 0.4
combined = alpha * efficacy + beta * np.array(response)
plt.figure(figsize=(6,4))
plt.plot(t, combined, label='Combined Protocol')
plt.xlabel('Time (months)')
plt.ylabel('Protocol Value')
plt.title('Combined DBS Protocol')
plt.legend()
plt.tight_layout()
plt.savefig('dbs/plots/plot_combined.png')
plt.close()
