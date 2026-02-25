# Finite Mathematical Derivations of Pulse Sequences

## 1. Standard Spin Echo (SE)

The Spin Echo sequence is the gold standard for T1 and T2 weighted imaging. The signal intensity $S$ is derived from the Bloch equations describing the magnetization vector $\vec{M}$.

The transverse magnetization magnitude at echo time $TE$ is given by:

$$ S_{SE} = k \cdot \rho \cdot \left( 1 - e^{-TR/T_1} \right) \cdot e^{-TE/T_2} $$

Where:
- $\rho$ is the proton density.
- $TR$ is the Repetition Time.
- $TE$ is the Echo Time.
- $T_1$ is the longitudinal relaxation time.
- $T_2$ is the transverse relaxation time.
- $k$ is a system constant (gain).

## 2. Gradient Recalled Echo (GRE)

Gradient Echo sequences utilize a flip angle $\alpha$ less than $90^\circ$ and do not use a $180^\circ$ refocusing pulse, making them sensitive to field inhomogeneities ($T_2^*$).

The steady-state signal intensity for a GRE sequence is:

$$ S_{GRE} = k \cdot \rho \cdot \frac{(1 - e^{-TR/T_1}) \sin \alpha}{1 - e^{-TR/T_1} \cos \alpha} \cdot e^{-TE/T_2^*} $$

The **Ernst Angle** $\alpha_E$ maximizes the signal for a given $TR$ and $T_1$:

$$ \alpha_E = \arccos(e^{-TR/T_1}) $$

## 3. Inversion Recovery (IR) & FLAIR

Inversion Recovery sequences begin with a $180^\circ$ inversion pulse to manipulate longitudinal magnetization contrast, defined by the Inversion Time $TI$.

The signal magnitude is:

$$ S_{IR} = k \cdot \rho \cdot \left| 1 - 2e^{-TI/T_1} + e^{-TR/T_1} \right| \cdot e^{-TE/T_2} $$

**Fluid Attenuated Inversion Recovery (FLAIR)** suppresses fluid (CSF) signal by setting $TI$ such that the longitudinal magnetization of CSF is null at the time of the $90^\circ$ excitation pulse:

$$ TI_{null} = T_{1,fluid} \cdot \ln(2) $$

## 4. Balanced Steady-State Free Precession (bSSFP)

bSSFP (or TrueFISP) maintains a coherent transverse steady state. The signal intensity is a complex function of $T_1/T_2$ ratio and the resonance offset angle $\beta$.

$$ S_{bSSFP} = M_0 \sin \alpha \cdot \frac{1 - e^{-TR/T_1}}{1 - (e^{-TR/T_1} - e^{-TR/T_2})\cos \alpha - e^{-TR/T_1}e^{-TR/T_2}} \cdot e^{-TE/T_2} $$

For $\alpha \approx 60-90^\circ$ and $TR \ll T_2$:

$$ S_{bSSFP} \approx \frac{M_0}{2} \sqrt{\frac{T_2}{T_1}} $$

## 5. Statistical Adaptive Sequences

Adaptive sequences optimize scan parameters ($\theta$) in real-time based on acquired k-space statistics. We define a Contrast-to-Noise (CNR) objective function $J(\theta)$ to be maximized:

$$ J(\theta_{TR, TE}) = \frac{|S_{GM}(\theta) - S_{WM}(\theta)|}{\sigma_{noise}} - \lambda \cdot TR $$

Where $\lambda$ is a penalty term for scan time.

### Bayesian Tissue Estimation
Tissue parameters $\hat{T}_1, \hat{T}_2$ are estimated from the histogram of image intensities $I(x)$ using a Gaussian Mixture Model (GMM):

$$ P(I(x) | \mu_k, \sigma_k) = \sum_{k=1}^{K} \pi_k \mathcal{N}(I(x) | \mu_k, \sigma_k) $$

The posterior update for the mean intensity $\mu$ uses a conjugate prior:

$$ \mu_{post} = \frac{\frac{\mu_{data}}{\sigma^2} + \frac{\mu_{prior}}{\sigma_{prior}^2}}{\frac{1}{\sigma^2} + \frac{1}{\sigma_{prior}^2}} $$

## 6. Stroke Imaging: Elliptic Modular Forms

The **Stroke Imaging (Elliptic Modular)** sequence models diffusion signal attenuation in ischemic tissue using concepts from number theory, specifically elliptic modular forms.

The signal decay is modulated by the modular discriminant $\Delta(\tau)$, where $\tau$ is a complex diffusion parameter:

$$ S_{Stroke} = S_0 \cdot e^{-b \cdot D} \cdot (1 + \gamma |\Delta(\tau)|) $$

Approximating the modular discriminant $\Delta(\tau)$ using the Fourier series expansion in terms of the nome $q = e^{2\pi i \tau}$:

$$ \Delta(\tau) \approx q \prod_{n=1}^{\infty} (1 - q^n)^{24} $$

Here, $\tau$ relates to the local texture heterogeneity (entropy) of the tissue. In the penumbra (ischemic but viable tissue), the microstructural disorder alters $\tau$, enhancing the contrast via the modular form weighting.
