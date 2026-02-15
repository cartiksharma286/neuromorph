.\" **** Custom macro definitions *********************************
.\" * Super/subscript
.\" (https://lists.gnu.org/archive/html/groff/2012-07/msg00046.html)
.ds { \v'-0.3m'\\s[\\n[.s]*9u/12u]
.ds } \s0\v'0.3m'
.ds < \v'0.3m'\s[\\n[.s]*9u/12u]
.ds > \s0\v'-0.3m'
.\" * Horizontal line
.de HLINE
.LP
.ce
\l'20'
..
.\" **** Settings *************************************************
.\" text width
.nr LL 5.5i
.\" left margin
.nr PO 1.25i
.\" top margin
.nr HM 1.25i
.\" bottom margin
.nr FM 1.25i
.\" header/footer width
.nr LT \n[LL]
.\" point size
.nr PS 10p
.\" line height
.nr VS 12p
.\" font family: A, BM, H, HN, N, P, T, ZCM
.fam T
.\" paragraph indent
.nr PI 2m
.\" interparagraph space
.nr PD 0.33v
.\" footnote width
.nr FL \n[LL]
.\" footnote point size
.nr FPS (\n[PS] - 2000)
.\" color used for strikeout
.defcolor strikecolor rgb 0.7 0.7 0.7
.\" color for links (rgb)
.ds PDFHREF.COLOUR   0.35 0.00 0.60
.\" border for links (default none)
.ds PDFHREF.BORDER   0 0 0
.\" point size difference between heading levels
.nr PSINCR 1p
.\" heading level above which point size no longer changes
.nr GROWPS 2
.\" comment these out if you want a dot after section numbers:
.als SN SN-NO-DOT
.als SN-STYLE SN-NO-DOT
.\" pdf outline fold level
.nr PDFOUTLINE.FOLDLEVEL 3
.\" start out in outline view
.pdfview /PageMode /UseOutlines
.\" ***************************************************************
.\" PDF metadata
.pdfinfo /Title ""
.pdfinfo /Author ""
.hy
.EQ
delim @@
.EN
.\" 1 column (use .2C for two column)
.1C
.SH 1
Quantum Phase MRI Simulation: Finite Mathematics & Circuit Derivations
.pdfhref O 1 "Quantum Phase MRI Simulation: Finite Mathematics & Circuit Derivations"
.pdfhref M "quantum-phase-mri-simulation-finite-mathematics-circuit-derivations"
.LP
\f[B]NeuroPulse Physics Engine v3.0\f[R]
.br
\f[B]Date:\f[R] January 10, 2026
.br
\f[B]Classification:\f[R] Advanced Quantum MRI Physics
.HLINE
.SH 2
Executive Summary
.pdfhref O 2 "Executive Summary"
.pdfhref M "executive-summary"
.LP
This report presents the complete finite mathematical framework for
quantum phase-based MRI reconstruction, including discrete Bloch
equation solutions, RF coil circuit topology derivations, and Berry
phase accumulation in topological pulse sequences.
All derivations are presented in discrete form suitable for numerical
implementation.
.HLINE
.SH 2
1. Discrete Bloch Equations for Quantum Phase Evolution
.pdfhref O 2 "1. Discrete Bloch Equations for Quantum Phase Evolution"
.pdfhref M "discrete-bloch-equations-for-quantum-phase-evolution"
.SH 3
1.1 Finite Difference Formulation
.pdfhref O 3 "1.1 Finite Difference Formulation"
.pdfhref M "finite-difference-formulation"
.LP
The continuous Bloch equations are discretized using forward Euler
method with time step @DELTA t@:
.PP
.EQ
M sub x sup {n + 1} = M sub x sup n + DELTA t left [ gamma ( M sub y sup n B sub z \[u2212] M sub z sup n B sub y ) \[u2212] {M sub x sup n} over {T sub 2} right ]
.EN
.PP
.EQ
M sub y sup {n + 1} = M sub y sup n + DELTA t left [ gamma ( M sub z sup n B sub x \[u2212] M sub x sup n B sub z ) \[u2212] {M sub y sup n} over {T sub 2} right ]
.EN
.PP
.EQ
M sub z sup {n + 1} = M sub z sup n + DELTA t left [ gamma ( M sub x sup n B sub y \[u2212] M sub y sup n B sub x ) \[u2212] {M sub z sup n \[u2212] M sub 0} over {T sub 1} right ]
.EN
.PP
where: - @M sup n = ( M sub x sup n , M sub y sup n , M sub z sup n )@
is the magnetization vector at time step @n@ -
@gamma = 42.58 roman " MHz/T"@ is the gyromagnetic ratio -
@B = ( B sub x , B sub y , B sub z )@ is the magnetic field vector -
@T sub 1 , T sub 2@ are relaxation time constants
.SH 3
1.2 Stability Criterion
.pdfhref O 3 "1.2 Stability Criterion"
.pdfhref M "stability-criterion"
.LP
For numerical stability, the time step must satisfy:
.PP
.EQ
DELTA t < 2 over {gamma B sub max + max ( 1 / T sub 1 , 1 / T sub 2 )}
.EN
.HLINE
.SH 2
2. Quantum Berry Phase Accumulation
.pdfhref O 2 "2. Quantum Berry Phase Accumulation"
.pdfhref M "quantum-berry-phase-accumulation"
.SH 3
2.1 Geometric Phase in Parameter Space
.pdfhref O 3 "2.1 Geometric Phase in Parameter Space"
.pdfhref M "geometric-phase-in-parameter-space"
.LP
The Berry phase @gamma sub B@ accumulated during adiabatic evolution
along a closed path @C@ in parameter space is:
.PP
.EQ
gamma sub B ( C ) = i \[u222E] sub C \[u27E8] psi sub n ( bold {R} ) \[u007C] grad sub {bold {R}} \[u007C] psi sub n ( bold {R} ) \[u27E9] \[u22C5] d bold {R}
.EN
.PP
For MRI applications with spatially varying gradients, this becomes:
.PP
.EQ
gamma sub B ( bold {r} ) = int sub 0 sup T bold {A} ( bold {r} , t ) \[u22C5] {d bold {r}} over {d t} d t
.EN
.PP
where @bold {A}@ is the Berry connection (gauge potential).
.SH 3
2.2 Discrete Berry Phase Calculation
.pdfhref O 3 "2.2 Discrete Berry Phase Calculation"
.pdfhref M "discrete-berry-phase-calculation"
.LP
In discrete form with @N@ time steps:
.PP
.EQ
gamma sub B approx sum sub {n = 0} sup {N \[u2212] 1} {bold {A}} sub n \[u22C5] DELTA {bold {r}} sub n
.EN
.PP
where:
.PP
.EQ
{bold {A}} sub n = roman "Im" left [ \[u27E8] psi sub n \[u007C] {psi sub {n + 1} \[u2212] psi sub n} over {DELTA t} \[u27E9] right ]
.EN
.SH 3
2.3 Berry Curvature and Topological Invariant
.pdfhref O 3 "2.3 Berry Curvature and Topological Invariant"
.pdfhref M "berry-curvature-and-topological-invariant"
.LP
The Berry curvature tensor is:
.PP
.EQ
OMEGA sub {i j} ( bold {k} ) = partial sub {k sub i} A sub j \[u2212] partial sub {k sub j} A sub i
.EN
.PP
The Chern number (topological invariant) is:
.PP
.EQ
C = 1 over {2 pi} int sub {B Z} OMEGA sub {x y} ( bold {k} ) d sup 2 k
.EN
.PP
For discrete k-space sampling on an @N times N@ grid:
.PP
.EQ
C approx 1 over {2 pi} sum sub {i = 1} sup N sum sub {j = 1} sup N OMEGA sub {x y} ( k sub i , k sub j ) DELTA k sub x DELTA k sub y
.EN
.HLINE
.SH 2
3. RF Coil Circuit Topology
.pdfhref O 2 "3. RF Coil Circuit Topology"
.pdfhref M "rf-coil-circuit-topology"
.SH 3
3.1 Birdcage Coil Impedance Matrix
.pdfhref O 3 "3.1 Birdcage Coil Impedance Matrix"
.pdfhref M "birdcage-coil-impedance-matrix"
.LP
For an @N@-rung birdcage coil, the impedance matrix @bold {Z}@ is:
.PP
.EQ
Z sub {i j} = left { matrix{
lcol{ {R + j omega L sub {l e g} + 1 over {j omega C sub {r i n g}}} above {1 over {j omega C sub {r i n g}}} above 0 }
lcol{ {i = j} above {\[u007C] i \[u2212] j \[u007C] = 1 roman " or " N \[u2212] 1} above {roman "otherwise"} }
} right ""
.EN
.PP
The resonant frequencies are:
.PP
.EQ
omega sub m = 1 over {sqrt {L sub {l e g} C sub {r i n g}}} {left \[u007C] 2 sin left ( {m pi} over N right ) right \[u007C]} sup {\[u2212] 1} , fwd 100 m = 1 , 2 , ... , N / 2
.EN
.SH 3
3.2 Mutual Inductance Calculation
.pdfhref O 3 "3.2 Mutual Inductance Calculation"
.pdfhref M "mutual-inductance-calculation"
.LP
For two circular loops with radii @a sub 1 , a sub 2@ separated by
distance @d@:
.PP
.EQ
M sub 12 = mu sub 0 sqrt {a sub 1 a sub 2} left [ left ( 2 \[u2212] k sup 2 right ) K ( k ) \[u2212] 2 E ( k ) right ]
.EN
.PP
where: -
@k sup 2 = {4 a sub 1 a sub 2} over {( a sub 1 + a sub 2 ) sup 2 + d sup 2}@
- @K ( k ) , E ( k )@ are complete elliptic integrals
.PP
Discrete approximation using mesh elements:
.PP
.EQ
M sub {i j} approx sum sub {p = 1} sup {N sub 1} sum sub {q = 1} sup {N sub 2} {mu sub 0} over {4 pi} {{bold {J}} sub i sup p \[u22C5] {bold {J}} sub j sup q} over {\[u007C] {bold {r}} sub {i j} sup {p q} \[u007C]} DELTA A sub p DELTA A sub q
.EN
.SH 3
3.3 Quantum Surface Lattice Coupling
.pdfhref O 3 "3.3 Quantum Surface Lattice Coupling"
.pdfhref M "quantum-surface-lattice-coupling"
.LP
For a hexagonal lattice of quantum coil elements, the coupling
Hamiltonian is:
.PP
.EQ
H sub {c o u p l i n g} = sum sub {\[u27E8] i , j \[u27E9]} J sub {i j} ( sigma sub i sup + sigma sub j sup \[u2212] + sigma sub i sup \[u2212] sigma sub j sup + ) + sum sub i h sub i sigma sub i sup z
.EN
.PP
The flux quantization condition:
.PP
.EQ
PHI = n PHI sub 0 , fwd 100 PHI sub 0 = h over {2 e} = 2.067 times 10 sup {\[u2212] 15} roman " Wb"
.EN
.HLINE
.SH 2
4. K-Space Trajectory and Reconstruction
.pdfhref O 2 "4. K-Space Trajectory and Reconstruction"
.pdfhref M "k-space-trajectory-and-reconstruction"
.SH 3
4.1 Discrete Fourier Transform
.pdfhref O 3 "4.1 Discrete Fourier Transform"
.pdfhref M "discrete-fourier-transform"
.LP
The k-space signal is related to the image by discrete Fourier
transform:
.PP
.EQ
S ( k sub x , k sub y ) = sum sub {m = 0} sup {M \[u2212] 1} sum sub {n = 0} sup {N \[u2212] 1} rho ( m , n ) e sup {\[u2212] 2 pi i ( k sub x m / M + k sub y n / N )}
.EN
.PP
The inverse transform for reconstruction:
.PP
.EQ
rho ( m , n ) = 1 over {M N} sum sub {k sub x = 0} sup {M \[u2212] 1} sum sub {k sub y = 0} sup {N \[u2212] 1} S ( k sub x , k sub y ) e sup {2 pi i ( k sub x m / M + k sub y n / N )}
.EN
.SH 3
4.2 Non-Uniform FFT (NUFFT)
.pdfhref O 3 "4.2 Non-Uniform FFT (NUFFT)"
.pdfhref M "non-uniform-fft-nufft"
.LP
For non-Cartesian trajectories, the NUFFT approximation:
.PP
.EQ
S ( {bold {k}} sub j ) approx sum sub {m \[u2208] {N} ( {bold {k}} sub j )} c sub m rho sub m e sup {\[u2212] 2 pi i {bold {k}} sub j \[u22C5] {bold {r}} sub m}
.EN
.PP
where @{N} ( {bold {k}} sub j )@ is the neighborhood of
@{bold {k}} sub j@ and @c sub m@ are Kaiser-Bessel interpolation
coefficients.
.SH 3
4.3 Parallel Imaging with SENSE
.pdfhref O 3 "4.3 Parallel Imaging with SENSE"
.pdfhref M "parallel-imaging-with-sense"
.LP
For @N sub c@ coils with sensitivity maps @C sub j ( bold {r} )@:
.PP
.EQ
S sub j ( bold {k} ) = int C sub j ( bold {r} ) rho ( bold {r} ) e sup {\[u2212] 2 pi i bold {k} \[u22C5] bold {r}} d bold {r}
.EN
.PP
The SENSE reconstruction solves:
.PP
.EQ
rho to \[u0302] = "arg" min sub rho sum sub {j = 1} sup {N sub c} \[u2225] {F} { C sub j rho } \[u2212] S sub j \[u2225] sub 2 sup 2 + lambda R ( rho )
.EN
.PP
Discrete form:
.PP
.EQ
{rho to \[u0302]} sub n = {left ( sum sub {j = 1} sup {N sub c} C sub j sup * C sub j + lambda {bold {L}} sup T bold {L} right )} sup {\[u2212] 1} sum sub {j = 1} sup {N sub c} C sub j sup * {F} sup {\[u2212] 1} { S sub j }
.EN
.HLINE
.SH 2
5. Quantum Low Energy Beam Focusing
.pdfhref O 2 "5. Quantum Low Energy Beam Focusing"
.pdfhref M "quantum-low-energy-beam-focusing"
.SH 3
5.1 Energy Distribution Function
.pdfhref O 3 "5.1 Energy Distribution Function"
.pdfhref M "energy-distribution-function"
.LP
The beam intensity profile with quantum focusing:
.PP
.EQ
I ( bold {r} ) = I sub 0 left [ 1 + eta {H} ( chi ( bold {r} ) ) right ] exp left ( \[u2212] {\[u007C] bold {r} \[u2212] {bold {r}} sub 0 \[u007C] sup 2} over {2 sigma sup 2} right )
.EN
.PP
where: - @{H} ( chi )@ is the entropy-based attention function -
@chi ( bold {r} )@ is the local image complexity measure
.SH 3
5.2 Entropy-Driven Denoising
.pdfhref O 3 "5.2 Entropy-Driven Denoising"
.pdfhref M "entropy-driven-denoising"
.LP
The local entropy at position @bold {r}@ in a window @W@:
.PP
.EQ
{H} ( bold {r} ) = \[u2212] sum sub {i = 1} sup L p sub i ( bold {r} ) log sub 2 p sub i ( bold {r} )
.EN
.PP
where @p sub i ( bold {r} )@ is the probability of intensity level @i@
in window @W@ centered at @bold {r}@.
.PP
Discrete implementation:
.PP
.EQ
{H} sub n = \[u2212] sum sub {i = 0} sup 255 {h sub i sup n} over {N sub w} log sub 2 left ( {h sub i sup n} over {N sub w} right )
.EN
.PP
where @h sub i sup n@ is the histogram count for intensity @i@ in the
@n@-th window.
.HLINE
.SH 2
6. Signal-to-Noise Ratio Analysis
.pdfhref O 2 "6. Signal-to-Noise Ratio Analysis"
.pdfhref M "signal-to-noise-ratio-analysis"
.SH 3
6.1 SNR for Sum-of-Squares Reconstruction
.pdfhref O 3 "6.1 SNR for Sum-of-Squares Reconstruction"
.pdfhref M "snr-for-sum-of-squares-reconstruction"
.LP
For @N sub c@ coils with uncorrelated noise:
.PP
.EQ
roman "SNR" = {sqrt {sum sub {j = 1} sup {N sub c} \[u007C] C sub j rho \[u007C] sup 2}} over {sigma sub n sqrt {N sub c}}
.EN
.SH 3
6.2 g-Factor for Parallel Imaging
.pdfhref O 3 "6.2 g-Factor for Parallel Imaging"
.pdfhref M "g-factor-for-parallel-imaging"
.LP
The geometry factor quantifying SNR loss:
.PP
.EQ
g ( bold {r} ) = sqrt {[ ( {bold {C}} sup H PSI sup {\[u2212] 1} bold {C} ) sup {\[u2212] 1} ] sub {i i} \[u22C5] [ {bold {C}} sup H PSI sup {\[u2212] 1} bold {C} ] sub {i i}}
.EN
.PP
where @PSI@ is the noise covariance matrix.
.PP
For @R@-fold acceleration:
.PP
.EQ
{roman "SNR"} sub {a c c} = {{roman "SNR"} sub {f u l l}} over {g sqrt R}
.EN
.HLINE
.SH 2
7. Shimming Optimization
.pdfhref O 2 "7. Shimming Optimization"
.pdfhref M "shimming-optimization"
.SH 3
7.1 B1+ Homogeneity Objective
.pdfhref O 3 "7.1 B1+ Homogeneity Objective"
.pdfhref M "b1-homogeneity-objective"
.LP
The shimming problem minimizes field inhomogeneity:
.PP
.EQ
{bold {w}} sub {o p t} = "arg" min sub {bold {w}} {left ∥ \[u007C] bold {A} bold {w} \[u007C] \[u2212] bold {T} right ∥} sub 2 sup 2 + lambda \[u2225] bold {w} \[u2225] sub 2 sup 2
.EN
.PP
where: - @bold {A}@ is the @N sub {v o x e l} times N sub {c o i l}@
sensitivity matrix - @bold {T}@ is the target field - @lambda@ is the
regularization parameter
.SH 3
7.2 Iterative Solution
.pdfhref O 3 "7.2 Iterative Solution"
.pdfhref M "iterative-solution"
.LP
Using gradient descent with step size @alpha@:
.PP
.EQ
{bold {w}} sup {k + 1} = {bold {w}} sup k \[u2212] alpha grad sub {bold {w}} left [ \[u2225] \[u007C] bold {A} {bold {w}} sup k \[u007C] \[u2212] bold {T} \[u2225] sub 2 sup 2 + lambda \[u2225] {bold {w}} sup k \[u2225] sub 2 sup 2 right ]
.EN
.PP
Discrete gradient:
.PP
.EQ
grad sub {bold {w}} = 2 {bold {A}} sup H roman "diag" left ( {\[u007C] bold {A} bold {w} \[u007C] \[u2212] bold {T}} over {\[u007C] bold {A} bold {w} \[u007C]} right ) bold {A} bold {w} + 2 lambda bold {w}
.EN
.HLINE
.SH 2
8. Numerical Implementation Details
.pdfhref O 2 "8. Numerical Implementation Details"
.pdfhref M "numerical-implementation-details"
.SH 3
8.1 Grid Resolution
.pdfhref O 3 "8.1 Grid Resolution"
.pdfhref M "grid-resolution"
.LP
For a field of view (FOV) of @L times L@ with @N times N@ pixels:
.PP
.EQ
DELTA x = DELTA y = L over N
.EN
.PP
.EQ
DELTA k sub x = DELTA k sub y = 1 over L
.EN
.SH 3
8.2 Gradient Strength
.pdfhref O 3 "8.2 Gradient Strength"
.pdfhref M "gradient-strength"
.LP
The gradient required for k-space traversal:
.PP
.EQ
G = {k sub {m a x}} over {gamma t sub {r e a d}} = {N / ( 2 L )} over {gamma t sub {r e a d}}
.EN
.PP
For @N = 128@, @L = 25@ cm, @t sub {r e a d} = 10@ ms:
.PP
.EQ
G = {128 / ( 2 times 0.25 )} over {42.58 times 10 sup 6 times 0.01} = 6.0 roman " mT/m"
.EN
.SH 3
8.3 Sampling Theorem
.pdfhref O 3 "8.3 Sampling Theorem"
.pdfhref M "sampling-theorem"
.LP
Nyquist criterion for artifact-free reconstruction:
.PP
.EQ
DELTA k <= 1 over {2 DELTA x}
.EN
.PP
.EQ
k sub {m a x} >= N over {2 L}
.EN
.HLINE
.SH 2
9. Quantum Entanglement Contrast
.pdfhref O 2 "9. Quantum Entanglement Contrast"
.pdfhref M "quantum-entanglement-contrast"
.SH 3
9.1 Entangled State Formulation
.pdfhref O 3 "9.1 Entangled State Formulation"
.pdfhref M "entangled-state-formulation"
.LP
The two-contrast entangled state:
.PP
.EQ
\[u007C] PSI \[u27E9] = 1 over {sqrt 2} ( \[u007C] T 1 \[u27E9] \[u2297] \[u007C] b r i g h t \[u27E9] + \[u007C] T 2 \[u27E9] \[u2297] \[u007C] d a r k \[u27E9] )
.EN
.PP
Measurement yields superposition:
.PP
.EQ
M sub {e n t a n g l e d} = alpha M sub {T 1} + beta M sub {T 2} + gamma M sub {T 1} \[u2299] M sub {T 2}
.EN
.PP
where @\[u2299]@ denotes element-wise product (quantum correlation
term).
.SH 3
9.2 Discrete Implementation
.pdfhref O 3 "9.2 Discrete Implementation"
.pdfhref M "discrete-implementation"
.LP
.EQ
M sub {e n t a n g l e d} sup {i j} = 0.5 M sub {b a s e} sup {i j} + 0.25 M sub {T 1} sup {i j} + 0.25 M sub {T 2} sup {i j}
.EN
.PP
where:
.PP
.EQ
M sub {T 1} sup {i j} = rho sup {i j} left ( 1 \[u2212] e sup {\[u2212] T R / T 1 sup {i j}} right )
.EN
.PP
.EQ
M sub {T 2} sup {i j} = rho sup {i j} e sup {\[u2212] T E / T 2 sup {i j}}
.EN
.HLINE
.SH 2
10. Computational Complexity
.pdfhref O 2 "10. Computational Complexity"
.pdfhref M "computational-complexity"
.SH 3
10.1 FFT Complexity
.pdfhref O 3 "10.1 FFT Complexity"
.pdfhref M "fft-complexity"
.LP
For @N times N@ image:
.PP
.EQ
{O} ( N sup 2 log N )
.EN
.SH 3
10.2 SENSE Reconstruction
.pdfhref O 3 "10.2 SENSE Reconstruction"
.pdfhref M "sense-reconstruction"
.LP
Matrix inversion for each pixel:
.PP
.EQ
{O} ( N sup 2 N sub c sup 3 )
.EN
.SH 3
10.3 Berry Phase Calculation
.pdfhref O 3 "10.3 Berry Phase Calculation"
.pdfhref M "berry-phase-calculation"
.LP
For @N sub t@ time steps and @N sub p@ spatial points:
.PP
.EQ
{O} ( N sub t N sub p )
.EN
.HLINE
.SH 2
11. Advanced Artifact Removal Algorithms
.pdfhref O 2 "11. Advanced Artifact Removal Algorithms"
.pdfhref M "advanced-artifact-removal-algorithms"
.SH 3
11.1 Speckle Reducing Anisotropic Diffusion (SRAD)
.pdfhref O 3 "11.1 Speckle Reducing Anisotropic Diffusion (SRAD)"
.pdfhref M "speckle-reducing-anisotropic-diffusion-srad"
.LP
The SRAD PDE for image @I ( x , y ; t )@ is given by:
.PP
.EQ
{partial I} over {partial t} = grad \[u22C5] [ c ( q ) grad I ]
.EN
.PP
where @c ( q )@ is the diffusion coefficient:
.PP
.EQ
c ( q ) = 1 over {1 + [ q sup 2 ( x , y ; t ) \[u2212] q sub 0 sup 2 ( t ) ] / ( q sub 0 sup 2 ( t ) ( 1 + q sub 0 sup 2 ( t ) ) )}
.EN
.PP
Discrete approximation (Lee Filter equivalent):
.PP
.EQ
{I to \[u0302]} sub {i , j} = {I to \[u203E]} sub {i , j} + k sub {i , j} ( I sub {i , j} \[u2212] {I to \[u203E]} sub {i , j} )
.EN
.PP
where the weighting factor @k sub {i , j}@ depends on local variance
@sigma sup 2@:
.PP
.EQ
k sub {i , j} = {sigma sub {i , j} sup 2} over {sigma sub {i , j} sup 2 + sigma sub {n o i s e} sup 2}
.EN
.SH 3
11.2 Non-Local Means (NLM) Denoising
.pdfhref O 3 "11.2 Non-Local Means (NLM) Denoising"
.pdfhref M "non-local-means-nlm-denoising"
.LP
The discrete NLM equation for a pixel @p@:
.PP
.EQ
N L ( I ) ( p ) = 1 over {Z ( p )} sum sub {q \[u2208] OMEGA} w ( p , q ) I ( q )
.EN
.PP
The weights @w ( p , q )@ depend on patch similarity:
.PP
.EQ
w ( p , q ) = exp left ( \[u2212] {\[u2225] {N} ( p ) \[u2212] {N} ( q ) \[u2225] sub 2 sup 2} over {h sup 2} right )
.EN
.PP
where @{N} ( p )@ is the local neighborhood patch and @h@ is the decay
parameter.
.HLINE
.SH 2
12. Pulse Sequence - Coil Interaction Dynamics
.pdfhref O 2 "12. Pulse Sequence - Coil Interaction Dynamics"
.pdfhref M "pulse-sequence---coil-interaction-dynamics"
.SH 3
12.1 Coil-Specific B1 Field Generation
.pdfhref O 3 "12.1 Coil-Specific B1 Field Generation"
.pdfhref M "coil-specific-b1-field-generation"
.LP
The transverse magnetic field @B sub 1 sup + ( bold {r} )@ generated by
a coil with current density @bold {J} ( bold {r} prime )@ is governed by
the Biot-Savart Law:
.PP
.EQ
B sub 1 sup + ( bold {r} ) = {mu sub 0} over {4 pi} int sub V {bold {J} ( bold {r} prime ) times ( bold {r} \[u2212] bold {r} prime )} over {\[u007C] bold {r} \[u2212] bold {r} prime \[u007C] sup 3} d V prime
.EN
.PP
For a \f[B]Birdcage Coil\f[R] (Quadrature mode), the idealized field is
homogeneous:
.PP
.EQ
B sub 1 sup + = {mu sub 0 I N} over {2 D} x to \[u0302] + i {mu sub 0 I N} over {2 D} y to \[u0302]
.EN
.PP
For a \f[B]Surface Loop Coil\f[R] of radius @a@ at height @z@:
.PP
.EQ
B sub z ( z ) = {mu sub 0 I a sup 2} over {2 ( a sup 2 + z sup 2 ) sup {3 / 2}}
.EN
.SH 3
12.2 Pulse Sequence Excitation Profiles
.pdfhref O 3 "12.2 Pulse Sequence Excitation Profiles"
.pdfhref M "pulse-sequence-excitation-profiles"
.LP
The local flip angle @alpha ( bold {r} )@ induced by a pulse sequence of
duration @tau@ depends on the coil’s sensitivity profile
@S ( bold {r} )@:
.PP
.EQ
alpha ( bold {r} ) = gamma int sub 0 sup tau \[u007C] B sub 1 sup + ( bold {r} , t ) \[u007C] d t approx gamma tau \[u007C] S ( bold {r} ) \[u007C] V sub {p u l s e}
.EN
.PP
\f[B]12.2.1 Spin Echo (SE) in Surface Coils\f[R] The refocusing
(@180 sup \[u2218]@) pulse is highly sensitive to @B sub 1 sup +@
inhomogeneity.
The effective signal @M sub {x y}@ is:
.PP
.EQ
M sub {x y} ( bold {r} ) = M sub 0 sin sup 3 ( alpha ( bold {r} ) ) \[u22C5] e sup {\[u2212] T E / T 2}
.EN
.PP
This cubic dependence implies rapid signal drop-off away from the coil
surface.
.PP
\f[B]12.2.2 Gradient Echo (GRE) Optimization\f[R] To maximize SNR given
coil sensitivity @S ( bold {r} )@, the local Ernst Angle is:
.PP
.EQ
alpha sub E ( bold {r} ) = "arccos" ( e sup {\[u2212] T R / T 1} )
.EN
.PP
Since @alpha@ is fixed by the transmit coil, local signal strength
varies as:
.PP
.EQ
M sub {G R E} ( bold {r} ) = M sub 0 {( 1 \[u2212] e sup {\[u2212] T R / T 1} ) sin ( alpha ( bold {r} ) )} over {1 \[u2212] cos ( alpha ( bold {r} ) ) e sup {\[u2212] T R / T 1}} \[u22C5] S sup * ( bold {r} )
.EN
.SH 3
12.3 Quantum Coil-Sequence Coupling
.pdfhref O 3 "12.3 Quantum Coil-Sequence Coupling"
.pdfhref M "quantum-coil-sequence-coupling"
.LP
For \f[B]Quantum Berry Phase sequences\f[R], the phase accumulation
@PHI sub B@ is topologically protected against linear coil
inhomogeneities (@grad B sub 1@).
The coupling Hamiltonian @H sub {i n t}@ is:
.PP
.EQ
H sub {i n t} = \[u2212] gamma bold {S} \[u22C5] ( {bold {B}} sub 0 + {bold {B}} sub {R F} ( t ) )
.EN
.PP
The geometric phase acquired is proportional to the solid angle @OMEGA@
swept by the effective field vector @{bold {B}} sub {e f f}@ generated
by the coil array:
.PP
.EQ
PHI sub B = \[u2212] 1 over 2 OMEGA ( C ) = \[u2212] 1 over 2 int sub S grad times {bold {A}} sub {c o i l} \[u22C5] d bold {S}
.EN
.PP
This implies that \[lq]Quantum\[rq] sequences are robust against the
@B sub 1@ drop-off characteristic of surface coils, maintaining contrast
fidelity even at depth.
.HLINE
.SH 2
Conclusion
.pdfhref O 2 "Conclusion"
.pdfhref M "conclusion"
.LP
This report provides the complete finite mathematical framework for
quantum phase MRI simulation.
It establishes the rigorous discrete formalism for: 1.
\f[B]Bloch Dynamics\f[R]: Finite difference time-domain evolution of
magnetization.
2.
\f[B]Topological Physics\f[R]: Berry phase accumulation and Chern number
invariants.
3.
\f[B]Hardware Modeling\f[R]: Discrete impedance matrices for Birdcage
and Surface coils.
4.
\f[B]Reconstruction\f[R]: SENSE parallel imaging and Quantum-Statistical
artifact removal.
5.
\f[B]Pulse-Coil Interaction\f[R]: Coupled equations describing
excitation inhomogeneity and quantum protection.
.PP
All equations are presented in forms directly implementable in the
NeuroPulse simulation engine (Python/NumPy), ensuring physically
accurate modeling of quantum topological effects in magnetic resonance
imaging.
.HLINE
.PP
\f[B]References:\f[R] 1.
Berry, M.
V.
(1984).
\[lq]Quantal Phase Factors Accompanying Adiabatic Changes\[rq] 2.
Pruessmann, K.
P.
et al.
(1999).
\[lq]SENSE: Sensitivity Encoding for Fast MRI\[rq] 3.
Hayes, C.
E.
et al.
(1985).
\[lq]An Efficient, Highly Homogeneous Radiofrequency Coil\[rq] 4.
Sodickson, D.
K.
& Manning, W.
J.
(1997).
\[lq]Simultaneous Acquisition of Spatial Harmonics\[rq] 5.
Perona, P.
& Malik, J.
(1990).
\[lq]Scale-space and edge detection using anisotropic diffusion\[rq] 6.
Buades, A.
et al.
(2005).
\[lq]A non-local algorithm for image denoising\[rq] 7.
Hoult, D.
I.
(2000).
\[lq]The Principle of Reciprocity in NMR Microscopy\[rq]
.PP
\f[B]Document ID:\f[R] NP-QP-MRI-2026-001
.br
\f[B]Classification:\f[R] Advanced Physics Research
.br
\f[B]Generated by:\f[R] NeuroPulse Physics Engine v3.0
.pdfsync
