Segmentation, registration, drilling and implant design for dental surgery robot with SLAM.

Research-only drill-cavity preprint
-----------------------------------
`generate_fea_cavity_preprint.py` produces
`finite_element_drill_cavity_recurrent_primes_preprint.pdf`. It documents the
finite-element formulation and the deterministic 2-D material-field surrogate
used by `drill_path_planner.py`, including its bounded recurrent-prime path.

Regenerate it from the repository root with:

    python3 dental_robot_surgery/generate_fea_cavity_preprint.py

The simulation and PDF are not patient-specific, are not connected to drill
hardware, and must not be used for clinical planning or decision-making.
