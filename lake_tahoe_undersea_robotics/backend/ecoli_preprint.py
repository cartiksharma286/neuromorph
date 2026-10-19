"""Independent Nature-style manuscript with finite NMR spectroscopy, chemometrics and SLAM equations."""

from io import BytesIO
from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.ecoli_analysis import Inputs, analyze


PDF_NAME = "Nature_Style_Ontario_Ecoli_NMR_SLAM_Preprint.pdf"


def render_preprint(report: dict) -> bytes:
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    story = []

    def text(value: str, style: str = "BodyText") -> None:
        story.extend([Paragraph(value, styles[style]), Spacer(1, 8)])

    def plot(title: str, series: list, labels: list, xlabel: str) -> None:
        drawing = Drawing(480, 220)
        drawing.add(String(20, 204, title, fontSize=10))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 55, 60, 395, 130
        chart.data = series
        for i, label in enumerate(labels):
            color = [colors.teal, colors.darkorange, colors.purple, colors.darkgreen][i % 4]
            chart.lines[i].strokeColor = color
            drawing.add(String(25 + 115*i, 12, label, fontSize=8, fillColor=color))
        drawing.add(chart)
        drawing.add(String(180, 34, xlabel, fontSize=9))
        story.append(drawing)

    text("Advanced Nuclear Magnetic Resonance (NMR) Spectroscopy and Joint Robotic SLAM for "
         "In-Situ E. coli Water-Sample Detection in Lake Ontario", "Title")
    text("Independent Nature-style research preprint | In lieu of quantum machine learning | "
         "Computational physics & chemometrics demonstrator | Not peer reviewed")
    text("Abstract", "Heading1")
    text("We formulate an in-situ water monitoring framework combining high-resolution 1H-NMR "
         "metabolomic profiling, Carr-Purcell-Meiboom-Gill (CPMG) spin-echo T2 relaxation editing, "
         "Diffusion-Ordered Spectroscopy (DOSY), and regularized chemometric deconvolution in lieu of "
         "quantum machine learning. Autonomous sampling is co-registered using planar Extended Kalman "
         "Filter Simultaneous Localization and Mapping (EKF-SLAM) with Joseph-stabilized covariance "
         "propagation. All spectral, relaxation, and kinematic trajectories are generated from reproducible "
         "finite equations. This research manuscript presents physical and numerical models; no environmental "
         "clinical diagnosis or water-safety clearance is implied.")
    text(report["disclaimer"])
    text("1. Water Sample Characteristics & Sensor Parameterization", "Heading1")
    table = Table([["Input Parameter (unit encoded in name)", "Assumed Value"]] +
                  [[k, str(v)] for k, v in report["inputs"].items()], colWidths=[320, 150])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey)]))
    story.append(table)
    text("Concentrations represent user-specified environmental scenarios: colony-forming units (CFU/100 mL) "
         "for E. coli burden, mg/L for nitrate, phosphate, and dissolved organic carbon (DOC), and NTU for "
         "turbidity. Spectroscopic observables are synthesized from these ground truths under finite signal-to-noise ratios.")
    for row in report["spectra"]["characteristics"]:
        text(f"<b>{row['band']}:</b> {row['assignment']}.")

    text("2. Finite NMR Spectroscopy Physics & Relaxation Equations", "Heading1")
    text("Nuclear magnetization dynamics under a static field B<sub>0</sub> along z and RF excitation B<sub>1</sub> "
         "are governed by the Bloch equations in the rotating frame (omega = omega<sub>0</sub>):<br/>"
         "dM<sub>x</sub>/dt = -M<sub>x</sub> / T<sub>2</sub>, &nbsp;&nbsp; "
         "dM<sub>y</sub>/dt = -M<sub>y</sub> / T<sub>2</sub> + gamma B<sub>1</sub> M<sub>z</sub>, &nbsp;&nbsp; "
         "dM<sub>z</sub>/dt = -gamma B<sub>1</sub> M<sub>y</sub> - (M<sub>z</sub> - M<sub>0</sub>) / T<sub>1</sub>. &nbsp;&nbsp; (1)")
    text("The complex Free Induction Decay (FID) signal s(t) is a superposition of damped rotating dipoles:<br/>"
         "s(t) = sum<sub>k=1</sub><sup>K</sup> A<sub>k</sub> exp(i 2 pi nu<sub>k</sub> t - t / T<sub>2,k</sub>) exp(i phi<sub>k</sub>). &nbsp;&nbsp; (2)")
    text("Fourier transformation yields the frequency spectrum S(nu), displaying Lorentzian absorption lineshapes:<br/>"
         "S(nu) = Re{ integral<sub>0</sub><sup>infinity</sup> s(t) exp(-i 2 pi nu t) dt } = "
         "sum<sub>k=1</sub><sup>K</sup> A<sub>k</sub> [ (Gamma<sub>k</sub> / 2 pi) / ((nu - nu<sub>k</sub>)<sup>2</sup> + (Gamma<sub>k</sub>/2)<sup>2</sup>) ]. &nbsp;&nbsp; (3)")
    text("Chemical shifts delta are normalized in parts-per-million (ppm): delta = (nu - nu<sub>ref</sub>) / nu<sub>0</sub> * 10<sup>6</sup>. "
         "Primary bacterial markers include the lactate methyl doublet at delta = 1.33 ppm (J = 7.0 Hz), acetate singlet at 1.92 ppm, "
         "peptidoglycan envelope fragments at 2.05 ppm, succinate singlet at 2.40 ppm, and formate singlet at 8.45 ppm.")
    text("To separate low-molecular-weight microbial metabolites from broad background humic/protein macromolecules, "
         "the Carr-Purcell-Meiboom-Gill (CPMG) pulse sequence [90<sub>x</sub> - (tau - 180<sub>y</sub> - tau)<sub>n</sub>] "
         "applies T<sub>2</sub> relaxation filtering:<br/>"
         "I(2n tau) = I<sub>macro</sub> exp(-2n tau / T<sub>2,macro</sub>) + I<sub>metabolite</sub> exp(-2n tau / T<sub>2,metabolite</sub>) + I<sub>solvent</sub> exp(-2n tau / T<sub>2,solvent</sub>), &nbsp;&nbsp; (4)<br/>"
         "where T<sub>2,macro</sub> approx 28 ms and T<sub>2,metabolite</sub> approx 220 ms.")
    text("Diffusion-Ordered Spectroscopy (DOSY) incorporates pulsed field gradients (PFG) to resolve translational mobility via the Stejskal-Tanner equation:<br/>"
         "ln[ I(g) / I(0) ] = -gamma<sup>2</sup> g<sup>2</sup> delta<sub>grad</sub><sup>2</sup> (Delta - delta<sub>grad</sub> / 3) D, &nbsp;&nbsp; (5)<br/>"
         "where intact bacterial cells exhibit D approx 1.2 * 10<sup>-12</sup> m<sup>2</sup>/s, while small organic acid metabolites exhibit D approx 0.85 * 10<sup>-9</sup> m<sup>2</sup>/s.")

    s = report["spectra"]
    plot("Figure 1. High-Resolution 1H-NMR Spectrum (0.0 to 10.0 ppm)",
         [list(zip(s["chemical_shift_ppm"], s["nmr_intensity_au"]))], ["1H-NMR Spectrum"], "Chemical shift delta (ppm)")
    plot("Figure 2. CPMG T2 Relaxation & DOSY Diffusion Attenuation",
         [list(zip(s["cpmg_time_ms"], s["cpmg_intensity_au"])),
          list(zip([t * 0.8 for t in s["cpmg_time_ms"]], s["dosy_attenuation_au"]))],
         ["CPMG T2 decay", "DOSY attenuation"], "Echo time tau (ms) / Normalized gradient")

    text("3. Regularized Chemometric Deconvolution in Lieu of Quantum Machine Learning", "Heading1")
    text("Rather than relying on speculative, non-entangled quantum feature maps (which offer no proven advantage for 1D chemical spectra), "
         "we implement physically interpretable regularized chemometric deconvolution and multivariate Ridge discriminant analysis.<br/>"
         "The measured spectral vector s in R<sup>M</sup> (M = 401 points) is projected onto a calibrated metabolite basis matrix Phi in R<sup>M x P</sup>:<br/>"
         "s = Phi c + epsilon, &nbsp;&nbsp; epsilon ~ N(0, sigma<sup>2</sup> I). &nbsp;&nbsp; (6)")
    text("The optimal metabolite loading vector c_hat is resolved through Tikhonov regularized inversion:<br/>"
         "c_hat = (Phi<sup>T</sup> Phi + lambda I)<sup>-1</sup> Phi<sup>T</sup> s. &nbsp;&nbsp; (7)")
    text("Extracted loadings and T<sub>2</sub>/DOSY invariants are concatenated into feature vector x in R<sup>D</sup>. "
         "The multivariate Ridge classifier evaluates:<br/>"
         "y = w<sup>T</sup> x + b, &nbsp;&nbsp; w = (X<sub>train</sub><sup>T</sup> X<sub>train</sub> + lambda<sub>2</sub> I)<sup>-1</sup> X<sub>train</sub><sup>T</sup> y<sub>train</sub>. &nbsp;&nbsp; (8)")
    ch = report.get("chemometrics", report.get("qml", {}))
    text(f"Chemometric validation (180 training / 60 held-out samples): Accuracy = {ch['accuracy']:.3f}, "
         f"Sensitivity = {ch['sensitivity']:.3f}, Specificity = {ch['specificity']:.3f}, "
         f"Unregularized baseline accuracy = {ch['linear_baseline_accuracy']:.3f}. Confusion counts: {ch['confusion']}. "
         f"Current sample discriminant score = {ch['decision_score']:.3f} ({ch['synthetic_class']}).")

    text("4. Underwater Robotic Planar SLAM Formulations", "Heading1")
    text("The autonomous submersible maintains joint pose and landmark state vector x = [p<sub>x</sub>, p<sub>y</sub>, l<sub>1x</sub>, l<sub>1y</sub>, ..., l<sub>3y</sub>]<sup>T</sup> in R<sup>8</sup>. "
         "Kinematic dead reckoning propagates state and covariance under control input u<sub>k</sub>:<br/>"
         "x<sub>k</sub><sup>-</sup> = x<sub>k-1</sub> + B u<sub>k</sub>, &nbsp;&nbsp; P<sub>k</sub><sup>-</sup> = P<sub>k-1</sub> + Q. &nbsp;&nbsp; (9)")
    text("Cartesian acoustic sonar observations of benthic landmarks z<sub>jk</sub> = l<sub>j</sub> - p<sub>k</sub> + v<sub>k</sub> update the filter via Joseph-stabilized covariance propagation:<br/>"
         "S<sub>k</sub> = H P<sub>k</sub><sup>-</sup> H<sup>T</sup> + R, &nbsp;&nbsp; "
         "K<sub>k</sub> = P<sub>k</sub><sup>-</sup> H<sup>T</sup> S<sub>k</sub><sup>-1</sup>, &nbsp;&nbsp; (10)<br/>"
         "x<sub>k</sub> = x<sub>k</sub><sup>-</sup> + K<sub>k</sub> (z<sub>k</sub> - H x<sub>k</sub><sup>-</sup>), &nbsp;&nbsp; "
         "P<sub>k</sub> = (I - K<sub>k</sub> H) P<sub>k</sub><sup>-</sup> (I - K<sub>k</sub> H)<sup>T</sup> + K<sub>k</sub> R K<sub>k</sub><sup>T</sup>. &nbsp;&nbsp; (11)")
    sl = report["slam"]
    text(f"Planar SLAM positional RMSE against synthetic ground truth: {sl['rmse_m']:.3f} m "
         f"(vs {sl['dead_reckoning_rmse_m']:.3f} m for uncorrected dead reckoning).")
    plot("Figure 3. Benthic Survey Track & Joint SLAM Landmark Estimation (m)",
         [sl["truth_xy_m"], sl["estimated_xy_m"], sl["dead_reckoning_xy_m"]],
         ["Ground Truth", "Joint SLAM Track", "Dead Reckoning"], "Local Easting x (m)")

    text("5. Temporal Water-Sample Kinetics & Long-Term Source Control Scenarios", "Heading1")
    text("Assumed first-order biological decay governs in-vitro sample kinetics over 72 hours:<br/>"
         "C(h) = C(0) exp(-k<sub>decay</sub> h), &nbsp;&nbsp; C<sub>bounds</sub>(h) = C(h) exp(+/- 0.25 h / 72). &nbsp;&nbsp; (12)")
    text("Hypothetical 10-year catchment remediation trajectory models watershed intervention at fractional reduction rate r:<br/>"
         "C<sub>remediated</sub>(y) = C(0) exp(-r<sub>intervention</sub> y), &nbsp;&nbsp; y in {0, 1, ..., 10}. &nbsp;&nbsp; (13)")
    trajectory = report["sample_trajectory"]
    plot("Figure 4. 72-Hour In-Vitro Sample Decay Dynamics (CFU/100 mL)",
         [[(v["hour"], v[key]) for v in trajectory] for key in ("ecoli_cfu_100ml", "low", "high")],
         ["Central model", "Lower bound", "Upper bound"], "Elapsed sample time (hours)")
    plot("Figure 5. Ten-Year Catchment Source-Control Horizon (CFU/100 mL)",
         [[(v["year"], v[key]) for v in report["recovery"]]
          for key in ("baseline_cfu_100ml", "intervention_cfu_100ml")],
         ["Status quo baseline", "Remediation scenario"], "Elapsed calendar year")

    text("6. Analytical Limitations & Regulatory Standards", "Heading1")
    text("Spectroscopic signals alone do not confirm microbial viability, serotype (e.g. O157:H7), or enterotoxigenic pathogenicity. "
         "Metabolic byproducts (acetate, lactate, formate) can arise from ubiquitous environmental bacteria, algal decomposition, or agricultural runoff. "
         "NMR sensitivity in natural waters typically requires pre-concentration (e.g. tangential flow filtration or solid-phase extraction) "
         "to overcome sub-micromolar detection limits. Confirmatory microbiology (e.g., US EPA Method 1603 membrane filtration on modified mTEC agar, "
         "or ISO 9308-1 chromogenic coliform culture) remains mandatory for regulatory water-safety determination.")

    text("References & Methodological Standards", "Heading1")
    text("1. Bloch, F. Nuclear induction. Phys. Rev. 70, 460-474 (1946).<br/>"
         "2. Meiboom, S. & Gill, D. Modified spin-echo method for measuring nuclear relaxation times. Rev. Sci. Instrum. 29, 688-691 (1958).<br/>"
         "3. Johnson, C. S. Diffusion ordered nuclear magnetic resonance spectroscopy: principles and applications. Prog. Nucl. Magn. Reson. Spectrosc. 34, 203-256 (1999).<br/>"
         "4. Durrant-Whyte, H. & Bailey, T. Simultaneous localization and mapping: part I. IEEE Robot. Autom. Mag. 13, 99-110 (2006).<br/>"
         "5. US EPA Method 1603: Escherichia coli in water by membrane filtration using modified membrane-thermotolerant Escherichia coli agar (2014).")

    SimpleDocTemplate(buffer, title="Ontario E. coli NMR Spectroscopy and SLAM Research Preprint",
                      author="Neuromorph Computational Undersea Robotics Consortium").build(story)
    return buffer.getvalue()


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / PDF_NAME
    path.write_bytes(render_preprint(analyze(Inputs())))
    print(path)
