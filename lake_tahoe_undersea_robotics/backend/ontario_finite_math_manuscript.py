"""Generate a separate, simulation-grounded research manuscript PDF."""

from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from backend.ontario_qubit_spectroscopy_engine import OntarioQubitSpectroscopyEngine


OUTPUT = Path(__file__).resolve().parents[1] / "Lake_Ontario_Finite_Math_Research_Preprint.pdf"


def generate_pdf(output: Path = OUTPUT) -> Path:
    recovery = OntarioQubitSpectroscopyEngine().compute_lake_ontario_10yr_recovery()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Equation", fontName="Times-Roman", fontSize=10,
        leading=16, spaceBefore=6, spaceAfter=8, leftIndent=10,
    ))
    story = []

    def text(content: str, style: str = "BodyText") -> None:
        story.append(Paragraph(content, styles[style]))
        story.append(Spacer(1, 5))

    def equation(content: str) -> None:
        text(content, "Equation")

    def plot(key: str, title: str, unit: str) -> None:
        drawing = Drawing(490, 175)
        drawing.add(String(35, 160, title, fontName="Helvetica-Bold", fontSize=10))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 48, 32, 420, 115
        chart.data = [[(i + 1, float(v)) for i, v in enumerate(recovery[key])]]
        chart.lines[0].strokeColor = colors.HexColor("#007c91")
        chart.lines[0].strokeWidth = 2
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 1, 10
        chart.xValueAxis.valueSteps = [1, 3, 6, 10]
        chart.yValueAxis.labelTextFormat = "%.1f"
        drawing.add(chart)
        drawing.add(String(195, 8, "Elapsed model year", fontSize=9))
        drawing.add(String(48, 148, unit, fontSize=8))
        story.append(drawing)

    text("Lake Ontario and Yellow Creek Ravine: finite mathematics for "
         "qubit-assisted robotic monitoring and ten-year remediation scenarios", "Title")
    text("Independent research preprint | Nature-style article structure")
    text("<b>Status:</b> Original simulation-based manuscript; not published, endorsed, "
         "submitted to, or peer-reviewed by Nature. No institutional affiliations or "
         "experimental authorship are asserted. The results below are outputs of the "
         "local application, not measurements or validated environmental forecasts.")
    text("Abstract", "Heading1")
    text("We formulate a finite-horizon framework connecting hypothetical superconducting "
         "qubit-assisted spectral analysis, autonomous water sampling, and watershed "
         "remediation scenarios. Transmon, coplanar-waveguide, fluid-flow, natural-gradient "
         "and discrete stability equations describe the proposed architecture. Numerical "
         "tables and plots are generated directly from the application's existing recovery "
         "model. At elapsed year 10, modeled phosphorus loading is 6,540.5 tonnes/year "
         "and Yellow Creek E. coli is 148.1 CFU/100 mL, not the asymptotic targets of "
         "6,200 and 78. Hardware performance, quantum advantage, hydraulic feasibility "
         "and regulatory compliance remain unvalidated.")

    text("1. Model provenance and finite horizon", "Heading1")
    text("The data source is OntarioQubitSpectroscopyEngine.compute_lake_ontario_10yr_recovery. "
         "The model uses ten annual samples n = 1,...,10. Its labels span 2026-2035; "
         "a baseline at n = 0 is mathematically separate. A literal 2026-2036 interval "
         "would need an explicitly defined baseline and calendar convention. No measured "
         "time series, fitted parameters, confidence intervals or intervention trials "
         "are supplied by this model.")
    equation("x<sub>n</sub> = x* + (x<sub>0</sub> - x*) exp(-k n), "
             "n in {0,...,10}. &nbsp; (1)")
    text("Equation (1) implies x(n) approaches x* asymptotically; the target is not "
         "attained at a finite year merely because it appears in the formula.")

    text("2. Qubit circuitry and finite-dimensional readout", "Heading1")
    equation("H = 4 E<sub>C</sub>(N - n<sub>g</sub>I)<sup>2</sup> "
             "- E<sub>J</sub> cos(Phi) + hbar omega<sub>r</sub>a<sup>dagger</sup>a "
             "+ hbar chi a<sup>dagger</sup>a sigma<sub>z</sub>. &nbsp; (2)")
    text("N and Phi denote charge and phase operators. A numerical implementation "
         "must specify a finite charge basis and a resonator photon cutoff. For six "
         "two-level qubits the register dimension is 2<sup>6</sup> = 64; this "
         "dimension alone does not imply a 64-fold computational advantage.")
    equation("epsilon<sub>m</sub> approximately (-1)<sup>m</sup>E<sub>C</sub>"
             "(2<sup>4m+5</sup>/m!) sqrt(2/pi)"
             "(E<sub>J</sub>/(2E<sub>C</sub>))<sup>m/2+3/4</sup>"
             " exp[-sqrt(8E<sub>J</sub>/E<sub>C</sub>)]. &nbsp; (3)")
    text("This asymptotic transmon charge-dispersion expression requires a large "
         "E<sub>J</sub>/E<sub>C</sub> ratio. It does not prove immunity to noise or "
         "a particular coherence time. The earlier listed energies are inconsistent: "
         "248.5 GHz / 0.280 GHz = 887.5, not 88.75. Energies quoted in GHz "
         "must mean E/h; angular frequencies require omega = 2 pi f.")
    equation("K(k) = integral<sub>0</sub><sup>pi/2</sup>"
             " (1 - k<sup>2</sup> sin<sup>2</sup>t)<sup>-1/2</sup> dt; "
             "K'(k) = K(sqrt(1-k<sup>2</sup>)). &nbsp; (4)")
    equation("Z<sub>0</sub> approximately (30 pi / sqrt(epsilon<sub>eff</sub>)) "
             "K'(k)/K(k), k = w/(w+2s). &nbsp; (5)")
    text("Equation (5) is the ideal quasi-static coplanar-waveguide convention "
         "for center width w and gap s, not a universal transmission-line formula. "
         "Substrate geometry and electromagnetic validation are required to establish "
         "50-ohm matching. Keeping an optical water sample separate from a millikelvin "
         "processor is essential; a practical cryogenic AUV has not been demonstrated here.")

    text("3. Hydraulic formulation and robotic collection", "Heading1")
    equation("rho (du/dt + u . grad u) = -grad p + mu laplacian(u) + f; "
             "div u = 0. &nbsp; (6)")
    equation("rho U<sub>j</sub> dU<sub>i</sub>/dx<sub>j</sub> = "
             "-dP/dx<sub>i</sub> + d/dx<sub>j</sub>"
             "[mu(dU<sub>i</sub>/dx<sub>j</sub> + dU<sub>j</sub>/dx<sub>i</sub>) "
             "-rho &lt;u'<sub>i</sub>u'<sub>j</sub>&gt;]. &nbsp; (7)")
    equation("Re = rho U D<sub>h</sub>/mu; "
             "mu<sub>t</sub> = rho C<sub>mu</sub> k<sup>2</sup>/epsilon. &nbsp; (8)")
    text("The app assigns U = 1.45 m/s, Re = 48,500, turbulent kinetic energy "
         "k = 0.185 m<sup>2</sup>/s<sup>2</sup>, and bed shear 8.42 Pa. These are "
         "scenario inputs, not outputs of a mesh-based RANS solver. Channel geometry, "
         "boundary conditions, dissipation and mesh-convergence studies are missing.")
    equation("Q<sub>SI</sub> = Q<sub>mL/min</sub> 10<sup>-6</sup>/60; "
             "v<sub>in</sub> = Q<sub>SI</sub>/A; "
             "Delta P = rho v<sub>in</sub><sup>2</sup>/2 + R<sub>h</sub>Q<sub>SI</sub>. "
             "&nbsp; (9)")
    text("For Q = 150 mL/min and A = 10<sup>-4</sup> m<sup>2</sup>, "
         "v<sub>in</sub> = 0.025 m/s and the dynamic pressure is 0.3125 Pa "
         "at rho = 1,000 kg/m<sup>3</sup>. The earlier 2,120 Pa claim is not "
         "supported by these SI conversions. Filter resistance must be measured.")
    equation("Stk = rho<sub>p</sub>d<sub>p</sub><sup>2</sup>U/(18 mu D); "
             "eta<sub>asp</sub> = C<sub>in</sub>/C<sub>ambient</sub>. &nbsp; (10)")
    text("Matched stream and intake velocities define ideal isokinetic sampling. "
         "A low Stokes number does not establish a universal 1.00 +/- 0.02 capture "
         "efficiency or prove absence of cell damage.")
    equation("r<sub>n+1</sub> = r<sub>n</sub> + Delta t v<sub>n</sub>; "
             "M(v<sub>n+1</sub>-v<sub>n</sub>)/Delta t = "
             "F<sub>thruster</sub> - F<sub>drag</sub> + F<sub>buoyancy</sub>. &nbsp; (11)")
    text("Equation (11) supplies a finite-step robot model to be calibrated. "
         "The existing intake animation is illustrative, not a validated navigation solver.")

    text("4. Natural-gradient spectral inversion", "Heading1")
    equation("g<sub>ij</sub> = Re[&lt;d<sub>i</sub>psi|d<sub>j</sub>psi&gt; "
             "- &lt;d<sub>i</sub>psi|psi&gt;&lt;psi|d<sub>j</sub>psi&gt;]; "
             "theta<sub>n+1</sub> = theta<sub>n</sub> "
             "- eta (g + lambda I)<sup>-1</sup> grad L. &nbsp; (12)")
    text("Here lambda &gt; 0 regularizes the finite metric matrix. A positive metric "
         "eigenvalue alone does not prevent barren plateaus or establish a quantum "
         "speedup. The app's fidelity and speedup readouts are synthetic, without a "
         "trained circuit, held-out samples or a matched classical benchmark. The "
         "345, 420, 582 and 680 nm peaks are synthetic optical features, not specific "
         "proof of pathogens, polymers or measured concentrations.")

    text("5. Ten-year results and conditional stability", "Heading1")
    equation("P<sub>n</sub> = 6200 + 5600 exp(-0.28n); "
             "E<sub>n</sub> = 78 + 2322 exp(-0.35n). &nbsp; (13)")
    equation("S<sub>n</sub> = 12.4 - 8.6 exp(-0.24n); "
             "T<sub>n</sub> = 2.4 + 46.1 exp(-0.38n). &nbsp; (14)")
    equation("D<sub>n</sub> = floor[120 + 4730(1-exp(-0.22n))]; "
             "I<sub>n</sub> = 44 - 27.5 exp(-0.26n). &nbsp; (15)")
    text("P: phosphorus tonnes/year; E: E. coli CFU/100 mL; S: Secchi depth m; "
         "T: turbidity NTU; D: Diporeia individuals/m<sup>2</sup>; I: illustrative "
         "benthic index. Baselines and rates are assumed, not site-calibrated.")
    rows = [["Model metric", "Year 3", "Year 6", "Year 10"]]
    for label, key in [
        ("Phosphorus (tonnes/yr)", "phosphorus_load_mt"),
        ("Secchi depth (m)", "secchi_depth_m"),
        ("Diporeia (individuals/m2)", "diporeia_density_indiv_m2"),
        ("Cladophora (g/m2)", "cladophora_biomass_g_m2"),
        ("E. coli (CFU/100 mL)", "yellow_creek_ecoli_cfu"),
        ("Turbidity (NTU)", "yellow_creek_turbidity_ntu"),
        ("Benthic index", "yellow_creek_bibi_score"),
        ("Illustrative value (CAD M)", "cumulative_economic_value_cad_m"),
    ]:
        rows.append([label] + [f"{float(recovery[key][i]):,.2f}" for i in (2, 5, 9)])
    table = Table(rows, colWidths=[220, 88, 88, 88], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0edf0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    text("Table 1. Values reproduced from the current application, with its rounding "
         "and integer truncation. Asymptotic targets are not substituted for year-10 values.")
    plot("phosphorus_load_mt", "Figure 1. Modeled lake-wide phosphorus trajectory", "tonnes/year")
    plot("yellow_creek_ecoli_cfu", "Figure 2. Modeled Yellow Creek microbial trajectory", "CFU/100 mL")
    plot("secchi_depth_m", "Figure 3. Modeled water transparency", "Secchi depth (m)")
    equation("z<sub>n+1</sub> = A z<sub>n</sub>, "
             "A = diag(exp(-0.28), exp(-0.35)); "
             "V<sub>n</sub> = z<sub>n</sub><sup>T</sup>z<sub>n</sub>/2. &nbsp; (16)")
    equation("V<sub>n+1</sub> - V<sub>n</sub> = "
             "z<sub>n</sub><sup>T</sup>(A<sup>T</sup>A-I)z<sub>n</sub>/2 &lt; 0 "
             "for z<sub>n</sub> != 0. &nbsp; (17)")
    text("For normalized deviations z = ((P-P*)/P*, (E-E*)/E*), "
         "Equation (17) proves convergence only for the assumed autonomous exponential "
         "model. Storm forcing, delays, saturation and ecological feedback invalidate "
         "any unconditional real-world guarantee. Year-10 outputs exceed the earlier "
         "illustrative 6,500 tonnes/year and 100 CFU/100 mL thresholds; neither these "
         "thresholds nor compliance is verified here.")

    text("6. Remediation staging and economic evaluation", "Heading1")
    text("Years 1-2: establish measured baselines, calibrate sampling and survey runoff "
         "hotspots. Years 3-5: evaluate source-control and riparian retention pilots "
         "under permits. Years 6-8: assess benthic response and adapt monitoring. "
         "Years 9-10: audit outcomes and sustain surveillance. Biological releases, "
         "species reintroductions and sediment interventions require independent "
         "ecological review; they are not validated deployment recommendations.")
    equation("J = sum<sub>n=0</sub><sup>9</sup>"
             "[z<sub>n</sub><sup>T</sup>Wz<sub>n</sub> + "
             "u<sub>n</sub><sup>T</sup>Ru<sub>n</sub>], "
             "subject to state, budget and environmental constraints. &nbsp; (18)")
    text("Equation (18) defines a possible finite-horizon optimization objective; "
         "the app does not solve it, so its staging is not demonstrated optimal.")
    equation("B(n) = 14.5 n<sup>1.35</sup> million CAD; "
             "NPV = sum<sub>n=1</sub><sup>10</sup>"
             "(benefit<sub>n</sub>-cost<sub>n</sub>)/(1+r)<sup>n</sup>. &nbsp; (19)")
    text("The application produces B(10) = 324.6 million CAD. B(n) is an assumed "
         "cumulative value curve, not discounted net present value or revenue. "
         "True NPV requires incremental benefits, capital and operating costs, "
         "a discount rate, counterfactuals and uncertainty analysis.")
    text("7. Discussion and validation requirements", "Heading1")
    text("This manuscript specifies mathematical hypotheses and reproduces plots, "
         "rather than reporting field discoveries. Next steps include measured "
         "bathymetry and hydraulics, laboratory spectroscopy calibration, classical "
         "versus quantum benchmarking, cryogenic power and thermal feasibility, "
         "robot navigation trials, independently sourced water-quality criteria, "
         "and sensitivity and uncertainty analysis of remediation and economics. "
         "The earlier claims of established quantum advantage, noise immunity, "
         "isokinetic capture and guaranteed ecological compliance are unsupported.")
    text("Reproducibility and methodological references", "Heading1")
    text("Local source: backend/ontario_qubit_spectroscopy_engine.py. "
         "This PDF's generator reads its recovery arrays without modifying them. "
         "Equations (2)-(5) are idealized circuit descriptions, (6)-(11) fluid/robot "
         "formulations, (12) a proposed learning update, and (13)-(19) finite-horizon "
         "model and economic definitions.")
    text("Koch et al., Charge-insensitive qubit design derived from the Cooper pair "
         "box, Physical Review A 76, 042319 (2007), doi:10.1103/PhysRevA.76.042319. "
         "Stokes et al., Quantum Natural Gradient, Quantum 4, 269 (2020), "
         "doi:10.22331/q-2020-05-25-269. These methodological references do not "
         "validate the Lake Ontario application or its numerical assumptions.")

    def footer(canvas, doc) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 24, "Independent simulation preprint - not a Nature publication")
        canvas.drawRightString(570, 24, str(doc.page))

    doc = SimpleDocTemplate(
        str(output), pagesize=letter, rightMargin=42, leftMargin=42,
        topMargin=42, bottomMargin=42,
        title="Lake Ontario: finite mathematics and ten-year remediation scenarios",
        author="Independent simulation manuscript",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == "__main__":
    print(generate_pdf())
