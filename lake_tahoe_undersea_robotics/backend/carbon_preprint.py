"""Nature-style independent scenario manuscript generated from tab results."""

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PDF_NAME = "Ontario_Carbon_Economics_PQC_Research_Preprint.pdf"


def render_preprint(report: dict, verification: dict) -> bytes:
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Equation", fontName="Times-Roman", fontSize=10,
                              leading=16, leftIndent=8, spaceAfter=8))
    story = []

    def text(value: str, style: str = "BodyText") -> None:
        story.extend([Paragraph(value, styles[style]), Spacer(1, 6)])

    def eq(value: str) -> None:
        text(value, "Equation")

    def chart(title: str, series: list, labels: list) -> None:
        drawing = Drawing(500, 195)
        drawing.add(String(40, 181, title, fontSize=10, fontName="Helvetica-Bold"))
        line = LinePlot()
        line.x, line.y, line.width, line.height = 65, 45, 405, 120
        line.data = series
        line.xValueAxis.valueMin, line.xValueAxis.valueMax = 0, 10
        line.xValueAxis.valueSteps = [0, 2, 4, 6, 8, 10]
        for i, label in enumerate(labels):
            color = [colors.teal, colors.darkorange][i]
            line.lines[i].strokeColor = color
            line.lines[i].strokeWidth = 2
            drawing.add(String(65 + i * 215, 12, label, fillColor=color, fontSize=9))
        drawing.add(line)
        drawing.add(String(210, 30, "Elapsed model year", fontSize=9))
        story.append(drawing)

    a = report["assumptions"]
    text("Carbon-aware economic viability of Ontario underwater robotics: "
         "finite-horizon accounting, quantum kernels and post-quantum report verification", "Title")
    text("Independent Nature-style research draft | Not a Nature publication or peer-reviewed study")
    text("Abstract", "Heading1")
    text("We couple an editable ten-year fleet cost and carbon-footprint scenario with a "
         "finite product-state kernel demonstrator and local post-quantum report verification. "
         "The model separates discounted financial value, carbon shadow value and lifecycle "
         "emissions. Arithmetic scheduling labels reuse a helper from Quantum Privacy Cloak, "
         "but its simulated security measures are not relied upon. Real ML-DSA-65 and ML-KEM-768 "
         "functional tests exercise signatures, tampering and shared-secret agreement. "
         "All economic inputs are assumptions: no field-calibrated lifecycle assessment, "
         "quantum advantage, carbon-credit eligibility or production security is established.")
    text("1. Scope, assumptions and reproducibility", "Heading1")
    text("The comparison is an incremental new robotic fleet against incumbent conventional "
         "monitoring. Robot and QML facility embodied emissions are charged at year zero. "
         "The baseline's existing assets are assumed sunk; its annual carbon inventory is "
         "user-entered. Both alternatives are assumed to provide comparable monitoring "
         "service, an unverified functional-equivalence hypothesis. Maintenance emissions, "
         "replacement batteries, transport, disposal and facility utilization beyond the "
         "entered hours are excluded unless absorbed into the inputs. All amounts are CAD "
         "and undiscounted carbon masses are kg CO2e.")
    rows = [["Assumption", "Value"]]
    rows.extend([[name, f"{value:,.4f}"] for name, value in a.items()])
    table = Table(rows, colWidths=[280, 215], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0edf0")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    text("Table 1. Exact tab input keys. qml_reduction is an assumed fractional robot-energy "
         "reduction (0 by default), not learned or experimentally demonstrated. qml_kw is "
         "total shared facility draw, including assumed cryogenic overhead; it is not per robot.")
    text("2. Finite economics and carbon derivations", "Heading1")
    eq("C<sub>0</sub> = N c<sub>robot</sub> + c<sub>QML</sub>; "
       "E = H [N P<sub>robot</sub>(1-s) + P<sub>QML</sub>]. &nbsp; (1)")
    eq("O = N o<sub>robot</sub> + E p<sub>electricity</sub>; "
       "F = C<sub>baseline</sub> + B - O. &nbsp; (2)")
    eq("NPV<sub>T</sub> = -C<sub>0</sub> + sum<sub>t=1</sub><sup>T</sup> "
       "F/(1+r)<sup>t</sup>, T = 10. &nbsp; (3)")
    text("Power is kW, H is operating hours/year, and E is kWh/year. B is a hypothetical "
         "annual non-carbon benefit; it must not double-count savings already in the avoided "
         "baseline cost. NPV is undiscounted net cash flow when r = 0. Constant cash flows, "
         "no inflation, salvage or replacements are assumed.")
    eq("G<sub>op</sub> = E gamma; G<sub>emb</sub> = N g<sub>robot</sub> + g<sub>QML</sub>; "
       "G<sub>project</sub>(T) = G<sub>emb</sub> + T G<sub>op</sub>. &nbsp; (4)")
    eq("Delta G(T) = T G<sub>baseline</sub> - G<sub>project</sub>(T); "
       "t<sub>carbon</sub> = G<sub>emb</sub>/(G<sub>baseline</sub>-G<sub>op</sub>). &nbsp; (5)")
    text("The break-even ratio is meaningful only for positive annual avoided emissions. "
         "If annual emissions savings are nonpositive and embodied emissions are positive, "
         "no finite carbon break-even exists under this stationary model.")
    eq("V<sub>carbon</sub> = p<sub>CO2</sub>(G<sub>baseline</sub>-G<sub>op</sub>)/1000; "
       "NPV<sub>shadow</sub> = -C<sub>0</sub> - p<sub>CO2</sub>G<sub>emb</sub>/1000 + "
       "sum<sub>t=1</sub><sup>10</sup>(F+V<sub>carbon</sub>)/(1+r)<sup>t</sup>. &nbsp; (6)")
    text("The carbon price is CAD/tonne CO2e. Carbon-valued NPV is a shadow-value scenario, "
         "not revenue or evidence of saleable offsets. Viability is defined here as "
         "financial NPV &gt;= 0 and Delta G(10) &gt;= 0, without claiming broader social viability.")
    text("3. Scenario results and sensitivity", "Heading1")
    for label, key, unit in [
        ("Initial capital", "capital_cad", "CAD"), ("Annual electricity", "annual_kwh", "kWh"),
        ("Annual operating cost", "annual_cost_cad", "CAD"),
        ("Annual operational emissions", "annual_operational_kg", "kg CO2e"),
        ("Embodied emissions", "embodied_kg", "kg CO2e"),
        ("Financial NPV", "financial_npv_cad", "CAD"),
        ("Carbon-valued NPV", "carbon_valued_npv_cad", "CAD"),
        ("Net avoided ten-year emissions", "net_avoided_kg_10yr", "kg CO2e"),
    ]:
        text(f"<b>{label}:</b> {report[key]:,.2f} {unit}.")
    be = report["carbon_break_even_years"]
    text("Carbon break-even: " + (f"{be:.2f} years." if be is not None else
                                 "No finite value under these assumptions."))
    text(f"Viable under the stated joint criterion: {report['viable_under_assumptions']}.")
    trajectory = report["trajectory"]
    chart("Figure 1. Discounted value (CAD)", [
        [(v["year"], v["npv_cad"]) for v in trajectory],
        [(v["year"], v["carbon_valued_npv_cad"]) for v in trajectory],
    ], ["Financial NPV", "Carbon shadow NPV"])
    chart("Figure 2. Lifecycle carbon (kg CO2e)", [
        [(v["year"], v["project_kg"]) for v in trajectory],
        [(v["year"], v["baseline_kg"]) for v in trajectory],
    ], ["Project incl. embodied", "Incumbent baseline"])
    eq("NPV(alpha C<sub>0</sub>) = NPV(C<sub>0</sub>) "
       "- (alpha-1)C<sub>0</sub>. &nbsp; (7)")
    text("The tab sweeps capital multipliers 0.5, 0.75, 1, 1.25 and 1.5 while holding other "
         "parameters fixed. This is sensitivity analysis, not a probabilistic confidence interval.")
    text("4. Quantum kernel and number-theoretic structure", "Heading1")
    eq("|phi(x)&gt; = tensor<sub>j=1</sub><sup>3</sup> R<sub>y</sub>(pi x<sub>j</sub>)|0&gt;; "
       "K(x,z) = product<sub>j=1</sub><sup>3</sup> "
       "cos<sup>2</sup>[pi(x<sub>j</sub>-z<sub>j</sub>)/2]. &nbsp; (8)")
    eq("alpha = (K + 0.01 I)<sup>-1</sup> y; f(x) = k(x)<sup>T</sup>alpha. &nbsp; (9)")
    text("Six synthetic training points have labels 0.6 x1 + 0.3 x2 + 0.1 x3. Query "
         "features are robot_kw/1000, hours/8760 and qml_reduction/0.5. The kernel is "
         "computed exactly on a classical computer, without entangling gates. Its "
         f"training RMSE is {report['qml']['training_rmse']:.6f} and query score is "
         f"{report['qml']['synthetic_score']:.6f}. Neither score is used in the financial "
         "forecast. Synthetic training accuracy is not evidence of useful QML prediction "
         "or speedup; real training data and held-out classical comparisons are required.")
    eq("group(i) = i mod p; c<sub>q</sub>(n) = "
       "sum<sub>d | gcd(q,n)</sub> d mu(q/d). &nbsp; (10)")
    text("Prime-based residue groups label station schedules. The reused Privacy Cloak "
         "helper yields c5(1) = -1. These identities are finite arithmetic, not optimal "
         "geographic placement, encryption or a security guarantee. Lattice parameters "
         "belong to the standardized PQC provider, not to user-selected scheduling primes.")
    text("5. Post-quantum report verification", "Heading1")
    eq("m = domain || canonicalJSON(report); sigma = Sign<sub>sk</sub>(m); "
       "Verify<sub>pk</sub>(m,sigma). &nbsp; (11)")
    eq("(ct,k1) = Encaps(pk); k2 = Decaps(sk,ct); "
       "test: k1 = k2 and k1 != Decaps(sk,altered(ct)). &nbsp; (12)")
    text("The implementation uses pqcrypto 1.0.0 with ML-DSA-65 signatures and "
         "ML-KEM-768 shared-secret functional tests. RFC 8785 JSON canonicalization "
         "normalizes keys and numeric representations across browser round-trips, "
         "and rejects nonfinite numbers. A domain prefix binds reports to "
         "this app. The verifier pins the server's current public key instead of trusting "
         "a supplied key. Keys are ephemeral and reports expire on server restart.")
    for name in ("signature_valid", "altered_report_rejected", "kem_shared_secret_agreed",
                 "altered_ciphertext_secret_mismatch"):
        text(f"{escape(name)}: <b>{verification[name]}</b>.")
    text("Public key fingerprint (SHA3-256): " + escape(verification["key_id"]))
    text("These checks establish local functional behavior only. HTTP is not encrypted "
         "and the demo does not provide PQ TLS, user authentication, replay resistance, "
         "durable identity, encrypted reports or FIPS-certified module operation. The PDF "
         "itself is not digitally signed; its narrative records a report self-test.")
    text("6. Discussion, limitations and data availability", "Heading1")
    text("Economic viability depends strongly on comparable service, cryogenic overhead, "
         "manufacturing inventory, fleet utilization and attributed benefits. Entering "
         "measured values is supported, but source provenance is not independently checked. "
         "Monitoring is not equivalent to remediation or carbon removal. Cryptographic "
         "integrity cannot validate scientific assumptions, carbon claims or data quality. "
         "The source functions evaluate, quantum_kernel_demo and ReportSigner reproduce "
         "the calculations. No experiments or Nature affiliation are asserted.")
    text("References", "Heading1")
    text("NIST FIPS 203, Module-Lattice-Based Key-Encapsulation Mechanism Standard "
         "(2024), doi:10.6028/NIST.FIPS.203. NIST FIPS 204, Module-Lattice-Based "
         "Digital Signature Standard (2024), doi:10.6028/NIST.FIPS.204. "
         "Schuld and Killoran, Quantum Machine Learning in Feature Hilbert Spaces, "
         "Physical Review Letters 122, 040504 (2019), doi:10.1103/PhysRevLett.122.040504. "
         "These references motivate methods, not the numerical assumptions or security "
         "certification of this application.")

    def footer(canvas, doc) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 24, "Independent scenario preprint - not a Nature publication")
        canvas.drawRightString(570, 24, str(doc.page))

    SimpleDocTemplate(buffer, pagesize=letter, leftMargin=42, rightMargin=42,
                      topMargin=42, bottomMargin=42,
                      title="Ontario carbon economics and PQC report verification",
                      author="Independent research draft").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def save_default() -> Path:
    from backend.carbon_economics import Assumptions, evaluate
    from backend.pqc_reports import ReportSigner
    report = evaluate(Assumptions())
    path = Path(__file__).resolve().parents[1] / PDF_NAME
    path.write_bytes(render_preprint(report, ReportSigner().self_test(report)))
    return path


if __name__ == "__main__":
    print(save_default())
