#!/usr/bin/env python3
"""
Publication-Quality Nature Preprint Generator:
'Quantum-Theoretic Economic Distributions and 10-Year Long-Term Revenue Projections
of Post-Quantum Privacy Ciphers Across MedTech, FinTech, Geospatial Navigation, and Oil & Gas'

Produces a 3-page Nature Computational Science / Nature Quantum Information
styled PDF manuscript with finite mathematical equations, quantum-theoretic
distributions (Wigner-Dyson GUE, Fermi-Dirac, Bose-Einstein, Lindblad density evolution),
4-panel empirical plots, and 10-year sectoral revenue tables (2027-2036).
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend_economic_revenue_quantum import (
    evaluate_10year_economic_revenue_projections,
)


def build_economic_revenue_nature_preprint(output_pdf_path: str) -> str:
    econ = evaluate_10year_economic_revenue_projections(
        epsilon=0.45,
        mmd2=3.8e-11,
        attenuation_db=98.6,
        k_anonymity=50,
        qrom_security_bits=159.0,
        discount_rate=0.085,
        temperature_q=0.65,
        horizon_years=10,
    )

    with PdfPages(output_pdf_path) as pdf:
        # =====================================================================
        # PAGE 1: TITLE, ABSTRACT & FINITE MATHEMATICAL EQUATIONS
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()
        fig.gca().axis("off")

        fig.text(
            0.5,
            0.964,
            "Nature Computational Science & Quantum Economics | Sovereign Cybersecurity Forecast",
            fontsize=9.6,
            ha="center",
            style="italic",
            color="#1e3a8a",
            weight="bold",
        )
        fig.text(
            0.5,
            0.916,
            "Quantum-Theoretic Distributions and 10-Year Economic Revenue Projections of\n"
            "Post-Quantum Privacy Ciphers for MedTech, FinTech, Geospatial & Oil and Gas",
            fontsize=12.6,
            ha="center",
            weight="bold",
            color="#0f172a",
            linespacing=1.22,
        )
        fig.text(
            0.5,
            0.868,
            "Cartik Sharma, PhD et al. | Neuromorph Quantum Econometrics & Critical Infrastructure Group",
            fontsize=8.8,
            ha="center",
            color="#475569",
        )

        # Abstract Box
        rect = FancyBboxPatch(
            (0.075, 0.728),
            0.85,
            0.125,
            boxstyle="round,pad=0.014",
            ec="#10b981",
            fc="#f8fafc",
            linewidth=1.2,
        )
        fig.gca().add_patch(rect)

        abstract_title = "ABSTRACT & 10-YEAR QUANTUM ECONOMETRIC SYNTHESIS (2027-2036)"
        abstract_body = (
            "The mandatory transition of critical infrastructure to post-quantum cryptography (NIST FIPS 203 ML-KEM, FIPS 204 ML-DSA) "
            "and zero-telemetry privacy cloaking creates a structural capital reallocation across MedTech, FinTech, Geospatial Navigation "
            "(PNT/GNSS), and Oil & Gas SCADA networks. Here we formulate a finite mathematical econometric model coupling technical privacy "
            "metrics—RKHS Maximum Mean Discrepancy (MMD^2_u <= 3.8e-11), differential privacy (epsilon = 0.45), Grassmannian k-anonymity "
            "(k >= 50), metamaterial RF attenuation (98.6 dB), and QROM signature unforgeability (2^-159)—to quantum-theoretic statistical "
            "distributions: Wigner-Dyson Gaussian Unitary Ensemble (GUE) eigenvalue spacing, Fermi-Dirac regulatory saturation, Bose-Einstein "
            "network condensation, and Lindblad master equation density evolution. Over the 10-year horizon (2027-2036), cumulative "
            f"multi-sector cipher revenue reaches USD {econ['headline_metrics']['cumulative_10yr_revenue_b']:.2f}B "
            f"(NPV USD {econ['headline_metrics']['net_present_value_10yr_b']:.2f}B, CAGR {econ['headline_metrics']['composite_cagr_pct']}%) "
            f"while preventing USD {econ['headline_metrics']['cumulative_breach_avoided_b']:.2f}B in systemic quantum breach liabilities."
        )
        fig.text(0.092, 0.838, abstract_title, fontsize=8.4, weight="bold", color="#065f46")
        fig.text(
            0.092,
            0.826,
            abstract_body,
            fontsize=7.25,
            va="top",
            wrap=True,
            linespacing=1.32,
            color="#1e293b",
        )

        # Section 1: Privacy-Metric Elasticity Functional
        fig.text(
            0.075,
            0.702,
            "1. Finite Mathematical Privacy-Metric Elasticity Multiplier",
            fontsize=10.0,
            weight="bold",
            color="#0f172a",
        )
        sec1_text = (
            "Enterprise willingness-to-pay and regulatory safe-harbor yield are governed by the composite privacy elasticity "
            "multiplier Phi_priv(epsilon, MMD^2, Att_dB, k, Adv_QROM), linking cryptographic invariants to sectoral growth g_i:"
        )
        fig.text(0.075, 0.662, sec1_text, fontsize=7.7, wrap=True, color="#334155", linespacing=1.22)

        eq1 = (
            r"$\Phi_{\mathrm{priv}} = \frac{1}{Z_0}\left(1 + \alpha_\varepsilon e^{-\varepsilon}\right)"
            r"\left(1 + \frac{\alpha_{\mathrm{MMD}}}{1 + 10^9\,\mathrm{MMD}^2_u}\right)"
            r"\left(1 + \alpha_{\mathrm{att}}\frac{\mathrm{Att}_{\mathrm{dB}}}{100}\right)"
            r"\left(1 + \alpha_k \ln k\right)"
            r"\left(1 + \alpha_{\mathrm{sig}}\frac{-\log_2 \mathrm{Adv}_{\mathrm{QROM}}}{160}\right)$"
        )
        fig.text(0.5, 0.620, eq1, fontsize=8.5, ha="center", color="#1e3a8a")

        # Section 2: Quantum-Theoretic Statistical Distributions (Wigner-Dyson GUE, Fermi-Dirac, Bose-Einstein)
        fig.text(
            0.075,
            0.578,
            "2. Quantum-Theoretic Statistical Distributions for Market Adoption & Volatility",
            fontsize=10.0,
            weight="bold",
            color="#0f172a",
        )
        sec2_text = (
            "Cross-sector correlation matrices H_market in GUE(N) exhibit eigenvalue spacing repulsion governed by the Wigner-Dyson "
            "distribution P_GUE(s), while statutory compliance adoption and network condensation follow Fermi-Dirac and Bose-Einstein laws:"
        )
        fig.text(0.075, 0.538, sec2_text, fontsize=7.7, wrap=True, color="#334155", linespacing=1.22)

        eq2 = (
            r"$P_{\mathrm{GUE}}(s) = \frac{32}{\pi^2} s^2 \exp\!\left(-\frac{4}{\pi}s^2\right), \quad "
            r"P_{\mathrm{GOE}}(s) = \frac{\pi}{2} s \exp\!\left(-\frac{\pi}{4}s^2\right), \quad "
            r"\sigma^2_{\mathrm{GUE}} = \frac{3\pi}{8} - 1 \approx 0.1781$"
        )
        fig.text(0.5, 0.498, eq2, fontsize=8.6, ha="center", color="#1e3a8a")

        eq2b = (
            r"$f_{\mathrm{FD}}(E_t; \mu, T_Q) = \frac{1}{\exp\!\left(\frac{E_t - \mu_{\mathrm{reg}}}{T_Q}\right) + 1}, \quad "
            r"f_{\mathrm{BE}}(E_t; \mu, T_Q) = \frac{1}{\exp\!\left(\frac{E_t - \mu_{\mathrm{net}}}{T_Q}\right) - 1}, \quad "
            r"P_{\mathrm{coh}}(n; |\alpha|^2) = e^{-|\alpha|^2}\frac{|\alpha|^{2n}}{n!}$"
        )
        fig.text(0.5, 0.456, eq2b, fontsize=8.4, ha="center", color="#1e3a8a")

        # Section 3: Lindblad Master Equation & 10-Year Revenue Functional
        fig.text(
            0.075,
            0.415,
            "3. Lindblad Open Quantum System Revenue Evolution & 10-Year NPV Functional",
            fontsize=10.0,
            weight="bold",
            color="#0f172a",
        )
        sec3_text = (
            "The 4-sector density matrix rho_R(t) for MedTech, FinTech, Geospatial Navigation, and Oil & Gas evolves under the "
            "Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation with jump operators L_k representing inter-sector spillovers:"
        )
        fig.text(0.075, 0.375, sec3_text, fontsize=7.7, wrap=True, color="#334155", linespacing=1.22)

        eq3 = (
            r"$\frac{d\rho_R(t)}{dt} = -i[H_{\mathrm{econ}}, \rho_R(t)] + \sum_{k=1}^4 \gamma_k \left(L_k \rho_R(t) L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho_R(t)\}\right)$"
        )
        fig.text(0.5, 0.335, eq3, fontsize=8.8, ha="center", color="#1e3a8a")

        eq4 = (
            r"$R_i(t) = R_{i,0}\left(1 + g_i \Phi_{\mathrm{priv}}\right)^{t-1} \left(1 - \beta_0 e^{-\lambda_i t}\right) \mathrm{Tr}(\rho_R(t)\Pi_i), \quad "
            r"\mathrm{NPV}_{10} = \sum_{t=1}^{10} \sum_{i=1}^4 \frac{R_i(t)}{(1 + r_{\mathrm{WACC}})^t}$"
        )
        fig.text(0.5, 0.292, eq4, fontsize=8.5, ha="center", color="#1e3a8a")

        # Section 4: Sectoral Cybersecurity Architecture Overview
        fig.text(
            0.075,
            0.250,
            "4. Four-Sector Sovereign Cipher Deployment & Breach-Avoidance Capital Yield",
            fontsize=10.0,
            weight="bold",
            color="#0f172a",
        )
        sec4_text = (
            "Each vertical leverages a specialized subset of the finite mathematical privacy cloak: (i) MedTech utilizes RKHS MMD "
            "quantum varieties and Ramanujan modular forms for 7T MRI & WGS biobanks; (ii) FinTech deploys hybrid Ed25519+ML-DSA-87 and "
            "multi-prime CRT hashing for CBDC/RTGS; (iii) Geospatial Navigation protects GNSS/LEO PNT ephemeris via 98.6 dB transformation "
            "cloaks; and (iv) Oil & Gas secures subsurface seismic reservoirs and ICS/SCADA via Grassmannian q-binomial subspace lattices."
        )
        fig.text(0.075, 0.195, sec4_text, fontsize=7.6, wrap=True, color="#334155", linespacing=1.25)

        foot_box = FancyBboxPatch(
            (0.075, 0.065),
            0.85,
            0.095,
            boxstyle="round,pad=0.012",
            ec="#38bdf8",
            fc="#f0f9ff",
            linewidth=1.1,
        )
        fig.gca().add_patch(foot_box)
        hm = econ["headline_metrics"]
        fig.text(
            0.092,
            0.135,
            f"10-Year Headline Forecast (2027-2036):  Cumulative Revenue = USD {hm['cumulative_10yr_revenue_b']:.3f}B  |  "
            f"Year-10 Annual Run-Rate = USD {hm['year_10_total_revenue_b']:.3f}B",
            fontsize=7.6,
            weight="bold",
            color="#0f172a",
        )
        fig.text(
            0.092,
            0.108,
            f"Discounted 10-Yr NPV (8.5% WACC) = USD {hm['net_present_value_10yr_b']:.3f}B  |  "
            f"Composite CAGR = {hm['composite_cagr_pct']}%  |  Quantum Breach Avoided = USD {hm['cumulative_breach_avoided_b']:.3f}B",
            fontsize=7.5,
            color="#1e3a8a",
        )
        fig.text(
            0.092,
            0.082,
            f"Privacy Multiplier Phi_priv = {econ['privacy_elasticity']['composite_privacy_multiplier']}x  |  "
            f"Quantum Sharpe Ratio = {hm['quantum_sharpe_ratio']}  |  Q-VaR (99% GUE) = {hm['quantum_var_99_pct']}",
            fontsize=7.4,
            color="#065f46",
        )

        fig.text(
            0.5,
            0.032,
            "Nature Computational Science | Quantum Economic Revenue Projections Preprint | Page 1 of 3",
            fontsize=7.8,
            ha="center",
            color="#94a3b8",
        )
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 2: 4-PANEL QUANTUM DISTRIBUTIONS & 10-YEAR REVENUE PLOTS
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()
        fig.gca().axis("off")

        fig.text(
            0.5,
            0.962,
            "5. Quantum-Theoretic Distributions & 10-Year Multi-Sector Revenue Trajectories",
            fontsize=11.0,
            ha="center",
            weight="bold",
            color="#0f172a",
        )
        p2_desc = (
            "Empirical and analytical plots of (a) 10-Year sectoral revenue trajectories (2027-2036), (b) Wigner-Dyson GUE/GOE vs "
            "Poisson quantum eigenvalue spacing distributions, (c) Fermi-Dirac & Bose-Einstein adoption occupancy, and (d) Cumulative revenue vs avoided breach loss."
        )
        fig.text(0.075, 0.922, p2_desc, fontsize=7.8, wrap=True, color="#334155")

        ax1 = fig.add_axes([0.10, 0.55, 0.36, 0.31])
        ax2 = fig.add_axes([0.56, 0.55, 0.36, 0.31])
        ax3 = fig.add_axes([0.10, 0.12, 0.36, 0.31])
        ax4 = fig.add_axes([0.56, 0.12, 0.36, 0.31])

        # Panel (a): 10-Year Sectoral Revenue Trajectories ($M USD)
        yrs = [r["year"] for r in econ["yearly_projections"]]
        med_y = [r["medtech_rev_m"] for r in econ["yearly_projections"]]
        fin_y = [r["fintech_rev_m"] for r in econ["yearly_projections"]]
        geo_y = [r["geospatial_rev_m"] for r in econ["yearly_projections"]]
        oil_y = [r["oil_gas_rev_m"] for r in econ["yearly_projections"]]

        ax1.plot(yrs, fin_y, "o-", color="#3b82f6", linewidth=1.8, markersize=4, label="FinTech & CBDC/RTGS")
        ax1.plot(yrs, med_y, "s-", color="#10b981", linewidth=1.8, markersize=4, label="MedTech & 7T/WGS Biobanks")
        ax1.plot(yrs, oil_y, "^-", color="#f59e0b", linewidth=1.8, markersize=4, label="Oil, Gas & Energy SCADA")
        ax1.plot(yrs, geo_y, "d-", color="#8b5cf6", linewidth=1.8, markersize=4, label="Geospatial PNT/GNSS")
        ax1.set_title("(a) 10-Year Annual Revenue by Sector ($M USD)", fontsize=8.0, weight="bold")
        ax1.set_xlabel("Forecast Year (2027-2036)", fontsize=7.2)
        ax1.set_ylabel("Annual Revenue ($M USD)", fontsize=7.2)
        ax1.legend(fontsize=5.8, loc="upper left")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Panel (b): Quantum Wigner-Dyson GUE / GOE vs Poisson Spectral Distribution
        qd = econ["quantum_distributions"]["distributions_grid"]
        s_vals = [g["s_parameter"] for g in qd]
        p_gue = [g["wigner_dyson_gue"] for g in qd]
        p_goe = [g["wigner_dyson_goe"] for g in qd]
        p_poi = [g["poisson_baseline"] for g in qd]

        ax2.plot(s_vals, p_gue, "-", color="#0284c7", linewidth=2.0, label=r"Wigner-Dyson GUE $\frac{32}{\pi^2}s^2 e^{-4s^2/\pi}$")
        ax2.plot(s_vals, p_goe, "--", color="#10b981", linewidth=1.7, label=r"Wigner-Dyson GOE $\frac{\pi}{2}s e^{-\pi s^2/4}$")
        ax2.plot(s_vals, p_poi, ":", color="#f43f5e", linewidth=1.6, label=r"Poisson Uncorrelated $e^{-s}$")
        ax2.fill_between(s_vals, 0, p_gue, color="#0284c7", alpha=0.12)
        ax2.set_title("(b) Quantum Random Matrix Spectral Distribution", fontsize=8.0, weight="bold")
        ax2.set_xlabel("Normalized Eigenvalue Spacing s", fontsize=7.2)
        ax2.set_ylabel("Probability Density P(s)", fontsize=7.2)
        ax2.legend(fontsize=5.8, loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.5)

        # Panel (c): Fermi-Dirac Regulatory Saturation & Glauber Coherent State Distribution
        f_fd = [g["fermi_dirac_adoption"] for g in qd]
        f_be = [g["bose_einstein_condensation"] for g in qd]
        ax3.plot(s_vals, f_fd, "o-", color="#6366f1", linewidth=1.8, markersize=3.5, label=r"Fermi-Dirac Mandate $f_{\mathrm{FD}}(E)$")
        ax3.plot(s_vals, f_be, "s--", color="#f59e0b", linewidth=1.7, markersize=3.5, label=r"Bose-Einstein Network $f_{\mathrm{BE}}(E)$")
        ax3.set_title("(c) Quantum Statistical Adoption Occupancy", fontsize=8.0, weight="bold")
        ax3.set_xlabel("Normalized Adoption Energy Barrier E", fontsize=7.2)
        ax3.set_ylabel("Quantum Occupancy Factor", fontsize=7.2)
        ax3.legend(fontsize=6.0, loc="upper right")
        ax3.grid(True, linestyle=":", alpha=0.5)

        # Panel (d): 10-Year Cumulative Revenue vs Breach-Avoided Capital ($B USD)
        sec_names = ["MedTech", "FinTech", "Geospatial", "Oil & Gas"]
        cum_rev_b = [s["cumulative_10yr_rev_b"] for s in econ["sector_summaries"]]
        breach_b = [s["breach_avoided_10yr_b"] for s in econ["sector_summaries"]]

        x_pos = np.arange(len(sec_names))
        width = 0.35
        ax4.bar(x_pos - width / 2, cum_rev_b, width, color="#0284c7", label="10-Yr Cumulative Revenue ($B)")
        ax4.bar(x_pos + width / 2, breach_b, width, color="#10b981", label="10-Yr Breach Loss Avoided ($B)")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(sec_names, fontsize=7.0)
        ax4.set_title("(d) 10-Year Cumulative Revenue & Breach Savings ($B)", fontsize=8.0, weight="bold")
        ax4.set_ylabel("Capital Volume ($ Billions USD)", fontsize=7.2)
        ax4.legend(fontsize=5.8, loc="upper right")
        ax4.grid(True, linestyle=":", alpha=0.5)

        fig.text(
            0.5,
            0.032,
            "Nature Computational Science | Quantum Economic Revenue Projections Preprint | Page 2 of 3",
            fontsize=7.8,
            ha="center",
            color="#94a3b8",
        )
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 3: 10-YEAR REVENUE SCHEDULE TABLE & SECTORAL CHARACTERISTICS
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()
        fig.gca().axis("off")

        fig.text(
            0.5,
            0.962,
            "6. Ten-Year Sectoral Revenue Schedule (2027-2036) & Quantum Privacy Characteristics",
            fontsize=11.0,
            ha="center",
            weight="bold",
            color="#0f172a",
        )

        fig.text(
            0.075,
            0.924,
            "TABLE 1: 10-YEAR ANNUAL & CUMULATIVE REVENUE PROJECTIONS BY CRITICAL SECTOR ($M USD)",
            fontsize=8.6,
            weight="bold",
            color="#1e3a8a",
        )

        yr_table_rows = [
            ["Year", "MedTech (USD M)", "FinTech (USD M)", "Geospatial (USD M)", "Oil & Gas (USD M)", "Total Annual (USD M)", "Breach Saved (USD M)"]
        ]
        for row in econ["yearly_projections"]:
            yr_table_rows.append([
                str(row["year"]),
                f"{row['medtech_rev_m']:,.1f}M",
                f"{row['fintech_rev_m']:,.1f}M",
                f"{row['geospatial_rev_m']:,.1f}M",
                f"{row['oil_gas_rev_m']:,.1f}M",
                f"{row['total_revenue_m']:,.1f}M",
                f"{row['total_breach_avoided_m']:,.1f}M",
            ])

        t1 = fig.gca().table(cellText=yr_table_rows, loc="upper center", bbox=[0.075, 0.565, 0.85, 0.34])
        t1.auto_set_font_size(False)
        t1.set_fontsize(6.9)
        for i in range(len(yr_table_rows)):
            for j in range(7):
                cell = t1[i, j]
                if i == 0:
                    cell.set_facecolor("#1e3a8a")
                    cell.set_text_props(color="white", weight="bold")
                else:
                    cell.set_facecolor("#f8fafc" if i % 2 == 1 else "#ffffff")
                    if j == 5:
                        cell.set_text_props(color="#0284c7", weight="bold")
                    elif j == 6:
                        cell.set_text_props(color="#065f46", weight="bold")

        fig.text(
            0.075,
            0.520,
            "TABLE 2: SECTORAL PRIVACY CIPHER CHARACTERISTICS, 10-YEAR CAGR, NPV & REGULATORY DRIVERS",
            fontsize=8.6,
            weight="bold",
            color="#1e3a8a",
        )

        sec_rows = [
            ["Sector", "10-Yr Cumul.", "10-Yr NPV", "CAGR", "Gross Margin", "Primary Privacy Cipher & Regulatory Stack"]
        ]
        for s in econ["sector_summaries"]:
            sec_rows.append([
                s["sector_name"].split("&")[0].strip(),
                f"USD {s['cumulative_10yr_rev_b']:.2f}B",
                f"USD {s['npv_10yr_m']/1000.0:.2f}B",
                f"{s['cagr_10yr_pct']}%",
                f"{s['gross_margin_pct']}%",
                f"{s['regulatory_drivers'][:46]}",
            ])

        t2 = fig.gca().table(cellText=sec_rows, loc="upper center", bbox=[0.075, 0.315, 0.85, 0.19])
        t2.auto_set_font_size(False)
        t2.set_fontsize(6.8)
        for i in range(len(sec_rows)):
            for j in range(6):
                cell = t2[i, j]
                if i == 0:
                    cell.set_facecolor("#0f172a")
                    cell.set_text_props(color="white", weight="bold")
                else:
                    cell.set_facecolor("#f1f5f9" if i % 2 == 1 else "#ffffff")
                    if j == 1:
                        cell.set_text_props(color="#10b981", weight="bold")

        # Concluding Synthesis Box
        synth_box = FancyBboxPatch(
            (0.075, 0.080),
            0.85,
            0.205,
            boxstyle="round,pad=0.014",
            ec="#6366f1",
            fc="#f5f3ff",
            linewidth=1.2,
        )
        fig.gca().add_patch(synth_box)
        fig.text(
            0.095,
            0.260,
            "QUANTUM ECONOMETRIC & PRIVACY METRIC CHARACTERIZATION SUMMARY",
            fontsize=8.5,
            weight="bold",
            color="#3730a3",
        )
        synth_text = (
            "1. MedTech & Biobanks: Driven by EHDS, GDPR Art. 9, and Canadian PHIPA/PIPEDA, RKHS MMD^2 <= 3.8e-11 and (eps=0.45)-DP\n"
            "   unlock cross-border secondary genomic & 7T fMRI monetization without re-identification risk.\n"
            "2. FinTech & CBDC Settlement: Hybrid Ed25519 + ML-DSA-87 signatures (QROM forgery risk <= 2^-159) and Carter-Wegman\n"
            "   multi-prime CRT hashing secure high-velocity institutional RTGS clearing under EU DORA and Canadian OSFI B-13.\n"
            "3. Geospatial Navigation & Oil/Gas SCADA: 98.6 dB metamaterial RF cloaks and Grassmannian Gr_q(k, n) subspace ciphers\n"
            "   protect LEO GNSS PNT ephemeris streams and proprietary 4D seismic hydrocarbon reservoir models against quantum espionage."
        )
        fig.text(
            0.095,
            0.238,
            synth_text,
            fontsize=7.25,
            va="top",
            linespacing=1.36,
            color="#1e1b4b",
        )

        fig.text(
            0.5,
            0.032,
            "Nature Computational Science | Quantum Economic Revenue Projections Preprint | Page 3 of 3",
            fontsize=7.8,
            ha="center",
            color="#94a3b8",
        )
        pdf.savefig(fig)
        plt.close()

    return output_pdf_path


if __name__ == "__main__":
    out_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Nature_Quantum_Economic_Revenue_Projections_Privacy_Ciphers.pdf",
    )
    build_economic_revenue_nature_preprint(out_file)
    print(f"Successfully generated economic revenue preprint PDF: {out_file}")
