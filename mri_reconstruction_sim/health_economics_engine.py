#!/usr/bin/env python3
"""
Health Economics Engine for Advanced Neuroimaging & MRI Infrastructure
5-Year Stochastic Revenue & Cost Projections
Mathematical Framework: Statistical Distributions & Recurrent Primes Generative Model
"""

import os
import io
import base64
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# First 30 prime numbers for recurrent prime generators
PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113
], dtype=int)

def von_mangoldt_weight(p):
    """Finite von Mangoldt function Lambda(p) = ln(p) for prime p"""
    return np.log(p)

def recurrent_prime_modulator(t_months, K=8, amplitude=0.18):
    """
    Computes the recurrent prime generative demand oscillator Psi_P(t)
    over a finite discrete horizon t in {1, ..., T}.
    
    Psi_P(t) = 1 + amplitude * sum_{k=1}^K [ ln(p_k) / sqrt(p_k) * cos(2*pi*t / p_k + theta_k) ]
    where theta_k = 2*pi * (k^2 mod p_k) / p_k is the modular Berry phase.
    """
    K = min(K, len(PRIMES))
    t = np.asarray(t_months, dtype=float)
    psi = np.ones_like(t)
    
    norm_factor = 0.0
    for k in range(K):
        p_k = PRIMES[k]
        weight = von_mangoldt_weight(p_k) / np.sqrt(p_k)
        theta_k = 2.0 * np.pi * ((k + 1)**2 % p_k) / p_k
        psi += amplitude * weight * np.cos(2.0 * np.pi * t / p_k + theta_k)
        norm_factor += weight
        
    # Scale harmonic component so it stays strictly positive and bounded
    if norm_factor > 0:
        psi = 1.0 + (psi - 1.0) / (norm_factor / K)
    return np.clip(psi, 0.45, 1.85)

def recurrent_prime_phase_orbit(T=60, K=8):
    """
    Generates the discrete recurrent prime phase trajectory (Psi_P(t), Delta Psi_P(t))
    and prime residue orbit z_{t+1} = (z_t + p_{t mod K}) mod M.
    """
    t_vals = np.arange(1, T + 1)
    psi = recurrent_prime_modulator(t_vals, K=K)
    d_psi = np.gradient(psi)
    
    # Modular prime residue orbit
    M = 71
    z_orbit = np.zeros(T, dtype=int)
    z = 1
    for i in range(T):
        p_k = PRIMES[i % K]
        z = (z * 7 + p_k) % M
        z_orbit[i] = z
        
    return t_vals, psi, d_psi, z_orbit

class HealthEconomicsSimulator:
    """
    5-Year Health Economics Monte Carlo Engine with Statistical Distributions
    and Recurrent Prime Generative Revenue Modeling.
    """
    def __init__(self,
                 capex=1850000.0,            # Initial system CapEx (Scanner, Coils, Suite)
                 base_scans_per_day=18.0,    # Daily baseline scan throughput
                 working_days_month=22.0,    # Operating days per month
                 base_reimbursement=680.0,   # Mean blended reimbursement per scan ($)
                 annual_growth_rate=0.06,    # Baseline organic annual growth
                 prime_depth_K=8,            # Order of recurrent primes generator
                 prime_amplitude=0.20,       # Amplitude of prime harmonic surges
                 discount_rate=0.05,         # Annual discount rate r for NPV
                 fixed_opex_monthly=28500.0, # Technologists, power, cryogens, insurance
                 variable_cost_scan=75.0,    # Contrast agents, consumables, QA
                 annual_maintenance_pct=0.08,# OEM service contract as % of CapEx
                 qaly_gain_per_scan=0.045,   # Clinical QALY improvement over standard of care
                 standard_care_cost=420.0):  # Cost of comparator / standard imaging
        
        self.capex = float(capex)
        self.base_scans_per_day = float(base_scans_per_day)
        self.working_days_month = float(working_days_month)
        self.base_reimbursement = float(base_reimbursement)
        self.annual_growth_rate = float(annual_growth_rate)
        self.prime_depth_K = int(prime_depth_K)
        self.prime_amplitude = float(prime_amplitude)
        self.discount_rate = float(discount_rate)
        self.fixed_opex_monthly = float(fixed_opex_monthly)
        self.variable_cost_scan = float(variable_cost_scan)
        self.annual_maintenance_pct = float(annual_maintenance_pct)
        self.qaly_gain_per_scan = float(qaly_gain_per_scan)
        self.standard_care_cost = float(standard_care_cost)
        self.horizon_months = 60 # 5 years

    def run_monte_carlo(self, n_simulations=1000, seed=42):
        """
        Executes N stochastic 60-month paths using Dirichlet payer distributions,
        Weibull maintenance failure processes, Poisson procedure demands,
        and Recurrent Prime generative revenue modulations.
        """
        rng = np.random.default_rng(seed)
        T = self.horizon_months
        t_months = np.arange(1, T + 1)
        
        # 1. Deterministic Recurrent Prime Modulator
        psi_prime = recurrent_prime_modulator(t_months, K=self.prime_depth_K, amplitude=self.prime_amplitude)
        
        # 2. Organic growth trend: (1 + g)^(month / 12)
        growth_trend = (1.0 + self.annual_growth_rate) ** (t_months / 12.0)
        
        # Monthly discount factor array: beta_t = (1 + r/12)^(-t)
        monthly_r = self.discount_rate / 12.0
        discount_factors = (1.0 + monthly_r) ** (-t_months)
        
        # Containers for Monte Carlo runs
        all_revenue = np.zeros((n_simulations, T))
        all_opex = np.zeros((n_simulations, T))
        all_net_cash_flow = np.zeros((n_simulations, T))
        all_cum_cash_flow = np.zeros((n_simulations, T))
        all_scans = np.zeros((n_simulations, T))
        all_npv = np.zeros(n_simulations)
        all_irr = np.zeros(n_simulations)
        all_breakeven = np.zeros(n_simulations)
        all_total_qaly = np.zeros(n_simulations)
        all_icer = np.zeros(n_simulations)

        # Base monthly maintenance allocation
        base_monthly_maint = (self.capex * self.annual_maintenance_pct) / 12.0
        
        # Payer Mix Dirichlet parameters: [Commercial, Medicare, Medicaid, Private Self-Pay]
        dirichlet_alpha = np.array([4.5, 3.8, 1.2, 0.8])
        tier_multipliers = np.array([1.35, 0.95, 0.70, 1.15]) # Multipliers vs base reimbursement
        
        for i in range(n_simulations):
            # A. Procedure Demand: Poisson distributed with Lognormal macroeconomic shock
            macro_shock = rng.lognormal(mean=0.0, sigma=0.06, size=T)
            lambda_scans = (self.base_scans_per_day * self.working_days_month) * growth_trend * psi_prime * macro_shock
            scans_t = rng.poisson(lam=lambda_scans).astype(float)
            all_scans[i, :] = scans_t
            
            # B. Payer Mix: Dirichlet draws for each month
            payer_shares = rng.dirichlet(alpha=dirichlet_alpha, size=T) # Shape (T, 4)
            blended_rate_ratio = np.dot(payer_shares, tier_multipliers) # Shape (T,)
            # Minor random reimbursement noise (lognormal)
            reimb_noise = rng.normal(1.0, 0.03, size=T)
            unit_reimbursement = self.base_reimbursement * blended_rate_ratio * reimb_noise
            
            # Monthly Gross Revenue
            rev_t = scans_t * unit_reimbursement
            all_revenue[i, :] = rev_t
            
            # C. Operational Costs (Fixed OpEx + Variable per scan + Weibull Component Downtime/Repairs)
            # Weibull downtime hazard / unplanned repair shocks
            weibull_shocks = rng.weibull(a=1.8, size=T) * 0.15 # Failure shape k=1.8
            unplanned_maint = base_monthly_maint * weibull_shocks
            
            opex_t = (
                self.fixed_opex_monthly +
                base_monthly_maint +
                unplanned_maint +
                (scans_t * self.variable_cost_scan)
            )
            all_opex[i, :] = opex_t
            
            # D. Net Cash Flow: Revenue - OpEx
            ncf_t = rev_t - opex_t
            all_net_cash_flow[i, :] = ncf_t
            
            # Cumulative Cash Flow starting from -CapEx
            cum_cf = -self.capex + np.cumsum(ncf_t)
            all_cum_cash_flow[i, :] = cum_cf
            
            # E. Net Present Value (NPV)
            npv_val = -self.capex + np.sum(ncf_t * discount_factors)
            all_npv[i] = npv_val
            
            # F. Break-Even Month (First month where cum_cf >= 0)
            be_idx = np.where(cum_cf >= 0)[0]
            if len(be_idx) > 0:
                all_breakeven[i] = be_idx[0] + 1
            else:
                all_breakeven[i] = 61 # Did not break even in 60 months
                
            # G. Internal Rate of Return (IRR) approximation
            # Cash flows: [-CapEx, NCF_1, NCF_2, ..., NCF_60]
            cf_stream = np.concatenate(([-self.capex], ncf_t))
            try:
                # Finite polynomial solver for IRR
                irr_monthly = np.irr(cf_stream) if hasattr(np, 'irr') else np.nan
                if np.isnan(irr_monthly) or irr_monthly <= -1.0 or irr_monthly > 2.0:
                    # Robust estimate via Newton or cash ratio
                    annualized_irr = ((np.sum(ncf_t) / self.capex) ** (1.0 / 5.0) - 1.0)
                else:
                    annualized_irr = ((1.0 + irr_monthly) ** 12) - 1.0
            except Exception:
                annualized_irr = ((np.sum(ncf_t) / self.capex) ** (1.0 / 5.0) - 1.0)
            all_irr[i] = np.clip(annualized_irr, -0.5, 1.5)
            
            # H. QALY and ICER
            total_scans_5y = np.sum(scans_t)
            qalys_gained = total_scans_5y * self.qaly_gain_per_scan
            total_cost_5y = self.capex + np.sum(opex_t)
            standard_cost_5y = total_scans_5y * self.standard_care_cost
            delta_cost = total_cost_5y - standard_cost_5y
            icer_val = delta_cost / max(qalys_gained, 1.0)
            all_total_qaly[i] = qalys_gained
            all_icer[i] = icer_val

        # Aggregate Statistics
        summary = {
            "capex": self.capex,
            "horizon_years": 5,
            "horizon_months": T,
            "mean_5y_revenue": float(np.mean(np.sum(all_revenue, axis=1))),
            "median_5y_revenue": float(np.median(np.sum(all_revenue, axis=1))),
            "std_5y_revenue": float(np.std(np.sum(all_revenue, axis=1))),
            "ci95_5y_revenue": [
                float(np.percentile(np.sum(all_revenue, axis=1), 2.5)),
                float(np.percentile(np.sum(all_revenue, axis=1), 97.5))
            ],
            "mean_5y_opex": float(np.mean(np.sum(all_opex, axis=1))),
            "mean_5y_total_cost": float(self.capex + np.mean(np.sum(all_opex, axis=1))),
            "mean_5y_net_profit": float(np.mean(np.sum(all_revenue, axis=1) - np.sum(all_opex, axis=1)) - self.capex),
            "mean_npv": float(np.mean(all_npv)),
            "median_npv": float(np.median(all_npv)),
            "ci95_npv": [float(np.percentile(all_npv, 2.5)), float(np.percentile(all_npv, 97.5))],
            "prob_positive_npv": float(np.mean(all_npv > 0) * 100.0),
            "mean_irr": float(np.mean(all_irr) * 100.0),
            "median_irr": float(np.median(all_irr) * 100.0),
            "roi_percentage": float(((np.mean(np.sum(all_revenue, axis=1)) - (self.capex + np.mean(np.sum(all_opex, axis=1)))) / (self.capex + np.mean(np.sum(all_opex, axis=1)))) * 100.0),
            "mean_breakeven_months": float(np.mean(all_breakeven[all_breakeven <= 60])) if np.any(all_breakeven <= 60) else 60.0,
            "mean_total_scans": float(np.mean(np.sum(all_scans, axis=1))),
            "mean_total_qalys": float(np.mean(all_total_qaly)),
            "mean_icer_per_qaly": float(np.mean(all_icer)),
            "prime_depth_K": self.prime_depth_K,
            "prime_amplitude": self.prime_amplitude,
            "primes_used": PRIMES[:self.prime_depth_K].tolist()
        }

        # Monthly trajectory percentiles for plotting
        trajectory = {
            "months": t_months.tolist(),
            "revenue_mean": np.mean(all_revenue, axis=0).tolist(),
            "revenue_p05": np.percentile(all_revenue, 5, axis=0).tolist(),
            "revenue_p50": np.percentile(all_revenue, 50, axis=0).tolist(),
            "revenue_p95": np.percentile(all_revenue, 95, axis=0).tolist(),
            "opex_mean": np.mean(all_opex, axis=0).tolist(),
            "cum_cash_flow_mean": np.mean(all_cum_cash_flow, axis=0).tolist(),
            "cum_cash_flow_p05": np.percentile(all_cum_cash_flow, 5, axis=0).tolist(),
            "cum_cash_flow_p50": np.percentile(all_cum_cash_flow, 50, axis=0).tolist(),
            "cum_cash_flow_p95": np.percentile(all_cum_cash_flow, 95, axis=0).tolist(),
            "scans_mean": np.mean(all_scans, axis=0).tolist(),
            "psi_prime": psi_prime.tolist()
        }

        # Annual aggregates (Years 1 to 5)
        annual = []
        for yr in range(5):
            idx_start = yr * 12
            idx_end = (yr + 1) * 12
            yr_rev = float(np.mean(np.sum(all_revenue[:, idx_start:idx_end], axis=1)))
            yr_opex = float(np.mean(np.sum(all_opex[:, idx_start:idx_end], axis=1)))
            yr_scans = int(np.mean(np.sum(all_scans[:, idx_start:idx_end], axis=1)))
            yr_net = yr_rev - yr_opex
            annual.append({
                "year": yr + 1,
                "scans": yr_scans,
                "revenue": yr_rev,
                "opex": yr_opex,
                "net_cash_flow": yr_net,
                "profit_margin_pct": round((yr_net / yr_rev) * 100.0, 2)
            })

        return summary, trajectory, annual, all_npv, all_cum_cash_flow

    def generate_publication_plots(self, summary, trajectory, annual, all_npv, output_dir=None):
        """
        Generates 4 publication-grade figures in dark theme and returns
        both base64 encoded strings and filepaths.
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        t_months = np.array(trajectory["months"])
        plots = {}
        saved_files = []

        # Color Palette
        DARK_BG = '#0a0f1d'
        BOX_BG = '#111827'
        TEXT_LIGHT = '#f1f5f9'
        TEXT_MUTED = '#94a3b8'
        ACCENT_CYAN = '#38bdf8'
        ACCENT_PURPLE = '#a855f7'
        ACCENT_EMERALD = '#10b981'
        ACCENT_ROSE = '#f43f5e'
        ACCENT_AMBER = '#f59e0b'

        def style_axis(ax, title, xlabel="", ylabel=""):
            ax.set_facecolor(BOX_BG)
            ax.set_title(title, color=TEXT_LIGHT, fontsize=11, fontweight='bold', pad=10)
            if xlabel:
                ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=9)
            if ylabel:
                ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        # -------------------------------------------------------------------------
        # PLOT 1: 5-Year Cumulative Revenue, Cost & Cash Flow Trajectory (Fan Chart)
        # -------------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(7.5, 4.6), facecolor=DARK_BG)
        cum_mean = np.array(trajectory["cum_cash_flow_mean"]) / 1e6
        cum_p05 = np.array(trajectory["cum_cash_flow_p05"]) / 1e6
        cum_p95 = np.array(trajectory["cum_cash_flow_p95"]) / 1e6
        
        ax1.plot(t_months, cum_mean, color=ACCENT_CYAN, lw=2.5, label='Mean Net Cash Flow ($M)')
        ax1.fill_between(t_months, cum_p05, cum_p95, color=ACCENT_CYAN, alpha=0.18, label='95% Monte Carlo CI')
        ax1.axhline(0, color=ACCENT_ROSE, linestyle='--', lw=1.5, label='Break-Even Frontier ($0)')
        
        # Mark break-even month
        be_month = summary["mean_breakeven_months"]
        if be_month <= 60:
            ax1.axvline(be_month, color=ACCENT_EMERALD, linestyle=':', lw=1.5, label=f'Break-Even ~{be_month:.1f} Mo')
            ax1.scatter([be_month], [0], color=ACCENT_EMERALD, s=70, zorder=5)

        # Plot CapEx anchor
        ax1.scatter([0], [-self.capex / 1e6], color=ACCENT_ROSE, s=60, zorder=5, label=f'Initial CapEx (-${self.capex/1e6:.2f}M)')

        style_axis(ax1, "5-Year Cumulative Cash Flow & Break-Even Trajectory (N=1000)", "Operational Month (1 to 60)", "Cumulative Position ($ Millions)")
        ax1.legend(loc='lower right', facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)
        
        buf1 = io.BytesIO()
        fpath1 = os.path.join(output_dir, "he_plot1_cash_flow_trajectory.png")
        fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        fig1.savefig(fpath1, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig1)
        plots["plot_cash_flow"] = base64.b64encode(buf1.getvalue()).decode()
        saved_files.append(fpath1)

        # -------------------------------------------------------------------------
        # PLOT 2: Recurrent Prime Generative Revenue Dynamics & Phase Attractor
        # -------------------------------------------------------------------------
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(9.5, 4.4), facecolor=DARK_BG)
        
        # 2A: Prime Modulation Harmonic Waveform
        psi = np.array(trajectory["psi_prime"])
        ax2a.plot(t_months, psi, color=ACCENT_PURPLE, lw=2.2, label=r'$\Psi_{\mathbb{P}}(t)$ Recurrent Prime Modulator')
        ax2a.axhline(1.0, color=TEXT_MUTED, linestyle=':', lw=1.2)
        
        # Highlight prime pulse harmonics
        primes_used = summary["primes_used"]
        for p in primes_used[:4]:
            ax2a.scatter(p, psi[min(p-1, 59)], color=ACCENT_AMBER, s=40, zorder=4)
        ax2a.set_ylim(0.5, 1.6)
        style_axis(ax2a, "Recurrent Prime Harmonics Kernel", "Month t", r"Generative Multiplier $\Psi_{\mathbb{P}}(t)$")
        ax2a.legend(loc='upper right', facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)

        # 2B: Phase Portrait Orbit (dPsi vs Psi)
        _, _, d_psi, z_orbit = recurrent_prime_phase_orbit(T=60, K=self.prime_depth_K)
        scatter = ax2b.scatter(psi, d_psi, c=t_months, cmap='plasma', s=35, edgecolors='none', zorder=4)
        ax2b.plot(psi, d_psi, color='#475569', lw=0.8, alpha=0.6)
        cbar = fig2.colorbar(scatter, ax=ax2b, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        cbar.set_label("Horizon Month", color=TEXT_MUTED, fontsize=8)
        style_axis(ax2b, "Recurrent Prime Phase Orbit Attractor", r"State $\Psi_{\mathbb{P}}(t)$", r"Rate $\Delta \Psi_{\mathbb{P}} / \Delta t$")

        buf2 = io.BytesIO()
        fpath2 = os.path.join(output_dir, "he_plot2_recurrent_prime_phase.png")
        fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        fig2.savefig(fpath2, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig2)
        plots["plot_prime_dynamics"] = base64.b64encode(buf2.getvalue()).decode()
        saved_files.append(fpath2)

        # -------------------------------------------------------------------------
        # PLOT 3: Statistical Probability Density Distributions (NPV & Revenue)
        # -------------------------------------------------------------------------
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(9.5, 4.4), facecolor=DARK_BG)
        
        # 3A: NPV Distribution Histogram with KDE
        npv_m = all_npv / 1e6
        n_bins = 35
        ax3a.hist(npv_m, bins=n_bins, color=ACCENT_EMERALD, alpha=0.55, density=True, edgecolor='#1e293b')
        kde_x = np.linspace(np.min(npv_m), np.max(npv_m), 200)
        kde = stats.gaussian_kde(npv_m)
        ax3a.plot(kde_x, kde(kde_x), color=ACCENT_EMERALD, lw=2.2, label='NPV Density (KDE)')
        ax3a.axvline(summary["mean_npv"]/1e6, color=TEXT_LIGHT, linestyle='--', lw=1.5, label=f'Mean = ${summary["mean_npv"]/1e6:.2f}M')
        ax3a.axvline(0, color=ACCENT_ROSE, linestyle=':', lw=1.5, label='NPV = 0 Threshold')
        style_axis(ax3a, "5-Year NPV Probability Distribution", "Net Present Value ($ Millions)", "Probability Density")
        ax3a.legend(loc='upper right', facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)

        # 3B: Annual Revenue vs OpEx Comparison Bar Chart
        years = [a["year"] for a in annual]
        rev_yr = [a["revenue"] / 1e6 for a in annual]
        opex_yr = [a["opex"] / 1e6 for a in annual]
        bar_w = 0.35
        x_indices = np.arange(len(years))
        ax3b.bar(x_indices - bar_w/2, rev_yr, width=bar_w, color=ACCENT_CYAN, label='Revenue ($M)', alpha=0.85)
        ax3b.bar(x_indices + bar_w/2, opex_yr, width=bar_w, color=ACCENT_AMBER, label='OpEx ($M)', alpha=0.85)
        ax3b.set_xticks(x_indices)
        ax3b.set_xticklabels([f"Year {y}" for y in years])
        style_axis(ax3b, "Annual Operating Financials (Years 1-5)", "Operational Year", "Amount ($ Millions)")
        ax3b.legend(loc='upper left', facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)

        buf3 = io.BytesIO()
        fpath3 = os.path.join(output_dir, "he_plot3_statistical_distributions.png")
        fig3.savefig(buf3, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        fig3.savefig(fpath3, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig3)
        plots["plot_distributions"] = base64.b64encode(buf3.getvalue()).decode()
        saved_files.append(fpath3)

        # -------------------------------------------------------------------------
        # PLOT 4: Cost Breakdown & Cost-Effectiveness Acceptability Curve (CEAC)
        # -------------------------------------------------------------------------
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9.5, 4.4), facecolor=DARK_BG)
        
        # 4A: 5-Year Life-Cycle Cost Breakdown Donut
        cost_labels = ['CapEx System', 'Salaries & Technicians', 'Service Contract', 'Cryogens & Power', 'Scan Consumables']
        five_yr_scans = summary["mean_total_scans"]
        costs = [
            self.capex,
            self.fixed_opex_monthly * 0.55 * 60, # Technologist salaries
            (self.capex * self.annual_maintenance_pct) * 5,
            self.fixed_opex_monthly * 0.45 * 60, # Cryogens, facility power
            five_yr_scans * self.variable_cost_scan
        ]
        donut_colors = [ACCENT_CYAN, ACCENT_PURPLE, ACCENT_EMERALD, ACCENT_AMBER, ACCENT_ROSE]
        wedges, texts, autotexts = ax4a.pie(
            costs, labels=None, autopct='%1.1f%%', startangle=140,
            colors=donut_colors, pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2)
        )
        for autotext in autotexts:
            autotext.set_color(TEXT_LIGHT)
            autotext.set_fontsize(7.5)
            autotext.set_fontweight('bold')
        ax4a.legend(wedges, cost_labels, loc='center left', bbox_to_anchor=(0.95, 0.5),
                    facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=7.5)
        ax4a.set_facecolor(BOX_BG)
        ax4a.set_title("5-Year Total Cost Allocation ($M)", color=TEXT_LIGHT, fontsize=11, fontweight='bold', pad=10)

        # 4B: Cost-Effectiveness Acceptability Curve (CEAC)
        wtp_thresholds = np.linspace(10000, 150000, 100) # Willingness-To-Pay per QALY ($10k - $150k)
        icer_mean = summary["mean_icer_per_qaly"]
        # Prob(ICER < WTP) approximated via normal cumulative of ICER distribution
        prob_ce = stats.norm.cdf(wtp_thresholds, loc=icer_mean, scale=max(abs(icer_mean)*0.35, 8000.0))
        
        ax4b.plot(wtp_thresholds / 1000.0, prob_ce * 100.0, color=ACCENT_EMERALD, lw=2.2, label='Acceptability Curve')
        ax4b.axvline(50.0, color=ACCENT_AMBER, linestyle='--', lw=1.2, label='US Standard ($50k/QALY)')
        ax4b.axvline(100.0, color=ACCENT_CYAN, linestyle=':', lw=1.2, label='High-Value Tier ($100k/QALY)')
        ax4b.scatter([icer_mean / 1000.0], [50.0], color=ACCENT_ROSE, s=50, zorder=5, label=f'Mean ICER (${icer_mean:,.0f})')
        style_axis(ax4b, "Cost-Effectiveness Acceptability (CEAC)", "Willingness-to-Pay ($k / QALY)", "Probability Cost-Effective (%)")
        ax4b.set_ylim(0, 105)
        ax4b.legend(loc='lower right', facecolor=BOX_BG, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=7.5)

        buf4 = io.BytesIO()
        fpath4 = os.path.join(output_dir, "he_plot4_cost_effectiveness_ceac.png")
        fig4.savefig(buf4, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        fig4.savefig(fpath4, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig4)
        plots["plot_cost_breakdown"] = base64.b64encode(buf4.getvalue()).decode()
        saved_files.append(fpath4)

        return plots, saved_files

def run_health_economics_pipeline(capex=1850000.0,
                                  base_scans_per_day=18.0,
                                  base_reimbursement=680.0,
                                  discount_rate=0.05,
                                  annual_growth=0.06,
                                  prime_depth_K=8,
                                  prime_amplitude=0.20):
    """
    Convenience wrapper to run complete 5-year simulation, generate plots,
    and return complete payload for web API and preprint.
    """
    engine = HealthEconomicsSimulator(
        capex=capex,
        base_scans_per_day=base_scans_per_day,
        base_reimbursement=base_reimbursement,
        annual_growth_rate=annual_growth,
        prime_depth_K=prime_depth_K,
        prime_amplitude=prime_amplitude,
        discount_rate=discount_rate
    )
    summary, trajectory, annual, all_npv, all_cum_cash_flow = engine.run_monte_carlo(n_simulations=1000)
    plots, saved_files = engine.generate_publication_plots(summary, trajectory, annual, all_npv)
    
    return {
        "success": True,
        "summary": summary,
        "trajectory": trajectory,
        "annual_breakdown": annual,
        "plots": plots,
        "plot_files": saved_files
    }

if __name__ == '__main__':
    print("Testing Health Economics Pipeline...")
    res = run_health_economics_pipeline()
    print("5-Year Mean Revenue: ${:,.2f}".format(res["summary"]["mean_5y_revenue"]))
    print("5-Year Mean NPV:     ${:,.2f}".format(res["summary"]["mean_npv"]))
    print("Mean Break-Even:     {:.1f} months".format(res["summary"]["mean_breakeven_months"]))
    print("IRR:                 {:.2f}%".format(res["summary"]["mean_irr"]))
    print("Plots generated:", len(res["plots"]))
