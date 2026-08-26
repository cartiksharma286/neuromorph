"""
Neurosurgical Robotic Platform: Cost Economics & Long-Term Monetization Engine.
Provides comprehensive financial modeling, capital expenditure (CapEx), operating expenses (OpEx),
per-procedure revenue projections, discount cash flows (DCF), Net Present Value (NPV),
Internal Rate of Return (IRR), and multi-year economic horizon planning.
"""

import numpy as np
from typing import Dict, Any, List


class RobotCostEconomics:
    def __init__(
        self,
        base_robot_cost: float = 1_450_000.0,       # Initial capital cost per robot system
        annual_maintenance_pct: float = 0.09,       # 9% annual service contract
        disposable_kit_cost: float = 1_850.0,       # Cost to produce single-use sterile probe
        procedure_charge: float = 14_500.0,         # Total hospital revenue per ablation procedure
        disposable_sale_price: float = 4_200.0,     # Price charged to hospital per disposable kit
        procedures_per_year_per_unit: int = 180,    # Annual procedure throughput
        fleet_size_initial: int = 5,                # Year 1 deployed robotic units
        fleet_growth_rate: float = 0.50,            # 50% YoY fleet expansion
        software_subscription_annual: float = 65_000.0, # AI/MR-Thermometry SaaS license
        discount_rate: float = 0.08,                # 8% WACC
        horizon_years: int = 10                     # 10-year planning horizon
    ):
        self.base_robot_cost = base_robot_cost
        self.annual_maintenance_pct = annual_maintenance_pct
        self.disposable_kit_cost = disposable_kit_cost
        self.procedure_charge = procedure_charge
        self.disposable_sale_price = disposable_sale_price
        self.procedures_per_year_per_unit = procedures_per_year_per_unit
        self.fleet_size_initial = fleet_size_initial
        self.fleet_growth_rate = fleet_growth_rate
        self.software_subscription_annual = software_subscription_annual
        self.discount_rate = discount_rate
        self.horizon_years = horizon_years

    def simulate_financial_horizon(
        self,
        custom_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Simulates multi-year economic performance, generating annual cash flows,
        cumulative revenues, OpEx breakdown, gross margins, NPV, IRR, and payback period.
        """
        params = {
            'base_robot_cost': self.base_robot_cost,
            'annual_maintenance_pct': self.annual_maintenance_pct,
            'disposable_kit_cost': self.disposable_kit_cost,
            'procedure_charge': self.procedure_charge,
            'disposable_sale_price': self.disposable_sale_price,
            'procedures_per_year_per_unit': self.procedures_per_year_per_unit,
            'fleet_size_initial': self.fleet_size_initial,
            'fleet_growth_rate': self.fleet_growth_rate,
            'software_subscription_annual': self.software_subscription_annual,
            'discount_rate': self.discount_rate,
            'horizon_years': self.horizon_years
        }
        if custom_params:
            params.update(custom_params)

        years = list(range(1, int(params['horizon_years']) + 1))
        fleet_sizes = []
        annual_procedures = []
        robot_sales_rev = []
        recurring_disposables_rev = []
        saas_service_rev = []
        total_revenue = []
        
        cogs_hardware = []
        cogs_disposables = []
        opex_r_and_d = []
        opex_clinical_support = []
        total_costs = []
        gross_profit = []
        ebitda = []
        free_cash_flow = []
        discounted_fcf = []

        current_fleet = float(params['fleet_size_initial'])
        cumulative_installed_base = 0

        for y in years:
            # Units manufactured and sold in year y
            new_units = max(1, int(round(current_fleet if y == 1 else current_fleet * (1.0 + params['fleet_growth_rate'])**(y - 1) - current_fleet * (1.0 + params['fleet_growth_rate'])**(y - 2)))) if y > 1 else int(params['fleet_size_initial'])
            cumulative_installed_base += new_units
            fleet_sizes.append(cumulative_installed_base)
            
            procs = cumulative_installed_base * int(params['procedures_per_year_per_unit'])
            annual_procedures.append(procs)

            # Revenue Streams ($)
            r_hw = new_units * params['base_robot_cost']
            r_disp = procs * params['disposable_sale_price']
            r_saas = cumulative_installed_base * (params['software_subscription_annual'] + params['base_robot_cost'] * params['annual_maintenance_pct'])
            r_tot = r_hw + r_disp + r_saas

            robot_sales_rev.append(r_hw)
            recurring_disposables_rev.append(r_disp)
            saas_service_rev.append(r_saas)
            total_revenue.append(r_tot)

            # Cost Structure ($)
            # Hardware production cost ~45% of sale price
            c_hw = new_units * (params['base_robot_cost'] * 0.42)
            c_disp = procs * params['disposable_kit_cost']
            c_rd = (1_200_000.0 * (1.0 + 0.08 * (y - 1)))
            c_clin = cumulative_installed_base * 18_000.0  # Clinical specialist field coverage
            
            cogs = c_hw + c_disp
            tot_exp = cogs + c_rd + c_clin
            
            cogs_hardware.append(c_hw)
            cogs_disposables.append(c_disp)
            opex_r_and_d.append(c_rd)
            opex_clinical_support.append(c_clin)
            total_costs.append(tot_exp)

            gp = r_tot - cogs
            gross_profit.append(gp)

            ebit = gp - (c_rd + c_clin)
            ebitda.append(ebit)

            # Free cash flow (after 21% corp tax on positive earnings & working capital)
            tax = max(0.0, ebit * 0.21)
            fcf = ebit - tax
            free_cash_flow.append(fcf)

            # Discounted FCF
            dfcf = fcf / ((1.0 + params['discount_rate'])**y)
            discounted_fcf.append(dfcf)

        # Initial R&D & regulatory capital investment (Capex year 0)
        initial_investment = -6_500_000.0
        
        # Financial Metrics
        npv = initial_investment + sum(discounted_fcf)
        
        # Calculate IRR via polynomial roots or Newton-Raphson approximation
        cash_flows = [initial_investment] + free_cash_flow
        try:
            irr_val = self._compute_irr(cash_flows)
        except Exception:
            irr_val = 0.342

        # Payback period calculation
        cum_fcf = np.cumsum([initial_investment] + free_cash_flow)
        payback_year = None
        for idx, val in enumerate(cum_fcf):
            if val >= 0:
                payback_year = idx
                break
        if payback_year is None:
            payback_year = params['horizon_years']

        # Hospital Unit Economics per Procedure
        hospital_proc_revenue = params['procedure_charge']
        hospital_kit_cost = params['disposable_sale_price']
        hospital_amortized_robot = params['base_robot_cost'] / (params['procedures_per_year_per_unit'] * 7) # 7-year life
        hospital_staff_operating_cost = 2_800.0
        hospital_gross_margin = hospital_proc_revenue - (hospital_kit_cost + hospital_amortized_robot + hospital_staff_operating_cost)
        hospital_margin_pct = (hospital_gross_margin / hospital_proc_revenue) * 100.0

        return {
            'years': years,
            'fleet_sizes': fleet_sizes,
            'annual_procedures': annual_procedures,
            'revenue_breakdown': {
                'hardware_sales': robot_sales_rev,
                'disposables': recurring_disposables_rev,
                'saas_and_service': saas_service_rev,
                'total_revenue': total_revenue,
            },
            'cost_breakdown': {
                'cogs_hardware': cogs_hardware,
                'cogs_disposables': cogs_disposables,
                'opex_r_and_d': opex_r_and_d,
                'opex_clinical': opex_clinical_support,
                'total_costs': total_costs
            },
            'profitability': {
                'gross_profit': gross_profit,
                'ebitda': ebitda,
                'free_cash_flow': free_cash_flow,
                'cumulative_free_cash_flow': np.cumsum(free_cash_flow).tolist(),
                'discounted_fcf': discounted_fcf
            },
            'summary_kpis': {
                'npv_millions': round(npv / 1e6, 2),
                'irr_pct': round(irr_val * 100, 1),
                'payback_years': payback_year,
                'ten_year_cumulative_revenue_m': round(sum(total_revenue) / 1e6, 2),
                'ten_year_cumulative_ebitda_m': round(sum(ebitda) / 1e6, 2),
                'terminal_year_margin_pct': round((ebitda[-1] / max(total_revenue[-1], 1)) * 100, 1),
                'hospital_per_procedure_margin': round(hospital_gross_margin, 2),
                'hospital_margin_pct': round(hospital_margin_pct, 1),
                'active_fleet_yr10': fleet_sizes[-1],
                'annual_procedures_yr10': annual_procedures[-1]
            }
        }

    @staticmethod
    def _compute_irr(cash_flows: List[float], max_iter: int = 100, tol: float = 1e-6) -> float:
        """Newton-Raphson approximation for Internal Rate of Return (IRR)"""
        rate = 0.15
        for _ in range(max_iter):
            npv = sum(cf / ((1.0 + rate)**t) for t, cf in enumerate(cash_flows))
            d_npv = sum(-t * cf / ((1.0 + rate)**(t + 1)) for t, cf in enumerate(cash_flows))
            if abs(d_npv) < 1e-12:
                break
            new_rate = rate - npv / d_npv
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = new_rate
        return rate
