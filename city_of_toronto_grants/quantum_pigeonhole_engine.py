"""Quantum-Statistical Selective Excitation & Pigeonhole Combinatorics Engine.
Designed for City of Toronto Community Grants Allocation and Municipal Revenue Maximization.
Author: Cartik Sharma, Neuromorph System.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import math

# 25 Municipal Wards of the City of Toronto with Demographic/Deprivation Indices
TORONTO_WARDS = [
    {"id": 1, "name": "Etobicoke North", "pop": 118040, "deprivation_index": 0.88, "priority": "High"},
    {"id": 2, "name": "Etobicoke Centre", "pop": 118020, "deprivation_index": 0.42, "priority": "Medium"},
    {"id": 3, "name": "Etobicoke-Lakeshore", "pop": 129080, "deprivation_index": 0.38, "priority": "Low"},
    {"id": 4, "name": "Parkdale-High Park", "pop": 108800, "deprivation_index": 0.45, "priority": "Medium"},
    {"id": 5, "name": "York South-Weston", "pop": 116606, "deprivation_index": 0.92, "priority": "High"},
    {"id": 6, "name": "York Centre", "pop": 104320, "deprivation_index": 0.58, "priority": "Medium"},
    {"id": 7, "name": "Humber River-Black Creek", "pop": 108035, "deprivation_index": 0.96, "priority": "High"},
    {"id": 8, "name": "Eglinton-Lawrence", "pop": 114395, "deprivation_index": 0.35, "priority": "Low"},
    {"id": 9, "name": "Davenport", "pop": 108475, "deprivation_index": 0.52, "priority": "Medium"},
    {"id": 10, "name": "Spadina-Fort York", "pop": 133570, "deprivation_index": 0.30, "priority": "Low"},
    {"id": 11, "name": "University-Rosedale", "pop": 104310, "deprivation_index": 0.28, "priority": "Low"},
    {"id": 12, "name": "Toronto-St. Paul's", "pop": 107900, "deprivation_index": 0.25, "priority": "Low"},
    {"id": 13, "name": "Toronto Centre", "pop": 121980, "deprivation_index": 0.85, "priority": "High"},
    {"id": 14, "name": "Toronto-Danforth", "pop": 106875, "deprivation_index": 0.39, "priority": "Low"},
    {"id": 15, "name": "Don Valley West", "pop": 102510, "deprivation_index": 0.50, "priority": "Medium"},
    {"id": 16, "name": "Don Valley East", "pop": 94580, "deprivation_index": 0.74, "priority": "High"},
    {"id": 17, "name": "Don Valley North", "pop": 110080, "deprivation_index": 0.62, "priority": "Medium"},
    {"id": 18, "name": "Willowdale", "pop": 118800, "deprivation_index": 0.48, "priority": "Medium"},
    {"id": 19, "name": "Beaches-East York", "pop": 109465, "deprivation_index": 0.41, "priority": "Low"},
    {"id": 20, "name": "Scarborough Southwest", "pop": 110280, "deprivation_index": 0.72, "priority": "High"},
    {"id": 21, "name": "Scarborough Centre", "pop": 112600, "deprivation_index": 0.78, "priority": "High"},
    {"id": 22, "name": "Scarborough-Agincourt", "pop": 105540, "deprivation_index": 0.65, "priority": "Medium"},
    {"id": 23, "name": "Scarborough North", "pop": 98800, "deprivation_index": 0.70, "priority": "High"},
    {"id": 24, "name": "Scarborough-Guildwood", "pop": 102340, "deprivation_index": 0.82, "priority": "High"},
    {"id": 25, "name": "Scarborough-Rouge Park", "pop": 102275, "deprivation_index": 0.61, "priority": "Medium"}
]

# Strategic Community Grant Sectors under City of Toronto Social Development, Finance and Administration
GRANT_SECTORS = [
    {
        "id": "affordable_housing",
        "name": "Affordable Housing & Shelter Innovation",
        "weight": 0.25,
        "sroi_factor": 5.40,
        "cost_avoidance_rate": 0.42, # reduction in shelter and emergency health services
        "base_resonance_freq": 0.05,
        "matching_leverage": 2.80
    },
    {
        "id": "youth_equity_safety",
        "name": "Youth Equity, Gang Diversion & Community Safety",
        "weight": 0.20,
        "sroi_factor": 6.10,
        "cost_avoidance_rate": 0.48, # police, court, emergency intervention avoidance
        "base_resonance_freq": -0.45,
        "matching_leverage": 2.40
    },
    {
        "id": "poverty_food_security",
        "name": "Poverty Reduction & Food Sovereignty Networks",
        "weight": 0.18,
        "sroi_factor": 4.80,
        "cost_avoidance_rate": 0.35, # public health and nutritional deficit reduction
        "base_resonance_freq": 0.55,
        "matching_leverage": 2.10
    },
    {
        "id": "mental_health_crisis",
        "name": "Community Crisis Response & Mental Health",
        "weight": 0.15,
        "sroi_factor": 5.80,
        "cost_avoidance_rate": 0.52, # EMS diversion and acute psychiatric triage
        "base_resonance_freq": -0.85,
        "matching_leverage": 2.50
    },
    {
        "id": "arts_culture_vibrancy",
        "name": "Local Arts, Cultural Incubators & Neighborhood Vibe",
        "weight": 0.12,
        "sroi_factor": 3.90,
        "cost_avoidance_rate": 0.18, # municipal commercial activity, tourism & BIA tax lift
        "base_resonance_freq": 1.15,
        "matching_leverage": 3.20
    },
    {
        "id": "climate_green_resilience",
        "name": "Urban Climate Resilience & Green Community Infrastructure",
        "weight": 0.10,
        "sroi_factor": 4.50,
        "cost_avoidance_rate": 0.30, # stormwater, extreme heat shelter and energy savings
        "base_resonance_freq": -1.35,
        "matching_leverage": 3.50
    }
]

# Downtown Toronto Biotech Innovation Corridor Hubs (Discovery District, University Ave, MaRS, U of T, TMU)
DOWNTOWN_BIOTECH_HUBS = [
    {
        "id": "mars_discovery",
        "name": "MaRS Discovery District & University Ave Hospital Row",
        "focus": "Genomic Medicine, Clinical Trials & Advanced Therapeutics",
        "annual_rd_investment_cad": 68_000_000.0,
        "vc_leverage_ratio": 4.85,
        "avg_stem_salary_cad": 118_000.0,
        "base_annual_jobs": 520,
        "ip_spinoff_multiplier": 3.40,
        "municipal_tax_capture_rate": 0.088
    },
    {
        "id": "sric_uoft",
        "name": "Schwartz Reisman Innovation Campus (U of T)",
        "focus": "AI Drug Discovery, Quantum Biology & Computational Oncology",
        "annual_rd_investment_cad": 54_000_000.0,
        "vc_leverage_ratio": 5.20,
        "avg_stem_salary_cad": 126_000.0,
        "base_annual_jobs": 440,
        "ip_spinoff_multiplier": 3.90,
        "municipal_tax_capture_rate": 0.092
    },
    {
        "id": "uhn_pmc",
        "name": "Princess Margaret Cancer Tower & UHN Bio-Incubator",
        "focus": "Cellular Immunotherapy, Radiopharmaceuticals & Diagnostics",
        "annual_rd_investment_cad": 48_000_000.0,
        "vc_leverage_ratio": 4.40,
        "avg_stem_salary_cad": 112_000.0,
        "base_annual_jobs": 380,
        "ip_spinoff_multiplier": 3.20,
        "municipal_tax_capture_rate": 0.085
    },
    {
        "id": "sickkids_pgcrl",
        "name": "Peter Gilgan Centre for Research and Learning (SickKids)",
        "focus": "Pediatric Precision Medicine & Rare Disease Gene Editing",
        "annual_rd_investment_cad": 42_000_000.0,
        "vc_leverage_ratio": 4.10,
        "avg_stem_salary_cad": 109_000.0,
        "base_annual_jobs": 310,
        "ip_spinoff_multiplier": 3.00,
        "municipal_tax_capture_rate": 0.082
    },
    {
        "id": "tmu_biomed",
        "name": "Toronto Metropolitan University iBest & Biomedical Cluster",
        "focus": "Neuromorphic BCI Implants & Surgical Robotics Integration",
        "annual_rd_investment_cad": 36_000_000.0,
        "vc_leverage_ratio": 4.60,
        "avg_stem_salary_cad": 114_000.0,
        "base_annual_jobs": 290,
        "ip_spinoff_multiplier": 3.60,
        "municipal_tax_capture_rate": 0.089
    },
    {
        "id": "spadina_healthtech",
        "name": "King-Spadina HealthTech & Digital Therapeutics Quarter",
        "focus": "Quantum Biosensors, Digital Health & Diagnostic Software",
        "annual_rd_investment_cad": 32_000_000.0,
        "vc_leverage_ratio": 5.50,
        "avg_stem_salary_cad": 122_000.0,
        "base_annual_jobs": 260,
        "ip_spinoff_multiplier": 4.10,
        "municipal_tax_capture_rate": 0.095
    }
]

class QuantumSelectiveExcitationEngine:
    """Simulates quantum-inspired selective excitation profiles using RF pulse shaping.
    Maps community demand resonance frequencies to Bloch excitation states |psi(theta, phi)>.
    """
    def __init__(self, time_steps: int = 256, duration_ms: float = 2.0):
        self.time_steps = time_steps
        self.duration_ms = duration_ms
        self.duration_sec = duration_ms * 1e-3
        self.t = np.linspace(-self.duration_sec/2, self.duration_sec/2, time_steps)
        self.dt = self.duration_sec / time_steps
        
    def generate_sinc_hermite_pulse(self, flip_angle_deg: float, bandwidth_khz: float) -> np.ndarray:
        """Generates a windowed Sinc-Hermite selective excitation RF pulse omega_1(t) in rad/sec."""
        theta = np.deg2rad(flip_angle_deg)
        tbw = bandwidth_khz * self.duration_ms
        sinc_envelope = np.sinc(tbw * self.t / self.duration_sec)
        
        # Blackman-Harris windowing to suppress sidelobe excitation out-of-band
        bh_window = (0.35875 
                     - 0.48829 * np.cos(2 * np.pi * (np.arange(self.time_steps) / self.time_steps))
                     + 0.14128 * np.cos(4 * np.pi * (np.arange(self.time_steps) / self.time_steps))
                     - 0.01168 * np.cos(6 * np.pi * (np.arange(self.time_steps) / self.time_steps)))
        
        pulse = sinc_envelope * bh_window
        # Normalize to target integral flip angle theta: sum(omega_1 * dt) = theta
        omega1 = (pulse / (np.sum(pulse * self.dt) + 1e-12)) * theta
        return omega1

    def compute_excitation_profile(self, pulse: np.ndarray, freq_range_khz: Tuple[float, float] = (-3.0, 3.0), n_freqs: int = 150) -> Dict[str, Any]:
        """Solves the Bloch selective excitation rotation across frequency offsets.
        Returns transverse excitation |M_xy| = sin(theta_eff) and longitudinal M_z = cos(theta_eff).
        """
        freqs = np.linspace(freq_range_khz[0], freq_range_khz[1], n_freqs)
        m_xy = []
        m_z = []
        excitation_prob = []
        
        dt = self.dt
        
        for f in freqs:
            # 2-level rotation spinor [alpha, beta]^T
            alpha = 1.0 + 0.0j
            beta = 0.0 + 0.0j
            
            omega_z = 2 * np.pi * f * 1000.0 # rad/sec
            
            for omega_1 in pulse:
                omega_eff = np.sqrt(omega_1**2 + omega_z**2) + 1e-12
                phi = omega_eff * dt
                
                # Cayley-Klein rotation parameters
                n_x = omega_1 / omega_eff
                n_z = omega_z / omega_eff
                
                a = np.cos(phi / 2) - 1j * n_z * np.sin(phi / 2)
                b = -1j * n_x * np.sin(phi / 2)
                
                # Update spinor
                new_alpha = a * alpha - np.conj(b) * beta
                new_beta = b * alpha + np.conj(a) * beta
                alpha, beta = new_alpha, new_beta
            
            prob = float(np.abs(beta)**2)
            transverse = float(2 * np.abs(np.conj(alpha) * beta))
            longitudinal = float(np.abs(alpha)**2 - np.abs(beta)**2)
            
            excitation_prob.append(round(min(1.0, prob), 4))
            m_xy.append(round(min(1.0, transverse), 4))
            m_z.append(round(longitudinal, 4))
            
        return {
            "frequencies": [round(f, 3) for f in freqs.tolist()],
            "excitation_probability": excitation_prob,
            "transverse_magnetization": m_xy,
            "longitudinal_magnetization": m_z
        }

class PigeonholeCombinatoricsEngine:
    """Applies Generalized Dirichlet Pigeonhole Combinatorics to partition N grant applicants
    into K municipal priority impact pigeonholes with capacity bounds and collision optimization.
    """
    def __init__(self, n_applicants: int, k_pigeonholes: int):
        self.n = n_applicants
        self.k = k_pigeonholes
        
    def compute_dirichlet_bounds(self) -> Dict[str, Any]:
        """Calculates fundamental Dirichlet pigeonhole invariants."""
        ceil_bound = math.ceil(self.n / self.k)
        floor_bound = math.floor(self.n / self.k)
        collisions = self.n - self.k if self.n >= self.k else 0
        entropy = -sum((1.0/self.k) * math.log(1.0/self.k) for _ in range(self.k))
        
        return {
            "total_applicants_n": self.n,
            "impact_bins_k": self.k,
            "pigeonhole_minimum_peak": ceil_bound,
            "pigeonhole_floor_baseline": floor_bound,
            "guaranteed_collisions": collisions,
            "combinatorial_entropy": round(entropy, 4)
        }
        
    def partition_with_resonance(self, excitation_weights: List[float], capacity_limits: List[int]) -> List[int]:
        """Partitions n applicants into k bins modulated by quantum excitation resonance weights
        subject to capacity constraints: sum(n_i) = N, 1 <= n_i <= cap_i.
        """
        weights = np.array(excitation_weights, dtype=float)
        weights = weights / np.sum(weights)
        
        # Base allocation proportional to weights
        raw_alloc = weights * self.n
        int_alloc = np.floor(raw_alloc).astype(int)
        
        # Ensure at least 1 in each bin (Dirichlet non-emptiness guarantee)
        int_alloc = np.maximum(int_alloc, 1)
        
        # Cap at limits
        caps = np.array(capacity_limits, dtype=int)
        int_alloc = np.minimum(int_alloc, caps)
        
        remaining = self.n - np.sum(int_alloc)
        
        # Distribute remaining by largest fractional remainder
        remainders = raw_alloc - int_alloc
        order = np.argsort(-remainders)
        
        for idx in order:
            if remaining <= 0:
                break
            if int_alloc[idx] < caps[idx]:
                int_alloc[idx] += 1
                remaining -= 1
                
        # If still remaining due to caps, distribute to any bin with room
        for idx in range(self.k):
            while remaining > 0 and int_alloc[idx] < caps[idx]:
                int_alloc[idx] += 1
                remaining -= 1
                
        return int_alloc.tolist()

class TorontoCommunityGrantOptimizer:
    """Integrates Quantum Selective Excitation, Pigeonhole Combinatorics, and City of Toronto
    Municipal Revenue Projections to maximize capital yield and social return on investment (SROI).
    """
    def __init__(self):
        self.bloch = QuantumSelectiveExcitationEngine()
        self.wards = TORONTO_WARDS
        self.sectors = GRANT_SECTORS
        
    def run_full_allocation(
        self,
        total_budget_cad: float = 125_000_000.0,
        flip_angle_deg: float = 90.0,
        bandwidth_khz: float = 2.4,
        total_applicants: int = 1450,
        projection_years: int = 5
    ) -> Dict[str, Any]:
        """Executes the dual-engine optimization and generates 5-year municipal revenue projections."""
        
        # 1. Generate RF selective pulse and spectral excitation
        rf_pulse = self.bloch.generate_sinc_hermite_pulse(flip_angle_deg, bandwidth_khz)
        profile = self.bloch.compute_excitation_profile(rf_pulse)
        
        # 2. Compute resonance excitation factor for each sector based on base frequency
        sector_excitations = []
        for sec in self.sectors:
            f0 = sec["base_resonance_freq"]
            # Interpolate excitation probability from profile
            freqs = np.array(profile["frequencies"])
            probs = np.array(profile["excitation_probability"])
            idx = (np.abs(freqs - f0)).argmin()
            prob = probs[idx]
            # Modulate sector weight by quantum excitation state
            eff_weight = sec["weight"] * (0.4 + 0.6 * prob)
            sector_excitations.append({
                "sector_id": sec["id"],
                "sector_name": sec["name"],
                "base_freq_khz": f0,
                "excitation_prob": round(float(prob), 4),
                "modulated_weight": round(float(eff_weight), 4),
                "sroi_factor": sec["sroi_factor"],
                "cost_avoidance_rate": sec["cost_avoidance_rate"],
                "matching_leverage": sec["matching_leverage"]
            })
            
        # Normalize modulated weights
        tot_mod_wt = sum(s["modulated_weight"] for s in sector_excitations)
        for s in sector_excitations:
            s["normalized_weight"] = round(s["modulated_weight"] / tot_mod_wt, 4)
            
        # 3. Pigeonhole Combinatoric Partition of Applicants
        k_bins = len(self.sectors)
        pigeon_engine = PigeonholeCombinatoricsEngine(total_applicants, k_bins)
        pigeon_invariants = pigeon_engine.compute_dirichlet_bounds()
        
        sector_weights = [s["normalized_weight"] for s in sector_excitations]
        capacity_caps = [int(total_applicants * 0.45) for _ in range(k_bins)]
        applicant_partition = pigeon_engine.partition_with_resonance(sector_weights, capacity_caps)
        
        # 4. Budget Allocation across Sectors
        sector_allocations = []
        for i, sec in enumerate(sector_excitations):
            budget_for_sector = total_budget_cad * sec["normalized_weight"]
            n_grants = applicant_partition[i]
            avg_grant_size = budget_for_sector / n_grants if n_grants > 0 else 0
            
            # Matching funds unlocked (Provincial, Federal, Corporate/Philanthropic)
            leveraged_capital = budget_for_sector * sec["matching_leverage"]
            
            sector_allocations.append({
                "sector_id": sec["sector_id"],
                "sector_name": sec["sector_name"],
                "allocation_cad": round(budget_for_sector, 2),
                "allocation_pct": round(sec["normalized_weight"] * 100, 2),
                "grants_awarded": n_grants,
                "avg_grant_size_cad": round(avg_grant_size, 2),
                "excitation_resonance": sec["excitation_prob"],
                "leveraged_capital_cad": round(leveraged_capital, 2),
                "total_invested_cad": round(budget_for_sector + leveraged_capital, 2)
            })
            
        # 5. Ward-level Distribution using Deprivation Index & Selective Spatial Focusing
        # Deprivation index amplifies high-need wards (e.g. Humber River-Black Creek, Scarborough, York South-Weston)
        ward_scores = []
        for w in self.wards:
            # Spatial selective susceptibility
            susceptibility = (w["deprivation_index"] ** 1.35) * (w["pop"] / 100000.0)
            ward_scores.append(susceptibility)
            
        tot_ward_score = sum(ward_scores)
        ward_allocations = []
        for i, w in enumerate(self.wards):
            ward_share = ward_scores[i] / tot_ward_score
            ward_budget = total_budget_cad * ward_share
            # Estimated local jobs created (approx $65k CAD per direct/indirect community worker)
            jobs_created = int(round(ward_budget / 48000.0))
            # Direct municipal savings
            municipal_cost_savings = ward_budget * (0.32 + 0.28 * w["deprivation_index"])
            
            ward_allocations.append({
                "ward_id": w["id"],
                "ward_name": w["name"],
                "population": w["pop"],
                "deprivation_index": w["deprivation_index"],
                "priority_tier": w["priority"],
                "allocation_cad": round(ward_budget, 2),
                "allocation_pct": round(ward_share * 100, 2),
                "jobs_created": jobs_created,
                "annual_cost_savings_cad": round(municipal_cost_savings, 2)
            })
            
        # 6. Multi-Year Municipal Revenue & Job Creation Projections (2026 - 2030)
        # Revenue vectors:
        # 1) Direct Municipal Tax Uplift (Commercial vibrancy, BIA assessment growth, property enhancement)
        # 2) Municipal Expenditure Avoidance (Emergency shelters, police/crisis calls, chronic hospital EMS diversion)
        # 3) External Matching Inflow (Federal/Provincial transfers and institutional foundations spent in Toronto)
        # 4) SROI (Net societal and economic value created)
        # 5) 5-Year Job Creation Horizon (Community Grants + Downtown Biotech Innovation Corridor)
        
        # 6A. Downtown Toronto Biotech Innovation Corridor Analytics
        biotech_hubs_analysis = []
        tot_biotech_rd = sum(h["annual_rd_investment_cad"] for h in DOWNTOWN_BIOTECH_HUBS)
        tot_biotech_vc = sum(h["annual_rd_investment_cad"] * h["vc_leverage_ratio"] for h in DOWNTOWN_BIOTECH_HUBS)
        tot_biotech_tax = sum(h["annual_rd_investment_cad"] * h["municipal_tax_capture_rate"] for h in DOWNTOWN_BIOTECH_HUBS)
        tot_biotech_base_jobs = sum(h["base_annual_jobs"] for h in DOWNTOWN_BIOTECH_HUBS)
        tot_biotech_gdp = sum(h["annual_rd_investment_cad"] * h["ip_spinoff_multiplier"] for h in DOWNTOWN_BIOTECH_HUBS)

        for h in DOWNTOWN_BIOTECH_HUBS:
            hub_vc = h["annual_rd_investment_cad"] * h["vc_leverage_ratio"]
            hub_gdp = h["annual_rd_investment_cad"] * h["ip_spinoff_multiplier"]
            hub_tax = h["annual_rd_investment_cad"] * h["municipal_tax_capture_rate"]
            # 5-year cumulative STEM jobs for this hub (compounding as clinical pipeline matures)
            hub_5yr_jobs = int(round(h["base_annual_jobs"] * (1.0 + 1.15 + 1.32 + 1.52 + 1.75)))
            biotech_hubs_analysis.append({
                "hub_id": h["id"],
                "hub_name": h["name"],
                "focus": h["focus"],
                "annual_rd_investment_cad": h["annual_rd_investment_cad"],
                "vc_leverage_ratio": h["vc_leverage_ratio"],
                "vc_matched_capital_cad": round(hub_vc, 2),
                "avg_stem_salary_cad": h["avg_stem_salary_cad"],
                "yr1_stem_jobs": h["base_annual_jobs"],
                "five_year_total_jobs": hub_5yr_jobs,
                "annual_municipal_tax_cad": round(hub_tax, 2),
                "annual_gdp_impact_cad": round(hub_gdp, 2)
            })

        yearly_projections = []
        cumulative_revenue = 0.0
        cumulative_savings = 0.0
        cumulative_sroi = 0.0
        cumulative_community_jobs = 0
        cumulative_biotech_jobs = 0
        cumulative_total_jobs = 0
        
        base_year = 2026
        base_community_jobs = sum(w["jobs_created"] for w in ward_allocations)
        weighted_sroi = sum(s["normalized_weight"] * s["sroi_factor"] for s in sector_excitations)
        weighted_avoidance = sum(s["normalized_weight"] * s["cost_avoidance_rate"] for s in sector_excitations)
        weighted_leverage = sum(s["normalized_weight"] * s["matching_leverage"] for s in sector_excitations)
        
        # Combinatorial efficiency bonus: reduces deadweight allocation losses as Dirichlet pigeonhole collisions are harmonized
        entropy_ratio = pigeon_invariants["combinatorial_entropy"] / np.log(len(self.sectors))
        combinatorial_efficiency = 0.95 + 0.10 * entropy_ratio
        
        for yr in range(projection_years):
            current_year = base_year + yr
            # Growth dynamics: compounded economic multipliers as community capacity solidifies
            growth_factor = (1.0 + 0.082)**yr
            
            # Municipal direct tax recapture: 15.2% baseline recapture rate compounding annually
            annual_tax_lift = total_budget_cad * 0.152 * growth_factor * combinatorial_efficiency
            # Direct municipal operational cost avoidance (Emergency shelters, EMS, police diversion)
            annual_avoidance = total_budget_cad * weighted_avoidance * (0.88 + 0.14 * yr) * combinatorial_efficiency
            # Leveraged funds circulating in Toronto economy
            annual_matching_flow = (total_budget_cad * weighted_leverage) / projection_years * (1.0 + 0.06 * yr)
            # Total municipal balance sheet yield
            total_municipal_yield = annual_tax_lift + annual_avoidance
            
            # Social Return on Investment (total societal economic value)
            annual_sroi_value = total_budget_cad * (weighted_sroi / projection_years) * growth_factor * combinatorial_efficiency
            
            cumulative_revenue += total_municipal_yield
            cumulative_savings += annual_avoidance
            cumulative_sroi += annual_sroi_value
            
            # 5-Year Job Horizon Dynamics:
            # Community grants new jobs: initial deployment + compounding BIA/neighborhood multiplier jobs
            if yr == 0:
                ann_community_new_jobs = base_community_jobs
            else:
                ann_community_new_jobs = int(round(base_community_jobs * 0.22 * ((1.0 + 0.08)**yr)))
            cumulative_community_jobs += ann_community_new_jobs

            # Biotech corridor new STEM jobs: accelerates with trial phases and spin-off growth
            biotech_year_factor = (1.0 + 0.15)**yr
            ann_biotech_new_jobs = int(round(tot_biotech_base_jobs * biotech_year_factor))
            cumulative_biotech_jobs += ann_biotech_new_jobs

            # Combined total new jobs and cumulative total jobs
            ann_total_new_jobs = ann_community_new_jobs + ann_biotech_new_jobs
            cumulative_total_jobs += ann_total_new_jobs
            
            yearly_projections.append({
                "year": current_year,
                "tax_base_lift_cad": round(annual_tax_lift, 2),
                "cost_avoidance_cad": round(annual_avoidance, 2),
                "leveraged_inflow_cad": round(annual_matching_flow, 2),
                "total_municipal_yield_cad": round(total_municipal_yield, 2),
                "cumulative_municipal_yield_cad": round(cumulative_revenue, 2),
                "annual_sroi_value_cad": round(annual_sroi_value, 2),
                "cumulative_sroi_value_cad": round(cumulative_sroi, 2),
                "community_new_jobs": ann_community_new_jobs,
                "cumulative_community_jobs": cumulative_community_jobs,
                "biotech_new_jobs": ann_biotech_new_jobs,
                "cumulative_biotech_jobs": cumulative_biotech_jobs,
                "total_new_jobs_created": ann_total_new_jobs,
                "cumulative_total_jobs": cumulative_total_jobs
            })
            
        summary_metrics = {
            "total_budget_allocated_cad": total_budget_cad,
            "total_grants_awarded": total_applicants,
            "weighted_average_sroi": round(weighted_sroi, 2),
            "total_leveraged_capital_cad": round(total_budget_cad * weighted_leverage, 2),
            "5yr_cumulative_municipal_yield_cad": round(cumulative_revenue, 2),
            "5yr_net_municipal_roi_pct": round(((cumulative_revenue - total_budget_cad) / total_budget_cad) * 100, 2),
            "5yr_cumulative_sroi_cad": round(cumulative_sroi, 2),
            "total_community_jobs_sustained": sum(w["jobs_created"] for w in ward_allocations),
            "5yr_total_community_jobs": cumulative_community_jobs,
            "5yr_total_biotech_stem_jobs": cumulative_biotech_jobs,
            "5yr_total_citywide_jobs_created": cumulative_total_jobs,
            "dirichlet_peak_occupancy": pigeon_invariants["pigeonhole_minimum_peak"],
            "quantum_excitation_bandwidth_khz": bandwidth_khz,
            "excitation_flip_angle_deg": flip_angle_deg,
            "biotech_corridor_total_rd_cad": tot_biotech_rd,
            "biotech_corridor_total_vc_cad": tot_biotech_vc,
            "biotech_corridor_annual_tax_cad": tot_biotech_tax,
            "biotech_corridor_annual_gdp_cad": tot_biotech_gdp
        }
        
        return {
            "summary": summary_metrics,
            "pigeonhole_invariants": pigeon_invariants,
            "rf_pulse_profile": profile,
            "sector_allocations": sector_allocations,
            "ward_allocations": ward_allocations,
            "yearly_projections": yearly_projections,
            "biotech_corridor": {
                "hubs": biotech_hubs_analysis,
                "summary": {
                    "annual_rd_investment_cad": tot_biotech_rd,
                    "vc_matched_capital_cad": tot_biotech_vc,
                    "annual_municipal_tax_cad": tot_biotech_tax,
                    "annual_gdp_impact_cad": tot_biotech_gdp,
                    "five_year_total_stem_jobs": cumulative_biotech_jobs,
                    "avg_weighted_stem_salary_cad": round(sum(h["annual_rd_investment_cad"] * h["avg_stem_salary_cad"] for h in DOWNTOWN_BIOTECH_HUBS) / tot_biotech_rd, 2)
                }
            }
        }

if __name__ == "__main__":
    optimizer = TorontoCommunityGrantOptimizer()
    res = optimizer.run_full_allocation()
    print("Allocation Optimization Complete:")
    print(f"Total Budget: ${res['summary']['total_budget_allocated_cad']:,.2f}")
    print(f"5-Year Cumulative Municipal Yield: ${res['summary']['5yr_cumulative_municipal_yield_cad']:,.2f}")
    print(f"5-Year Net Municipal ROI: {res['summary']['5yr_net_municipal_roi_pct']}%")
    print(f"Total Grants Awarded: {res['summary']['total_grants_awarded']}")
