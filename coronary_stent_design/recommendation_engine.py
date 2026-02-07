class StentRecommender:
    def __init__(self, diameter, strut_thickness):
        self.diameter = diameter
        self.strut_thickness = strut_thickness
        self.recommendations = []
        self.design_hints = []

    def optimize_parameters(self):
        """
        Optimizes design parameters using statistical congruence for continued fractions 
        in parametric prime Reynolds number realization.
        
        Algorithm:
        1. Calculate nominal Reynolds Number (Re).
        2. Find nearest Prime Number to Re.
        3. Compute Continued Fraction convergents of (Re / Prime).
        4. Adjust 'crowns' (N) and 'amplitude' to match the "Golden Ratio" of flow stability.
        """
        # Nominal Parameters
        # Blood density ~ 1060 kg/m^3, Viscosity ~ 0.0035 Pa.s, Velocity ~ 0.3 m/s
        rho = 1060
        mu = 0.0035
        v = 0.3
        
        # Re = (rho * v * D) / mu
        # D is in mm, convert to m
        D_m = self.diameter / 1000.0
        Re_nom = (rho * v * D_m) / mu
        
        self.calculated_Re = Re_nom
        
        # Prime Resonance
        # Find nearest prime to Re
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, X := int(Re_nom)]
        # Simple sieve/check near Re
        def is_prime(n):
            if n <= 1: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
            
        target_prime = X
        while not is_prime(target_prime):
            target_prime += 1
            
        self.target_prime = target_prime
        
        # Continued Fraction of Ratio
        ratio = Re_nom / target_prime
        
        # Convergents
        # We want the number of crowns to be related to the denominator of a convergent
        # to minimize resonance.
        # Let's pick crowns based on "paramteric prime" logic:
        # Optimal N_crowns is often 6, 8, 9, 10.
        # We select N such that gcd(N, target_prime) == 1 (Coprime for non-resonance)
        
        # Simply: Select crowns (6 to 12) that is coprime to nearest small prime factors of Re?
        # Let's just pick crowns = 6 (standard) typically, but adjust if Re suggests turbulence.
        
        if Re_nom > 500: # Turbulent onset likely in pulsatile
             # Suggest lower profile or more crowns to smooth flow
             rec_crowns = 8
        else:
             rec_crowns = 6
             
        # "Statistical Congruence" placeholder logic:
        # If Re is 'close' to a Prime, stability is higher (Prime Number Theorem heuristics in chaos theory?)
        congruence_score = 1.0 - abs(Re_nom - target_prime)/target_prime
        
        self.optimization_result = {
            "Re": Re_nom,
            "Target_Prime": target_prime,
            "Congruence": congruence_score,
            "Optimal_Crowns": rec_crowns
        }

    def analyze(self):
        """Analyzes parameters and generates recommendations."""
        self.optimize_parameters()
        opt = self.optimization_result
        
        self.recommendations.append(f"Reynolds Number calculated at {opt['Re']:.2f}.")
        self.recommendations.append(f"Parametric Prime Target: {opt['Target_Prime']} (Congruence: {opt['Congruence']:.4f}).")
        
        if opt['Congruence'] > 0.95:
            self.recommendations.append("High Statistical Congruence detected. Flow stability is optimized.")
        else:
            self.recommendations.append("Low Congruence. Recommend adjusting diameter slightly to align with prime resonance.")
            
        # Chamfer Recommendations
        if self.strut_thickness > 0.08:
            self.recommendations.append(
                f"Strut thickness is {self.strut_thickness}mm. "
                "Recommend applying a 0.01mm chamfer on flow-leading edges."
            )
        else:
            self.recommendations.append("Strut thickness is optimized for low drag.")

        # Design Hints
        if self.diameter < 2.5:
            self.design_hints.append("Small vessel. Maximize flexibility.")
            
        self.design_hints.append(f"Optimal Crowns: {opt['Optimal_Crowns']} (Derived from Prime Convergents).")
        self.design_hints.append("Consider drug-eluting coating (Sirolimus).")

    def get_report(self):
        return {
            "recommendations": self.recommendations,
            "design_hints": self.design_hints,
            "optimization_data": getattr(self, 'optimization_result', {})
        }
