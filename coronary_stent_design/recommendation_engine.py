class StentRecommender:
    def __init__(self, diameter, strut_thickness):
        self.diameter = diameter
        self.strut_thickness = strut_thickness
        self.recommendations = []
        self.design_hints = []

    def analyze(self):
        """Analyzes parameters and generates recommendations."""
        # Chamfer Recommendations
        if self.strut_thickness > 0.08:
            self.recommendations.append(
                f"Strut thickness is {self.strut_thickness}mm. "
                "Recommend applying a 0.01mm chamfer on flow-leading edges to reduce turbulence."
            )
        else:
            self.recommendations.append("Strut thickness is low. Standard electropolishing should suffice.")

        # Design Hints
        if self.diameter < 2.5:
            self.design_hints.append(
                "Small vessel diameter detected. Ensure high radial strength but maximize flexibility."
            )
        elif self.diameter > 4.0:
            self.design_hints.append(
                "Large vessel diameter. Ensure adequate scaffold coverage to prevent recoil."
            )
        
        self.design_hints.append("Consider drug-eluting coating (e.g., Sirolimus/Paclitaxel) for anti-restenosis.")
        self.design_hints.append("Verify fatigue life for at least 400 million cycles (approx 10 years).")

    def get_report(self):
        return {
            "recommendations": self.recommendations,
            "design_hints": self.design_hints
        }
