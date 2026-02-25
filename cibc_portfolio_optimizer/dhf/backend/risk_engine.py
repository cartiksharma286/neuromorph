import numpy as np
from scipy.stats import norm

class RiskEngine:
    def __init__(self):
        # 5x5 Matrix Definitions
        self.severity_map = {
            'Negligible': 1, 'Minor': 2, 'Serious': 3, 'Critical': 4, 'Catastrophic': 5
        }
        self.prob_map = {
            'Improbable': 1, 'Remote': 2, 'Occasional': 3, 'Probable': 4, 'Frequent': 5
        }

    def calculate_risk_score(self, severity: str, probability: str) -> int:
        s = self.severity_map.get(severity, 0)
        p = self.prob_map.get(probability, 0)
        return s * p

    def get_risk_level(self, score: int) -> str:
        if score <= 4:
            return "Acceptable"
        elif score <= 12:
            return "ALARP"  # As Low As Reasonably Practicable
        else:
            return "Unacceptable"

class StatisticalClassifier:
    """
    Analyzes component performance data to predict risk probability based on spread/variance.
    """
    def __init__(self):
        pass

    def classify_component_risk(self, data_points: list, limit_upper: float, limit_lower: float = 0):
        """
        Calculates probability of exceeding limits using Normal Distribution assumption.
        """
        arr = np.array(data_points)
        mean = np.mean(arr)
        std_dev = np.std(arr)
        
        # Z-scores
        z_upper = (limit_upper - mean) / std_dev if std_dev > 0 else 0
        z_lower = (limit_lower - mean) / std_dev if std_dev > 0 else 0
        
        # Prob of being OUTSIDE limits
        prob_fail_upper = 1 - norm.cdf(z_upper)
        prob_fail_lower = norm.cdf(z_lower)
        total_fail_prob = prob_fail_upper + prob_fail_lower
        
        # Map probability to Descriptive Categories
        if total_fail_prob < 0.0001: return "Improbable"
        if total_fail_prob < 0.001: return "Remote"
        if total_fail_prob < 0.01: return "Occasional"
        if total_fail_prob < 0.1: return "Probable"
        return "Frequent"

# Example usage for Stent Fatigue
# fatigue_cycles = [400e6, 410e6, 390e6, 405e6]
# limit = 380e6 # Minimum requirement
# classifier = StatisticalClassifier()
# risk = classifier.classify_component_risk(fatigue_cycles, limit_lower=380e6, limit_upper=float('inf'))
