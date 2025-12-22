import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import time

class ClimateEngine:
    def __init__(self):
        self.classifier = LogisticRegression()
        self.is_trained = False
        self.latest_metrics = {}

    def train_classifier(self, X, y):
        """
        Trains a statistical classifier (Logistic Regression) for fire prediction.
        """
        # Split data (simple 80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        duration = time.time() - start_time
        
        preds = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        self.is_trained = True
        self.latest_metrics = {
            "accuracy": acc,
            "training_time": duration,
            "model_type": "Logistic Regression (Statistical)",
            "coefficients": self.classifier.coef_.tolist()[0]
        }
        return self.latest_metrics

    def predict_risk(self, features):
        if not self.is_trained:
            return 0.0
        # features: [temp, humidity, wind, biomass] (raw 0-1 scales from frontend inputs)
        # Need to scale them to match training data distribution roughly
        # Our training gen: Temp(10-40), Hum(10-90), Wind(0-50), Bio(0-1)
        # Input features assumed to be passed in correct units from UI or normalized.
        # Let's assume UI passes normalized 0-1 and we scale up.
        
        scaled = np.array([
            features[0] * 30 + 10,
            features[1] * 80 + 10,
            features[2] * 50,
            features[3]
        ]).reshape(1, -1)
        
        prob = self.classifier.predict_proba(scaled)[0][1]
        return float(prob)

    def optimize_fire_spread(self, grid_state):
        """
        Optimized Forest Fire Cellular Automata.
        Uses NumPy vectorization for high performance spread simulation.
        
        grid_state: dict with 'density', 'moisture', 'fire_state' (0=none, 1=burning, 2=burnt)
        matrices (lists of lists).
        """
        density = np.array(grid_state['density'])
        moisture = np.array(grid_state['moisture'])
        fire = np.array(grid_state['fire_state'])
        
        rows, cols = fire.shape
        
        # IGNITE: Find burning cells
        burning_mask = (fire == 1)
        
        if not np.any(burning_mask):
            return grid_state # Nothing happening
        
        # SPREAD PROBABILITY
        # Prob = Density * (1 - Moisture) * Constant
        spread_prob = density * (1.0 - moisture) * 0.8
        
        # Neighbor check using shifting (Vectorized Neighbor Access)
        # Pad fire grid to handle boundaries
        padded_fire = np.pad(fire, 1, mode='constant', constant_values=0)
        
        # Neighbors: N, S, E, W
        n = padded_fire[:-2, 1:-1] == 1
        s = padded_fire[2:, 1:-1] == 1
        e = padded_fire[1:-1, 2:] == 1
        w = padded_fire[1:-1, :-2] == 1
        
        # If any neighbor is burning, cell is exposed
        exposed = (n | s | e | w) & (fire == 0)
        
        # Stochastic ignition
        ignition_roll = np.random.random((rows, cols))
        new_ignitions = exposed & (ignition_roll < spread_prob)
        
        # UPDATE STATE
        # Old burning becomes Burnt (2)
        # New ignitions become Burning (1)
        
        next_fire = fire.copy()
        next_fire[burning_mask] = 2 # Burn out
        next_fire[new_ignitions] = 1 # Ignite
        
        return {
            "density": density.tolist(),
            "moisture": moisture.tolist(),
            "fire_state": next_fire.tolist(),
            "nt_ignitions": int(np.sum(new_ignitions)),
            "nt_burnt": int(np.sum(next_fire == 2))
        }

engine = ClimateEngine()
