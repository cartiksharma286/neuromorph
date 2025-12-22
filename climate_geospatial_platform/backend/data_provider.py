import numpy as np
import random

class DataProvider:
    def __init__(self):
        self.parks = {
            "Banff": {"lat": 51.1784, "lon": -115.5708, "biome": "Coniferous"},
            "Jasper": {"lat": 52.8737, "lon": -118.0814, "biome": "Coniferous"},
            "Rideau": {"lat": 45.4215, "lon": -75.6972, "biome": "Deciduous"},
            "CapeBreton": {"lat": 46.8138, "lon": -60.6698, "biome": "Boreal"},
            "PacificRim": {"lat": 49.0307, "lon": -125.7126, "biome": "Rainforest"}
        }

    def generate_lidar_scan(self, location, size=50):
        """
        Generates a grid of forest data.
        Returns a structured dictionary or list of points.
        """
        grid_size = size
        # Generate 2D grid of features
        # Features: Elevation, Vegetation Density, Moisture, Wind Exposure
        
        # Coherent noise driven generation (simplified with random for now, or numpy)
        elevation = np.random.normal(500, 150, (grid_size, grid_size))
        density = np.random.uniform(0.3, 0.9, (grid_size, grid_size))
        moisture = np.random.uniform(0.1, 0.8, (grid_size, grid_size))
        
        # Biome adjustments
        biome = self.parks.get(location, {}).get('biome', 'Mixed')
        if biome == 'Rainforest':
            moisture += 0.3
            density += 0.2
        elif biome == 'Boreal':
            moisture -= 0.1
        
        # Clip to valid ranges
        moisture = np.clip(moisture, 0, 1)
        density = np.clip(density, 0, 1)
        
        points = []
        for x in range(grid_size):
            for y in range(grid_size):
                points.append({
                    "x": x, "y": y,
                    "elevation": float(elevation[x, y]),
                    "density": float(density[x, y]),
                    "moisture": float(moisture[x, y]),
                    "lat": 0, "lon": 0 # Placeholder if needed
                })
        
        return {
            "grid_metrics": {
                "elevation": elevation.tolist(),
                "density": density.tolist(),
                "moisture": moisture.tolist()
            },
            "points": points,
            "location": location
        }

    def generate_training_data(self, n_samples=1000):
        """
        Generates synthetic data for training the Fire Risk Classifier.
        Target: 0 (No Fire), 1 (Fire)
        """
        # Features: Temp, Humidity, WindSpeed, Biomass
        X = np.random.rand(n_samples, 4)
        # Scale features:
        # Temp: 10-40C -> X[:,0] * 30 + 10
        # Humidity: 10-90% -> X[:,1] * 80 + 10
        # Wind: 0-50kmh -> X[:,2] * 50
        # Biomass: 0-1 -> X[:,3]
        
        temps = X[:, 0] * 30 + 10
        humidity = X[:, 1] * 80 + 10
        wind = X[:, 2] * 50
        biomass = X[:, 3]
        
        # Risk Function (Ground Truth for synthetic labels)
        # Fire likely if High Temp, Low Humidity, High Wind, High Biomass
        risk_score = (temps / 40.0) * 0.4 + \
                     ((100 - humidity) / 100.0) * 0.3 + \
                     (wind / 50.0) * 0.2 + \
                     biomass * 0.1
        
        noise = np.random.normal(0, 0.05, n_samples)
        risk_score += noise
        
        y = (risk_score > 0.6).astype(int)
        
        return X.tolist(), y.tolist()

data_provider = DataProvider()
