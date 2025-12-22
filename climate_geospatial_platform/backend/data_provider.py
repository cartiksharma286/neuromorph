import numpy as np
import random

class DataProvider:
    def __init__(self):
        self.parks = {
            "Banff": {"lat": 51.1784, "lon": -115.5708, "biome": "Coniferous", "radius_km": 50, "restricted": False},
            "Jasper": {"lat": 52.8737, "lon": -118.0814, "biome": "Coniferous", "radius_km": 60, "restricted": False},
            "Rideau": {"lat": 45.4215, "lon": -75.6972, "biome": "Deciduous", "radius_km": 20, "restricted": False},
            "CapeBreton": {"lat": 46.8138, "lon": -60.6698, "biome": "Boreal", "radius_km": 40, "restricted": False},
            "PacificRim": {"lat": 49.0307, "lon": -125.7126, "biome": "Rainforest", "radius_km": 30, "restricted": True} # Protected Zone
        }

    def check_geofence(self, location):
        """
        Verifies if access to the location is permitted within the current geofence context.
        """
        park = self.parks.get(location)
        if not park:
            return {"allowed": False, "reason": "Unknown Location"}
        
        if park.get("restricted", False):
            # In a real app, check user credentials here.
            # For simulation, we allow but flag it.
            return {
                "allowed": True, 
                "warning": "Entering Protected Geo-Zone (Pacific Rim Type-1)",
                "geofence_id": f"GF-{location.upper()}-884"
            }
            
        return {"allowed": True, "geofence_id": "PUBLIC-ACCESS"}

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
        
        geofence_status = self.check_geofence(location)
        
        return {
            "grid_metrics": {
                "elevation": elevation.tolist(),
                "density": density.tolist(),
                "moisture": moisture.tolist()
            },
            "points": points,
            "location": location,
            "geofence": geofence_status
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
