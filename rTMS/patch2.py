import re

with open("logic/rtms_engine.py", "r") as f:
    data = f.read()

new_bem = """def _dementia_bem_simulation(n_layers=4, resolution=30, include_dbs=True):
    \"\"\"
    Boundary Element Method simulation for dementia-specific head model.
    Models concentric tissue boundaries (scalp, skull, CSF, grey matter)
    and computes rTMS-induced electric field attenuation at each layer.
    Also models Deep Brain Stimulation (DBS) outwards field from the newly added
    Boston Scientific Vercise Genus device.
    \"\"\"
    conductivities = {
        "Scalp": 0.33,
        "Skull": 0.0042,
        "CSF": 1.79,
        "Grey Matter": 0.33,
        "Deep Brain (DBS Target)": 0.45
    }
    radii = [1.0, 0.92, 0.87, 0.80, 0.25]
    
    layers = []
    for idx, (name, sigma) in enumerate(conductivities.items()):
        r = radii[idx]
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution // 2)
        TH, PH = np.meshgrid(theta, phi)

        if name == "Grey Matter":
            R_layer = r + 0.04 * np.sin(6 * TH) * np.cos(5 * PH) - 0.02
        elif name == "Deep Brain (DBS Target)":
            R_layer = r + 0.01 * np.sin(2 * TH) * np.cos(2 * PH)
        else:
            R_layer = r + 0.02 * np.sin(3 * TH) * np.cos(3 * PH)

        X = R_layer * np.sin(PH) * np.cos(TH)
        Y = R_layer * np.sin(PH) * np.sin(TH)
        Z = R_layer * np.cos(PH)

        # Baseline rTMS incoming potential
        potential_rtms = sigma * np.exp(-Z**2 / 0.5) * (1 + 0.3 * np.sin(4 * TH))
        
        # DBS out-radiating potential from center (z=0, x=0, y=0)
        # The Boston Scientific Vercise Genus (MICC) delivers highly directional currents
        dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
        dbs_potential = 0.0
        if include_dbs:
            # High intensity near center (directional field via MICC tuning)
            dbs_potential = 1.5 * np.exp(-dist_from_center / 0.3) * (1 + 0.5 * np.cos(PH))

        potential = potential_rtms + dbs_potential

        layers.append({
            "name": name,
            "conductivity": sigma,
            "radius": r,
            "x": X.tolist(),
            "y": Y.tolist(),
            "z": Z.tolist(),
            "potential": potential.tolist()
        })

    depths = np.linspace(0, 1, 50)
    attenuation = []
    for d in depths:
        v = 100.0
        for idx, (name, sigma) in enumerate(conductivities.items()):
            boundary = radii[idx]
            if d >= (1 - boundary):
                v *= np.exp(-sigma * (d - (1 - boundary)) * 5)
        # Add DBS amplification deep
        if include_dbs and d > 0.6:
            v += 45.0 * np.exp(-(d - 0.75)**2 / 0.05)
            
        attenuation.append(round(float(v), 2))

    return {
        "layers": layers,
        "attenuation": {
            "depths": [round(float(d), 3) for d in depths],
            "field_pct": attenuation
        }
    }"""

data = re.sub(
    r"def _dementia_bem_simulation.*?return \{\s*\"layers\": layers,\s*\"attenuation\": \{\s*\"depths\": \[.*?field_pct\": attenuation\s*\}\s*\}",
    new_bem,
    data,
    flags=re.DOTALL
)

with open("logic/rtms_engine.py", "w") as f:
    f.write(data)
