# Ontario Wildfire QML App

A Flask-based wildfire management simulator for Ontario and Northern Ontario with three integrated views:

- Regional command forecasting for active burn area, suppression efficiency, AQI, and quantum alignment.
- Wildfire containment strategies with ecological land restoration, convergence characteristics, and 24-month summer mitigation plots.
- Smoke propagation to the US East Coast with AQI safeguard metrics, plume half-life analytics, and city-by-city PM2.5 exposure.

## Core modeling themes

- Statistical quantum machine learning alignment kernel for command and dissipation control.
- Seasonal summer wildfire pressure over a 24-month horizon starting in May 2026.
- Ecological restoration and land recovery tied to containment strategy and prescribed burn capacity.
- Air quality protection through smoke transport and dissipation tracking across East Coast cities.

## Files

- `app.py` - Flask backend and simulation model.
- `templates/index.html` - Tabbed dashboard UI with Plotly charts.
- `requirements.txt` - Python dependencies for the simulator.

## Run locally

From this folder:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
PORT=9100 ./.venv/bin/python app.py
```

Then open `http://127.0.0.1:9100/`.

If you want to reuse the repository root environment instead, install the same dependencies there and run the app with that interpreter.
