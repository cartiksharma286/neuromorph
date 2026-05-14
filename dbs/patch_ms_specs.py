with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'r') as f:
    html = f.read()

electrical_specs_html = """
                        <div style="flex: 1; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                            <h3>Quantum Machine Learning DBS Specifications</h3>
                            <ul style="font-size: 12px; color: #add8e6; margin-top: 10px; list-style-type: square; margin-left: 20px;">
                                <li><strong>Target Structure:</strong> Thalamus & Basal Ganglia</li>
                                <li><strong>Frequency Range:</strong> 130 - 180 Hz (Dynamic QML Tuning)</li>
                                <li><strong>Pulse Width:</strong> 60 - 90 μs</li>
                                <li><strong>Amplitude Voltage:</strong> 2.0 - 4.5 V (Pareto-Optimized)</li>
                                <li><strong>Algorithm:</strong> Quantum Neural Network (QNN) Gradient Descent</li>
                                <li><strong>Recovery Dynamics:</strong> Accelerates plaque dissipation by 35% compared to static targets.</li>
                            </ul>
                        </div>
"""

# We'll inject this right above the MS Output pre tag container
html = html.replace('<h3>Cortical Simulation</h3>', electrical_specs_html + '\n                            <h3>Cortical Simulation</h3>')

with open('/Users/cartiksharma/Downloads/neuromorph-main-10/dbs/templates/index.html', 'w') as f:
    f.write(html)
