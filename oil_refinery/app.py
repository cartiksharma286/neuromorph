
from flask import Flask, render_template, jsonify
import sys
import os
import matplotlib
matplotlib.use('Agg')

# Ensure we can import the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from refinery_design import RefineryDesigner
from cfd_simulation import CFDSolver
from pipeline_network import PipelineManager
from trading_engine import OilTradingDesk

app = Flask(__name__)

# Initialize components
cfd_solver = CFDSolver()
pipeline_manager = PipelineManager()
trading_desk = OilTradingDesk()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/drawings/heat_exchanger')
def draw_heat_exchanger():
    svg_content = RefineryDesigner.draw_heat_exchanger()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/reactor')
def draw_reactor():
    svg_content = RefineryDesigner.draw_refinery_reactor()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/valve')
def draw_valve():
    svg_content = RefineryDesigner.draw_control_valve()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/plant_isometric')
def draw_plant_isometric():
    svg_content = RefineryDesigner.draw_isometric_plant()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/plant_orthographic')
def draw_plant_orthographic():
    svg_content = RefineryDesigner.draw_orthographic_plant()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/pipeline_assembly')
def draw_pipeline_assembly():
    svg_content = RefineryDesigner.draw_pipeline_assembly()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/master_plan')
def draw_master_plan():
    svg_content = RefineryDesigner.draw_master_plan_layout()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/chemical_plant')
def draw_chemical_plant():
    svg_content = RefineryDesigner.draw_chemical_plant_schematic()
    return jsonify({'svg': svg_content})

@app.route('/api/drawings/abbc_systematic')
def draw_abbc_systematic():
    svg_content = RefineryDesigner.draw_abbc_pipeline_systematic()
    return jsonify({'svg': svg_content})

@app.route('/api/simulation/cfd')
def simulation_cfd():
    # Helper to generate a new frame
    img_base64 = cfd_solver.generate_flow_plot()
    return jsonify({'image': img_base64})

@app.route('/api/network/jamnagar')
def network_jamnagar():
    img_base64 = pipeline_manager.visualize_network("jamnagar")
    return jsonify({'image': img_base64})

@app.route('/api/network/abbc')
def network_abbc():
    img_base64 = pipeline_manager.visualize_network("abbc")
    return jsonify({'image': img_base64})

@app.route('/api/trading/market')
def trading_market():
    data = trading_desk.generate_market_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
