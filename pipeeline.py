import schemdraw
from schemdraw import flow
import matplotlib.pyplot as plt

def draw_refinery_schematic():
    # Set global drawing parameters for a blueprint look
    schemdraw.theme('default')
    
    with schemdraw.Drawing(file='refinery_flow.png', show=False) as d:
        d.config(fontsize=10, unit=1)
        
        # --- Main Incoming Line ---
        # Start: Crude from Pipeline
        start = d.add(flow.Start(w=2, h=1.5, label='Incoming Crude\nfrom 
AB Pipeline'))
        d.add(flow.Line(l=2).right().at(start.E).add_label('Crude Feed', 
loc='top'))
        
        # --- Desalter Unit ---
        desalter = d.add(flow.Box(w=3, h=2, label='Desalter Unit\n(Remove 
salts/water)'))
        d.add(flow.Line(l=2).right().at(desalter.E))
        
        # --- Furnace ---
        furnace = d.add(flow.Box(w=2.5, h=2.5, label='Atmospheric\nFurnace 
Heater').fill('#FFCCCB'))
        d.add(flow.Line(l=2).right().at(furnace.E).add_label('Heated 
Crude\n(350°C+)', loc='top'))

        # --- Distillation Column ---
        # A tall box representing the fractionation tower
        column = d.add(flow.Box(w=3, h=8, 
label='Atmospheric\nDistillation\nColumn').fill('#E0E0E0'))

        # --- Products leaving the column at different heights ---

        # Top: Gases / Naphtha
        d.add(flow.Line(l=3).right().at(column.E[1]).add_label('LPG / 
Naphtha\n(Gasoline Base)', loc='top'))
        d.add(flow.Box(w=2, h=1, label='Stabilizer').anchor('W'))
        
        # Upper Middle: Jet Fuel / Kerosene
        
d.add(flow.Line(l=4).right().at(column.E[3]).add_label('Kerosene\n(Jet 
Fuel)', loc='top'))
        d.add(flow.Tank(w=1.5, h=1.5, label='Jet Storage').anchor('W'))

        # Middle: Diesel
        d.add(flow.Line(l=4).right().at(column.E[5]).add_label('Light Gas 
Oil\n(Diesel)', loc='top'))
        d.add(flow.Tank(w=1.5, h=1.5, label='Diesel Storage').anchor('W'))
        
        # Bottom: Heavy oils / Residuum
        
d.add(flow.Line(l=3).right().at(column.E[7]).add_label('Atmospheric 
Residue\n(To Vacuum Unit/Asphalt)', loc='top'))
        d.add(flow.Box(w=2, h=1, label='Further 
Processing').anchor('W').fill('#A9A9A9'))

    # Display the generated image
    # Note: In a Jupyter notebook, d.draw() works better than plt.show() 
for schemdraw
    d.draw()

if __name__ == "__main__":
    draw_refinery_schematic()
