import schemdraw
import schemdraw.elements as elm

def draw_eeg_circuit():
    schemdraw.theme('default')
    with schemdraw.Drawing(file='eeg_schematic.png', show=False) as d:
        # 1. Input Node (Scalp electrode potential)
        d += elm.Dot().label('Scalp\nElectrode', 'left')
        
        # Line to the matching node
        d += elm.Line().right().length(1.0)
        
        # 2. Impedance Matching Section (Parallel RC to ground)
        d += elm.Dot()
        d.push()
        d += elm.Resistor().down().label('R_match\n5.0 kΩ')
        d += elm.Capacitor().down().label('C_match\n14.0 pF')
        d += elm.Ground()
        d.pop()
        
        # Continue to filters
        d += elm.Line().right().length(1.5)
        
        # 3. Active High-pass Filter (Series C, shunt R to ground)
        d += elm.Capacitor().right().label('C_highpass\n0.1 μF')
        d += elm.Dot()
        d.push()
        d += elm.Resistor().down().label('R_highpass\n3.18 MΩ')
        d += elm.Ground()
        d.pop()
        
        # Continue to Low-pass Filter
        d += elm.Line().right().length(1.5)
        
        # 4. Active Low-pass Filter (Series R, shunt C to ground)
        d += elm.Resistor().right().label('R_lowpass\n10.0 kΩ')
        d += elm.Dot()
        d.push()
        d += elm.Capacitor().down().label('C_lowpass\n35.4 nF')
        d += elm.Ground()
        d.pop()
        
        # Continue to Op-Amp
        d += elm.Line().right().length(1.5)
        
        # 5. Instrumentation Pre-amplifier stage (AD8221)
        d += elm.Opamp().right().label('AD8221\nPre-Amp\nGain=150', 'center')
        
        # Output
        d += elm.Line().right().length(1.0).label('Denoised\nOutput', 'right')
        
    print("EEG active biosensing circuit schematic successfully drawn and saved as eeg_schematic.png!")

if __name__ == '__main__':
    draw_eeg_circuit()
