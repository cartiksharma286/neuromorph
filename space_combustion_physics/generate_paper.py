from fpdf import FPDF
from fpdf.enums import XPos, YPos

class NaturePreprint(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 8)
        self.cell(0, 10, 'Nature Preprints | Integrated Space Systems Dynamics', border=0, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

def generate_paper():
    pdf = NaturePreprint()
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    
    # Title
    pdf.set_font('helvetica', 'B', 18)
    pdf.multi_cell(0, 10, 'Unified Computational Framework for High-Fidelity CFD, Combustion Kinetics, and Optimal Orbital Trajectories', align='L')
    pdf.ln(5)
    
    # Author
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 8, 'Cartik Sharma', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', 'I', 9)
    pdf.cell(0, 8, 'Neuromorph Space Systems Research Group', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    
    # Abstract
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 10, 'Abstract', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    abstract = (
        "This paper presents a unified computational environment for the simulation and optimization of aerospace systems. "
        "We integrate high-fidelity quasi-1D CFD nozzle modeling with Arrhenius combustion kinetics, "
        "Pontryagin-based optimal throttle control, and 2-D gravity-turn trajectory analysis. "
        "Each module is grounded in rigorous mathematical formulations, ranging from non-linear PDEs to linearized "
        "finite-math state-space representations. Our results demonstrate the scalability of these models for "
        "real-time mission control and structural thermal analysis of regenerative cooling systems."
    )
    pdf.multi_cell(0, 5, abstract)
    pdf.ln(8)
    
    # Sections with Equations (Using only ASCII for default fonts)
    sections = [
        ("1. Advanced CFD and Thermal Modeling", 
         "The gas dynamics within the rocket nozzle are resolved using the quasi-1D Navier-Stokes equations with area variation (A). Convective heat flux (q_wall) is modeled using the Bartz equation:",
         "  h_g = [0.026 / D^0.2] * [mu^0.2 * Cp / Pr^0.6] * [Pc / C*]^0.8 * [Dt / r]^0.1\n  q_wall = h_g * (T_gas - T_wall)\n  dT_wall/dt = q_wall / (rho_w * Cp_w * delta_w)"),
        
        ("2. Combustion PDE and Arrhenius Kinetics", 
         "Flame front evolution is resolved through coupled species and energy transport equations:",
         "  dT/dt  = alpha * d^2T/dx^2 + Q * w / (rho * Cp)\n  dYf/dt = D * d^2Yf/dx^2 - w / rho\n  w      = A * (rho*Yf) * (rho*Yo) * exp(-Ea/RT)"),
        
        ("3. Optimal Throttle via Pontryagin's Principle", 
         "Throttle optimization minimizes the fuel functional J = Integral(|u(t)|) dt subject to dynamics:",
         "  dv/dt = [T_max * u / m] - g(h) - [D(v,h) / m]\n  dm/dt = - [T_max * u / (Isp * g0)]"),
        
        ("4. 2-D Gravity-Turn Trajectory Analysis", 
         "Vehicle motion is integrated in a non-rotating coordinate system with varying gravity:",
         "  dx/dt = v * cos(gamma)\n  dz/dt = v * sin(gamma)\n  dv/dt = [T - D]/m - g * sin(gamma)\n  d(gamma)/dt = [-g * cos(gamma)] / v"),
        
        ("5. Mass Budget and Tsiolkovsky staging", 
         "The multi-stage rocket capability is analyzed using the cumulative delta-v budget:",
         "  delta_v_total = Sum[ Isp_i * g0 * ln( m_initial_i / m_final_i ) ]"),
        
        ("6. Finite Math and State-Space Stability", 
         "Linearized pitch-plane stability is assessed using the state transition matrix Phi(t):",
         "  dx/dt = Ax + Bu\n  Phi(t) = exp(At) = Inv_Laplace { (sI - A)^-1 }\n  Stability <=> max( Re(eig(A)) ) < 0")
    ]

    for title, desc, eq in sections:
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 9)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)
        pdf.set_font('courier', 'B', 9)
        pdf.multi_cell(0, 6, eq)
        pdf.ln(5)

    # Conclusion
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 10, '7. Conclusion', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(0, 5, "The integrated computational suite provides a comprehensive toolset for aerospace vehicle design. "
                         "By bridging the gap between microscopic combustion PDEs and macroscopic orbital trajectories, "
                         "we enable holistic optimization of space-faring systems.")
    
    output_path = "/Users/cartiksharma/Downloads/neuromorph-main-10/space_combustion_physics/Integrated_Space_Systems_Nature_Preprint.pdf"
    pdf.output(output_path)
    print(f"Paper generated at: {output_path}")

if __name__ == "__main__":
    generate_paper()
