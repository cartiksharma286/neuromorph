from fpdf import FPDF
from fpdf.enums import XPos, YPos

class NaturePreprint(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 8)
        self.cell(0, 10, 'Nature Preprints | Mathematical Foundations of Aerospace Dynamics', border=0, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

def generate_paper():
    pdf = NaturePreprint()
    pdf.set_margins(20, 25, 20)
    pdf.add_page()
    
    # Title
    pdf.set_font('helvetica', 'B', 16)
    pdf.multi_cell(170, 10, 'Finite Mathematical Derivations and Numerical Stability in Multi-Physics Aerospace Simulations', align='L')
    pdf.ln(5)
    
    # Author
    pdf.set_font('helvetica', '', 10)
    pdf.cell(170, 6, 'Cartik Sharma', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', 'I', 8)
    pdf.cell(170, 6, 'Neuromorph Space Systems, Advanced Computation Division', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    
    # Abstract
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(170, 8, 'Abstract', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    abstract = (
        "This report provides the rigorous finite mathematical derivations supporting the Space Combustion Physics platform. "
        "We detail the discretization of the Navier-Stokes equations for quasi-1D flow, the linearization of pitch-plane "
        "dynamics for state-space control, and the implementation of Runge-Kutta numerical integration for high-performance "
        "trajectory modeling. These derivations ensure the numerical stability and physical accuracy of the integrated "
        "aerospace mission control environment."
    )
    pdf.multi_cell(170, 5, abstract)
    pdf.ln(8)
    
    # Derivations Section
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(170, 8, '1. Finite Difference Discretization of Combustion PDEs', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "To solve the energy and species transport equations, we apply a central difference scheme. The Laplacian operator for temperature T at node i is approximated as:")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  d2T/dx2 |i approx [ T(i+1) - 2T(i) + T(i-1) ] / (dx^2)")
    pdf.ln(2)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "Substituting this into the heat equation with Arrhenius source terms (w):")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  dT/dt = alpha * [ (T_i+1 - 2T_i + T_i-1)/dx^2 ] + Q*w/rho*Cp")
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(170, 8, '2. Linearized State-Space Derivation', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "The non-linear rocket dynamics f(x,u) are linearized using a first-order Taylor expansion about an equilibrium state (x_ref, u_ref):")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  delta_dot_x = A * delta_x + B * delta_u\n  A_ij = df_i / dx_j | (x_ref, u_ref)")
    pdf.ln(2)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "For pitch dynamics, this leads to the state vector [theta, q, v, alpha]T and the matrix A:")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  A = [ [0, 1, 0, 0], [M_alpha, M_q, M_v, 0], [0, 0, Dv, D_alpha], [0, 0, 0, 0] ]")
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(170, 8, '3. Numerical Stability in CFD (Lax-Friedrichs)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "For the hyperbolic conservation laws in nozzle flow, we use the Lax-Friedrichs scheme to maintain stability at the sonic point (Ma=1):")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  U_i^{n+1} = 0.5 * (U_{i+1}^n + U_{i-1}^n) - (dt/2dx) * (F_{i+1}^n - F_{i-1}^n) + dt*S_i")
    pdf.ln(2)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "Stability is maintained if the Courant-Friedrichs-Lewy (CFL) condition is met:")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  CFL = (u + c) * dt / dx <= 1.0")
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(170, 8, '4. High-Performance Integration (RK45)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "Trajectory integration uses the 4th-order Runge-Kutta method with error estimation (Dormand-Prince):")
    pdf.ln(2)
    pdf.set_font('courier', 'B', 9)
    pdf.multi_cell(170, 6, "  k1 = f(t_n, y_n)\n  k2 = f(t_n + 0.5h, y_n + 0.5h*k1)\n  y_{n+1} = y_n + (h/6)*(k1 + 2k2 + 2k3 + k4)")
    pdf.ln(8)

    # Conclusion
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(170, 10, 'Conclusion', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 9)
    pdf.multi_cell(170, 5, "These derivations provide the mathematical infrastructure for the mission control dashboard. By implementing these finite math constructs, we achieve a balance between computational efficiency and high-fidelity physical realism.")
    
    output_path = "/Users/cartiksharma/Downloads/neuromorph-main-10/space_combustion_physics/Space_Combustion_Math_Derivations_Nature_Preprint.pdf"
    pdf.output(output_path)
    print(f"Paper generated at: {output_path}")

if __name__ == "__main__":
    generate_paper()
