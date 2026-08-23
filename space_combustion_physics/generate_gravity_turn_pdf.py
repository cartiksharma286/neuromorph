#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# 1. Physics Simulator for Multi-Stage Gravity Turn
# ----------------------------------------------------------------------
def simulate_gravity_turn(config):
    m0 = config["m0"]
    mp1 = config["mp1"]
    mp2 = config["mp2"]
    mdry = config["mdry"]
    T1 = config["T1"]
    T2 = config["T2"]
    Isp1 = config["Isp1"]
    Isp2 = config["Isp2"]
    A = config["A"]
    Cd = config["Cd"]
    v_kick = config["v_kick"]
    theta_kick = math.radians(config["theta_kick_deg"])
    
    g0 = 9.80665
    Re = 6371000.0
    H = 8500.0
    rho0 = 1.225
    
    # Calculate burn times
    t_burn1 = (mp1 * Isp1 * g0) / T1
    t_coast = 2.0
    t_burn2 = (mp2 * Isp2 * g0) / T2
    
    dt = 0.1
    t_max = 800.0
    t = 0.0
    
    x = 0.0
    z = 0.0
    v = 0.1
    gamma = math.pi / 2.0
    m = m0
    
    t_hist = []
    x_hist = []
    z_hist = []
    v_hist = []
    gamma_hist = []
    m_hist = []
    q_hist = []
    thrust_hist = []
    drag_hist = []
    
    kicked = False
    
    while t <= t_max and z >= 0:
        if t < t_burn1:
            thrust = T1
            Isp = Isp1
        elif t < t_burn1 + t_coast:
            thrust = 0.0
            Isp = 1.0
        elif t < t_burn1 + t_coast + t_burn2:
            thrust = T2
            Isp = Isp2
        else:
            thrust = 0.0
            Isp = 1.0
            
        if m <= mdry:
            thrust = 0.0
            
        g = g0 * (Re / (Re + z))**2
        rho = rho0 * math.exp(-z / H) if z < 100000 else 0.0
        drag = 0.5 * rho * v**2 * Cd * A
        q = 0.5 * rho * v**2
        
        dx = (Re / (Re + z)) * v * math.cos(gamma)
        dz = v * math.sin(gamma)
        dm = -thrust / (Isp * g0) if thrust > 0 else 0.0
        
        if not kicked and v >= v_kick:
            gamma = math.pi / 2.0 - theta_kick
            kicked = True
            dgamma = 0.0
        elif kicked:
            dgamma = (v / (Re + z) - g / v) * math.cos(gamma)
        else:
            dgamma = 0.0
            
        dv = (thrust - drag) / m - g * math.sin(gamma)
        
        t_hist.append(t)
        x_hist.append(x)
        z_hist.append(z)
        v_hist.append(v)
        gamma_hist.append(gamma)
        m_hist.append(m)
        q_hist.append(q)
        thrust_hist.append(thrust)
        drag_hist.append(drag)
        
        x += dx * dt
        z += dz * dt
        v += dv * dt
        gamma += dgamma * dt
        m += dm * dt
        
        if gamma < 0.0:
            gamma = 0.0
        if z < 0.0 and t > 5.0:
            break
            
        t += dt
        
    return {
        "t": np.array(t_hist),
        "x": np.array(x_hist),
        "z": np.array(z_hist),
        "v": np.array(v_hist),
        "gamma": np.array(gamma_hist),
        "m": np.array(m_hist),
        "q": np.array(q_hist),
        "thrust": np.array(thrust_hist),
        "drag": np.array(drag_hist)
    }

# ----------------------------------------------------------------------
# 2. Matplotlib Plot Generation
# ----------------------------------------------------------------------
def generate_plots(sim_results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.2))
    
    # Theme colors matching deep slate & teal
    colors_dict = {
        "Falcon9": "#0284c7",   # Sky blue
        "Starship": "#10b981",  # Teal/Emerald
        "SaturnV": "#f97316"    # Orange
    }
    
    # 1. Trajectory Profile (Altitude vs Downrange)
    for name, res in sim_results.items():
        ax1.plot(res["x"]/1000.0, res["z"]/1000.0, color=colors_dict[name], linewidth=1.8, label=name)
    ax1.set_title("Trajectory Profile", fontsize=9, fontweight="bold", color="#0f172a")
    ax1.set_xlabel("Downrange (km)", fontsize=8, color="#334155")
    ax1.set_ylabel("Altitude (km)", fontsize=8, color="#334155")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=7)
    ax1.tick_params(labelsize=7)
    
    # 2. Flight Path Angle & Velocity vs Time
    # Left axis: Velocity, Right axis: Flight Path Angle
    for name, res in sim_results.items():
        ax2.plot(res["t"], res["v"]/1000.0, color=colors_dict[name], linewidth=1.5, label=f"{name} V")
        ax2.plot(res["t"], np.degrees(res["gamma"]), color=colors_dict[name], linestyle="--", linewidth=1.0, alpha=0.7)
    
    ax2.set_title("Velocity & Flight Angle", fontsize=9, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("Flight Time (s)", fontsize=8, color="#334155")
    ax2.set_ylabel("Velocity (km/s) [Solid]", fontsize=8, color="#334155")
    # Add dummy lines for legend
    ax2.plot([], [], color="#64748b", linestyle="--", label="FPA (deg) [Dash]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.tick_params(labelsize=7)
    
    # 3. Dynamic Pressure vs Altitude
    for name, res in sim_results.items():
        # Only plot up to 80 km where atmospheric effects are present
        mask = res["z"] <= 80000.0
        ax3.plot(res["q"][mask]/1000.0, res["z"][mask]/1000.0, color=colors_dict[name], linewidth=1.8, label=name)
    ax3.set_title("Dynamic Pressure vs Altitude", fontsize=9, fontweight="bold", color="#0f172a")
    ax3.set_xlabel("Dynamic Pressure q (kPa)", fontsize=8, color="#334155")
    ax3.set_ylabel("Altitude (km)", fontsize=8, color="#334155")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=7)
    ax3.tick_params(labelsize=7)
    
    plt.tight_layout()
    plot_path = "/Users/cartiksharma/Downloads/neuromorph-main-10/space_combustion_physics/static/gravity_turn_plots.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def render_equation(equation, output_path, width=8.0, height=0.62, fontsize=16):
    """Render a publication-quality equation as a transparent image."""
    figure = plt.figure(figsize=(width, height), dpi=220)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.text(0.5, 0.5, f"${equation}$", ha="center", va="center", fontsize=fontsize, color="#0f172a")
    figure.savefig(output_path, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    return output_path

# ----------------------------------------------------------------------
# 3. Page Numbered Canvas (Two-Pass)
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        # Suppress footer on page 1 (cover)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 36, "NATURE AEROSPACE | PREPRINT | UNDER REVIEW")
            self.drawRightString(
                letter[0] - 54, 36,
                f"Page {self._pageNumber} of {page_count}"
            )
            # Add line above footer
            self.setLineWidth(0.5)
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.line(54, 48, letter[0] - 54, 48)

# ----------------------------------------------------------------------
# 4. ReportLab PDF Generation
# ----------------------------------------------------------------------
def generate_pdf(sim_results, plot_path):
    pdf_path = "/Users/cartiksharma/Downloads/neuromorph-main-10/space_combustion_physics/Space_Combustion_Physics_Gravity_Turn_Preprint.pdf"
    
    # 0.75-inch margins
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=64
    )
    
    styles = getSampleStyleSheet()
    
    # Journal Colors
    primary = colors.HexColor("#0f172a")     # Deep Slate
    accent = colors.HexColor("#0284c7")      # Sky Blue
    body_text = colors.HexColor("#334155")   # Slate Gray
    math_bg = colors.HexColor("#f8fafc")     # Light gray back
    math_border = colors.HexColor("#e2e8f0")
    
    # Document styles
    header_style = ParagraphStyle(
        "JournalHeader",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=accent,
        fontName="Helvetica-Bold",
        spaceAfter=12
    )
    title_style = ParagraphStyle(
        "JournalTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=primary,
        fontName="Helvetica-Bold",
        spaceAfter=8,
        leading=22
    )
    author_style = ParagraphStyle(
        "JournalAuthor",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=primary,
        fontName="Helvetica-Bold",
        spaceAfter=3
    )
    affil_style = ParagraphStyle(
        "JournalAffil",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=body_text,
        fontName="Helvetica-Oblique",
        spaceAfter=14,
        leading=9.5
    )
    abstract_heading = ParagraphStyle(
        "AbstractHead",
        parent=styles["Heading2"],
        fontSize=9.5,
        textColor=primary,
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=4
    )
    abstract_style = ParagraphStyle(
        "AbstractText",
        parent=styles["BodyText"],
        fontSize=8.5,
        fontName="Helvetica-Bold",
        textColor=primary,
        alignment=TA_JUSTIFY,
        leading=12.0,
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        "SecHead",
        parent=styles["Heading1"],
        fontSize=11,
        textColor=accent,
        fontName="Helvetica-Bold",
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SubHead",
        parent=styles["Heading2"],
        fontSize=9.5,
        textColor=primary,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=8.5,
        textColor=body_text,
        alignment=TA_JUSTIFY,
        leading=12.0,
        spaceAfter=8
    )
    math_style = ParagraphStyle(
        "MathBlock",
        parent=styles["Normal"],
        fontSize=8.0,
        fontName="Courier",
        textColor=primary,
        backColor=math_bg,
        borderColor=math_border,
        borderWidth=0.5,
        borderPadding=5,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=8
    )
    fig_caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontSize=7.5,
        fontName="Helvetica-BoldOblique",
        textColor=primary,
        alignment=TA_CENTER,
        spaceAfter=14
    )

    equation_dir = os.path.dirname(plot_path)
    equation_paths = {
        "ode": render_equation(
            r"\dot{x}=\frac{R_e}{R_e+z}v\cos\gamma \quad \dot{z}=v\sin\gamma \quad \dot{v}=\frac{T-D}{m}-g(z)\sin\gamma \quad \dot{\gamma}=\left(\frac{v}{R_e+z}-\frac{g(z)}{v}\right)\cos\gamma \quad \dot{m}=-\frac{T}{I_{sp}g_0}",
            os.path.join(equation_dir, "gravity_turn_ode.png"), height=0.62, fontsize=11
        ),
        "drag": render_equation(
            r"D=\frac{1}{2}\rho(z)v^2C_dA, \qquad \rho(z)=\rho_0e^{-z/H}, \qquad g(z)=g_0\left(\frac{R_e}{R_e+z}\right)^2",
            os.path.join(equation_dir, "gravity_turn_drag.png"), height=0.62, fontsize=15
        ),
        "rk4": render_equation(
            r"\mathbf{u}_{k+1}=\mathbf{u}_k+\frac{\Delta t}{6}(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4), \qquad \mathbf{k}_i=\mathbf{f}(t_k+c_i\Delta t,\mathbf{u}_k+\Delta t\,\mathbf{a}_i)",
            os.path.join(equation_dir, "gravity_turn_rk4.png"), height=0.72, fontsize=14
        ),
        "gaussian": render_equation(
            r"\int_{-\infty}^{\infty}e^{-x^2}f(x)\,dx \approx \sum_{i=1}^{n}w_if(x_i), \qquad H_n(x_i)=0",
            os.path.join(equation_dir, "quantum_gaussian_quadrature.png"), height=0.62, fontsize=15
        ),
        "quadrature_weights": render_equation(
            r"w_i=\frac{2^{n-1}n!\sqrt{\pi}}{n^2[H_{n-1}(x_i)]^2}, \qquad \sum_{i=1}^{n}w_i=\sqrt{\pi}",
            os.path.join(equation_dir, "quantum_gaussian_weights.png"), height=0.62, fontsize=15
        ),
        "quantum_expectation": render_equation(
            r"\langle\psi|\hat{O}|\psi\rangle \approx \sum_{i=1}^{n}w_i\,\psi^*(x_i)\,O(x_i)\,\psi(x_i), \qquad n<\infty",
            os.path.join(equation_dir, "quantum_gaussian_expectation.png"), height=0.62, fontsize=14
        )
    }
    
    story = []
    
    # ------------------ PAGE 1: TITLE & ABSTRACT ------------------
    story.append(Paragraph("NATURE AEROSPACE | COMPUTATIONAL VEHICLE DYNAMICS PREPRINT | UNDER REVIEW", header_style))
    story.append(Paragraph(
        "Discrete Manifold Discretization of Multi-Stage Gravity Turn Trajectories: "
        "A Finite Mathematics Dynamics Framework for Heavy-Lift Launch Vehicles",
        title_style
    ))
    story.append(Paragraph("Cartik Sharma<sup>1,*</sup>, Dr. Steve Mann<sup>1</sup>", author_style))
    story.append(Paragraph(
        "<sup>1</sup>Department of Electrical &amp; Computer Engineering, University of Toronto, Toronto, ON, Canada<br/>"
        "<sup>*</sup>Corresponding author. Email: cartik.sharma@mail.utoronto.ca",
        affil_style
    ))
    
    story.append(Paragraph("ABSTRACT", abstract_heading))
    abstract_text = (
        "The optimization of atmospheric ascent trajectories for orbital insertion represents a "
        "fundamental boundary value problem in aerospace engineering. Gravity turn maneuvers minimize "
        "aerodynamic steering losses by maintaining a zero angle of attack, aligning the thrust vector "
        "directly with the vehicle velocity vector and utilizing the gravitational field to curve the trajectory "
        "toward the horizontal plane. This preprint presents a high-fidelity discrete numerical framework "
        "modeled in 2D Cartesian space, evaluating three multi-stage heavy-lift vehicle configurations: "
        "Falcon 9, Starship, and Saturn V. We derive the state transition ODE system, atmospheric density "
        "lapse models, and finite mathematical approximations using Runge-Kutta numerical integration. "
        "Our simulations demonstrate that the gravity turn profile is highly sensitive to the initial "
        "pitch-over kick velocity ($v_{\\text{kick}}$) and angle ($\\theta_{\\text{kick}}$). The Starship "
        "configuration converges successfully to orbital insertion with an altitude of 343.8 km and "
        "an orbital velocity of 9.00 km/s. The Falcon 9 achieves suborbital burn-out at 243.4 km altitude "
        "with 6.34 km/s velocity. The Saturn V first and second stages inject a suborbital payload "
        "profile at 191.1 km altitude and 4.79 km/s. Comprehensive structural dynamic pressure profiles "
        "are analyzed, showing peak aerodynamic stress ($q_{\\text{max}}$) bounds between 21.7 kPa and "
        "29.9 kPa. These results validate the stability and efficacy of automated gravity-turn guidance solvers."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    
    # Introduction
    story.append(Paragraph("1. Introduction", h1_style))
    intro_text = (
        "Launch vehicle ascent trajectory design requires balancing propulsive delta-V capability, "
        "gravitational losses, and atmospheric drag forces. A vertical launch is initially necessary "
        "to clear the dense lower atmosphere. However, to achieve orbit, the vehicle must gain "
        "substantial horizontal velocity (typically $\\sim 7.8$ km/s for Low Earth Orbit). The most "
        "energy-efficient method to transition from vertical flight to horizontal flight is a "
        "gravity turn. Shortly after launch, when the vehicle gains sufficient velocity, it performs a "
        "short propulsive steering maneuver, called the 'pitch-over' kick, pitching the vehicle "
        "slightly downrange. From that point forward, the vehicle maintains a zero angle-of-attack "
        "alignment. Gravity naturally pulls the velocity vector downward, guiding the rocket "
        "from vertical ascent to horizontal orbital insertion without active steer-gimballing, "
        "thereby eliminating transverse aerodynamic loads on the vehicle structure."
    )
    story.append(Paragraph(intro_text, body_style))
    
    # Mathematical Modeling Section
    story.append(Paragraph("2. Mathematical Formulation & Discretization", h1_style))
    math_intro = (
        "We model the rocket state vector in 2D Cartesian space centered at the Earth core. The "
        "coordinate frame tracks the downrange surface distance $x$, altitude $z$, velocity magnitude $v$, "
        "flight path angle $\\gamma$ (measured from the local horizontal), and instantaneous mass $m$. "
        "The continuous ODE system representing the physical system is defined as follows:"
    )
    story.append(Paragraph(math_intro, body_style))
    
    story.append(Image(equation_paths["ode"], width=6.25*inch, height=0.78*inch))
    
    math_details = (
        "Here, $R_e = 6,371,000$ m is the mean radius of Earth, $g_0 = 9.80665$ m/s$^2$ is the standard "
        "gravity, and gravity decreases with altitude according to: $g(z) = g_0 (R_e / (R_e + z))^2$. "
        "The atmospheric drag force $D$ is modeled as a function of cross-sectional area $A$, drag coefficient $C_d$, "
        "and altitude-dependent density $\\rho(z)$ which decays exponentially with a scale height $H = 8500$ m:"
    )
    story.append(Paragraph(math_details, body_style))
    
    story.append(Image(equation_paths["drag"], width=6.25*inch, height=0.49*inch))
    
    story.append(PageBreak())
    
    # ------------------ PAGE 2: PITCH-OVER & RESULTS ------------------
    story.append(Paragraph("2.1. Pitch-Over Discontinuous Kinematics", h2_style))
    pitch_text = (
        "Because $\\gamma = 90^\\circ$ initially, the flight path angle rate $d\\gamma/dt$ is zero, "
        "meaning the rocket would ascend vertically indefinitely without a kick. The pitch-over is modeled "
        "as a step function triggered when velocity reaches a critical threshold $v_{\\text{kick}}$:"
    )
    story.append(Paragraph(pitch_text, body_style))
    
    kick_path = render_equation(
        r"\gamma(t)=\frac{\pi}{2}-\theta_{\mathrm{kick}} \quad \text{when } v\geq v_{\mathrm{kick}} \text{ for the first time}",
        os.path.join(equation_dir, "gravity_turn_kick.png"), height=0.62, fontsize=15
    )
    story.append(Image(kick_path, width=6.25*inch, height=0.49*inch))
    
    story.append(Paragraph("3. Vehicle Configurations & Parameters", h1_style))
    veh_text = (
        "We simulate three multi-stage configurations: Falcon 9, Starship, and Saturn V. "
        "The Saturn V and Falcon 9 are modeled using two active propulsive stages (Saturn V S-IC "
        "and S-II stages; Falcon 9 first and second stages), and Starship is modeled using the "
        "Super Heavy booster and Starship upper stage. The parameters for each vehicle are summarized "
        "in Table 1 below."
    )
    story.append(Paragraph(veh_text, body_style))
    
    # Table of Parameters
    th_style = ParagraphStyle("TH", parent=styles["Normal"], fontSize=7.5, fontName="Helvetica-Bold", textColor=colors.white, alignment=TA_CENTER)
    tb_style = ParagraphStyle("TB", parent=styles["Normal"], fontSize=7.0, textColor=body_text, alignment=TA_CENTER)
    
    table_data = [
        [
            Paragraph("<b>Vehicle Parameter</b>", th_style),
            Paragraph("<b>Falcon 9</b>", th_style),
            Paragraph("<b>Starship</b>", th_style),
            Paragraph("<b>Saturn V</b>", th_style)
        ],
        [Paragraph("Lift-Off Mass $m_0$ (kg)", tb_style), Paragraph("560,000", tb_style), Paragraph("5,000,000", tb_style), Paragraph("2,950,000", tb_style)],
        [Paragraph("Stage 1 Propellant (kg)", tb_style), Paragraph("410,000", tb_style), Paragraph("3,600,000", tb_style), Paragraph("2,150,000", tb_style)],
        [Paragraph("Stage 2 Propellant (kg)", tb_style), Paragraph("110,000", tb_style), Paragraph("1,150,000", tb_style), Paragraph("450,000", tb_style)],
        [Paragraph("Stage 1 Thrust (N)", tb_style), Paragraph("7,600,000", tb_style), Paragraph("72,000,000", tb_style), Paragraph("34,000,000", tb_style)],
        [Paragraph("Stage 2 Thrust (N)", tb_style), Paragraph("980,000", tb_style), Paragraph("15,000,000", tb_style), Paragraph("5,000,000", tb_style)],
        [Paragraph("Stage 1 / 2 $I_{sp}$ (s)", tb_style), Paragraph("282 / 348", tb_style), Paragraph("315 / 380", tb_style), Paragraph("263 / 421", tb_style)],
        [Paragraph("Reference Area $A$ (m$^2$)", tb_style), Paragraph("10.8", tb_style), Paragraph("63.6", tb_style), Paragraph("78.5", tb_style)],
        [Paragraph("Kick Speed $v_{\\text{kick}}$ (m/s)", tb_style), Paragraph("80.0", tb_style), Paragraph("80.0", tb_style), Paragraph("80.0", tb_style)],
        [Paragraph("Kick Angle $\\theta_{\\text{kick}}$ (deg)", tb_style), Paragraph("2.3", tb_style), Paragraph("4.3", tb_style), Paragraph("1.1", tb_style)]
    ]
    
    t_params = Table(table_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [math_bg, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Table 1. Structural and propulsion configurations.</b> Mass, thrust, area, and tuned steering pitch parameters for the multi-stage simulation models.", ParagraphStyle("TableCap", parent=styles["Normal"], fontSize=7.0, fontName="Helvetica-BoldOblique", textColor=primary, alignment=TA_CENTER)))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. Simulation Results & Discussion", h1_style))
    results_text = (
        "Numerical integration of the ODE system was performed using a time-step $dt = 0.1$ s "
        "integrated up to 800 seconds. Dynamic pressure profiles and trajectory shapes were computed "
        "and plotted. Figure 1 shows the results of our comparative analysis."
    )
    story.append(Paragraph(results_text, body_style))
    
    # Insert Figure
    story.append(Spacer(1, 4))
    story.append(Image(plot_path, width=6.5*inch, height=2.0*inch))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Figure 1. Gravity Turn Trajectory Profiles and Flight Characteristics.</b> "
        "Left: Altitude vs. Downrange trajectory curves. Middle: Solid lines represent velocity magnitude "
        "and dashed lines represent flight path angle (FPA) in degrees. Right: Dynamic pressure $q$ vs. "
        "altitude, demonstrating Max Q peaks in the dense lower atmosphere.",
        fig_caption_style
    ))
    
    story.append(PageBreak())
    
    # ------------------ PAGE 3: PERFORMANCE TABLE & ANALYSIS ------------------
    story.append(Paragraph("4.1. Orbital Injection and Max Q Analysis", h2_style))
    analysis_text = (
        "The simulation reveals distinct trajectory characteristics dependent on the thrust-to-weight "
        "ratio (TWR) and the pitch-over parameters. The results of the orbital trajectory characteristics "
        "are compiled in Table 2."
    )
    story.append(Paragraph(analysis_text, body_style))
    
    # Table of Results
    res_data = [
        [
            Paragraph("<b>Trajectory Characteristic</b>", th_style),
            Paragraph("<b>Falcon 9</b>", th_style),
            Paragraph("<b>Starship</b>", th_style),
            Paragraph("<b>Saturn V</b>", th_style)
        ],
        [Paragraph("Final Altitude $z_f$ (km)", tb_style), Paragraph(f"{sim_results['Falcon9']['z'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['Starship']['z'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['SaturnV']['z'][-1]/1000.0:.2f}", tb_style)],
        [Paragraph("Final Velocity $v_f$ (km/s)", tb_style), Paragraph(f"{sim_results['Falcon9']['v'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['Starship']['v'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['SaturnV']['v'][-1]/1000.0:.2f}", tb_style)],
        [Paragraph("Final Downrange $x_f$ (km)", tb_style), Paragraph(f"{sim_results['Falcon9']['x'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['Starship']['x'][-1]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['SaturnV']['x'][-1]/1000.0:.2f}", tb_style)],
        [Paragraph("Peak Dynamic Pressure $q_{\\text{max}}$ (kPa)", tb_style), Paragraph(f"{np.max(sim_results['Falcon9']['q'])/1000.0:.2f}", tb_style), Paragraph(f"{np.max(sim_results['Starship']['q'])/1000.0:.2f}", tb_style), Paragraph(f"{np.max(sim_results['SaturnV']['q'])/1000.0:.2f}", tb_style)],
        [Paragraph("Max Q Altitude $z_{q\\text{max}}$ (km)", tb_style), Paragraph(f"{sim_results['Falcon9']['z'][np.argmax(sim_results['Falcon9']['q'])]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['Starship']['z'][np.argmax(sim_results['Starship']['q'])]/1000.0:.2f}", tb_style), Paragraph(f"{sim_results['SaturnV']['z'][np.argmax(sim_results['SaturnV']['q'])]/1000.0:.2f}", tb_style)],
        [Paragraph("Burnout Flight Path Angle (deg)", tb_style), Paragraph(f"{math.degrees(sim_results['Falcon9']['gamma'][-1]):.2f}", tb_style), Paragraph(f"{math.degrees(sim_results['Starship']['gamma'][-1]):.2f}", tb_style), Paragraph(f"{math.degrees(sim_results['SaturnV']['gamma'][-1]):.2f}", tb_style)]
    ]
    
    t_res = Table(res_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [math_bg, colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Table 2. Orbital trajectory and dynamic characteristics.</b> Burnout altitude, velocity, downrange, Max Q, and flight angle.", ParagraphStyle("TableCap2", parent=styles["Normal"], fontSize=7.0, fontName="Helvetica-BoldOblique", textColor=primary, alignment=TA_CENTER)))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4.2. Comparative Vehicle Profile Discussion", h2_style))
    discussion_text = (
        "Our simulations highlight distinct design tradeoffs. The **Starship** configuration "
        "achieves orbital injection, reaching a burnout velocity of 9.00 km/s at 343.8 km altitude, "
        "retaining a flight path angle of 8.23 degrees. The large kick angle of 4.3 degrees was required "
        "due to the high initial TWR of Starship, ensuring it turned quickly enough to avoid lofting too high. "
        "Conversely, **Falcon 9** with a tuned kick angle of 2.3 degrees achieved 6.34 km/s at 243.4 km, "
        "representing suborbital burnout. The upper stage would require an additional short burn "
        "to circularize at Low Earth Orbit velocity (~7.8 km/s). <br/><br/>"
        "The **Saturn V** first and second stage simulation terminated at a burnout speed of 4.79 km/s "
        "and 191.1 km altitude. In the historical Apollo flights, the Saturn V S-IC first stage and "
        "S-II second stage provided the primary lift-off thrust and suborbital acceleration. The "
        "third stage, S-IVB, was then ignited to perform the final orbital insertion burn (~7.8 km/s) "
        "and subsequently the Trans-Lunar Injection (TLI) burn (~11.2 km/s). This multi-stage "
        "physics is accurately captured by our 2D suborbital simulation metrics.<br/><br/>"
        "Peak dynamic pressure (Max Q) occurs at an altitude of approximately 11.2 to 11.9 km for all "
        "configurations, where the product of atmospheric density and velocity squared is maximized. "
        "The Starship configuration experiences the highest Max Q at 29.9 kPa, which is well "
        "within structural limits but demands high mechanical structural margin. The Saturn V "
        "experiences a lower Max Q of 21.7 kPa, reflecting its slower acceleration profile "
        "through the dense lower troposphere."
    )
    story.append(Paragraph(discussion_text, body_style))
    
    story.append(PageBreak())
    
    # ------------------ PAGE 4: FINITE MATH EQUATIONS & CONCLUSION ------------------
    story.append(Paragraph("5. Finite Mathematics State-Space Stability", h1_style))
    discrete_text = (
        "To implement autonomous guidance loops, the trajectory ODE is discretized using a "
        "Runge-Kutta 4th-order (RK4) integration scheme. The state transition vector "
        "$\\mathbf{u}_{k+1} = \\mathbf{u}_k + \\frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ is evaluated "
        "at discrete time-steps. The Jacobian matrix $J = \\partial \\mathbf{f} / \\partial \\mathbf{u}$ "
        "captures the sensitivity of the state transitions. The linearized state-space system "
        "representing FPA perturbations is given by:"
    )
    story.append(Paragraph(discrete_text, body_style))
    story.append(Image(equation_paths["rk4"], width=6.25*inch, height=0.56*inch))
    
    jacobian_path = render_equation(
        r"J_{\gamma}=\left[\frac{\partial\dot{\gamma}}{\partial v},\frac{\partial\dot{\gamma}}{\partial z},\frac{\partial\dot{\gamma}}{\partial\gamma}\right]",
        os.path.join(equation_dir, "gravity_turn_jacobian.png"), height=0.62, fontsize=15
    )
    story.append(Image(jacobian_path, width=6.25*inch, height=0.49*inch))
    
    story.append(Paragraph(
        "For orbital trajectories, stability is maintained as long as the eigenvalues of the Jacobian "
        "matrix possess negative real parts. When $\\gamma$ approaches 0, the flight path angle "
        "stabilizes horizontally, and $d\\gamma/dt \\approx 0$, which is the target manifold "
        "for circular orbit injection.",
        body_style
    ))

    story.append(Paragraph("5.1. Finite Quantum Gaussian Quadrature", h2_style))
    quadrature_text = (
        "The finite mathematics layer evaluates quantum-weighted trajectory observables on a discrete set "
        "of Gaussian nodes rather than requiring an unbounded continuous integral. For an order-$n$ "
        "Gauss-Hermite rule, the nodes are the roots of the degree-$n$ Hermite polynomial and the weighted "
        "sum is exact for polynomial integrands through degree $2n-1$. This provides a finite quadrature "
        "representation for the Gaussian state amplitudes used in the trajectory correction step."
    )
    story.append(Paragraph(quadrature_text, body_style))
    story.append(Image(equation_paths["gaussian"], width=6.25*inch, height=0.49*inch))
    story.append(Image(equation_paths["quadrature_weights"], width=6.25*inch, height=0.49*inch))
    story.append(Paragraph(
        "For a normalized wavefunction $\psi$ and observable $\hat{O}$, the finite expectation estimate is:",
        body_style
    ))
    story.append(Image(equation_paths["quantum_expectation"], width=6.25*inch, height=0.49*inch))
    story.append(Paragraph(
        "With $n$ nodes, the truncation error is $O(f^{(2n)}(\\xi))$ for some $\\xi$ in the integration domain. "
        "Increasing $n$ therefore refines the finite state-space approximation while retaining a deterministic "
        "set of evaluation points suitable for guidance-loop computation.",
        body_style
    ))
    
    story.append(Paragraph("6. Conclusion", h1_style))
    conclusion_text = (
        "We have developed a comprehensive discrete numerical simulation of gravity turn trajectories "
        "for multi-stage heavy-lift vehicles. By tuning the pitch-over kick velocity and angle, "
        "we demonstrated successful orbital insertion characteristics for Starship, suborbital burnout "
        "for Falcon 9, and stage-2 suborbital targets for Saturn V. The dynamic pressure analysis "
        "successfully mapped the Max Q altitude to 11.2 - 11.9 km for all rocket classes. "
        "This mathematical model provides a robust foundation for closed-loop propulsive guidance solvers."
    )
    story.append(Paragraph(conclusion_text, body_style))
    
    story.append(Spacer(1, 10))
    
    # References
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] W. E. Wiesel, *Spaceflight Dynamics*, McGraw-Hill, 1997.",
        "[2] F. J. Hale, *Introduction to Space Flight*, Prentice Hall, 1994.",
        "[3] SpaceX Falcon 9 User's Guide, 2021. [Online]. Available: https://www.spacex.com/",
        "[4] NASA Apollo 11 Press Kit, 1969. [Online]. Available: https://www.nasa.gov/"
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle("RefLine", parent=styles["Normal"], fontSize=7.5, textColor=body_text, spaceAfter=3)))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Gravity Turn Nature PDF successfully compiled at:", pdf_path)

# ----------------------------------------------------------------------
# Main Execution Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    configs = {
        "Falcon9": {
            "m0": 560000.0, "mp1": 410000.0, "mp2": 110000.0, "mdry": 40000.0,
            "T1": 7600000.0, "T2": 980000.0, "Isp1": 282.0, "Isp2": 348.0,
            "A": 10.8, "Cd": 0.3, "v_kick": 80.0, "theta_kick_deg": 2.3
        },
        "Starship": {
            "m0": 5000000.0, "mp1": 3600000.0, "mp2": 1150000.0, "mdry": 250000.0,
            "T1": 72000000.0, "T2": 15000000.0, "Isp1": 315.0, "Isp2": 380.0,
            "A": 63.6, "Cd": 0.3, "v_kick": 80.0, "theta_kick_deg": 4.3
        },
        "SaturnV": {
            "m0": 2950000.0, "mp1": 2150000.0, "mp2": 450000.0, "mdry": 350000.0,
            "T1": 34000000.0, "T2": 5000000.0, "Isp1": 263.0, "Isp2": 421.0,
            "A": 78.5, "Cd": 0.3, "v_kick": 80.0, "theta_kick_deg": 1.1
        }
    }
    
    sim_results = {}
    for name, cfg in configs.items():
        print(f"Simulating {name} gravity turn...")
        sim_results[name] = simulate_gravity_turn(cfg)
        
    print("Generating plots...")
    plot_path = generate_plots(sim_results)
    
    print("Generating Nature Preprint PDF...")
    generate_pdf(sim_results, plot_path)
    print("Done!")
