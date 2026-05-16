from flask import Flask, render_template, jsonify, request
import numpy as np
from scipy import integrate, linalg
from scipy.interpolate import interp1d

app = Flask(__name__)

# ─────────────────────────────────────────────
# COMBUSTION PDE  (1-D premixed laminar flame)
# ∂T/∂t  = α ∂²T/∂x² + Q·ω/(ρ Cp)
# ∂Yf/∂t = D ∂²Yf/∂x² − ω/ρ
# ∂Yo/∂t = D ∂²Yo/∂x² − ν ω/ρ
# ω = A·(ρYf)(ρYo)·exp(−Ea/RT)   [Arrhenius]
# ─────────────────────────────────────────────
FUEL_DB = {
    "H2-O2":   {"A":1e10,"Ea":28000,"nu":0.5,"Q":120e6,"D":4e-4,"alpha":3.5e-4,"T_ad":2800,"SL":2.5},
    "CH4-Air": {"A":3e9, "Ea":34000,"nu":4.0,"Q":50e6, "D":2e-4,"alpha":2e-4,  "T_ad":2230,"SL":0.4},
    "RP1-LOX": {"A":6e10,"Ea":31000,"nu":3.4,"Q":43e6, "D":1.5e-4,"alpha":1.6e-4,"T_ad":3460,"SL":1.8},
}

def run_combustion_pde(fuel, phi, P):
    p   = FUEL_DB.get(fuel, FUEL_DB["H2-O2"])
    A,Ea,nu,Q,D,al = p["A"],p["Ea"],p["nu"],p["Q"],p["D"],p["alpha"]
    T_ad= p["T_ad"];  R=8314.0;  rho=1.2*P;  Cp=1300.0
    n=120; L=0.04; x=np.linspace(0,L,n); dx=x[1]-x[0]
    T0=300.0
    YF_max=1.0/(1+nu/phi);  YO_max=1-YF_max
    sig=lambda z: 1/(1+np.exp(-100*(z-L/2)))
    T_i=T0+(T_ad-T0)*sig(x); YF_i=YF_max*(1-sig(x)); YO_i=YO_max*(1-sig(x))

    def rhs(t,y):
        T=np.maximum(y[:n],T0); YF=np.maximum(y[n:2*n],0); YO=np.maximum(y[2*n:],0)
        om=np.minimum(A*(rho*YF)*(rho*YO)*np.exp(-Ea/(R*T))*P**0.5, 5e7)
        def lap(f):
            d=np.zeros_like(f); d[1:-1]=(f[2:]-2*f[1:-1]+f[:-2])/dx**2
            d[0]=d[1]; d[-1]=d[-2]; return d
        return np.concatenate([al*lap(T)+Q*om/(rho*Cp), D*lap(YF)-om/rho, D*lap(YO)-nu*om/rho])

    y0=np.concatenate([T_i,YF_i,YO_i])
    sol=integrate.solve_ivp(rhs,[0,0.005],y0,method="RK23",max_step=5e-5,rtol=1e-3,atol=1e-5)
    Tf=sol.y[:n,-1]; YFf=np.clip(sol.y[n:2*n,-1],0,1); YOf=np.clip(sol.y[2*n:,-1],0,1)
    YPf=np.clip(1-YFf-YOf,0,1)
    fi=np.argmax(np.gradient(Tf))
    SL=p["SL"]*phi**0.3*np.exp(-0.5*(phi-1)**2)*P**(-0.2)
    eta=(np.max(Tf)-T0)/(T_ad-T0)*100
    return {"x":(x*100).tolist(),"temperature":Tf.tolist(),"fuel":YFf.tolist(),
            "oxidizer":YOf.tolist(),"products":YPf.tolist(),
            "flame_speed":round(SL,3),"peak_temperature":round(float(np.max(Tf)),1),
            "adiabatic_temperature":T_ad,"combustion_efficiency":round(eta,1),
            "flame_position_cm":round(float(x[fi]*100),2)}

@app.route("/api/combustion", methods=["POST"])
def combustion_api():
    d=request.json
    return jsonify(run_combustion_pde(d.get("fuel","H2-O2"),float(d.get("phi",1.0)),float(d.get("P",1.0))))


# ─────────────────────────────────────────────
# OPTIMAL THROTTLE  (Pontryagin bang-bang / smooth)
# ─────────────────────────────────────────────
def compute_throttle(Isp, m0, mode):
    g0=9.80665; T_max=1.5e6; t=np.linspace(0,300,200)
    if mode=="fuel_optimal":
        th=np.ones(200); th[120:170]=np.linspace(1,.7,50); th[170:]=np.linspace(.7,.5,30)
    elif mode=="time_optimal":
        th=np.ones(200)
    else:
        th=np.clip(0.7+0.3*np.cos(np.pi*t/300),0.3,1.0)

    thi=interp1d(t,th,fill_value="extrapolate")
    def eom(tv,s):
        v,h,m=s; Tc=thi(tv)*T_max
        drag=0.5*1.225*np.exp(-h/8500)*v**2*10; gc=g0*(6371000/(6371000+max(h,0)))**2
        return [(Tc-drag)/max(m,1)-gc, v, -Tc/(Isp*g0)]
    sol=integrate.solve_ivp(eom,[0,300],[0,0,m0],t_eval=t,method="RK45",rtol=1e-4,atol=1e-6)
    mf=sol.y[2,-1]; fc=m0-mf; mr=m0/max(mf,1)
    dv=Isp*g0*np.log(mr)
    return {"time":t.tolist(),"throttle":th.tolist(),
            "velocity":sol.y[0].tolist(),"altitude_km":(sol.y[1]/1000).tolist(),
            "mass":sol.y[2].tolist(),"fuel_consumed_kg":round(fc,0),
            "delta_v_ideal":round(dv,0),"mass_ratio":round(mr,3),
            "final_velocity_ms":round(float(sol.y[0,-1]),0),
            "max_altitude_km":round(float(np.max(sol.y[1]/1000)),1)}

@app.route("/api/throttle", methods=["POST"])
def throttle_api():
    d=request.json
    return jsonify(compute_throttle(float(d.get("Isp",450)),float(d.get("m0",100000)),d.get("mode","fuel_optimal")))


# ─────────────────────────────────────────────
# ROCKET TRAJECTORY  (gravity-turn, 2-D)
# dx/dt = v cos γ   dz/dt = v sin γ
# dv/dt = T/m − g sin γ − D/m
# dγ/dt = (−g cos γ)/v   dm/dt = −T/(Isp g0)
# ─────────────────────────────────────────────
VEHICLE_DB = {
    "Falcon9": {"m0":549054,"T":7607000,"Isp":282,"A_ref":10.75},
    "SaturnV": {"m0":2970000,"T":34020000,"Isp":263,"A_ref":80},
    "Starship": {"m0":5000000,"T":74000000,"Isp":363,"A_ref":63.6},
}

def compute_trajectory(vehicle, m_payload, orbit):
    v=VEHICLE_DB.get(vehicle,VEHICLE_DB["Falcon9"])
    g0=9.80665; R_e=6.371e6; mu=3.986e14
    m0=v["m0"]+m_payload; T=v["T"]; Isp=v["Isp"]; A=v["A_ref"]; Cd=0.3
    orbit_dv={"LEO":9400,"GTO":11800,"TLI":12800}.get(orbit,9400)

    def eom(t,s):
        x,z,vx,vz,m=s
        r=np.sqrt(x**2+(R_e+z)**2); speed=np.sqrt(vx**2+vz**2)+1e-6
        grav=mu/r**2; gx=-grav*x/r; gz=-grav*(R_e+z)/r
        rho_atm=1.225*np.exp(-z/8500)
        drag=0.5*rho_atm*Cd*A*speed**2/max(m,1)
        Tc=T if m>m0*0.15 else 0
        dm=-Tc/(Isp*g0)
        ax=Tc*vx/(speed*max(m,1))-drag*vx/speed+gx
        az=Tc*vz/(speed*max(m,1))-drag*vz/speed+gz
        return [vx,vz,ax,az,dm]

    gamma0=np.radians(89)
    v0=10.0
    s0=[0,0,v0*np.cos(gamma0),v0*np.sin(gamma0),m0]
    t_end=600
    sol=integrate.solve_ivp(eom,[0,t_end],s0,method="RK45",
                            t_eval=np.linspace(0,t_end,400),rtol=1e-4,atol=1e-6,
                            events=lambda t,s: s[2] if s[2]>0 else -1)
    x=sol.y[0]/1000; z=sol.y[1]/1000
    speed=np.sqrt(sol.y[2]**2+sol.y[3]**2)
    mf=sol.y[4,-1]; mr=m0/max(mf,1)
    dv_achieved=Isp*g0*np.log(mr)
    orbit_achieved=dv_achieved>=orbit_dv*0.95
    return {"x_km":x.tolist(),"z_km":z.tolist(),
            "speed_ms":speed.tolist(),"time":sol.t.tolist(),
            "mass":sol.y[4].tolist(),"max_altitude_km":round(float(np.max(z)),1),
            "final_speed_ms":round(float(speed[-1]),0),"mass_ratio":round(mr,3),
            "delta_v_achieved":round(dv_achieved,0),"orbit_dv_required":orbit_dv,
            "orbit_achieved":bool(orbit_achieved)}

@app.route("/api/trajectory", methods=["POST"])
def trajectory_api():
    d=request.json
    return jsonify(compute_trajectory(d.get("vehicle","Falcon9"),float(d.get("payload",5000)),d.get("orbit","LEO")))


# ─────────────────────────────────────────────
# PAYLOAD CHARACTERISTICS  (Tsiolkovsky + staging)
# Δv = Isp·g0·ln(m0/mf)
# ─────────────────────────────────────────────
@app.route("/api/payload", methods=["POST"])
def payload_api():
    d=request.json
    Isp=float(d.get("Isp",450)); m0=float(d.get("m0",500000))
    m_struct=float(d.get("m_struct",50000)); m_payload=float(d.get("m_payload",20000))
    stages=int(d.get("stages",2))
    g0=9.80665
    m_prop=m0-m_struct-m_payload
    mr=m0/(m_struct+m_payload)
    dv=Isp*g0*np.log(mr)
    mass_frac=m_prop/m0*100
    struct_frac=m_struct/m0*100
    pay_frac=m_payload/m0*100
    # Multi-stage Tsiolkovsky continued-fraction approximation
    dv_stages=[]
    m_curr=m0
    m_per_stage=m_prop/stages; ms_per_stage=m_struct/stages
    for i in range(stages):
        m_stage_end=m_curr-m_per_stage
        dv_stages.append(round(Isp*g0*np.log(m_curr/m_stage_end),0))
        m_curr=m_stage_end-ms_per_stage
    return {"total_dv":round(dv,0),"mass_ratio":round(mr,3),
            "propellant_fraction":round(mass_frac,1),
            "structure_fraction":round(struct_frac,1),
            "payload_fraction":round(pay_frac,1),
            "stage_dvs":dv_stages,
            "specific_impulse":Isp,
            "payload_mass_kg":m_payload,
            "pie":{"Propellant":round(mass_frac,1),"Structure":round(struct_frac,1),"Payload":round(pay_frac,1)}}

# ─────────────────────────────────────────────
# CFD MODELING  (Quasi-1D Compressible Flow with Combustion)
# ∂U/∂t + ∂F/∂x = S
# U = [ρA, ρuA, ρEA, ρYfA]ᵀ
# S = [0, P dA/dx, Q_heat * ω * A, -ω * A]ᵀ
# ─────────────────────────────────────────────
def run_cfd_simulation(throttle_level, fuel_type):
    nx = 100
    L = 2.0  # Chamber + Nozzle length (m)
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    
    # Area profile: Convergent-Divergent Nozzle
    # x=0 to 0.5: Chamber, 0.5 to 1.2: Convergent, 1.2 to 2.0: Divergent
    A = np.where(x < 0.5, 0.5, 
                 np.where(x < 1.2, 0.5 - 0.4*(x-0.5)/0.7, 
                          0.1 + 0.4*(x-1.2)/0.8))
    
    # Initial conditions
    rho = np.ones(nx) * 1.2
    u = np.ones(nx) * 0.1
    P = np.ones(nx) * 101325 * (1 + 10 * throttle_level) # Inlet pressure depends on throttle
    T = np.ones(nx) * 300
    Yf = np.ones(nx) * 0.1
    
    gamma = 1.4
    R = 287.0
    Cp = R * gamma / (gamma - 1)
    Q_heat = 43e6 # Fuel heating value
    
    # Time stepping (simplified steady-state approach via pseudo-time)
    dt = 1e-5
    for _ in range(500):
        # Primitive to Conservative
        U1 = rho * A
        U2 = rho * u * A
        e_int = P / (rho * (gamma - 1))
        U3 = rho * (e_int + 0.5 * u**2) * A
        U4 = rho * Yf * A
        
        # Fluxes
        F1 = rho * u * A
        F2 = (rho * u**2 + P) * A
        F3 = (U3 + P * A) * u
        F4 = rho * u * Yf * A
        
        # Source terms (Combustion)
        # Simple reaction rate based on T and Yf
        om = 5.0 * throttle_level * rho * Yf * np.exp(-5000/T)
        S1 = np.zeros(nx)
        # S2: Pressure term for area change
        S2 = np.zeros(nx)
        S2[1:-1] = P[1:-1] * (A[2:] - A[:-2]) / (2*dx)
        S3 = Q_heat * om * A
        S4 = -om * A
        
        # Update (Central difference + damping)
        def step(U, F, S):
            Unew = np.copy(U)
            Unew[1:-1] = U[1:-1] - dt/dx * (F[2:] - F[:-2])/2 + dt * S[1:-1]
            return Unew
            
        U1 = step(U1, F1, S1)
        U2 = step(U2, F2, S2)
        U3 = step(U3, F3, S3)
        U4 = step(U4, F4, S4)
        
        # Conservative to Primitive
        rho = U1 / A
        u = U2 / (rho * A)
        Yf = np.clip(U4 / (rho * A), 0, 1)
        e_total = U3 / (rho * A)
        e_int = e_total - 0.5 * u**2
        T = np.maximum(e_int * (gamma - 1) / R, 300)
        P = rho * R * T
    
    mach = u / np.sqrt(gamma * R * T)
    thrust = (rho[-1] * u[-1]**2 + P[-1]) * A[-1]
    
    return {
        "x": x.tolist(),
        "pressure": (P / 1e5).tolist(), # bar
        "velocity": u.tolist(),
        "temperature": T.tolist(),
        "mach": mach.tolist(),
        "area": A.tolist(),
        "fuel_fraction": Yf.tolist(),
        "thrust_kN": round(thrust / 1000, 2),
        "exit_mach": round(float(mach[-1]), 2),
        "chamber_temp": round(float(T[25]), 0),
        "peak_pressure": round(float(np.max(P/1e5)), 2)
    }

@app.route("/api/cfd", methods=["POST"])
def cfd_api():
    d = request.json
    return jsonify(run_cfd_simulation(float(d.get("throttle", 0.5)), d.get("fuel", "RP1")))


# ─────────────────────────────────────────────
# FINITE MATH  —  state-space rocket dynamics
# ẋ = Ax + Bu,  Φ(t) = exp(At)
# State: [altitude, velocity, pitch, pitch_rate]
# ─────────────────────────────────────────────
@app.route("/api/finite", methods=["POST"])
def finite_api():
    d=request.json
    v_ref=float(d.get("v_ref",500)); h_ref=float(d.get("h_ref",50000))
    g=9.81
    # Linearised A matrix (pitch-plane dynamics)
    A=np.array([[0,1,0,0],
                [-g/v_ref,0,g,0],
                [0,0,0,1],
                [0,0,-5,-2]], dtype=float)
    B=np.array([[0],[0],[0],[10]], dtype=float)
    eigs=np.linalg.eigvals(A)
    stable=bool(np.all(eigs.real<0))
    # State transition matrices at t = 0,1,2,5,10 s
    times=[0,0.5,1,2,5]
    phis=[]
    for t in times:
        Phi=linalg.expm(A*t)
        phis.append({"t":t,"matrix":Phi.tolist()})
    return {"A":A.tolist(),"B":B.tolist(),
            "eigenvalues":[{"re":round(e.real,4),"im":round(e.imag,4)} for e in eigs],
            "stable":stable,"transition_matrices":phis,
            "condition_number":round(float(np.linalg.cond(A)),2)}


@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True, port=5099)
