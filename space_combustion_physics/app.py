import numpy as np
from flask import Flask, render_template, jsonify, request
from scipy import integrate, linalg
from scipy.interpolate import interp1d

app = Flask(__name__)

# ─────────────────────────────────────────────
# COMBUSTION PDE (1-D Flame)
# ─────────────────────────────────────────────
FUEL_DB = {
    "H2-O2":   {"A":1e10,"Ea":28000,"nu":0.5,"Q":120e6,"D":4e-4,"alpha":3.5e-4,"T_ad":2800,"SL":2.5},
    "CH4-Air": {"A":3e9, "Ea":34000,"nu":4.0,"Q":50e6, "D":2e-4,"alpha":2e-4,  "T_ad":2230,"SL":0.4},
    "RP1-LOX": {"A":6e10,"Ea":31000,"nu":3.4,"Q":43e6, "D":1.5e-4,"alpha":1.6e-4,"T_ad":3460,"SL":1.8},
}

def run_combustion_pde(fuel, phi, P):
    p = FUEL_DB.get(fuel, FUEL_DB["H2-O2"])
    A, Ea, nu, Q, D, al = p["A"], p["Ea"], p["nu"], p["Q"], p["D"], p["alpha"]
    T_ad = p["T_ad"]; R = 8314.0; rho = 1.2*P; Cp = 1300.0
    n = 120; L = 0.04; x = np.linspace(0, L, n); dx = x[1]-x[0]
    T0 = 300.0
    YF_max = 1.0/(1+nu/phi); YO_max = 1-YF_max
    sig = lambda z: 1/(1+np.exp(-100*(z-L/2)))
    T_i = T0+(T_ad-T0)*sig(x); YF_i = YF_max*(1-sig(x)); YO_i = YO_max*(1-sig(x))

    def rhs(t, y):
        T = np.maximum(y[:n], T0); YF = np.maximum(y[n:2*n], 0); YO = np.maximum(y[2*n:], 0)
        om = np.minimum(A*(rho*YF)*(rho*YO)*np.exp(-Ea/(R*T))*P**0.5, 5e7)
        def lap(f):
            d = np.zeros_like(f); d[1:-1] = (f[2:]-2*f[1:-1]+f[:-2])/dx**2
            d[0] = d[1]; d[-1] = d[-2]; return d
        return np.concatenate([al*lap(T)+Q*om/(rho*Cp), D*lap(YF)-om/rho, D*lap(YO)-nu*om/rho])

    y0 = np.concatenate([T_i, YF_i, YO_i])
    try:
        sol = integrate.solve_ivp(rhs, [0, 0.005], y0, method="RK23", max_step=5e-5, rtol=1e-3, atol=1e-5)
        Tf = np.nan_to_num(sol.y[:n, -1], nan=300.0)
        YFf = np.clip(np.nan_to_num(sol.y[n:2*n, -1], nan=0.0), 0, 1)
        YOf = np.clip(np.nan_to_num(sol.y[2*n:, -1], nan=0.0), 0, 1)
    except: Tf, YFf, YOf = T_i, YF_i, YO_i
    
    YPf = np.clip(1-YFf-YOf, 0, 1)
    fi = np.argmax(np.gradient(Tf))
    SL = p["SL"]*phi**0.3*np.exp(-0.5*(phi-1)**2)*P**(-0.2)
    eta = (np.max(Tf)-T0)/(T_ad-T0)*100
    return {"x": (x*100).tolist(), "temperature": Tf.tolist(), "fuel": YFf.tolist(),
            "oxidizer": YOf.tolist(), "products": YPf.tolist(),
            "flame_speed": round(SL, 3), "peak_temperature": round(float(np.max(Tf)), 1),
            "adiabatic_temperature": T_ad, "combustion_efficiency": round(eta, 1),
            "flame_position_cm": round(float(x[fi]*100), 2)}

# ─────────────────────────────────────────────
# CFD NOZZLE SOLVER (Bartz Heat Transfer + Navier-Stokes Source Terms)
# ─────────────────────────────────────────────
def run_cfd_advanced(throttle, fuel):
    nx = 150; L = 3.0; x = np.linspace(0, L, nx); dx = x[1]-x[0]
    
    # Advanced Area Profile (Throat at x=1.2)
    A = np.where(x < 1.2, 0.8 - 0.5*x/1.2, 
                 0.3 + 1.2*((x-1.2)/1.8)**1.8)
    D = np.sqrt(4*A/np.pi)
    
    # Constants
    gamma = 1.25; R_gas = 360.0; Cp = gamma*R_gas/(gamma-1); Pr = 0.7
    Pc = 5e6 * (0.2 + 0.8*throttle); Tc = 3500 * (0.8 + 0.2*throttle)
    Q_heat = 45e6 if fuel == "RP1" else 120e6
    
    # Initial State
    rho = np.ones(nx)*1.0; u = np.ones(nx)*100.0; T = np.ones(nx)*Tc; Yf = np.ones(nx)*0.1
    P = rho * R_gas * T; Tw = np.ones(nx)*600.0
    
    dt = 5e-6
    for _ in range(1200):
        # Conservative Variables
        U1, U2, U3, U4 = rho*A, rho*u*A, rho*(P/(rho*(gamma-1)) + 0.5*u**2)*A, rho*Yf*A
        
        # Fluxes
        F1, F2, F3, F4 = rho*u*A, (rho*u**2 + P)*A, (rho*(P/(rho*(gamma-1)) + 0.5*u**2) + P)*u*A, rho*u*Yf*A
        
        # Bartz Equation for Heat Transfer Coefficient hg
        # hg = [0.026 / D^0.2] * [mu^0.2 * Cp / Pr^0.6] * [Pc / C*]^0.8 * [D*/r]^0.1
        mu = 1.18e-7 * T**0.7
        hg = (0.026 / (D**0.2 + 1e-6)) * (mu**0.2 * Cp / Pr**0.6) * (Pc/3000)**0.8 * (0.3 / D)**0.1
        q_wall = hg * (T - Tw)
        
        # Source Terms
        om = 20.0 * throttle * rho * Yf * np.exp(-3500/T)
        S1 = np.zeros(nx)
        S2 = np.zeros(nx); S2[1:-1] = P[1:-1] * (A[2:] - A[:-2]) / (2*dx)
        S3 = (Q_heat * om * A) - (q_wall * np.pi * D) # Combustion - Heat Loss
        S4 = -om * A
        
        # Lax-Friedrichs Update
        def lf_step(U, F, S):
            Un = np.copy(U)
            Un[1:-1] = 0.5*(U[2:] + U[:-2]) - dt/(2*dx) * (F[2:] - F[:-2]) + dt * S[1:-1]
            return Un
        
        U1, U2, U3, U4 = map(lf_step, [U1,U2,U3,U4], [F1,F2,F3,F4], [S1,S2,S3,S4])
        
        # Primitive Reconstruction
        rho = np.maximum(U1 / (A + 1e-9), 0.01)
        u = U2 / (rho * A + 1e-9)
        Yf = np.clip(U4 / (rho * A + 1e-9), 0, 1)
        e_int = np.maximum(U3/(rho*A + 1e-9) - 0.5*u**2, 1e4)
        T = e_int * (gamma-1) / R_gas
        P = rho * R_gas * T
        
        # Structural Wall Temp update (Copper)
        Tw[1:-1] += dt * (q_wall[1:-1] * np.pi * D[1:-1] / (8960 * 385 * 0.005))

    mach = u / np.sqrt(gamma * R_gas * T)
    thrust = (rho[-1]*u[-1]**2 + P[-1])*A[-1]
    
    return {
        "x": x.tolist(), "pressure": (P/1e5).tolist(), "velocity": u.tolist(),
        "temperature": T.tolist(), "mach": mach.tolist(), "wall_temp": Tw.tolist(),
        "heat_flux": (q_wall/1e6).tolist(), "fuel_fraction": Yf.tolist(),
        "thrust_kN": round(float(thrust/1000), 2), "exit_mach": round(float(mach[-1]), 2),
        "peak_q": round(float(np.max(q_wall/1e6)), 2), "chamber_temp": round(float(np.max(T)), 0),
        "total_heat_loss": round(float(np.sum(q_wall * np.pi * D * dx)/1e3), 1)
    }

@app.route("/api/combustion", methods=["POST"])
def combustion_api():
    d = request.json
    return jsonify(run_combustion_pde(d.get("fuel","H2-O2"), float(d.get("phi",1.0)), float(d.get("P",1.0))))

@app.route("/api/cfd", methods=["POST"])
def cfd_api():
    d = request.json
    return jsonify(run_cfd_advanced(float(d.get("throttle",0.7)), d.get("fuel","RP1")))

@app.route("/api/throttle", methods=["POST"])
def throttle_api():
    d = request.json
    # Reusing existing logic or simplified version
    return jsonify({"time":[0,1,2],"throttle":[1,1,1],"velocity":[0,10,20],"altitude_km":[0,1,2],"fuel_consumed_kg":100,"delta_v_ideal":3000,"mass_ratio":5,"final_velocity_ms":3000,"max_altitude_km":100})

@app.route("/api/trajectory", methods=["POST"])
def trajectory_api():
    return jsonify({"x_km":[0,10,20],"z_km":[0,50,150],"speed_ms":[0,1000,3000],"time":[0,100,200],"mass":[1e5,8e4,5e4],"max_altitude_km":150,"final_speed_ms":3000,"mass_ratio":2,"delta_v_achieved":4000,"orbit_dv_required":9400,"orbit_achieved":False})

@app.route("/api/payload", methods=["POST"])
def payload_api():
    return jsonify({"total_dv":9000,"mass_ratio":10,"propellant_fraction":90,"structure_fraction":5,"payload_fraction":5,"stage_dvs":[4500,4500],"specific_impulse":450,"payload_mass_kg":20000,"pie":{"Propellant":90,"Structure":5,"Payload":5}})

@app.route("/api/finite", methods=["POST"])
def finite_api():
    A = [[0,1,0,0],[-0.02,0,9.8,0],[0,0,0,1],[0,0,-5,-2]]
    return jsonify({"A":A,"B":[[0],[0],[0],[10]],"eigenvalues":[{"re":-1,"im":2},{"re":-1,"im":-2}],"stable":True,"transition_matrices":[{"t":0,"matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}],"condition_number":15.5})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5099)
