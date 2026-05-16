import numpy as np
from flask import Flask, render_template, jsonify, request
from scipy import integrate, linalg
from scipy.interpolate import interp1d

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. COMBUSTION PDE (1-D Flame Dynamics)
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
# 2. ADVANCED CFD NOZZLE SOLVER
# ─────────────────────────────────────────────
def run_cfd_advanced(throttle, fuel):
    nx = 150; L = 3.0; x = np.linspace(0, L, nx); dx = x[1]-x[0]
    A = np.where(x < 1.2, 0.8 - 0.5*x/1.2, 0.3 + 1.2*((x-1.2)/1.8)**1.8)
    D = np.sqrt(4*A/np.pi)
    gamma = 1.25; R_gas = 360.0; Cp = gamma*R_gas/(gamma-1); Pr = 0.7
    Pc = 5e6 * (0.2 + 0.8*throttle); Tc = 3500 * (0.8 + 0.2*throttle)
    Q_heat = 45e6 if fuel == "RP1" else 120e6
    rho = np.ones(nx)*1.0; u = np.ones(nx)*100.0; T = np.ones(nx)*Tc; Yf = np.ones(nx)*0.1
    P = rho * R_gas * T; Tw = np.ones(nx)*600.0
    dt = 5e-6
    for _ in range(1200):
        U1, U2, U3, U4 = rho*A, rho*u*A, rho*(P/(rho*(gamma-1)) + 0.5*u**2)*A, rho*Yf*A
        F1, F2, F3, F4 = rho*u*A, (rho*u**2 + P)*A, (rho*(P/(rho*(gamma-1)) + 0.5*u**2) + P)*u*A, rho*u*Yf*A
        mu = 1.18e-7 * T**0.7
        hg = (0.026 / (D**0.2 + 1e-6)) * (mu**0.2 * Cp / Pr**0.6) * (Pc/3000)**0.8 * (0.3 / D)**0.1
        q_wall = hg * (T - Tw)
        om = 20.0 * throttle * rho * Yf * np.exp(-3500/T)
        S1, S2, S3, S4 = np.zeros(nx), np.zeros(nx), (Q_heat * om * A) - (q_wall * np.pi * D), -om * A
        S2[1:-1] = P[1:-1] * (A[2:] - A[:-2]) / (2*dx)
        def lf_step(U, F, S):
            Un = np.copy(U); Un[1:-1] = 0.5*(U[2:] + U[:-2]) - dt/(2*dx) * (F[2:] - F[:-2]) + dt * S[1:-1]
            return Un
        U1, U2, U3, U4 = map(lf_step, [U1,U2,U3,U4], [F1,F2,F3,F4], [S1,S2,S3,S4])
        rho = np.maximum(U1 / (A + 1e-9), 0.01); u = U2 / (rho * A + 1e-9); Yf = np.clip(U4 / (rho * A + 1e-9), 0, 1)
        e_int = np.maximum(U3/(rho*A + 1e-9) - 0.5*u**2, 1e4); T = e_int * (gamma-1) / R_gas; P = rho * R_gas * T
        Tw[1:-1] += dt * (q_wall[1:-1] * np.pi * D[1:-1] / (8960 * 385 * 0.005))
    mach = u / np.sqrt(gamma * R_gas * T)
    thrust = (rho[-1]*u[-1]**2 + P[-1])*A[-1]
    return {
        "x": x.tolist(), "pressure": (P/1e5).tolist(), "velocity": u.tolist(), "temperature": T.tolist(), "mach": mach.tolist(), "wall_temp": Tw.tolist(),
        "heat_flux": (q_wall/1e6).tolist(), "fuel_fraction": Yf.tolist(), "thrust_kN": round(float(thrust/1000), 2), "exit_mach": round(float(mach[-1]), 2),
        "peak_q": round(float(np.max(q_wall/1e6)), 2), "chamber_temp": round(float(np.max(T)), 0), "total_heat_loss": round(float(np.sum(q_wall * np.pi * D * dx)/1e3), 1)
    }

# ─────────────────────────────────────────────
# 3. OPTIMAL THROTTLE CONTROL (Pontryagin)
# ─────────────────────────────────────────────
def run_optimal_throttle(Isp, m0, mode):
    g0 = 9.81; T_max = 2e6; m_final = m0 * 0.15; dt = 0.5; t_max = 400
    t = np.arange(0, t_max, dt); n = len(t)
    v = np.zeros(n); h = np.zeros(n); m = np.ones(n)*m0; u = np.ones(n)
    
    for i in range(n-1):
        # Pontryagin-like logic: switch throttle to minimize fuel
        if mode == "fuel_optimal":
            u[i] = 1.0 if m[i] > m_final else 0.0
        else: # Time optimal
            u[i] = 1.0
            
        drag = 0.5 * 1.225 * np.exp(-h[i]/8500) * v[i]**2 * 10.0
        accel = (T_max * u[i] / m[i]) - g0 - (drag / m[i])
        v[i+1] = v[i] + accel * dt
        h[i+1] = h[i] + v[i] * dt
        m[i+1] = max(m[i] - (T_max * u[i] / (Isp * g0)) * dt, m_final)
        if h[i+1] < 0: h[i+1]=0; v[i+1]=0
        
    return {
        "time": t.tolist(), "velocity": v.tolist(), "altitude_km": (h/1000).tolist(), "throttle": u.tolist(),
        "fuel_consumed_kg": round(float(m0 - m[-1]), 0), "delta_v_ideal": round(float(Isp * g0 * np.log(m0/m[-1])), 0),
        "mass_ratio": round(float(m0/m[-1]), 2), "final_velocity_ms": round(float(v[-1]), 1), "max_altitude_km": round(float(np.max(h/1000)), 1)
    }

# ─────────────────────────────────────────────
# 4. HIGH-PERFORMANCE TRAJECTORY (3-DOF RK45)
# ─────────────────────────────────────────────
VEHICLES = {
    "Falcon9": {"m0": 549054, "mp": 500000, "T": 7607000, "Isp": 311, "A": 10.8},
    "Starship": {"m0": 5000000, "mp": 4500000, "T": 72000000, "Isp": 330, "A": 63.6},
    "SaturnV": {"m0": 2970000, "mp": 2600000, "T": 34000000, "Isp": 263, "A": 78.5}
}

def run_trajectory_hpc(vehicle_name, payload_mass, orbit_type):
    v_data = VEHICLES.get(vehicle_name, VEHICLES["Falcon9"])
    m0 = v_data["m0"] + payload_mass
    T_thrust = v_data["T"]
    Isp = v_data["Isp"]
    A_ref = v_data["A"]
    
    g0 = 9.80665; Re = 6371000
    
    # State: [x, z, v, gamma, m]
    # x: downrange, z: altitude, v: velocity, gamma: flight path angle, m: mass
    def derivs(t, y):
        x, z, v, gamma, m = y
        if m < (m0 - v_data["mp"]): thrust = 0
        else: thrust = T_thrust
        
        # Atmospheric model (Exponential)
        rho = 1.225 * np.exp(-z / 8500.0) if z < 100000 else 0
        drag = 0.5 * rho * v**2 * 0.3 * A_ref
        
        r = Re + z
        g = g0 * (Re / r)**2
        
        d_x = (Re / r) * v * np.cos(gamma)
        d_z = v * np.sin(gamma)
        d_v = (thrust - drag) / m - g * np.sin(gamma)
        
        if v > 10:
            d_gamma = (v / r - g / v) * np.cos(gamma)
        else: d_gamma = 0
            
        d_m = -(thrust / (Isp * g0)) if thrust > 0 else 0
        
        return [d_x, d_z, d_v, d_gamma, d_m]

    # Stop when altitude hits 0 or orbit achieved
    def hit_ground(t, y): return y[1] if t > 10 else 1
    hit_ground.terminal = True; hit_ground.direction = -1

    sol = integrate.solve_ivp(derivs, [0, 800], [0, 0, 0.1, np.pi/2, m0], 
                              method="RK45", t_eval=np.linspace(0, 800, 400), 
                              events=hit_ground, rtol=1e-6)
    
    res = sol.y
    t = sol.t
    
    # Calculate Dynamic Pressure (Max Q)
    rho_arr = 1.225 * np.exp(-res[1] / 8500.0)
    q_dyn = 0.5 * rho_arr * res[2]**2
    
    dv_achieved = Isp * g0 * np.log(m0 / res[4][-1])
    target_v = np.sqrt(g0 * Re**2 / (Re + np.max(res[1])))
    
    return {
        "time": t.tolist(),
        "x_km": (res[0]/1000).tolist(),
        "z_km": (res[1]/1000).tolist(),
        "speed_ms": res[2].tolist(),
        "gamma_deg": np.degrees(res[3]).tolist(),
        "mass": res[4].tolist(),
        "dynamic_pressure": (q_dyn/1000).tolist(), # kPa
        "max_altitude_km": round(float(np.max(res[1]/1000)), 2),
        "final_speed_ms": round(float(res[2][-1]), 2),
        "max_q_kpa": round(float(np.max(q_dyn/1000)), 2),
        "delta_v_achieved": round(float(dv_achieved), 0),
        "orbit_dv_required": round(float(target_v + 2000), 0),
        "orbit_achieved": bool(res[2][-1] > target_v * 0.95 and res[1][-1] > 150000)
    }

# ─────────────────────────────────────────────
# 5. PAYLOAD BUDGET (Tsiolkovsky)
# ─────────────────────────────────────────────
def run_payload_budget(Isp, m0, ms, mp, stages):
    m_stage_init = m0 / stages
    dv_per_stage = Isp * 9.81 * np.log(m0 / (m0 - (m0-ms-mp)/stages))
    total_dv = dv_per_stage * stages
    return {
        "total_dv": round(total_dv, 0), "mass_ratio": round(m0/ms, 2), "propellant_fraction": round((m0-ms-mp)/m0*100, 1),
        "payload_fraction": round(mp/m0*100, 1), "stage_dvs": [round(dv_per_stage, 0)]*stages, "pie": {"Propellant": m0-ms-mp, "Structure": ms, "Payload": mp}
    }

# ─────────────────────────────────────────────
# 6. FINITE MATH (State-Space Stability)
# ─────────────────────────────────────────────
def run_finite_math(v_ref, h_ref):
    # Linearized pitch dynamics matrix
    q = 0.5 * 1.225 * np.exp(-h_ref/8500) * v_ref**2
    A = np.array([[0, 1, 0, 0], [q*0.01, -0.1, 9.8, 0], [0, 0, 0, 1], [0, 0, -2.0, -0.5]])
    eigs = linalg.eigvals(A)
    phi_list = []
    for dt in [0, 1, 5]:
        phi = linalg.expm(A * dt)
        phi_list.append({"t": dt, "matrix": phi.real.tolist()})
    return {
        "A": A.tolist(), "eigenvalues": [{"re": round(float(e.real), 3), "im": round(float(e.imag), 3)} for e in eigs],
        "stable": all(e.real < 0 for e in eigs), "transition_matrices": phi_list, "condition_number": round(float(np.linalg.cond(A)), 2)
    }

@app.route("/api/combustion", methods=["POST"])
def combustion_api():
    d = request.json; return jsonify(run_combustion_pde(d.get("fuel","H2-O2"), float(d.get("phi",1.0)), float(d.get("P",1.0))))

@app.route("/api/cfd", methods=["POST"])
def cfd_api():
    d = request.json; return jsonify(run_cfd_advanced(float(d.get("throttle",0.7)), d.get("fuel","RP1")))

@app.route("/api/throttle", methods=["POST"])
def throttle_api():
    d = request.json; return jsonify(run_optimal_throttle(float(d.get("Isp",450)), float(d.get("m0",1e5)), d.get("mode","fuel_optimal")))

@app.route("/api/trajectory", methods=["POST"])
def trajectory_api():
    d = request.json
    return jsonify(run_trajectory_hpc(d.get("vehicle","Falcon9"), float(d.get("payload",5000)), d.get("orbit","LEO")))

@app.route("/api/payload", methods=["POST"])
def payload_api():
    d = request.json; return jsonify(run_payload_budget(float(d.get("Isp",450)), float(d.get("m0",5e5)), float(d.get("m_struct",5e4)), float(d.get("m_payload",2e4)), int(d.get("stages",2))))

@app.route("/api/finite", methods=["POST"])
def finite_api():
    d = request.json; return jsonify(run_finite_math(float(d.get("v_ref",500)), float(d.get("h_ref",50000))))

@app.route("/")
def index(): return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5099)
