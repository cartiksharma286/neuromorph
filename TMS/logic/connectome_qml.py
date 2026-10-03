"""Human cortical connectome with modular harmonic forms and a quantum-kernel (QML) connectome matrix.

Healthy vs. Alzheimer-type dementia, on a 68-region Desikan-Killiany-style parcellation.

Everything below is a computational model on a *synthetic, stylised* connectome (approximate region
centroids, distance-decay + lobe/homotopic structure, seeded noise). It is not subject data.

Pipeline
  1. W_H: healthy structural connectome.  W_D: dementia connectome from Braak-staged regional atrophy
     a_i = 0.85 s w(Braak_i) and long-range disconnection (severity s in [0, 1]).
  2. Graph/finite-math invariants on a common-threshold graph: strength, Onnela clustering, global
     efficiency, algebraic connectivity, log weighted spanning-tree count (Kirchhoff), Betti numbers.
  3. Modular harmonic forms: eigenmodes (lambda_n, psi_n) of the normalised Laplacian filtered by the
     Hecke-normalised Ramanujan tau coefficients and the nome q = exp(2 pi i tau):
         c_n(tau) = tau(n) n^(-11/2) Re[q^lambda_n],   K_mod = sum_n c_n psi_n psi_n^T.
     Eisenstein E4, E6, Delta, eta and j are evaluated at the same tau.
  4. QML connectome: a data-re-uploading RY/CZ circuit on 6 qubits (statevector simulation) maps each
     region's 6 features to a state; the fidelity kernel Q_ij = |<phi_i|phi_j>|^2 is trained by SPSA on
     centred kernel-target alignment with the structural connectome.
"""

import math
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np

# name: (lobe, braak_stage, (x, y, z) right-hemisphere approximate MNI centroid in mm)
DK_REGIONS: List[Tuple[str, str, int, Tuple[float, float, float]]] = [
    ("bankssts", "temporal", 4, (50, -42, 8)),
    ("caudalanteriorcingulate", "cingulate", 5, (6, 18, 30)),
    ("caudalmiddlefrontal", "frontal", 5, (38, 10, 48)),
    ("cuneus", "occipital", 6, (8, -80, 22)),
    ("entorhinal", "temporal", 1, (24, -8, -32)),
    ("fusiform", "temporal", 3, (36, -52, -16)),
    ("inferiorparietal", "parietal", 5, (44, -66, 34)),
    ("inferiortemporal", "temporal", 4, (52, -34, -24)),
    ("isthmuscingulate", "cingulate", 4, (8, -48, 14)),
    ("lateraloccipital", "occipital", 6, (34, -86, 2)),
    ("lateralorbitofrontal", "frontal", 5, (28, 32, -14)),
    ("lingual", "occipital", 6, (12, -70, -4)),
    ("medialorbitofrontal", "frontal", 5, (8, 42, -14)),
    ("middletemporal", "temporal", 4, (58, -30, -10)),
    ("parahippocampal", "temporal", 2, (24, -32, -16)),
    ("paracentral", "frontal", 6, (8, -30, 56)),
    ("parsopercularis", "frontal", 5, (50, 14, 12)),
    ("parsorbitalis", "frontal", 5, (44, 38, -10)),
    ("parstriangularis", "frontal", 5, (48, 30, 6)),
    ("pericalcarine", "occipital", 6, (10, -78, 6)),
    ("postcentral", "parietal", 6, (46, -24, 48)),
    ("posteriorcingulate", "cingulate", 4, (6, -14, 36)),
    ("precentral", "frontal", 6, (40, -10, 48)),
    ("precuneus", "parietal", 5, (10, -62, 40)),
    ("rostralanteriorcingulate", "cingulate", 5, (8, 36, 4)),
    ("rostralmiddlefrontal", "frontal", 5, (34, 40, 26)),
    ("superiorfrontal", "frontal", 5, (14, 30, 52)),
    ("superiorparietal", "parietal", 5, (24, -62, 56)),
    ("superiortemporal", "temporal", 5, (58, -12, 2)),
    ("supramarginal", "parietal", 5, (56, -40, 32)),
    ("frontalpole", "frontal", 5, (10, 60, -4)),
    ("temporalpole", "temporal", 3, (40, 12, -32)),
    ("transversetemporal", "temporal", 6, (46, -20, 10)),
    ("insula", "insular", 4, (38, 0, 2)),
]
HUBS = {"precuneus", "superiorfrontal", "posteriorcingulate", "insula", "isthmuscingulate",
        "rostralmiddlefrontal", "supramarginal"}
BRAAK_WEIGHT = {1: 1.0, 2: 0.95, 3: 0.8, 4: 0.6, 5: 0.32, 6: 0.1}
LOBE_ORDER = ["frontal", "parietal", "temporal", "occipital", "cingulate", "insular"]
CDR_LEVELS = [(0.0, "CDR 0 - cognitively normal"), (0.5, "CDR 0.5 - very mild / MCI"),
              (1.0, "CDR 1 - mild dementia"), (2.0, "CDR 2 - moderate dementia"),
              (3.0, "CDR 3 - severe dementia")]
N_QUBITS = 6


# ───────────────────────────── parcellation & connectomes ─────────────────────────────

def build_parcellation() -> Dict[str, Any]:
    names, hemis, lobes, braak, coords = [], [], [], [], []
    for hemi, sign in (("rh", 1.0), ("lh", -1.0)):
        for nm, lobe, br, (x, y, z) in DK_REGIONS:
            names.append(f"{hemi}_{nm}")
            hemis.append(hemi)
            lobes.append(lobe)
            braak.append(br)
            coords.append((sign * x, y, z))
    return {"names": names, "hemis": hemis, "lobes": lobes, "braak": np.array(braak),
            "coords": np.array(coords, dtype=float)}


def healthy_connectome(parc: Dict[str, Any], seed: int = 11) -> np.ndarray:
    coords, names = parc["coords"], parc["names"]
    n = len(names)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    lobes = np.array(parc["lobes"])
    hemis = np.array(parc["hemis"])
    base = [nm.split("_", 1)[1] for nm in names]
    same_lobe = (lobes[:, None] == lobes[None, :]).astype(float)
    inter = (hemis[:, None] != hemis[None, :])
    homotopic = inter & (np.array(base)[:, None] == np.array(base)[None, :])
    hub = np.array([1.4 if b in HUBS else 1.0 for b in base])
    W = np.exp(-d / 40.0) * (1.0 + 0.5 * same_lobe + 0.9 * homotopic)
    W = W * np.where(inter & ~homotopic, 0.55, 1.0) * np.sqrt(hub[:, None] * hub[None, :])
    rng = np.random.default_rng(seed)
    noise = np.exp(rng.normal(0.0, 0.25, size=(n, n)))
    W = W * np.sqrt(noise * noise.T)
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    return W / W.max()


def dementia_connectome(W: np.ndarray, parc: Dict[str, Any], severity: float) -> Tuple[np.ndarray, np.ndarray]:
    w = np.array([BRAAK_WEIGHT[b] for b in parc["braak"]])
    atrophy = np.clip(0.85 * severity * w, 0.0, 0.9)
    d = np.linalg.norm(parc["coords"][:, None, :] - parc["coords"][None, :, :], axis=2)
    keep = np.sqrt(np.outer(1.0 - atrophy, 1.0 - atrophy)) * np.exp(-0.8 * severity * d / 120.0)
    Wd = W * keep
    np.fill_diagonal(Wd, 0.0)
    return Wd, atrophy


def cdr_label(severity: float) -> str:
    target = severity * 3.0
    return min(CDR_LEVELS, key=lambda t: abs(t[0] - target))[1]


# ───────────────────────────── graph / finite-math invariants ─────────────────────────────

def _components(A: np.ndarray) -> List[List[int]]:
    n = len(A)
    seen, comps = np.zeros(n, dtype=bool), []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where((A[u] > 0) & ~seen)[0]:
                seen[v] = True
                stack.append(int(v))
        comps.append(comp)
    return comps


def _onnela_clustering(A: np.ndarray) -> np.ndarray:
    k = (A > 0).sum(axis=1).astype(float)
    cube = np.cbrt(A)
    tri = np.einsum("ij,jk,ki->i", cube, cube, cube)
    denom = k * (k - 1.0)
    return np.where(denom > 0, tri / np.maximum(denom, 1e-12), 0.0)


def _global_efficiency(A: np.ndarray) -> float:
    n = len(A)
    D = np.where(A > 0, 1.0 / np.maximum(A, 1e-12), np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    off = ~np.eye(n, dtype=bool)
    inv = np.where(np.isfinite(D) & off, 1.0 / np.where(D > 0, D, np.inf), 0.0)
    return float(inv[off].mean())


def _normalised_laplacian(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    L = -(inv_sqrt[:, None] * A * inv_sqrt[None, :])
    L[np.diag_indices_from(L)] = np.where(deg > 0, 1.0, 0.0)
    return L


def graph_invariants(A: np.ndarray, spectrum: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    comps = _components(A)
    big = max(comps, key=len)
    sub = A[np.ix_(big, big)]
    Lc = np.diag(sub.sum(axis=1)) - sub
    sign, logdet = np.linalg.slogdet(Lc[1:, 1:]) if len(big) > 1 else (1.0, 0.0)
    n_edges = int(np.count_nonzero(np.triu(A, 1)))
    iso = sum(1 for c in comps if len(c) == 1)
    lam = spectrum[0]
    positive = lam[lam > 1e-9]
    return {
        "edges": n_edges,
        "density": float(n_edges / (len(A) * (len(A) - 1) / 2)),
        "mean_strength": float(A.sum(axis=1).mean()),
        "clustering": float(_onnela_clustering(A).mean()),
        "global_efficiency": _global_efficiency(A),
        "algebraic_connectivity": float(positive.min()) if (len(comps) == 1 and positive.size) else 0.0,
        "log_spanning_trees": float(logdet) if sign > 0 else None,
        "betti_0": len(comps),
        "betti_1": n_edges - len(A) + len(comps),
        "isolated_nodes": iso,
    }


# ───────────────────────────── modular harmonic forms ─────────────────────────────

def ramanujan_tau(n_max: int) -> List[int]:
    """tau(1..n_max) from q * prod (1 - q^k)^24 in exact integer arithmetic (index 0 unused)."""
    poly = [1] + [0] * (n_max - 1)
    for k in range(1, n_max):
        for _ in range(24):
            for i in range(n_max - 1, k - 1, -1):
                poly[i] -= poly[i - k]
    return [0] + poly[:n_max]


def _sigma(n: int, p: int) -> int:
    return sum(d ** p for d in range(1, n + 1) if n % d == 0)


def modular_forms_at(tau: complex, terms: int = 60) -> Dict[str, Any]:
    q = np.exp(2j * np.pi * tau)
    qn = q ** np.arange(1, terms + 1)
    e4 = 1 + 240 * np.sum(np.array([_sigma(n, 3) for n in range(1, terms + 1)]) * qn)
    e6 = 1 - 504 * np.sum(np.array([_sigma(n, 5) for n in range(1, terms + 1)]) * qn)
    delta = (e4 ** 3 - e6 ** 2) / 1728.0
    eta = np.exp(2j * np.pi * tau / 24.0) * np.prod(1 - qn)
    j = 1728.0 * e4 ** 3 / (e4 ** 3 - e6 ** 2) if abs(e4 ** 3 - e6 ** 2) > 1e-300 else complex("nan")
    prod_delta = q * np.prod((1 - qn) ** 24)
    c = lambda z: [float(np.real(z)), float(np.imag(z))]
    return {"nome_abs": float(abs(q)), "E4": c(e4), "E6": c(e6), "Delta": c(delta), "eta": c(eta),
            "j": c(j), "delta_product_residual": float(abs(delta - prod_delta))}


def modular_harmonics(L: np.ndarray, tau: complex, tau_hat: np.ndarray) -> Dict[str, Any]:
    lam, psi = np.linalg.eigh(L)
    n = len(lam)
    q_pow = np.exp(2j * np.pi * tau * lam)
    c = np.zeros(n)
    c[1:] = tau_hat[1:n] * np.real(q_pow[1:])
    K = (psi * c) @ psi.T
    amp = np.sqrt((psi ** 2) @ (c ** 2))
    return {"lam": lam, "psi": psi, "c": c, "K": K, "amp": amp,
            "Z_abs": float(abs(np.sum(q_pow))), "energy": float(np.sum(c ** 2))}


# ───────────────────────────── QML connectome (statevector) ─────────────────────────────

_IDX = np.arange(2 ** N_QUBITS)
_FLIP = [_IDX ^ (1 << (N_QUBITS - 1 - k)) for k in range(N_QUBITS)]
_SGN = [np.where(((_IDX >> (N_QUBITS - 1 - k)) & 1) == 0, -1.0, 1.0) for k in range(N_QUBITS)]


def _ry_apply(state: np.ndarray, k: int, ang: np.ndarray) -> np.ndarray:
    """RY(ang) on qubit k, per-sample angles: out = cos(a/2) psi + sin(a/2) (sgn * psi[flip_k])."""
    c = np.cos(ang / 2)[:, None]
    s = np.sin(ang / 2)[:, None]
    return c * state + s * (_SGN[k][None, :] * state[:, _FLIP[k]])


def _cz_ring_signs() -> np.ndarray:
    idx = np.arange(2 ** N_QUBITS)
    bits = [(idx >> (N_QUBITS - 1 - k)) & 1 for k in range(N_QUBITS)]
    sign = np.ones_like(idx, dtype=float)
    for k in range(N_QUBITS):
        sign *= np.where(bits[k] & bits[(k + 1) % N_QUBITS] == 1, -1.0, 1.0)
    return sign


_CZ = _cz_ring_signs()


def quantum_states(features01: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Data re-uploading circuit; theta has shape (layers, qubits, 2)."""
    B = len(features01)
    state = np.zeros((B, 2 ** N_QUBITS))
    state[:, 0] = 1.0
    phi = np.pi * features01
    for layer in theta:
        for k in range(N_QUBITS):
            state = _ry_apply(state, k, layer[k, 0] + layer[k, 1] * phi[:, k])
        state = state * _CZ[None, :]
    return state


def fidelity_kernel(states: np.ndarray) -> np.ndarray:
    G = states @ states.T
    return G ** 2


def _centred_alignment(K: np.ndarray, T: np.ndarray) -> float:
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Tc = H @ K @ H, H @ T @ H
    return float(np.sum(Kc * Tc) / (np.linalg.norm(Kc) * np.linalg.norm(Tc) + 1e-30))


def train_qml_kernel(features01: np.ndarray, target: np.ndarray, layers: int, iters: int,
                     seed: int = 5) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    theta = np.stack([rng.normal(0.0, 0.1, (layers, N_QUBITS)),
                      1.0 + rng.normal(0.0, 0.1, (layers, N_QUBITS))], axis=-1)
    T = target.copy()
    np.fill_diagonal(T, 1.0)

    def score(th):
        return _centred_alignment(fidelity_kernel(quantum_states(features01, th)), T)

    init = score(theta)
    best_theta, best = theta.copy(), init
    history = [{"iter": 0, "alignment": init}]
    A_s, a, c = max(1, iters // 10), 0.5, 0.12
    for k in range(iters):
        ak = a / (k + 1 + A_s) ** 0.602
        ck = c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        gp, gm = score(theta + ck * delta), score(theta - ck * delta)
        theta = theta + ak * (gp - gm) / (2 * ck) * delta
        if (k + 1) % max(1, iters // 25) == 0 or k == iters - 1:
            cur = score(theta)
            history.append({"iter": k + 1, "alignment": cur})
            if cur > best:
                best, best_theta = cur, theta.copy()
    Kbest = fidelity_kernel(quantum_states(features01, best_theta))
    return {"K": Kbest, "alignment_initial": init, "alignment_final": best, "history": history,
            "num_params": int(theta.size)}


# ───────────────────────────── orchestration ─────────────────────────────

def _node_features(A: np.ndarray, coords: np.ndarray) -> np.ndarray:
    strength = A.sum(axis=1)
    clust = _onnela_clustering(A)
    vals, vecs = np.linalg.eigh(A)
    ev = np.abs(vecs[:, -1])
    c01 = (coords - coords.min(axis=0)) / np.ptp(coords, axis=0)
    return np.column_stack([strength, clust, ev, c01])


def _scale_with(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)


def _edge_list(A: np.ndarray, top: int) -> List[List[float]]:
    iu = np.triu_indices(len(A), 1)
    w = A[iu]
    order = np.argsort(-w)[:top]
    return [[int(iu[0][o]), int(iu[1][o]), round(float(w[o]), 4)] for o in order if w[o] > 0]


def _brain_surface(nu: int = 40, nv: int = 30) -> Dict[str, Any]:
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.05, np.pi - 0.05, nv)
    U, V = np.meshgrid(u, v)
    ripple = 1.0 + 0.035 * np.sin(5 * U) * np.sin(4 * V) + 0.02 * np.sin(9 * U + 2 * V)
    X = 74.0 * ripple * np.sin(V) * np.cos(U)
    Y = -18.0 + 98.0 * ripple * np.sin(V) * np.sin(U)
    Z = 18.0 + 66.0 * ripple * np.cos(V)
    Z = np.where(Z < -15.0, -15.0 + 0.15 * (Z + 15.0), Z)
    rr, cc = np.meshgrid(np.arange(nv - 1), np.arange(nu - 1), indexing="ij")
    v0 = (rr * nu + cc).ravel()
    v1, v2 = v0 + 1, v0 + nu
    v3 = v2 + 1
    faces = np.vstack([np.column_stack([v0, v1, v2]), np.column_stack([v1, v3, v2])])
    return {"x": np.round(X.ravel(), 1).tolist(), "y": np.round(Y.ravel(), 1).tolist(),
            "z": np.round(Z.ravel(), 1).tolist(), "i": faces[:, 0].tolist(),
            "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist()}


def _round(a: np.ndarray, nd: int = 4) -> List:
    return np.round(a, nd).tolist()


def _validate(severity, tau_x, tau_y, layers, iters, thr_pct):
    return (round(float(min(1.0, max(0.0, severity))), 3), round(float(min(1.0, max(0.0, tau_x))), 3),
            round(float(min(2.5, max(0.2, tau_y))), 3), int(min(5, max(1, layers))),
            int(min(300, max(10, iters))), round(float(min(95.0, max(50.0, thr_pct))), 1))


@lru_cache(maxsize=32)
def _cached(severity, tau_x, tau_y, layers, iters, thr_pct):
    parc = build_parcellation()
    names, coords = parc["names"], parc["coords"]
    n = len(names)
    Wh = healthy_connectome(parc)
    Wd, atrophy = dementia_connectome(Wh, parc, severity)

    nz = Wh[np.triu_indices(n, 1)]
    thr = float(np.percentile(nz, thr_pct))
    Ah, Ad = np.where(Wh >= thr, Wh, 0.0), np.where(Wd >= thr, Wd, 0.0)

    tau_hat_all = np.array([0.0] + [t / (k ** 5.5) for k, t in enumerate(ramanujan_tau(n)[1:], start=1)])
    tau = complex(tau_x, tau_y)
    Lh, Ld = _normalised_laplacian(Ah), _normalised_laplacian(Ad)
    Hh, Hd = modular_harmonics(Lh, tau, tau_hat_all), modular_harmonics(Ld, tau, tau_hat_all)
    inv_h = graph_invariants(Ah, (Hh["lam"], Hh["psi"]))
    inv_d = graph_invariants(Ad, (Hd["lam"], Hd["psi"]))

    Fh_raw, Fd_raw = _node_features(Ah, coords), _node_features(Ad, coords)
    Fh, Fd = _scale_with(Fh_raw, Fh_raw), _scale_with(Fh_raw, Fd_raw)
    qh = train_qml_kernel(Fh, Wh, layers, iters)
    qd = train_qml_kernel(Fd, Wd, layers, iters)
    Qh, Qd = qh["K"].copy(), qd["K"].copy()
    np.fill_diagonal(Qh, 0.0)
    np.fill_diagonal(Qd, 0.0)
    off = ~np.eye(n, dtype=bool)
    corr = lambda Q, W: float(np.corrcoef(Q[off], W[off])[0, 1])

    order = sorted(range(n), key=lambda i: (LOBE_ORDER.index(parc["lobes"][i]), parc["hemis"][i], names[i]))
    P = np.array(order)

    def ordered(M):
        return _round(M[np.ix_(P, P)], 3)

    strength_h, strength_d = Ah.sum(axis=1), Ad.sum(axis=1)
    nodes = []
    for i in range(n):
        base = names[i].split("_", 1)[1]
        loss = 0.0 if strength_h[i] == 0 else 100.0 * (1.0 - strength_d[i] / strength_h[i])
        nodes.append({
            "id": i, "name": names[i], "label": base, "hemisphere": parc["hemis"][i],
            "lobe": parc["lobes"][i], "braak_stage": {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}[int(parc["braak"][i])],
            "hub": base in HUBS,
            "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]),
            "atrophy_pct": round(100.0 * float(atrophy[i]), 1),
            "strength_healthy": round(float(strength_h[i]), 4),
            "strength_dementia": round(float(strength_d[i]), 4),
            "strength_loss_pct": round(loss, 1),
            "modular_amp_healthy": round(float(Hh["amp"][i]), 5),
            "modular_amp_dementia": round(float(Hd["amp"][i]), 5),
            "qml_strength_healthy": round(float(Qh[i].sum()), 3),
            "qml_strength_dementia": round(float(Qd[i].sum()), 3),
        })

    metrics = {
        "healthy": {**inv_h, "spectral_Z_abs": Hh["Z_abs"], "modular_energy": Hh["energy"],
                    "qml_alignment": qh["alignment_final"], "qml_vs_structural_corr": corr(Qh, Wh)},
        "dementia": {**inv_d, "spectral_Z_abs": Hd["Z_abs"], "modular_energy": Hd["energy"],
                     "qml_alignment": qd["alignment_final"], "qml_vs_structural_corr": corr(Qd, Wd)},
    }
    mf = modular_forms_at(tau)
    tau_list = ramanujan_tau(12)[1:]
    return {
        "status": "success",
        "model_notes": ("Synthetic stylised connectome on approximate Desikan-Killiany centroids; "
                        "Braak weights are illustrative. Not subject data and not a diagnostic."),
        "parameters": {"severity": severity, "cdr_label": cdr_label(severity), "tau": [tau_x, tau_y],
                       "qml_layers": layers, "qml_iters": iters, "threshold_pct": thr_pct,
                       "threshold_weight": round(thr, 4), "num_regions": n, "num_qubits": N_QUBITS},
        "nodes": nodes,
        "order": order,
        "edges_healthy": _edge_list(Ah, 260),
        "edges_dementia": _edge_list(Ad, 260),
        "surface": _brain_surface(),
        "matrices": {"order_names": [names[i] for i in order],
                     "order_lobes": [parc["lobes"][i] for i in order],
                     "W_healthy": ordered(Wh), "W_dementia": ordered(Wd),
                     "Q_healthy": ordered(Qh), "Q_dementia": ordered(Qd)},
        "metrics": metrics,
        "spectrum": {"mode": list(range(n)), "lambda_healthy": _round(Hh["lam"]),
                     "lambda_dementia": _round(Hd["lam"]), "c_healthy": _round(Hh["c"], 5),
                     "c_dementia": _round(Hd["c"], 5), "tau_hat": _round(tau_hat_all[:n], 5)},
        "modular_forms": {**mf, "tau_coefficients": [int(t) for t in tau_list]},
        "qml_training": {"healthy": qh["history"], "dementia": qd["history"],
                         "num_params": qh["num_params"]},
        "braak_summary": [{"stage": s, "regions": int(np.sum(parc["braak"] == k)),
                           "mean_atrophy_pct": round(100.0 * float(atrophy[parc["braak"] == k].mean()), 1)}
                          for k, s in {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}.items()],
    }


def simulate_connectome_qml(severity: float = 0.6, tau_x: float = 0.25, tau_y: float = 0.85,
                            qml_layers: int = 3, qml_iters: int = 80,
                            threshold_pct: float = 75.0) -> Dict[str, Any]:
    return _cached(*_validate(severity, tau_x, tau_y, qml_layers, qml_iters, threshold_pct))
