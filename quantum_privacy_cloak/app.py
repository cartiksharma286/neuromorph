"""Quantum privacy cloak research prototype.

The QML and cloak-field calculations are simulations for interface and workflow
validation. The cryptographic demo uses standard-library primitives and is not
a substitute for a reviewed ML-KEM/ML-DSA implementation.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def recurrent_primes(seed: int, count: int = 8) -> list[int]:
    """Generate a deterministic, auditable prime schedule from a seed."""
    candidate = max(3, int(seed) | 1)
    values = []
    while len(values) < count:
        if is_prime(candidate):
            values.append(candidate)
        candidate += 2
    return values


def digest_label(value: str) -> str:
    return hashlib.sha3_256(value.encode()).hexdigest()[:16].upper()


def simulate_cloak(radius: float, layers: int, attenuation: float) -> dict:
    points = []
    samples = 48
    for index in range(samples):
        angle = 2 * math.pi * index / samples
        radial = radius * (1 + 0.08 * math.sin(layers * angle))
        field = math.exp(-attenuation * abs(math.cos(layers * angle)))
        points.append({"x": round(radial * math.cos(angle), 4), "y": round(radial * math.sin(angle), 4), "field": round(field, 4)})
    convergence = []
    loss = 1.0 + attenuation * 0.35
    for iteration in range(1, 21):
        loss *= 0.78 + (layers / 1000)
        convergence.append({"iteration": iteration, "loss": round(loss, 5), "visibility": round(100 - (1 - loss / (1 + attenuation * 0.35)) * 100, 2)})
    return {
        "points": points,
        "visibility": round(max(0.1, 100 - attenuation * 72 - layers * 1.7), 1),
        "scattering": round(max(0.2, attenuation * 10 + layers * 0.8), 2),
        "qml_confidence": round(min(99.9, 86 + layers * 1.1 + attenuation * 3.4), 1),
        "convergence": convergence,
        "cloak_characteristics": {
            "cloak_gain_db": round(layers * attenuation * 1.85, 2),
            "field_stability": round(100 - attenuation * 18 + layers * 0.4, 1),
            "prime_scale_factor": round(sum(1 / prime for prime in recurrent_primes(1009, layers)) * 100, 3),
        },
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    radius = float(payload.get("radius", 1.0))
    layers = max(2, min(12, int(payload.get("layers", 6))))
    attenuation = max(0.1, min(1.0, float(payload.get("attenuation", 0.65))))
    seed = max(3, int(payload.get("seed", 1009)))
    primes = recurrent_primes(seed, 8)
    cloak = simulate_cloak(radius, layers, attenuation)
    return jsonify({
        **cloak,
        "prime_schedule": primes,
        "schedule_id": digest_label("-".join(map(str, primes))),
        "mode": "simulated quantum machine-learning field optimization",
        "privacy_characteristics": {
            "data_minimization": "local-only inputs",
            "rotation_period": f"{max(5, primes[0] % 97)} min",
            "schedule_entropy_bits": round(math.log2(primes[-1]), 2),
        },
    })


@app.post("/api/session")
def session():
    payload = request.get_json(silent=True) or {}
    subject = str(payload.get("subject", "canadian-research-session"))[:120]
    seed = int(payload.get("seed", 1009))
    primes = recurrent_primes(seed, 8)
    nonce = secrets.token_hex(16)
    transcript = f"{subject}|{nonce}|{','.join(map(str, primes))}"
    # Demonstration transcript binding only; use a vetted PQC library in production.
    commitment = hmac.new(primes[0].to_bytes(8, "big"), transcript.encode(), hashlib.sha3_256).hexdigest()
    return jsonify({
        "session_id": f"NCL-{digest_label(nonce)}",
        "commitment": commitment[:32].upper(),
        "key_exchange": "ML-KEM-768 interface placeholder",
        "signature": "ML-DSA-65 interface placeholder",
        "privacy_basis": "PIPEDA + provincial privacy review required",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prime_schedule": primes,
        "session_characteristics": {
            "prime_count": len(primes),
            "rotation_period": f"{max(5, primes[0] % 97)} min",
            "transcript_bound": True,
            "data_residency": "local sandbox",
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7900, debug=False)
