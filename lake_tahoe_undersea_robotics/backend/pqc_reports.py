"""Local report-signing demonstrator with process-pinned ephemeral keys."""

import base64
import hashlib
import hmac
import importlib.util
from pathlib import Path

import rfc8785
from pqcrypto import InvalidSignatureError
from pqcrypto.kem import ml_kem_768
from pqcrypto.sign import ml_dsa_65


DOMAIN = b"ONTARIO-CARBON-REPORT-V1\x00"
CLOAK_SOURCE = Path(__file__).resolve().parents[2] / "quantum_privacy_cloak" / "number_theoretic_invisibility.py"
spec = importlib.util.spec_from_file_location("ontario_cloak_number_theory", CLOAK_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load Privacy Cloak helper: {CLOAK_SOURCE}")
cloak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloak)


def canonical_bytes(report: dict) -> bytes:
    if not isinstance(report, dict):
        raise ValueError("report must be an object.")
    encoded = rfc8785.dumps(report)
    if len(encoded) > 65536:
        raise ValueError("Report exceeds 64 KiB.")
    return DOMAIN + encoded


class ReportSigner:
    def __init__(self) -> None:
        self.public_key, self._secret_key = ml_dsa_65.keygen()
        self.key_id = hashlib.sha3_256(self.public_key).hexdigest()

    def sign(self, report: dict) -> dict:
        message = canonical_bytes(report)
        signature = ml_dsa_65.sign(self._secret_key, message)
        return {
            "report": report, "algorithm": "ML-DSA-65", "key_id": self.key_id,
            "canonicalization": "RFC 8785",
            "public_key_b64": base64.b64encode(self.public_key).decode("ascii"),
            "signature_b64": base64.b64encode(signature).decode("ascii"),
            "message_sha3_256": hashlib.sha3_256(message).hexdigest(),
        }

    def verify(self, envelope: dict) -> bool:
        if not isinstance(envelope, dict):
            raise ValueError("Envelope must be an object.")
        if (envelope.get("key_id") != self.key_id
                or envelope.get("algorithm") != "ML-DSA-65"
                or envelope.get("canonicalization") != "RFC 8785"):
            raise ValueError("Unknown signing key or algorithm; keys expire on server restart.")
        raw = envelope.get("signature_b64")
        if not isinstance(raw, str) or len(raw) > 10000:
            raise ValueError("signature_b64 must be a bounded base64 string.")
        signature = base64.b64decode(raw, validate=True)
        message = canonical_bytes(envelope.get("report"))
        try:
            ml_dsa_65.verify(self.public_key, message, signature)
        except InvalidSignatureError:
            return False
        return True

    def self_test(self, report: dict) -> dict:
        envelope = self.sign(report)
        valid = self.verify(envelope)
        changed = dict(envelope)
        changed["report"] = {**report, "tamper_marker": True}
        rejected = not self.verify(changed)
        public, secret = ml_kem_768.keygen()
        ciphertext, shared = ml_kem_768.encaps(public)
        agreed = hmac.compare_digest(shared, ml_kem_768.decaps(secret, ciphertext))
        altered = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        kem_tamper_rejected = not hmac.compare_digest(shared, ml_kem_768.decaps(secret, altered))
        if not (valid and rejected and agreed and kem_tamper_rejected):
            raise RuntimeError("Post-quantum self-test failed.")
        return {
            "signature_valid": valid, "altered_report_rejected": rejected,
            "kem_shared_secret_agreed": agreed,
            "altered_ciphertext_secret_mismatch": kem_tamper_rejected,
            "signature_bytes": len(base64.b64decode(envelope["signature_b64"])),
            "public_key_bytes": len(self.public_key), "ciphertext_bytes": len(ciphertext),
            "key_id": self.key_id, "provider": "pqcrypto 1.0.0",
            "algorithms": ["ML-DSA-65", "ML-KEM-768"],
            "cloak_integration": {
                "source": "quantum_privacy_cloak/number_theoretic_invisibility.py",
                "ramanujan_c5_1": cloak.compute_ramanujan_sum(5, 1),
                "prime_3329": cloak.is_prime(3329),
                "scope": "Reused arithmetic helper only; Cloak's simulated security scores are not evidence.",
            },
            "limitations": [
                "Local functional tests only; not FIPS module certification or a security audit.",
                "Ephemeral process-pinned report key; no external identity or durable trust anchor.",
                "HTTP remains unencrypted; no PQ TLS, authentication, access control or replay protection.",
                "KEM test does not encrypt reports or install a secure transport.",
            ],
        }
