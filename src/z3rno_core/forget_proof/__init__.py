"""Phase F slice 5 — forget-with-proof.

Cryptographic receipt for ``forget()``. Build a Merkle tree over the
leaf hashes of every Memo touched by a forget(), sign the root with
an operator-controlled ed25519 key, and persist a row in
``forget_certificates``. An auditor later asks ``GET /v1/forget/{id}``,
runs the CLI verifier with the public key, and gets a yes/no on
whether the proof matches the audit chain.

Pure-Python builders here; the actual DB write happens in
``z3rno_core.engine.forget`` when ``FORGET_PROOF_ENABLED=true``.
"""

from __future__ import annotations

from z3rno_core.forget_proof.certificate import (
    ForgetCertificate,
    Leaf,
    SigningKeyMissingError,
    build_leaves,
    build_merkle_root,
    canonical_payload,
    load_signing_key,
    load_verifying_key,
    sign_certificate,
    verify_certificate,
)

__all__ = [
    "ForgetCertificate",
    "Leaf",
    "SigningKeyMissingError",
    "build_leaves",
    "build_merkle_root",
    "canonical_payload",
    "load_signing_key",
    "load_verifying_key",
    "sign_certificate",
    "verify_certificate",
]
