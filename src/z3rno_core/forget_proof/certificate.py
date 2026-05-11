"""Merkle tree + ed25519 signing for forget certificates.

Leaf format: ``sha256(memory_id || ":" || content_hash || ":" || audit_hash)``
where missing fields are replaced with the literal byte string ``-``
(so the leaf is well-defined even for Memos with no audit row, e.g.
in tests). Leaves are then sorted lexicographically before tree
construction so the root is deterministic regardless of the input
ordering — auditors recomputing from the cert's ``memory_ids`` list
land on the same root without needing a separate ordering hint.

Tree construction: standard balanced binary Merkle. For an odd
trailing node at any level, the node is duplicated (concatenated
with itself) — the convention used by Bitcoin / RFC 9162. We do
*not* take the empty-tree case; callers must hand in ≥ 1 leaf
(matched by the ``ck_forget_certificates_nonempty`` constraint).

Signing: ed25519 over ``canonical_payload(...)`` — a sorted-keys
JSON encoding of the cert metadata + base64'd merkle root. The
verifier rebuilds the exact same byte string from the cert row
without needing the original Memos.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class SigningKeyMissingError(Exception):
    """Raised when ``FORGET_PROOF_ENABLED=true`` but the key file is
    absent or unreadable. The engine treats this as a hard error so a
    misconfigured deploy can't silently emit unsigned certs."""


@dataclass(frozen=True)
class Leaf:
    """One Merkle-tree leaf — the auditable fact about one Memo."""

    memory_id: UUID
    content_hash: str = ""
    audit_entry_hash: str = ""

    def digest(self) -> bytes:
        parts = (
            str(self.memory_id),
            self.content_hash or "-",
            self.audit_entry_hash or "-",
        )
        return hashlib.sha256(":".join(parts).encode("utf-8")).digest()


@dataclass(frozen=True)
class ForgetCertificate:
    """In-memory shape of a row about to be (or already) persisted."""

    cert_id: UUID
    org_id: UUID
    memory_ids: tuple[UUID, ...]
    merkle_root: bytes
    signer_key_id: str
    signed_at: datetime
    hard_delete: bool = False
    audit_seq_start: int | None = None
    audit_seq_end: int | None = None
    signature: bytes = b""
    agent_id: UUID | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Merkle construction
# ---------------------------------------------------------------------------


def build_leaves(
    *,
    memory_ids: list[UUID] | tuple[UUID, ...],
    content_hashes: dict[UUID, str] | None = None,
    audit_hashes: dict[UUID, str] | None = None,
) -> list[Leaf]:
    """Build leaves for the supplied Memo IDs.

    Missing content / audit hashes are tolerated and replaced with the
    placeholder ``-`` so the leaf hash is still well-defined.
    """
    chash = content_hashes or {}
    ahash = audit_hashes or {}
    return [
        Leaf(
            memory_id=mid,
            content_hash=chash.get(mid, ""),
            audit_entry_hash=ahash.get(mid, ""),
        )
        for mid in memory_ids
    ]


def build_merkle_root(leaves: list[Leaf]) -> bytes:
    """Compute the Merkle root over ``leaves``.

    Leaves are sorted by their digest so callers don't have to preserve
    the original Memo ordering — auditors recomputing from cert data
    land on the same root.
    """
    if not leaves:
        raise ValueError("Merkle tree requires at least one leaf")

    level: list[bytes] = sorted(leaf.digest() for leaf in leaves)
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
    return level[0]


# ---------------------------------------------------------------------------
# Canonical payload + signing
# ---------------------------------------------------------------------------


def canonical_payload(cert: ForgetCertificate) -> bytes:
    """The exact bytes signed by ed25519. Verifier rebuilds this.

    Stable sorted JSON over the field set — any drift in keys/order
    breaks verification, which is the point.
    """
    body = {
        "cert_id": str(cert.cert_id),
        "org_id": str(cert.org_id),
        "memory_ids": sorted(str(m) for m in cert.memory_ids),
        "merkle_root": base64.b64encode(cert.merkle_root).decode("ascii"),
        "signer_key_id": cert.signer_key_id,
        "signed_at": cert.signed_at.isoformat(),
        "hard_delete": cert.hard_delete,
        "audit_seq_start": cert.audit_seq_start,
        "audit_seq_end": cert.audit_seq_end,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_certificate(cert: ForgetCertificate, signing_key: Ed25519PrivateKey) -> bytes:
    return signing_key.sign(canonical_payload(cert))


def verify_certificate(
    cert: ForgetCertificate,
    verifying_key: Ed25519PublicKey,
) -> bool:
    try:
        verifying_key.verify(cert.signature, canonical_payload(cert))
    except InvalidSignature:
        return False
    return True


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------


def load_signing_key(path: str | Path) -> Ed25519PrivateKey:
    """Load an unencrypted PEM-encoded ed25519 private key.

    Operators stage this on disk (mounted secret in K8s, sidecar in
    Modal, etc.) and point ``FORGET_PROOF_SIGNING_KEY_PATH`` at it.
    Unencrypted on purpose — the engine needs to sign without a
    passphrase prompt. Protect at the filesystem layer instead.
    """
    p = Path(path)
    if not p.exists():
        raise SigningKeyMissingError(f"signing key not found at {p}")
    data = p.read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningKeyMissingError(f"signing key at {p} is not an Ed25519 private key")
    return key


def load_verifying_key(path: str | Path) -> Ed25519PublicKey:
    """Load a PEM-encoded ed25519 public key. Used by the verifier CLI."""
    p = Path(path)
    if not p.exists():
        raise SigningKeyMissingError(f"verifying key not found at {p}")
    data = p.read_bytes()
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise SigningKeyMissingError(f"verifying key at {p} is not an Ed25519 public key")
    return key
