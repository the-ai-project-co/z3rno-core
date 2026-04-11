"""``api_keys`` table — hashed API keys for SDK authentication.

Separated from the ``tenants`` table so that:
  1. Keys can be rotated without touching tenant rows.
  2. Multiple keys can exist per tenant (e.g. one per environment).
  3. Revocation is a simple ``revoked_at = now()`` UPDATE.

The plaintext key is **never stored** — only the BCrypt hash. The key
prefix is stored separately for fast lookup (we hash the suffix and compare).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin


class ApiKey(Base, OrgScopedMixin, TimestampMixin):
    """A hashed API key associated with a tenant.

    Schema reference: Doc 02 Week 1 Tuesday F1 task (``api_keys`` table).

    Authentication flow:
      1. Client sends ``Authorization: Bearer z3rno_sk_live_<32-char-suffix>``.
      2. Server extracts the prefix (``z3rno_sk_live_``) and the suffix.
      3. Server looks up keys by prefix + bcrypt-compares the suffix to ``key_hash``.
      4. If match and not revoked and not expired, server retrieves ``org_id``.
      5. Server sets ``app.current_org_id = '<uuid>'`` for RLS.
    """

    __tablename__ = "api_keys"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    # Human-readable name for the dashboard ("CI server", "Local dev").
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # ``z3rno_sk_live_`` or ``z3rno_sk_test_`` — used for fast lookup before
    # the BCrypt comparison. Indexed.
    prefix: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # BCrypt hash of the key suffix. Never stored in plaintext.
    key_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        status = "revoked" if self.revoked_at else "active"
        return f"<ApiKey id={self.id} org_id={self.org_id} name={self.name!r} status={status}>"
