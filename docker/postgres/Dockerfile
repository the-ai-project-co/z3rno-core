# syntax=docker/dockerfile:1.7
#
# z3rno-postgres — PostgreSQL 17 + extensions required by Z3rno
#
# Bundled extensions:
#   - pgvector 0.8+        (vector similarity search)    apt: postgresql-17-pgvector
#   - pgvectorscale        (DiskANN index)               binary from Timescale release
#   - Apache AGE           (openCypher graph)            compiled from source (master)
#   - pg_cron 1.6+         (in-db scheduler)             apt: postgresql-17-cron
#   - pgaudit              (audit logging)               apt: postgresql-17-pgaudit
#   - pgcrypto             (built-in contrib)            included with postgres-17-contrib
#
# Published as: ghcr.io/the-ai-project-co/z3rno-postgres:17
# Used by: z3rno-server docker-compose, Helm charts (via z3rno-helm), CloudNativePG cluster
#
# Apache AGE PG17 support lives on the `PG17` branch (not `master`, which is
# tracking PG18). Latest release candidate as of Apr 2026: PG17/v1.7.0-rc0.
# We compile from the `PG17` branch head for now; pin to a stable tag once
# AGE ships 1.7.0 GA for PG17.

FROM postgres:17-bookworm

LABEL org.opencontainers.image.title="z3rno-postgres" \
      org.opencontainers.image.description="PostgreSQL 17 + pgvector + pgvectorscale + Apache AGE + pg_cron + pgaudit, pre-configured for the Z3rno memory database" \
      org.opencontainers.image.source="https://github.com/the-ai-project-co/z3rno-core" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="The AI Project Co."

ARG AGE_REF=PG17
ARG PGVECTORSCALE_VERSION=0.5.1

# ---------------------------------------------------------------------------
# 1. Install runtime extensions from the PGDG apt repo (already configured in
#    the official postgres image) + build deps needed to compile AGE from src.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        postgresql-server-dev-17 \
        libreadline-dev \
        zlib1g-dev \
        flex \
        bison \
        postgresql-17-pgvector \
        postgresql-17-cron \
        postgresql-17-pgaudit \
    ; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Compile Apache AGE from the PG17 branch (see note at top of file).
# ---------------------------------------------------------------------------
RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 --branch "${AGE_REF}" https://github.com/apache/age.git; \
    cd age; \
    make PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config -j"$(nproc)"; \
    make install PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config; \
    cd /; \
    rm -rf /tmp/age

# ---------------------------------------------------------------------------
# 3. Install pgvectorscale from the Timescale pre-built debian package.
#    Falls back to building from source if the pre-built .deb isn't available
#    for this PG version (rare — Timescale ships PG17 binaries).
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    curl -fSL -o /tmp/pgvectorscale.deb \
        "https://github.com/timescale/pgvectorscale/releases/download/${PGVECTORSCALE_VERSION}/pgvectorscale-${PGVECTORSCALE_VERSION}-pg17-${ARCH}.deb" \
        || { echo "pgvectorscale binary not available for PG17/${ARCH}, building from source..."; \
             apt-get update && apt-get install -y --no-install-recommends rustc cargo clang libclang-dev pkg-config; \
             cd /tmp && git clone --depth 1 --branch "${PGVECTORSCALE_VERSION}" https://github.com/timescale/pgvectorscale.git; \
             cd pgvectorscale/pgvectorscale; \
             cargo install --locked cargo-pgrx --version "$(cargo pgrx --version 2>/dev/null || echo 0.12.5)" || true; \
             cargo pgrx init --pg17 /usr/lib/postgresql/17/bin/pg_config; \
             cargo pgrx install --release --pg-config /usr/lib/postgresql/17/bin/pg_config; \
             cd / && rm -rf /tmp/pgvectorscale; \
             exit 0; } ; \
    dpkg -i /tmp/pgvectorscale.deb; \
    rm /tmp/pgvectorscale.deb

# ---------------------------------------------------------------------------
# 4. Strip build deps to slim the image. Keeps apt-installed runtime libs.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get purge -y --auto-remove \
        build-essential \
        git \
        curl \
        postgresql-server-dev-17 \
        libreadline-dev \
        zlib1g-dev \
        flex \
        bison \
    ; \
    rm -rf /var/lib/apt/lists/* /tmp/*

# ---------------------------------------------------------------------------
# 5. initdb hooks — set shared_preload_libraries + pg_cron/pgaudit config via
#    ALTER SYSTEM on first boot. The postgres entrypoint stops and restarts
#    the server after initdb scripts run, so the preload-time settings take
#    effect cleanly. See docker-entrypoint-initdb.d/00-z3rno-preload.sh for
#    the full explanation.
#
#    Extension CREATE statements are deliberately NOT here — that is the job
#    of Alembic migration 001_create_extensions.py in z3rno-core, which runs
#    against the live database on first application startup.
# ---------------------------------------------------------------------------
COPY docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/*.sh

EXPOSE 5432

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" || exit 1
