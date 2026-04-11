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
ARG PGVECTORSCALE_VERSION=0.9.0

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
        unzip \
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
# 3. Install pgvectorscale from the Timescale pre-built release zip.
#    Timescale publishes zipped shared objects (not .deb) per PG/arch combo:
#      https://github.com/timescale/pgvectorscale/releases
#    Each zip contains vectorscale.so + vectorscale--*.sql + vectorscale.control.
#    We extract them into the standard PostgreSQL extension directories.
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    mkdir -p /tmp/pvs; \
    curl -fSL -o /tmp/pvs.zip \
        "https://github.com/timescale/pgvectorscale/releases/download/${PGVECTORSCALE_VERSION}/pgvectorscale-${PGVECTORSCALE_VERSION}-pg17-${ARCH}.zip"; \
    unzip -o /tmp/pvs.zip -d /tmp/pvs; \
    find /tmp/pvs -name "*.so" -exec cp -v {} /usr/lib/postgresql/17/lib/ \; ; \
    find /tmp/pvs -name "*.sql" -exec cp -v {} /usr/share/postgresql/17/extension/ \; ; \
    find /tmp/pvs -name "*.control" -exec cp -v {} /usr/share/postgresql/17/extension/ \; ; \
    rm -rf /tmp/pvs /tmp/pvs.zip

# ---------------------------------------------------------------------------
# 4. Strip build deps to slim the image. Keeps apt-installed runtime libs.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get purge -y --auto-remove \
        build-essential \
        git \
        curl \
        unzip \
        postgresql-server-dev-17 \
        libreadline-dev \
        zlib1g-dev \
        flex \
        bison \
    ; \
    rm -rf /var/lib/apt/lists/* /tmp/*

# ---------------------------------------------------------------------------
# 5. Set shared_preload_libraries via sed on postgresql.conf.sample.
#
#    Why not ALTER SYSTEM?
#    ALTER SYSTEM writes 'vectorscale,age,pg_cron,pgaudit' as a single quoted
#    string to postgresql.auto.conf, and the startup parser treats the whole
#    string as one library name — causing FATAL: could not access file
#    "vectorscale,age,pg_cron,pgaudit". The postgresql.conf.sample format
#    (main config, not auto.conf) parses comma-separated stringlists correctly.
#    So we use the build-time sed for the list, and ALTER SYSTEM (via the
#    initdb hook below) for single-value settings only.
# ---------------------------------------------------------------------------
RUN set -eux; \
    sed -ri \
        "s|^#?shared_preload_libraries = .*$|shared_preload_libraries = 'vectorscale,age,pg_cron,pgaudit'|" \
        /usr/share/postgresql/postgresql.conf.sample

# ---------------------------------------------------------------------------
# 6. initdb hooks — single-value pg_cron/pgaudit settings via ALTER SYSTEM.
#    Runs once on a fresh data directory; the postgres entrypoint restarts
#    the cluster afterwards so preload-time settings take effect.
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
