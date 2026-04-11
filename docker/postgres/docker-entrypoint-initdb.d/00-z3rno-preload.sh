#!/usr/bin/env bash
#
# 00-z3rno-preload.sh
#
# Runs once during initdb on a fresh PostgreSQL data directory. Configures:
#   - shared_preload_libraries for pgvectorscale, age, pg_cron, pgaudit
#   - pg_cron database target (points to the main POSTGRES_DB)
#   - pgaudit log settings (DDL only by default — tune per-tenant later)
#
# Why this is a .sh and not a .sql:
#   - We need to set cron.database_name BEFORE the cron extension is created
#   - ALTER SYSTEM writes to postgresql.auto.conf which is re-read on restart,
#     but shared_preload_libraries requires a full restart to take effect.
#     The postgres:17 Docker entrypoint does stop-and-restart after running
#     initdb scripts, so this timing works cleanly.
#   - psql -v ON_ERROR_STOP=1 makes any failure here abort the container boot.
#
# Notes:
#   - This file does NOT create the extensions themselves. That is the job of
#     Alembic migration 001_create_extensions.py in z3rno-core, which runs
#     against the live database on first application startup.

set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<-EOSQL
    -- shared_preload_libraries must include all extensions that need to hook
    -- into Postgres startup. Order matters for some extensions; pgvectorscale
    -- must come before age so its planner hooks register first.
    ALTER SYSTEM SET shared_preload_libraries = 'vectorscale,age,pg_cron,pgaudit';

    -- pg_cron must target exactly one database. We point it at the main app DB
    -- so z3rno-server can schedule recurring jobs (TTL expiry, decay, etc.).
    ALTER SYSTEM SET cron.database_name = '${POSTGRES_DB}';

    -- pgaudit: log DDL events by default. Applications can dial this up for
    -- compliance-sensitive tenants (e.g., pgaudit.log = 'write, ddl, role').
    ALTER SYSTEM SET pgaudit.log = 'ddl';
    ALTER SYSTEM SET pgaudit.log_catalog = 'off';
    ALTER SYSTEM SET pgaudit.log_client = 'on';
    ALTER SYSTEM SET pgaudit.log_parameter = 'on';

    -- Enable statistics collector for pg_stat_statements (already loaded by default
    -- in postgres:17, but make sure tracking is enabled).
    ALTER SYSTEM SET track_activities = 'on';
    ALTER SYSTEM SET track_counts = 'on';
EOSQL

echo "z3rno: shared_preload_libraries and pg_cron/pgaudit config written to postgresql.auto.conf"
echo "z3rno: postgres will restart after this initdb step and pick up the new config"
