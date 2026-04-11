#!/usr/bin/env bash
#
# 00-z3rno-preload.sh
#
# Runs once during initdb on a fresh PostgreSQL data directory. Configures
# the single-value runtime settings for pg_cron and pgaudit via ALTER SYSTEM.
#
# Not handled here:
#   - shared_preload_libraries — set at build time via sed on
#     postgresql.conf.sample in the Dockerfile. ALTER SYSTEM does not parse
#     comma-separated stringlists correctly (it writes a single quoted string
#     to postgresql.auto.conf and the startup parser treats the whole thing
#     as one library name, causing FATAL at boot). Using postgresql.conf
#     directly sidesteps the quoting issue entirely.
#
# Why this is a .sh and not a .sql:
#   - Shell interpolation lets us bind cron.database_name to the runtime
#     POSTGRES_DB env var, so the image works with any database name.
#   - psql -v ON_ERROR_STOP=1 makes any SQL failure here abort container boot.
#
# Restart timing:
#   - ALTER SYSTEM writes to postgresql.auto.conf.
#   - The postgres entrypoint stops the cluster after all initdb scripts run,
#     then starts it again. On the restart, the postmaster reads both
#     postgresql.conf (for shared_preload_libraries) and postgresql.auto.conf
#     (for cron.database_name and pgaudit settings) and applies everything.
#
# Not handled here (separation of concerns):
#   - CREATE EXTENSION statements. Those are the job of Alembic migration
#     001_create_extensions.py in z3rno-core, which runs against the live
#     database on first application startup.

set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<-EOSQL
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

echo "z3rno: cron.database_name and pgaudit config written to postgresql.auto.conf"
echo "z3rno: shared_preload_libraries was already baked into postgresql.conf at image build time"
echo "z3rno: postgres will restart after this initdb step and pick up the full config"
