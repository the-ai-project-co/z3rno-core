# z3rno-core — developer Makefile
#
# Alembic-specific targets for the z3rno-core memory engine library. Run
# these against a running postgres (typically launched via `make dev-up`
# in the z3rno-server repo next door).
#
# Most common workflow:
#   cd ../z3rno-server && make dev-up   # start postgres (in the other repo)
#   cd ../z3rno-core && make migrate    # apply all pending migrations
#   make migrate-status                 # show current revision
#   make migrate-new name=add_my_table  # generate a new migration skeleton

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

BOLD := \033[1m
DIM := \033[2m
RESET := \033[0m

.DEFAULT_GOAL := help

## help                 Show this help message
.PHONY: help
help:
	@printf "$(BOLD)z3rno-core — developer Makefile$(RESET)\n\n"
	@printf "$(BOLD)Usage:$(RESET) make $(DIM)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^## / {sub(/^## /, ""); print "  " $$0}' $(MAKEFILE_LIST)

# =============================================================================
# Alembic migrations (skeleton-mode until alembic.ini and models land)
# =============================================================================
#
# DATABASE_URL defaults to the local dev postgres brought up by the
# z3rno-server docker-compose stack. Override via env var if needed:
#   DATABASE_URL=postgresql+asyncpg://... make migrate
#
DATABASE_URL ?= postgresql+psycopg://z3rno:z3rno_dev_password@localhost:5432/z3rno

## migrate              Apply all pending Alembic migrations (upgrade head)
.PHONY: migrate
migrate:
	DATABASE_URL="$(DATABASE_URL)" uv run alembic upgrade head

## migrate-down         Roll back the most recent Alembic migration
.PHONY: migrate-down
migrate-down:
	DATABASE_URL="$(DATABASE_URL)" uv run alembic downgrade -1

## migrate-status       Show the current Alembic revision on the live database
.PHONY: migrate-status
migrate-status:
	DATABASE_URL="$(DATABASE_URL)" uv run alembic current

## migrate-history      Show the full Alembic migration history
.PHONY: migrate-history
migrate-history:
	DATABASE_URL="$(DATABASE_URL)" uv run alembic history --verbose

## migrate-new          Generate a new Alembic migration (usage: make migrate-new name=add_my_table)
.PHONY: migrate-new
migrate-new:
	@if [ -z "$(name)" ]; then \
		echo "ERROR: 'name' is required. Usage: make migrate-new name=add_my_table"; \
		exit 1; \
	fi
	DATABASE_URL="$(DATABASE_URL)" uv run alembic revision --autogenerate -m "$(name)"

# =============================================================================
# Python developer targets (skeleton-mode until pyproject.toml lands)
# =============================================================================

## seed                 Load dev seed data (2 tenants, 500 memories, 1000 audit entries)
.PHONY: seed
seed:
	DATABASE_URL="$(DATABASE_URL)" uv run python -m seeds.dev_seed

## test                 Run pytest (skeleton-mode until pyproject.toml lands)
.PHONY: test
test:
	@if [ -f pyproject.toml ]; then \
		uv run pytest -v; \
	else \
		echo "skeleton mode: no pyproject.toml yet"; \
	fi

## lint                 Run ruff check + mypy (skeleton-mode until pyproject.toml lands)
.PHONY: lint
lint:
	@if [ -f pyproject.toml ]; then \
		uv run ruff check . && uv run mypy .; \
	else \
		echo "skeleton mode: no pyproject.toml yet"; \
	fi

## format               Run ruff format (skeleton-mode until pyproject.toml lands)
.PHONY: format
format:
	@if [ -f pyproject.toml ]; then \
		uv run ruff format .; \
	else \
		echo "skeleton mode: no pyproject.toml yet"; \
	fi

## install              Install Python dependencies via uv sync
.PHONY: install
install:
	@if [ -f pyproject.toml ]; then \
		uv sync --all-extras --dev; \
	else \
		echo "skeleton mode: no pyproject.toml yet"; \
	fi

# =============================================================================
# Docker image (z3rno-postgres)
# =============================================================================

## image-build          Build the z3rno-postgres Docker image locally (tag :dev)
.PHONY: image-build
image-build:
	@cd docker/postgres && ./build.sh --tag dev

## image-push           Build and push the z3rno-postgres image to GHCR (tag :17)
.PHONY: image-push
image-push:
	@cd docker/postgres && ./build.sh --tag 17 --push
