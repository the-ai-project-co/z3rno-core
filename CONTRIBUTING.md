# Contributing to z3rno-core

Thank you for your interest in contributing to Z3rno. This guide covers the development workflow for `z3rno-core`.

## Getting Started

1. Fork and clone the repository.
2. Install Python 3.11+ and [uv](https://docs.astral.sh/uv/).
3. Install dependencies:

```bash
uv sync --dev
```

4. Start a PostgreSQL instance with the required extensions (pgvector, Apache AGE). The easiest way is via the dev Docker Compose in `z3rno-server`.

5. Run the checks:

```bash
uv run ruff check .
uv run mypy .
uv run pytest
```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Write your code with tests.
3. Run linting, type checking, and tests (see above).
4. Commit using conventional commit messages.
5. Open a pull request against `main`.

## Code Style

This project uses **ruff** for linting and formatting. Configuration lives in `pyproject.toml`.

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy .
```

## Testing

Tests use `pytest` with `pytest-asyncio`. Integration tests require a running PostgreSQL container and are marked with `@pytest.mark.integration`.

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m "not integration"

# With coverage
uv run pytest --cov=src/z3rno_core
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding or updating tests
- `refactor:` code change that neither fixes a bug nor adds a feature
- `chore:` maintenance (deps, CI, tooling)

Examples:
- `feat: add graph traversal to recall pipeline`
- `fix: prevent duplicate audit entries on retry`

## Pull Request Process

1. Ensure all checks pass (`ruff check`, `mypy`, `pytest`).
2. Keep PRs focused -- one logical change per PR.
3. Update or add tests for any changed behavior.
4. Fill out the PR template description.
5. A maintainer will review and merge.

## Adding Migrations

If your change modifies the database schema:

```bash
uv run alembic revision --autogenerate -m "description of change"
```

Review the generated migration before committing. Migrations must be backward-compatible.

## Questions?

Open a [GitHub Discussion](https://github.com/the-ai-project-co/z3rno-core/discussions) or reach out at engineering@z3rno.dev.
