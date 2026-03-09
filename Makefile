.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
APP_MODULE   := app/main.py
PORT         := 8000

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install all dependencies (including dev) via uv
	uv sync

.PHONY: dev
dev: ## Start dev server with hot-reload
	fastapi dev $(APP_MODULE) --port $(PORT)

.PHONY: run
run: ## Start production server
	fastapi run $(APP_MODULE) --host 0.0.0.0 --port $(PORT)

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

.PHONY: up
up: ## Start Postgres, Redis, FalkorDB containers
	docker compose up -d postgres redis falkordb

.PHONY: down
down: ## Stop all containers
	docker compose down

.PHONY: up-all
up-all: ## Start all containers including the app
	docker compose up -d

.PHONY: logs
logs: ## Tail docker compose logs
	docker compose logs -f

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

.PHONY: migrate
migrate: ## Run Alembic migrations to head
	alembic upgrade head

.PHONY: migration
migration: ## Auto-generate a new migration (usage: make migration msg="add foo table")
	alembic revision --autogenerate -m "$(msg)"

.PHONY: migrate-down
migrate-down: ## Downgrade one migration revision
	alembic downgrade -1

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run test suite
	pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest --cov=app --cov-report=term-missing

.PHONY: lint
lint: ## Run linter (ruff check)
	ruff check .

.PHONY: format
format: ## Auto-format code (ruff)
	ruff format .
	ruff check --fix .

.PHONY: typecheck
typecheck: ## Run mypy type checking
	mypy app/

.PHONY: check
check: lint typecheck test ## Run lint + typecheck + tests

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist *.egg-info

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
