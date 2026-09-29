.PHONY: help install install-dev update lock format lint type-check test test-cov clean clean-all run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	poetry install --no-dev

install-dev: ## Install project with dev dependencies
	poetry install

update: ## Update dependencies
	poetry update

lock: ## Lock dependencies
	poetry lock

format: ## Format code with black
	poetry run black src tests

lint: ## Lint code with ruff
	poetry run ruff check src tests
	poetry run black --check src tests

lint-fix: ## Auto-fix linting issues
	poetry run ruff check --fix src tests
	poetry run black src tests

type-check: ## Run type checking with mypy
	poetry run mypy src

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=src --cov-report=term-missing --cov-report=html

test-watch: ## Run tests in watch mode
	poetry run pytest-watch

clean: ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +

clean-all: clean ## Remove all generated files including virtualenv
	rm -rf .venv
	poetry env remove --all

setup: install-dev ## Initial setup: install dev dependencies
	poetry run pre-commit install

check: format lint type-check test ## Run all checks (format, lint, type-check, test)

ci: lint type-check test-cov ## Run CI pipeline checks

shell: ## Open poetry shell
	poetry shell

run: ## Run the main script
	poetry run python -m src.main

.DEFAULT_GOAL := help

