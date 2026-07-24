# Athlete Injury Risk Detection — common tasks.
#
# This file is the single description of "how to run this project". CI and the
# Docker image call these targets rather than re-encoding the pipeline themselves,
# so there is one definition to keep correct instead of three.

.DEFAULT_GOAL := help
.PHONY: help setup data train tune benchmark shap pipeline run docker-build docker-up serve test smoke notebooks audit hooks lint format check lock clean

PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- environment ----------------------------------------------------------- #

setup:  ## Install the package with its dev extras
	$(PY) -m pip install -e ".[dev]"

lock:  ## Regenerate the pinned dependency locks from pyproject
	$(PY) -m piptools compile --quiet --strip-extras --extra dashboard --extra api -o requirements.lock pyproject.toml
	$(PY) -m piptools compile --quiet --strip-extras --extra dev -o requirements-dev.lock pyproject.toml

# --- pipeline -------------------------------------------------------------- #

data:  ## Generate the synthetic dataset (injury events included)
	injury-risk data

train:  ## Train the models and write the metrics report
	injury-risk train

tune:  ## Search hyperparameters
	injury-risk tune

benchmark:  ## Compare the baselines under one protocol
	injury-risk benchmark

shap:  ## Generate the SHAP plots
	injury-risk shap --track synthetic

pipeline: data train shap  ## Run the whole pipeline end to end

run:  ## Launch the Streamlit dashboard
	injury-risk dashboard

docker-build:  ## Build the container image (trains a model in)
	docker build -t injury-risk:local .

docker-up:  ## Run the API + dashboard together (docker compose)
	docker compose up --build

serve:  ## Serve the REST API (docs at /docs)
	injury-risk serve --reload

# --- quality --------------------------------------------------------------- #

test:  ## Run the test suite with its coverage floor
	pytest

smoke:  ## Guard model quality (fails if the metrics regress)
	injury-risk smoke-test

notebooks:  ## Execute the notebooks (as CI does)
	jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=600 notebooks/*.ipynb > /dev/null

audit:  ## Scan pinned dependencies for known vulnerabilities
	pip-audit --requirement requirements.lock --strict

hooks:  ## Install the pre-commit hooks
	pre-commit install

lint:  ## Lint and type-check
	ruff check .
	mypy

format:  ## Format the code
	black .
	ruff check --fix .

check: lint test  ## Everything CI runs

clean:  ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
