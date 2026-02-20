# MHWF-PIKAN MRI Reconstruction Project Makefile
.PHONY: help install install-dev install-gpu format lint test test-cov clean docker-build docker-run docs

# Default Python interpreter
PYTHON = python
VENV_DIR = .venv
VENV_BIN = $(VENV_DIR)/Scripts
SRC_DIR = src/mhwf_pikan
TEST_DIR = tests

# Help target
help:
	@echo "MHWF-PIKAN MRI Reconstruction Project"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install package with base dependencies"
	@echo "  install-dev      Install package with development dependencies"
	@echo "  install-gpu      Install package with GPU dependencies"
	@echo "  format           Format code with black and ruff"
	@echo "  lint             Run ruff linter"
	@echo "  type-check       Run mypy type checker"
	@echo "  test             Run pytest"
	@echo "  test-cov         Run pytest with coverage"
	@echo "  test-slow        Run all tests including slow ones"
	@echo "  clean            Clean build artifacts and cache files"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  serve            Start FastAPI development server"
	@echo "  train            Run training script with default config"

# Installation targets
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

install-gpu:
	$(PYTHON) -m pip install -e ".[gpu]"

# Code quality targets
format:
	black $(SRC_DIR) $(TEST_DIR) scripts
	ruff check --fix $(SRC_DIR) $(TEST_DIR) scripts

lint:
	ruff check $(SRC_DIR) $(TEST_DIR) scripts
	ruff format --check $(SRC_DIR) $(TEST_DIR) scripts

type-check:
	mypy $(SRC_DIR)

# Testing targets
test:
	pytest -m "not slow and not gpu" $(TEST_DIR)

test-cov:
	pytest -m "not slow and not gpu" --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

test-slow:
	pytest --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

test-gpu:
	pytest -m "gpu" $(TEST_DIR)

# Training targets
train:
	python -m mhwf_pikan.training.train --config configs/default.yaml

train-multi:
	python -m mhwf_pikan.training.train --config configs/multi_coil.yaml

# API server targets
serve:
	uvicorn mhwf_pikan.api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	uvicorn mhwf_pikan.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker targets
docker-build:
	docker build -t mhwf-pikan-mri:latest .

docker-run:
	docker run -p 8000:8000 --gpus all mhwf-pikan-mri:latest

docker-run-cpu:
	docker run -p 8000:8000 mhwf-pikan-mri:latest

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Cleaning target
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	rm -rf $(SRC_DIR)/__pycache__ $(SRC_DIR)/*/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Data preparation targets
prepare-data:
	python scripts/prepare_fastmri_data.py --data-path data/fastmri

download-sample:
	python scripts/download_sample_data.py --output data/sample

# Model evaluation targets
evaluate:
	python scripts/evaluate.py --checkpoint checkpoints/best.pt --data-path data/test

benchmark:
	python scripts/benchmark.py --checkpoint checkpoints/best.pt --data-path data/test

# Release targets
build:
	python -m build

twine-check:
	twine check dist/*

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*
