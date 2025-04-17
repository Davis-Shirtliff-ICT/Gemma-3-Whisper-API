# Makefile for Gemma 3 API project

.PHONY: setup run debug clean test help check-ollama

PYTHON := python
PIP := pip
VENV := venv
PORT := 8000

# Check if we're in a virtual environment
IN_VENV = $(shell python -c "import sys; print(hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))")

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make run         - Run the FastAPI server"
	@echo "  make debug       - Run the FastAPI server in debug mode with auto-reload"
	@echo "  make clean       - Remove Python cache files"
	@echo "  make test        - Run tests"
	@echo "  make check-ollama - Check if Ollama is running and Gemma model is available"
	@echo "  make help        - Show this help message"

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(VENV)/bin/$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate the virtual environment with:"
	@echo "  source $(VENV)/bin/activate (Linux/macOS)"
	@echo "  .\\$(VENV)\\Scripts\\activate (Windows)"

run:
	@if [ "$(IN_VENV)" = "False" ]; then \
		echo "Not in a virtual environment. Please activate it first:"; \
		echo "  source $(VENV)/bin/activate (Linux/macOS)"; \
		echo "  .\\$(VENV)\\Scripts\\activate (Windows)"; \
		exit 1; \
	fi
	@echo "Starting FastAPI server on port $(PORT)..."
	$(PYTHON) main.py

debug:
	@if [ "$(IN_VENV)" = "False" ]; then \
		echo "Not in a virtual environment. Please activate it first."; \
		exit 1; \
	fi
	@echo "Starting FastAPI server in debug mode on port $(PORT)..."
	uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

test:
	@if [ "$(IN_VENV)" = "False" ]; then \
		echo "Not in a virtual environment. Please activate it first."; \
		exit 1; \
	fi
	@echo "Running tests..."
	pytest

check-ollama:
	@echo "Checking if Ollama is running..."
	@curl -s http://localhost:11434/api/tags > /dev/null && \
		echo "✓ Ollama is running" || \
		(echo "✗ Ollama is not running. Please start it with 'ollama serve'"; exit 1)
	@echo "Checking if Gemma model is available..."
	@curl -s http://localhost:11434/api/tags | grep -q "gemma3" && \
		echo "✓ Gemma model is available" || \
		echo "✗ Gemma model not found. Please run 'ollama run gemma3:1b' to download it"