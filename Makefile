.PHONY: install test lint clean

## Install the package and development tools (pytest, ruff)
install:
	pip install -e ".[dev]"

## Run the test suite
test:
	pytest -q

## Check code style
lint:
	ruff check .

## Remove compiled Python files and caches
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
