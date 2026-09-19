.PHONY: format format-check lint test

format:
	ruff format --line-length 79 src tests

format-check:
	ruff format --check --line-length 79 src tests

lint:
	flake8 src

test:
	PYTHONPATH=src MPLBACKEND=Agg pytest -q
