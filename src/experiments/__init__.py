"""Expose repository experiments when ``src/main.py`` is run as a script."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / 'experiments')]
