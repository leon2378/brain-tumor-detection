# Developer shortcuts. Without make (e.g. plain Windows), run the command after each target directly.
PYTHON ?= python

.PHONY: help lock upgrade check-lock lint test

help:
	@echo make lock        - re-lock requirements/serve-*.txt after changing pyproject.toml
	@echo make upgrade     - move every locked pin to the newest compatible release
	@echo make check-lock  - fail if the lock files are out of date (CI runs this)
	@echo make lint        - ruff + mypy
	@echo make test        - unit tests with coverage

lock:
	$(PYTHON) scripts/lock.py

upgrade:
	$(PYTHON) scripts/lock.py --upgrade

check-lock:
	$(PYTHON) scripts/lock.py --check

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest tests/unit -m "not pipeline" --cov=btd --cov-fail-under=75
