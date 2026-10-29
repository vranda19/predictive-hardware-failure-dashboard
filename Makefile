.PHONY: help init-db collect collect-once api test export-parquet clean

PYTHON ?= python

help:
	@echo "Predictive Hardware Failure Dashboard — Command Runner"
	@echo "Targets:"
	@echo "  make init-db        - Initialize SQLite database schema and indexes"
	@echo "  make collect        - Run edge telemetry collector daemon"
	@echo "  make collect-once   - Sample a single telemetry tick and exit"
	@echo "  make api            - Run FastAPI development server"
	@echo "  make test           - Run pytest test suite"
	@echo "  make export-parquet - Export SQLite telemetry to Parquet file"
	@echo "  make clean          - Remove temporary caches and build artifacts"

init-db:
	$(PYTHON) run.py init-db

collect:
	$(PYTHON) run.py collect

collect-once:
	$(PYTHON) run.py collect --once

api:
	$(PYTHON) run.py api

test:
	$(PYTHON) run.py test

export-parquet:
	$(PYTHON) run.py export-parquet

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
