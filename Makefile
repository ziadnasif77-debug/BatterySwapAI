# BatterySwapAI 2026 — every target is deterministic and non-interactive.
# `make submit` must run raw data -> submission on a clean checkout.

PY ?= python

.PHONY: help install data audit dry-run cv sweep features train calibrate decide plan submit simulate test lint

help:
	@echo "Targets: install | data | audit | dry-run | cv | sweep | features | train | calibrate | decide | plan | submit | simulate | test | lint"

# Full-pipeline rehearsal on synthetic data — runs with NO official files.
dry-run:
	$(PY) -m batteryswap.cli dry-run --report --record

# Grouped (GroupKFold by building) CV, per-fold + variance.
cv:
	$(PY) -m batteryswap.cli cv

install:
	$(PY) -m pip install -e ".[dev]"

# Validates that official raw files exist and conform to the documented schema.
data:
	$(PY) -m batteryswap.cli validate-data

audit: data
	$(PY) -m batteryswap.cli audit

features: data
	$(PY) -m batteryswap.cli features

train: features
	$(PY) -m batteryswap.cli train

calibrate: train
	$(PY) -m batteryswap.cli calibrate

decide: calibrate
	$(PY) -m batteryswap.cli decide

plan: decide
	$(PY) -m batteryswap.cli plan

# Raw official data -> submission, deterministic, no manual step.
submit:
	$(PY) -m batteryswap.cli submit --record

# Sweep the selection bar against the scorer (the section-11 q sweep).
sweep:
	$(PY) -m batteryswap.cli sweep

simulate:
	$(PY) -m batteryswap.cli simulate

test:
	$(PY) -m pytest

lint:
	ruff check src tests
