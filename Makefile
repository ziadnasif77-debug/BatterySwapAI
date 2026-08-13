# BatterySwapAI 2026 — every target is deterministic and non-interactive.
# `make submit` must run raw data -> submission on a clean checkout.

PY ?= python

.PHONY: help install data audit features train calibrate decide plan submit simulate test lint

help:
	@echo "Targets: install | data | audit | features | train | calibrate | decide | plan | submit | simulate | test | lint"

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

submit:
	$(PY) -m batteryswap.cli submit

simulate:
	$(PY) -m batteryswap.cli simulate

test:
	$(PY) -m pytest

lint:
	ruff check src tests
