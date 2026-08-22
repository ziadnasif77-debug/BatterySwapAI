# docs/official — the source of truth

Place here, verbatim and unmodified:

- `evaluator/` — the official scoring/baseline code. `batteryswap.evaluate`
  imports it directly; reimplementing it is forbidden.
- competition rules and dataset documentation.

Phase 0 (reverse-engineering the evaluator) starts the moment these files
exist: transcribe every constant into `configs/cost_model.yaml`, record source
lines in `reports/cost_model.md`, run the sensitivity study, and flip the
corresponding entries in `configs/unknowns.yaml` to `resolved`.
