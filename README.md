# BatterySwapAI 2026

Competition-grade, reproducible system for the **BatterySwapAI 2026 Challenge**
(NORA × NMBU × Soundsensing): probabilistic remaining-life estimation for
CR2450-powered IoT vibration sensors, coupled to a cost-optimal maintenance
schedule.

```
noisy sensor time-series → probabilistic remaining-life estimates
                         → cost-optimal maintenance schedule
                         → single scalar cost (the score)
```

**The objective is total operational cost.** Prediction error (MAE/RMSE) is a
diagnostic, never a target. See §Architecture below for how the two stages are
decoupled through an explicit decision layer.

## ⚠️ Project status: skeleton — awaiting official competition files

The official dataset, evaluator, and rules are **not yet in this repository**.
Under the no-invention rule, every competition-owned constant lives in
[`configs/cost_model.yaml`](configs/cost_model.yaml) as `UNKNOWN` and **raises
`UnknownValueError` at access time**. Nothing in this codebase runs on a
guessed penalty coefficient, wrench time, worker count, or failure threshold.

Check current blockers:

```bash
python -m batteryswap.cli status
```

### To unblock Phase 0, place the official artifacts here

| What | Where |
|---|---|
| training + evaluation time-series (Parquet), locations, travel matrix | `data/raw/` |
| official evaluator / baseline code | `docs/official/evaluator/` |
| rules, dataset documentation | `docs/official/` |

## Architecture

| Stage | Module | Status |
|---|---|---|
| Load + schema validation | `src/batteryswap/io.py` | ✅ implemented (schemas provisional until verified) |
| Data audit | `src/batteryswap/audit.py` | ✅ implemented (gaps, truncation shape, knee, seasonal confound) |
| EOL labels + censoring | `src/batteryswap/labels.py` | ✅ machinery done; threshold gated on official rule |
| Cutoff-matched sampling | `src/batteryswap/sampling.py` | ✅ implemented |
| Leak-free features | `src/batteryswap/features.py` | ✅ implemented (multi-scale, residualized vs temperature, knee group) |
| Models B0–B1 | `src/batteryswap/models/baseline.py` | ✅ implemented |
| Model B3 (primary): LightGBM quantile | `src/batteryswap/models/gbm_quantile.py` | ✅ implemented — monotone constraints + non-crossing |
| Model B4: Weibull AFT | `src/batteryswap/models/survival.py` | ✅ implemented |
| Model B6: sequence | `src/batteryswap/models/sequence.py` | ⛔ gated on measured final-cost win over B3 |
| Conformal calibration (CQR, grouped) | `src/batteryswap/calibration.py` | ✅ implemented + coverage diagnostics |
| Decision layer g_b(d) | `src/batteryswap/decision.py` | ✅ implemented; penalties gated on official constants |
| Official scorer wrapper | `src/batteryswap/evaluate.py` | ✅ wrapper ready; official code missing |
| Regret-k construction | `src/batteryswap/optimizer/construct.py` | ✅ implemented |
| ALNS | `src/batteryswap/optimizer/alns.py` | ✅ implemented (adaptive weights, SA acceptance) |
| Per-day routing | `src/batteryswap/optimizer/routing.py` | ✅ implemented (NN + 2-opt, day-cap accounting) |
| CP-SAT assignment | `src/batteryswap/optimizer/assign.py` | ⏳ deferred until constants exist |
| Offline harness | `src/batteryswap/simulate.py` | ⏳ wired in Phase 8 |

## Reproduce

```bash
python -m pip install -e ".[dev]"
make test          # runs now, on synthetic data — no official files needed
make data          # validates official files once placed in data/raw/
make audit         # writes reports/data_audit.md
make submit        # raw data → submission, deterministic (wired phase by phase)
```

## Testing (§16 of the project brief)

| # | Test | File | State |
|---|---|---|---|
| 1 | No identifier leaks into features | `tests/test_no_leakage.py` | ✅ active |
| 2 | No future data at cutoff | `tests/test_no_future.py` | ✅ active |
| 3 | Non-crossing quantiles | `tests/test_quantile_ordering.py` | ✅ active |
| 4 | Surrogate ≡ official scorer (≥1000 plans) | `tests/test_evaluator_agreement.py` | ⏸ self-activates when evaluator lands |
| 5 | Schedule feasibility | `tests/test_feasibility.py` | ✅ routing arithmetic active; caps gated |
| 6 | Determinism | `tests/test_determinism.py` | ✅ active for features/sampling/ALNS |
| 7 | Submission schema | `tests/test_schema.py` | ⏸ self-activates when format resolved |

## Results

_No results yet — blocked on official data. This table will report **final
cost** as the primary column for every model × calibration × q × optimizer
combination, with MAE/pinball/coverage as adjacent diagnostics._

## Known limitations / outstanding UNKNOWNs

All entries in [`configs/unknowns.yaml`](configs/unknowns.yaml) are unresolved
pending the official release. `python -m batteryswap.cli status` prints the
live list.

## License

MIT — see [LICENSE](LICENSE) (required for prize eligibility).
