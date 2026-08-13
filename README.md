# BatterySwapAI 2026

[![ci](https://github.com/ziadnasif77-debug/BatterySwapAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ziadnasif77-debug/BatterySwapAI/actions/workflows/ci.yml)

Competition-grade, reproducible system for the **BatterySwapAI 2026 Challenge**
(NORA × NMBU × Soundsensing): probabilistic remaining-life estimation for
CR2450-powered IoT vibration sensors, coupled to a cost-optimal maintenance
schedule.

**Official timeline** (see [docs/COMPETITION.md](docs/COMPETITION.md)):
dataset release **2026-08-14**, final submission **2026-08-21**.

```
noisy sensor time-series → probabilistic remaining-life estimates
                         → cost-optimal maintenance schedule
                         → single scalar cost (the score)
```

**The objective is total operational cost.** Prediction error (MAE/RMSE) is a
diagnostic, never a target. See §Architecture below for how the two stages are
decoupled through an explicit decision layer.

## ⚠️ Project status: pipeline wired — awaiting official files (release 2026-08-14)

The official dataset, evaluator, and rules are **not yet in this repository**.
Under the no-invention rule, every competition-owned constant lives in
[`configs/cost_model.yaml`](configs/cost_model.yaml) as `UNKNOWN` and **raises
`UnknownValueError` at access time**. Nothing in this codebase runs on a
guessed penalty coefficient, wrench time, worker count, or failure threshold.

Check current blockers:

```bash
python -m batteryswap.cli status
```

### Synthetic dry run — the whole pipeline, today

The full chain (labels → cutoff-matched sampling → features → B3 quantiles →
CQR calibration → decision cost curves → **q sweep** → regret-k → ALNS →
**opportunistic swaps** → per-day routing → cost + KPIs) runs end-to-end on
synthetic data with an explicitly synthetic cost model, so data day is pure
transcription work:

```bash
python -m batteryswap.cli dry-run --report --record
```

Sample output (~12 s, synthetic cost units): independent per-battery plan
`1310.7` → regret-k `1255.2` → ALNS `1194.4` → **`1192.3` final** after the q
sweep and opportunistic consolidation. Figures land in `reports/`, the run is
recorded under `experiments/` with its git SHA. Nothing synthetic can leak
into a submission: the real path still raises on every `UNKNOWN`.

Grouped cross-validation runs the same way:

```bash
python -m batteryswap.cli cv
```

### Findings already banked (before data day)

| Finding | Where |
|---|---|
| LightGBM **rejects** `monotone_constraints` with its built-in quantile objective. B3 therefore enforces non-crossing post-hoc; the constraints ship in **B2** (L2 objective), and monotone-quantile moves to a gated smoothed-pinball experiment. A test pins both halves so a LightGBM upgrade can't silently change either. | `models/gbm_quantile.py`, `models/gbm_point.py`, `tests/test_models.py` |
| **Raw B3 quantiles are badly miscalibrated on unseen buildings** — across 5 grouped folds the nominal 0.05 level captured 23 % of truths and the 0.95 level only 82 %. This is §10's claim, measured. CQR fixes it. Reason enough never to feed raw quantiles to the decision layer. | `make cv` |
| Fold-to-fold spread is large (MAE 422–569 h, σ≈49 h over 5 folds), so single-number CV summaries would be misleading here. Per-fold results are always reported. | `cv.summarize_folds` |
| With linear penalties, SAA scenario curves converge to the analytic integral (max deviation 2.6 %), confirming the scenario path is wired correctly. Its real value arrives if the official penalty is non-linear. | `saa.py`, `tests/test_decision_rules.py` |

⚠️ **What the dry run cannot yet tell us.** In the synthetic fleet each planned
day happens to serve a single building, so realized travel is 0 minutes and
there is almost no trip-sharing pressure. The q sweep therefore bottoms out
essentially at the newsvendor value (0.10 vs q\*=0.091) — §11's central claim,
that shared trips push the optimum *materially above* q\*, is **untested here
and must be re-measured on the official fleet geometry**. The sweep curve's
shape is informative, though: cost is flat below q≈0.3 and explodes above
q≈0.75, which is the late-penalty asymmetry doing exactly what it should.

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
| Grouped CV harness (GroupKFold by building) | `src/batteryswap/cv.py` | ✅ implemented — per-fold + variance |
| Models B0–B1 | `src/batteryswap/models/baseline.py` | ✅ implemented |
| Model B2: LightGBM point regression | `src/batteryswap/models/gbm_point.py` | ✅ implemented — monotone constraints active here |
| Model B3 (primary): LightGBM quantile | `src/batteryswap/models/gbm_quantile.py` | ✅ implemented — non-crossing enforced; monotone constraints gated (LightGBM rejects them with quantile objective) |
| Model B4: Weibull AFT | `src/batteryswap/models/survival.py` | ✅ implemented |
| Model B6: sequence | `src/batteryswap/models/sequence.py` | ⛔ gated on measured final-cost win over B3 |
| Conformal calibration (CQR, grouped) | `src/batteryswap/calibration.py` | ✅ implemented + coverage diagnostics |
| Decision layer g_b(d) + q-rule | `src/batteryswap/decision.py` | ✅ implemented; penalties gated on official constants |
| q sweep (§11) | `src/batteryswap/tuning.py` | ✅ implemented — swept with the shipped post-processing |
| SAA lifetime scenarios | `src/batteryswap/saa.py` | ✅ implemented — swaps into any optimizer unchanged |
| Experiment tracking (config + git SHA + env) | `src/batteryswap/experiments.py` | ✅ implemented |
| Report figures (coverage, q sweep) | `src/batteryswap/plots.py` | ✅ implemented |
| Official scorer wrapper | `src/batteryswap/evaluate.py` | ✅ wrapper ready; official code missing |
| Regret-k construction | `src/batteryswap/optimizer/construct.py` | ✅ implemented |
| ALNS | `src/batteryswap/optimizer/alns.py` | ✅ implemented (adaptive weights, SA acceptance) |
| Opportunistic-swap rule (amortized per building) | `src/batteryswap/optimizer/opportunistic.py` | ✅ implemented; trip cost is a proxy until the Phase 0 sensitivity study |
| Per-day routing | `src/batteryswap/optimizer/routing.py` | ✅ implemented (NN + 2-opt, day-cap accounting) |
| CP-SAT assignment | `src/batteryswap/optimizer/assign.py` | ⏳ deferred until constants exist |
| Offline harness | `src/batteryswap/simulate.py` | ⏳ wired in Phase 8 (real data) |
| Synthetic dry-run harness | `src/batteryswap/dryrun.py` | ✅ full chain end-to-end, `make dry-run` |

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
