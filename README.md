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

## ⚠️ Project status: end-to-end on the official data — with one recorded exception

The 2026-08-14 release landed and the pipeline runs raw data → submission.
**16 of the original 22 unknowns are resolved**, transcribed verbatim from
`data/raw/scenarios.json` and pinned by a test.

```bash
python -m batteryswap.cli status     # what is still unresolved and why
make submit                          # raw official data -> submissions/submission.csv
```

### ⛔ The exception you must read before trusting any number here

The official **scoring code never shipped with the data**, and the brief
(§18) forbids reimplementing it. With the deadline one day out, the project
owner explicitly authorised the fallback, so
[`src/batteryswap/scoring.py`](src/batteryswap/scoring.py) is **our
reimplementation of the cost function, not the official scorer**. Its
constants are certain — read straight from the official file — but eight
assumptions (A1–A8, listed in that module) are ours, and each one can move
the score. The most consequential is **A1**: the exchange rate between
technician-hours and penalty units is never stated in the release, and we
charge 1 cost unit per hour.

Consequences, stated plainly: every cost figure below is *self-consistent*,
not *externally validated*; the optimizer is tuned against our own objective;
and `tests/test_evaluator_agreement.py` — the check that would catch a
mismatch — still cannot run. If the official scorer appears, import it in
`evaluate.py`, diff it on ≥1000 plans, and re-tune before trusting anything.

### What the official release changed about the brief

| The brief assumed | The release says |
|---|---|
| travel matrix in **minutes** | **hours** (`travel_costs[].hours`), 0.03 h → 10.25 h |
| penalties per **hour** | per **day**: early 0.5, late 10.0 → **ratio 20:1**, q\* = 0.048 |
| a **worker count** constraint | no headcount at all — **hour caps**: 24 h/day *and* 24 h/week, flat 100 penalty each, plus 2× overtime beyond 8 h/day |
| identifiers `battery_id`, timestamp `timestamp` | `device_id`, and `end_time` (end of each 1-hour averaging window) |
| ~200 buildings, few thousand devices | **24 buildings, 461 devices**, 79 rooms |
| estimate the truncation distribution | the **48 weekly scenarios are the cutoffs** — no estimation needed |
| — (never mentioned) | there is a **depot** (`base_location`/`base_room`) per scenario |

Full transcription with every JSON key: [reports/cost_model.md](reports/cost_model.md).

### What the data itself forced

- **82 % of devices are right-censored** — only 82 of 461 carry a recorded
  EOL. Ground truth is assembled in three tiers (recorded label → sustained
  threshold crossing → censored) in [`labels.py`](src/batteryswap/labels.py).
- The official **EOL rule was never published**, so the threshold is *measured*
  against the 82 known labels rather than guessed: swept 2.30–2.50,
  **2.42 V** is least biased (median error −0.7 d) and tightest (MAD 10.1 d).
- Consequently **a median of ~10 devices fail in any 42-day window** out of
  461. The task is therefore *find the handful that will fail and batch their
  visits*, not *schedule everything efficiently*. Replacing a healthy device
  pays early penalty and burns the hour budget for nothing.

### Synthetic dry run (still passes, still useful)

The pre-release rehearsal runs unchanged on synthetic data with a synthetic
cost model, and remains the fastest way to check the plumbing:

```bash
python -m batteryswap.cli dry-run --report --record
python -m batteryswap.cli cv
```

## Architecture

| Stage | Module | Status |
|---|---|---|
| Load + schema validation | `src/batteryswap/io.py` | ✅ verified against the real release |
| Data audit | `src/batteryswap/audit.py` | ✅ `make audit` → reports/data_audit.md |
| EOL ground truth (label → threshold → censored) | `src/batteryswap/labels.py` | ✅ threshold calibrated against the 82 recorded labels |
| Cost function | `src/batteryswap/scoring.py` | ⚠️ **OURS, not the official scorer** — see the exception above |
| Production pipeline (raw → submission) | `src/batteryswap/pipeline.py` | ✅ `make submit` |
| Cutoff-matched sampling | `src/batteryswap/sampling.py` | ✅ superseded on official data — the 48 scenarios are the cutoffs |
| Leak-free features | `src/batteryswap/features.py` | ✅ implemented (multi-scale, residualized vs temperature, knee group) |
| Grouped CV harness (GroupKFold by building) | `src/batteryswap/cv.py` | ✅ implemented — per-fold + variance |
| Models B0–B1 | `src/batteryswap/models/baseline.py` | ✅ implemented |
| Model B2: LightGBM point regression | `src/batteryswap/models/gbm_point.py` | ✅ implemented — monotone constraints active here |
| Model B3 (primary): LightGBM quantile | `src/batteryswap/models/gbm_quantile.py` | ✅ implemented — non-crossing enforced; monotone constraints gated (LightGBM rejects them with quantile objective) |
| Model B4: Weibull AFT | `src/batteryswap/models/survival.py` | ✅ implemented |
| Model B6: sequence | `src/batteryswap/models/sequence.py` | ⛔ gated on measured final-cost win over B3 |
| Conformal calibration (CQR, grouped) | `src/batteryswap/calibration.py` | ✅ implemented + coverage diagnostics |
| Decision layer g_b(d) + q-rule | `src/batteryswap/decision.py` | ✅ running on the official penalties |
| q sweep (§11) | `src/batteryswap/tuning.py` | ✅ implemented — swept with the shipped post-processing |
| SAA lifetime scenarios | `src/batteryswap/saa.py` | ✅ implemented — swaps into any optimizer unchanged |
| Experiment tracking (config + git SHA + env) | `src/batteryswap/experiments.py` | ✅ implemented |
| Report figures (coverage, q sweep) | `src/batteryswap/plots.py` | ✅ implemented |
| Official scorer wrapper | `src/batteryswap/evaluate.py` | ⛔ ready, but the official code never shipped |
| Regret-k construction | `src/batteryswap/optimizer/construct.py` | ✅ implemented |
| ALNS | `src/batteryswap/optimizer/alns.py` | ✅ implemented (adaptive weights, SA acceptance) |
| Opportunistic-swap rule (amortized per building) | `src/batteryswap/optimizer/opportunistic.py` | ✅ implemented; trip cost is a proxy until the Phase 0 sensitivity study |
| Per-day routing | `src/batteryswap/optimizer/routing.py` | ✅ implemented (NN + 2-opt, day-cap accounting) |
| CP-SAT assignment | `src/batteryswap/optimizer/assign.py` | ⏳ deferred (§20 cut order) |
| Offline harness | `src/batteryswap/simulate.py` | ⏳ wired in Phase 8 (real data) |
| Synthetic dry-run harness | `src/batteryswap/dryrun.py` | ✅ full chain end-to-end, `make dry-run` |

## Reproduce

```bash
python -m pip install -e ".[dev]"
make test          # 67 tests; the synthetic ones need no official files
make data          # validates the official files in data/raw/
make audit         # writes reports/data_audit.md
make submit        # raw data → submissions/submission.csv, deterministic
make sweep         # selection-bar sweep against the scorer
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

Backtest over **all 48 official scenarios**, scored with
[`scoring.py`](src/batteryswap/scoring.py) — read the exception above before
reading the numbers.

| Plan | Total cost | vs do-nothing |
|---|---:|---:|
| Do nothing | 1,168,360 | — |
| **Ours** (B3 + CQR + decision layer + batching + ALNS) | **156,067** | **−86.6 %** |
| Hindsight oracle (true EOL, same planner) | 44,359 | −96.2 % |

Reproduce: `make submit` → `submissions/submission.csv` (4,496 rows),
`reports/score_breakdown.csv` per scenario.

### What actually moved the number

Four changes, in the order they were measured. Each cost figure is the total
across all 48 scenarios:

| # | Change | Cost | Why |
|---|---|---:|---|
| 0 | first end-to-end run | 3,212,961 | worse than doing nothing |
| 1 | drop mechanically-shrinking censored targets | 2,107,349 | targets computed as `last_observed + grace − cutoff` shrink as the cutoff advances, teaching the model that **every device dies when the dataset ends** |
| 2 | charge downtime **inside the window only** | 574,856 | a device that died 300 days before the window was being charged 3,000+ to replace on day 1, so no plan could beat inaction (assumption A9) |
| 3 | let the model see already-dead devices + hard override when observed below threshold at the cutoff | 202,296 | it had never seen a dead device, so it scheduled them weeks late — **93 % of the remaining cost was late penalty** |
| 4 | sweep the selection bar | **156,067** | the §11 sweep, in the form this problem takes |

An intermediate attempt — training only on devices that eventually failed —
made things *worse* (3,340,042): the model never saw a long survivor and
predicted short for everything. The fix that worked uses the fact that a
censored device observed alive for a **full horizon** after the cutoff
*provably* did not fail in that window; that is a real label, not an
assumption. Rows where the remaining observation is shorter than the horizon
are genuinely unknown and are dropped. See
[`pipeline.training_targets`](src/batteryswap/pipeline.py).

### Selection-bar sweep

The bar is `gain > marginal_work_cost × (1 − slack)`. Swept against the
scorer on every 2nd scenario:

| slack | −14 | −13 | **−12** | −11 | −10 | −9 |
|---|---:|---:|---:|---:|---:|---:|
| cost | 75,471 | 76,310 | **75,463** | 75,653 | 76,906 | 77,994 |

Flat between −14 and −11, so **−12** is a stable choice rather than a
knife-edge fit. Note the bar sits far above the naive break-even: with
lateness at 20× earliness and labour cheap under A1, expected-value selection
alone replaces far too much.

### Diagnostics (they explain the cost, they do not select)

Conformal calibration is close to nominal on held-out buildings — 0.05→0.075,
0.10→0.105, 0.25→0.252, 0.50→0.500, 0.75→0.750, 0.90→0.900 — while median
MAE is 42 days. **A model with worse MAE produced the cheaper schedule**:
change #3 raised MAE from 29 to 42 days (it added hard-to-fit already-dead
rows) and cut cost by 64 %. That is the brief's §13 prediction, observed.

### What the data audit found

`make audit` → [reports/data_audit.md](reports/data_audit.md). The two
protective audits:

- **Seasonal confound is real and large.** Median voltage-on-temperature
  coefficient **+0.0063 V/°C**, positive for **90 %** of devices. Across a
  30 °C Norwegian swing that is ~0.19 V — comparable to the entire margin
  between plateau and failure. Raw voltage is partly a thermometer, which is
  why features carry residualised voltage and slope.
- **Knee detection is reported but not trusted** — the current trigger fires
  on ordinary plateau decay. Documented in the report rather than quietly
  dropped; nothing downstream depends on it.

## Known limitations / outstanding UNKNOWNs

Ranked by how much they could invalidate the results above.

1. **⛔ The cost function is ours.** The official scorer never shipped. Every
   cost figure is self-consistent, not externally validated, and the optimizer
   is tuned against our own objective. Assumption **A1** (1 cost unit per
   technician-hour) has no basis in the release at all — it is the exchange
   rate between the two halves of the objective, and a different rate changes
   which trips are worth taking. `tests/test_evaluator_agreement.py` still
   cannot run.
2. **The submission format is a guess.** Columns `scenario,day,device` with
   1-based days (A8). The FAQ describes "an ordered work plan (CSV)" and
   nothing more precise. If the real header differs, the file is wrong even if
   the plan is right — fix `pipeline.submission_frame`.
3. **The EOL rule is inferred, not given.** 2.42 V is calibrated against 82
   recorded labels, but only 41 % of crossings land within 7 days of their
   label. Ground truth for the other 379 devices inherits that uncertainty.
4. **The censored-survivor target is a lower bound**, so long RULs are
   understated. Harmless for a 42-day decision, wrong for lifetime estimates.
5. **B4 (Weibull AFT) is implemented but unused in the shipped path** —
   handling censoring natively is the principled fix for #4 and was cut for
   time, not for evidence. Same for CP-SAT and SAA on the official path.
6. **Grouped CV is not run on the official data.** Calibration holds out
   buildings, but the full per-fold variance table (`make cv`) still runs on
   synthetic data only.
7. Six entries remain in [`configs/unknowns.yaml`](configs/unknowns.yaml);
   `python -m batteryswap.cli status` prints them.

## License

MIT — see [LICENSE](LICENSE) (required for prize eligibility).
