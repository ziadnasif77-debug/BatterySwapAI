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

### The official evaluator — found, and now the objective

For most of this project the official scoring code was missing: it did not
ship with the 2026-08-14 data release, and the work ran against a
reimplementation. **It has since been located.** It is the PyPI package
`batteryswap_public`, pulled in by the
[official example repository](https://huggingface.co/batteryswapaichallenge/BatterySwapAI2026-Example)
(MIT). [`evaluate.py`](src/batteryswap/evaluate.py) now imports and calls it
directly, as §5.3 always required, and every number below comes from it.

```bash
pip install batteryswap_public fastparquet structlog pydantic-settings
python -m batteryswap.cli official        # train, plan, score against the real scorer
```

The reimplementation survives as [`scoring.py`](src/batteryswap/scoring.py),
confined to the synthetic dry run. Checked against the official source, it was
right about the things that were documented and wrong about the things that
were not:

| Assumption | Verdict |
|---|---|
| A1 — 1 cost unit per technician-hour | ✅ correct (the official field docs say *"In hour-equivalent"*) |
| A2 — each day starts and ends at the depot | ✅ correct |
| A6 — unobserved EOL = last data + 30 d | ✅ correct |
| A4 — overtime | ❌ official adds `factor × overtime` **on top of** the hours |
| A3 — room/building overheads | ❌ official charges per **transition**, not per visit |
| A8 — day indexing | ❌ days are **dates**, not 1-based integers |
| A9 — late penalty capped at the window edge | ❌ **uncapped** in both directions |

`tests/test_evaluator_agreement.py` — §16.4, dormant for the whole project —
is finally live, and measures that divergence instead of assuming it away.

### The submission is a Planner, not a CSV

The official harness (`batteryswap_public.utils.make_submissions`) calls
`planner.plan(battery_data, locations, travel_costs, settings)` once per
scenario and assembles `submission.csv` itself. So the deliverable is a
**pickled `Planner`** committed to a HuggingFace model repo, not a file we
write. Two consequences shape [`planner.py`](src/batteryswap/planner.py):

- **No leakage is possible.** The official `iterate_scenarios` truncates each
  series at the scenario start and drops already-dead devices before the
  planner ever sees them.
- **Every battery must appear exactly once**, with a real date. "Skipping" a
  battery means giving it a date past the planning window, which the evaluator
  cuts. A battery with a *recorded* EOL inside the window that goes unscheduled
  triggers a forced emergency visit: its own day, its own round trip, plus
  late penalty.

### ⛔ Two submission-blocking traps in the container

The image copies **only** `requirements.txt`, `script.py` and
`batteryswap_example/`, and installs a package set the competition fixes
("changing these will *not* affect the competition runtime environment").
Both of the following would have failed on the submission machine while
passing every local test:

1. **LightGBM is not installed there.** The runtime has scikit-learn, scipy,
   statsmodels, lifelines, ortools, polars and torch — and no other
   gradient-boosting library. A pickle carrying LightGBM boosters raises
   ImportError on load. The shipped model is therefore
   [`SklearnQuantileModel`](src/batteryswap/models/sklearn_quantile.py). The
   forced move is an upgrade: `HistGradientBoostingRegressor` accepts
   `monotonic_cst` **together with** quantile loss, so §9's monotone
   constraints are finally active on the primary model — LightGBM refuses that
   combination.
2. **PyYAML is not installed there either**, and `config.py` imported it at
   module level, which would break the whole vendored package on import. The
   import is now function-local.

[`bundle.py`](src/batteryswap/bundle.py) builds a ready-to-push repository and
verifies it: it vendors only the modules `plan()` actually reaches, and walks
the **AST** of each one to reject module-level imports of packages the runtime
lacks (function-local and `try`-guarded imports are fine, which is exactly how
`config.py` still reaches PyYAML during development).

```bash
python -m batteryswap.cli official --bundle submission
```

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

Full testing guide: **[TESTING.md](TESTING.md)** — from a 3-second
smoke test to running the competition container.

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

Scored by the **official evaluator** (`batteryswap_public.evaluate`) over all
48 scenarios in the released split. Its metric is `total_cost` in
hour-equivalents; the competition reports the mean across scenarios.

| Plan | Mean cost/scenario | |
|---|---:|---|
| No planning (nothing scheduled in-window) | 3,324.7 | forced emergency visits only |
| **Ours** | **1,047.6** | **−68.5 %**, wins **48 of 48** scenarios |
| Hindsight oracle (true EOL, same batching) | ~409 | measured on the first 6 scenarios |

Confirmed end to end through the competition's own entry point, **inside the
official Docker image** (scikit-learn 1.7.2, numpy 2.2.6, Python 3.10):
`script.py` → `make_submissions` → `batteryswap_public.metric` reports
`train_score.total_time = 1047.6282118055558` — matching the local backtest
to 10 decimal places.

Reproduce: `python -m batteryswap.cli official --bundle submission`.

### §11 confirmed: the swept q sits above the theoretical one

The newsvendor identity gives `q* = 0.5/(0.5+10) = 0.0476`. Swept against the
official scorer, the optimum is **`q = 0.12`, 2.5× the theoretical value** —
the effect §11 predicts once trips are shared, and one the pre-release
synthetic rehearsal could not demonstrate because its fleet had no travel
pressure. Final knobs: `q=0.12`, batching window 14 days, voltage gate 0.50 V,
cap 25.

### A subset winner that lost on the full set

The sweep ran on every 4th scenario, and its winner was `cap=15` (937.0 there).
Re-scored on **all 48** it came second, and its late penalty was 21 % worse:

| config | mean (48) | late | |
|---|---:|---:|---|
| q=0.12, margin=0.50, **cap=25** | **1,047.6** | 428.8 | shipped |
| q=0.12, margin=0.50, **cap=15** | 1,061.0 | 521.2 | won the 12-scenario sweep |
| q=0.15, margin=0.35, cap=25 | 1,107.1 | 540.2 | |
| q=0.20, margin=0.20, cap=25 | 1,182.9 | 734.8 | pre-retune |

The tell was structural, not statistical: at most **19** devices ever hold a
recorded EOL inside a window, so a cap of 15 must skip genuinely-due
devices — and skipping one costs a forced emergency visit. Selecting on the
subset alone would have shipped the worse configuration.

### What actually moved the number

| Fix | Effect | Why |
|---|---|---|
| Skip **stale** devices | −27,700 on one scenario | a device that stopped reporting has assumed EOL `last data + 30 d`, often already past; swapping it is pure late penalty while leaving it alone is free. `locations.end_time` makes this exact, not inferred |
| **Voltage gate** (0.20 V above threshold) | one late scenario went from 260 swaps to a handful | only a battery that actually crosses the failure voltage earns a *recorded* EOL, and only a recorded EOL can force an emergency visit. The regression target cannot see this: the evaluator's assumed EOL creeps toward the cutoff as the year advances, making healthy batteries look near death |
| **Cap on swaps per scenario** | fixed the 3 scenarios that lost to doing nothing | at most 19 batteries ever hold a recorded EOL inside a window — a structural bound the model has no way to learn |
| Sort rows by building/room within a day | — | row order *is* the route; the evaluator charges a building change on every consecutive pair that differs |

### Diagnostics (they explain the cost, they do not select)

Calibrated coverage on held-out buildings tracks nominal closely (0.05→0.075,
0.10→0.105, 0.25→0.252, 0.50→0.500, 0.75→0.750, 0.90→0.900) at a median MAE
of 29 days. Note that MAE barely moved across the fixes above while cost fell
by more than half — the §13 point, observed again on real data: **accuracy
explains the cost, it does not select the plan.**

### What the data audit found

`make audit` → [reports/data_audit.md](reports/data_audit.md).

- **Seasonal confound is real and large.** Median voltage-on-temperature
  coefficient **+0.0063 V/°C**, positive for **90 %** of devices. Across a
  30 °C Norwegian swing that is ~0.19 V — comparable to the whole margin
  between plateau and failure.
- **82 % of devices are right-censored** (only 82 of 461 carry a recorded EOL).
- **Knee detection is reported but not trusted** — the current trigger fires on
  ordinary plateau decay. Documented rather than quietly dropped; nothing
  downstream depends on it.

## Known limitations / outstanding UNKNOWNs

1. **Only one split is available to us.** The release gave a flat set of files
   — one split. The official metric scores `public` and `private`; our numbers
   are a backtest on what we have, so they are an estimate of leaderboard
   performance, not a leaderboard result. ⛔ §15 still applies: do not tune
   against the public board when it appears.
2. ~~The submission has not been run through the official Docker image.~~
   **Done.** The official image was built and the full path — `script.py` →
   `make_submissions` → `batteryswap_public.metric` — ran **inside the
   container** against the train split: `total_time = 1047.6282118055558`,
   matching the local run to 10 decimal places. Notably the pickle, written
   with scikit-learn 1.9.0, loaded cleanly on the container's 1.7.2.
3. **The swap cap (25) is a structural constant, not a learned quantity.** It
   is justified by the observed maximum of 19 recorded EOLs per window in this
   split; a fleet with a different failure rate would need it re-derived.
4. **The EOL threshold (2.42 V) is inferred.** Calibrated against the 82
   recorded labels, but only 41 % of crossings land within 7 days of their
   label. It feeds `labels.py` ground truth and the planner's voltage gate.
5. **Survival modelling was cut for time.** B4 (Weibull AFT) is implemented but
   unused on the official path; it is the principled way to handle the 82 %
   censoring that the current lower-bound target handles crudely. So were
   CP-SAT and SAA.
6. **Grouped CV is not run on the official data.** Calibration holds out
   buildings, but the full per-fold variance table (`make cv`) still runs on
   synthetic data only.
7. Six entries remain in [`configs/unknowns.yaml`](configs/unknowns.yaml);
   `python -m batteryswap.cli status` prints them. Several are now answerable
   from the official source and simply have not been transcribed back.

## License

MIT — see [LICENSE](LICENSE) (required for prize eligibility).
