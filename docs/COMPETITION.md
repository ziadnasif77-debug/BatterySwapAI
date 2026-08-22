# BatterySwapAI 2026 — official competition intel

Transcribed from official NORA web pages on **2026-08-13**. Everything here is
provisional until the official dataset/evaluator release; where the released
files disagree, they win.

## Sources

- Competition page: <https://www.nora.ai/competitions/batteryswapai/batteryswapai2026.html>
- FAQ: <https://www.nora.ai/competitions/batteryswapai/batteryswapai-faq.html>
- Registration: <https://nettskjema.no/a/batteryswapai2026>
- Q&A webinar: <https://www.youtube.com/watch?v=g7Wy1QvrzUw>
- Contact: kushtrim.visoka@nora.ai

## Timeline

| Date | Event |
|---|---|
| 2026-05-13 | Launch, registration opens |
| 2026-05-29 | Q&A webinar |
| 2026-08-11 | **Registration deadline** (⚠ already passed as of this note) |
| **2026-08-14** | **Dataset release** — Phase 0 starts the moment files land |
| **2026-08-21** | **Final submission deadline** |
| 2026-09-01 | Winners announced |

## Eligibility & prizes

- Norway residents; teams up to 8; individual participation allowed.
- NOK 50,000 total: 40,000 first, 10,000 second.
- Full prize eligibility requires ≥1 Master's/PhD student or equivalent
  Norwegian academic affiliation.
- Prize submissions expected open-source under **MIT** (LICENSE already in repo).
  Opt-out possible (keep IP, forgo prize).

## Facts confirmed by the FAQ (evidence for unknowns.yaml notes)

- Schema as briefed: Parquet, hourly averages of 1-minute samples, voltage +
  temperature; battery/building/room IDs "not meaningful to use directly as a
  feature"; travel-time matrix between buildings; no coordinates;
  ~3–10 batteries/room, ~1–10 rooms/building, ~200 buildings, few thousand
  devices; CR2450 LiMnO₂, ~2-year nominal life.
- "Variable end-time cutoffs per series to prevent leakage" — confirms the
  truncation design in `sampling.py`.
- Cost components: wrench time per swap, room-change overhead, building-change
  overhead, travel from matrix, **overtime beyond an 8-hour workday**, early
  penalty "per missed hour of ideal lifetime", late penalty "per device-hour
  of downtime; expected to be larger".
- ⚠ "Final scoring weights are not yet locked down 100% and will be tuned" —
  so NO numeric constant may be trusted until read from the released evaluator.
- Submission: ordered work plan (data frame / CSV) of daily replacements,
  uploaded with code to a submission portal. Public leaderboard on submission;
  hidden set for final scoring (never tune against the leaderboard).
- Released with the dataset: example Python code (loading, baseline
  prognostics, baseline planning) and evaluation/scoring code → these go in
  `docs/official/evaluator/` verbatim.

## Data-day checklist (2026-08-14)

Ordered by what unblocks the most downstream work. Everything not listed here
is already built and tested against synthetic data.

1. Download release → `data/raw/` + `docs/official/` (verbatim, unmodified).
2. `make data` — schema validation. Pin the real filenames and fix
   `io.py` schemas to match the documentation field by field.
3. **Phase 0, in this order:**
   - Run the official baseline unmodified end-to-end; record its score (floor).
   - Transcribe every constant into `configs/cost_model.yaml`; flip the
     matching `unknowns.yaml` entries to `resolved`.
   - Pin the real entry point in `evaluate.py` (the loader currently guesses
     among score/evaluate/compute_cost/main).
   - Sensitivity study → the two governing numbers: empirical
     `late/early` ratio, and the **measured per-building** marginal trip cost
     (this replaces the proxy in `optimizer/opportunistic.py`).
   - Write `reports/cost_model.md`.
4. `make audit` — Phase 1 report (knee distribution, seasonal confound).
5. Estimate the truncation distribution from the evaluation series lengths and
   feed it to `sampling.py` (currently a synthetic stand-in).
6. Build the two things that cannot exist before the release: the **submission
   writer** (format unknown) and the **fast surrogate** + its
   agreement test (≥1000 plans).
7. Re-run `make cv` and the q sweep on real data — ⛔ the synthetic fleet has
   near-zero travel pressure, so the swept optimum has NOT yet been shown to
   sit above q\*. Re-measure before trusting any q.
8. Wire the CLI pipeline commands (`features`/`train`/…/`submit`) to the real
   data path; the synthetic dry run stays synthetic by design.

## The official code — where it actually lives (found 2026-08-20)

Not in the data release. The pieces:

| What | Where |
|---|---|
| Dataset (gated) | <https://huggingface.co/datasets/batteryswapaichallenge/BatterySwapAI-2026-Public> |
| **Example repo + submission harness** (MIT) | <https://huggingface.co/batteryswapaichallenge/BatterySwapAI2026-Example> |
| **The scorer** | PyPI package `batteryswap_public` (listed in the example repo's `requirements.txt`) |

```bash
pip install batteryswap_public fastparquet structlog pydantic-settings
```

Modules that matter: `batteryswap_public.evaluate.evaluate_plan` (the cost
function), `.utils.load_dataset` / `.iterate_scenarios` (loading + truncation),
`.utils.make_submissions` (builds submission.csv from a Planner),
`.interfaces.Planner` (the ABC to implement), `.metric.compute` (the
HuggingFace entry point).

### Dataset layout the official loader expects

One directory per split — the flat files from the release are **one split**:

```
<root>/<split>/devices.csv
<root>/<split>/eol_times.csv
<root>/<split>/battery_metrics.parquet
<root>/<split>/scenarios.json
```

`metric.compute` scores `public` and `private` splits; scoring reports
`total_time` — the mean of `total_cost` across a split's scenarios.

### How submission actually works

1. Implement `batteryswap_public.interfaces.Planner`.
2. Pickle it into the model repo (the example uses
   `batteryswap_example/planners/best.pickle`).
3. `script.py` loads the pickle and calls `make_submissions`, which writes
   `submission.csv`.
4. Commit and push to your HuggingFace **model** repo, then use *New
   submission* in the competition app.

Local check, mirroring the submission machine:

```bash
docker run --name batteryswapai -v ./dataset:/tmp/data batteryswapai-2026-example bash -c "/app/env/bin/python3 script.py && /app/env/bin/python3 -m batteryswap_public.metric"
```

### Rules the evaluator enforces on a plan

- every battery exactly once, no duplicates, no extras
- `day` must be datetime, **date-only** (no time of day), non-decreasing
- nothing before `start_time`; rows after `start_time + planning_window_days`
  are cut (this is how you skip a battery)
- a plain `RangeIndex`
- **row order inside a day is the route** — a building change is charged
  whenever consecutive rows differ
