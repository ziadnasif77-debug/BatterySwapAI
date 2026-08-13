# BatterySwapAI 2026 — official competition intel

Transcribed from official NORA web pages on **2026-08-13**. Everything here is
provisional until the official dataset/evaluator release; where the released
files disagree, they win (§0 source-of-truth rule).

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
| **2026-08-21** | **Final submission deadline** (7-day sprint, §20) |
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
  hidden set for final scoring (⛔ §15: never tune against the leaderboard).
- Released with the dataset: example Python code (loading, baseline
  prognostics, baseline planning) and evaluation/scoring code → these go in
  `docs/official/evaluator/` verbatim.

## Data-day checklist (2026-08-14)

Ordered by what unblocks the most downstream work. Everything not listed here
is already built and tested against synthetic data.

1. Download release → `data/raw/` + `docs/official/` (verbatim, unmodified).
2. `make data` — schema validation. Pin the real filenames and fix
   `io.py` schemas to match the documentation field by field.
3. **Phase 0 (§5), in this order:**
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
   agreement test (§16.4, ≥1000 plans).
7. Re-run `make cv` and the q sweep on real data — ⛔ the synthetic fleet has
   near-zero travel pressure, so the swept optimum has NOT yet been shown to
   sit above q\*. Re-measure before trusting any q.
8. Wire the CLI pipeline commands (`features`/`train`/…/`submit`) to the real
   data path; the synthetic dry run stays synthetic by design.
