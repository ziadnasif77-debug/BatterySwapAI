# Phase 0 — the official cost model

**Source of truth:** `data/raw/scenarios.json` from the official release
(Soundsensing, all rights reserved, challenge use only — never committed).
Verified by `tests/test_cost_model_transcription.py`.

## Scenario structure

The release contains **48 scenarios** (`s_0` … `s_47`), one per week with
`start_time` running **2025-09-01 → 2026-07-27**. Each scenario is one
independent planning problem:

| Field | Meaning | Varies? |
|---|---|---|
| `start_time` | first day of the planning window | 48 distinct, weekly |
| `settings.base_location` | **depot building** the crew starts/ends at | 16 distinct |
| `settings.base_room` | depot room | 27 distinct |
| `travel_costs[]` | `{from, to, hours}`, full 24×24 matrix | same matrix every scenario |
| `settings.*` (13 constants below) | economics + operations | **identical in all 48** |

The travel matrix is **complete and in hours**: 576 = 24² rows, including
**self-loops of 0.0333 h (2 min)** — moving inside one building is not free.
Range 0.033 h → **10.25 h** (median 0.70 h): the fleet has a genuinely remote
cluster, so per-building marginal trip cost varies enormously.

## The constants (transcribed verbatim)

| Official JSON key | Value | Unit | Where it binds |
|---|---|---|---|
| `time_per_battery_hours` | **0.25** | h | wrench time per swap (15 min) |
| `time_per_room_change_hours` | **0.5** | h | room→room inside a building |
| `time_per_building_change_hours` | **1.0** | h | building→building overhead |
| `travel_costs[].hours` | matrix | **h** | ⚠ the brief guessed *minutes* — it is hours |
| `overtime_start` | **8.0** | h/day | overtime threshold |
| `overtime_penalty_factor` | **2.0** | × | overtime multiplier |
| `worker_limit_daily_hours` | **24.0** | h | daily cap |
| `worker_limit_daily_penalty` | **100.0** | cost | flat penalty for breaching it |
| `worker_limit_weekly_hours` | **24.0** | h | **weekly cap — the binding one** |
| `worker_limit_weekly_penalty` | **100.0** | cost | flat penalty for breaching it |
| `early_replacement_penalty_daily` | **0.5** | cost/day | life wasted by replacing early |
| `late_replacement_penalty_daily` | **10.0** | cost/day | downtime from replacing late |
| `planning_window_days` | **42** | days | 6-week horizon |
| `unobserved_eol_days` | **30.0** | days | handling of devices with no observed EOL |

## The two governing quantities (§5.4)

1. **`late / early = 10.0 / 0.5 = 20.0`.** Read directly off the official
   constants, not estimated. Newsvendor
   `q* = c_early/(c_early+c_late) = 0.5/10.5 = **0.0476**` — the theoretical
   single-battery optimum is the **4.8th percentile** of the failure
   distribution. §11 still applies: `q` is swept, not adopted.

2. **Marginal cost of one additional building visit** — ⛔ still UNKNOWN as a
   *measured* quantity, because the sensitivity study requires the official
   scorer. What is already known from the constants: a dedicated extra visit
   costs `1.0 h` building overhead plus round-trip travel, i.e. **≈1.07 h for
   the nearest building and up to ≈21.5 h for the remote one** — a 20× spread
   that makes per-building amortization (already implemented in
   `optimizer/opportunistic.py`) essential rather than cosmetic.

### The weekly cap is the real constraint

`worker_limit_weekly_hours = 24` with a 42-day window = 6 weeks ⇒ roughly
**144 crew-hours total**. At 0.25 h per swap plus overheads and travel, this
is what makes the problem a scheduling problem rather than a prediction
problem. Overtime beyond 8 h/day costs 2×; breaching 24 h in a day or a week
costs a flat 100 each.

## What Phase 0 still cannot deliver

| Missing | Blocks | Why it cannot be inferred |
|---|---|---|
| **Official scorer code** | §5.1 baseline floor, §5.3 wrapper, §5.4 sensitivity study, §16.4 surrogate-agreement test | ⛔ §18 forbids reimplementing it |
| **Submission format** | `make submit`, §16.7 schema test | column names, day indexing base, repetition rule all unknown |
| **EOL definition / `unobserved_eol_days` semantics** | label construction for the 379 censored devices | see `reports/data_audit.md` |
