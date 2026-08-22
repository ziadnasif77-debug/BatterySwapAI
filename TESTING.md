# Testing guide

Everything here runs from the repository root. Commands are ordered by how
long they take, so you can go as deep as you have time for.

## 0. One-time setup

```bash
python -m pip install -e ".[dev]"
```

The competition's own scorer is a separate package — needed for anything that
touches the official evaluator:

```bash
pip install batteryswap_public fastparquet structlog pydantic-settings
```

### Dataset layout

The licensed dataset is **not** in the repository. The official loader expects
one directory per split:

```
data/raw/train/devices.csv
data/raw/train/eol_times.csv
data/raw/train/battery_metrics.parquet
data/raw/train/scenarios.json
```

Tests that need it skip cleanly when it is absent, so the suite passes on a
fresh clone with no data at all.

---

## 1. Fast checks (seconds)

```bash
python -m pytest -q          # 101 tests
python -m ruff check src tests
python -m batteryswap.cli status
```

`status` lists what is still unresolved in `configs/unknowns.yaml`.

### What the suite covers

| Area | File | What breaks if it fails |
|---|---|---|
| No identifier leaks into features | `test_no_leakage.py` | model learns the fleet, not the physics |
| No future data at a cutoff | `test_no_future.py` | leakage; offline scores become fiction |
| Quantiles never cross | `test_quantile_ordering.py` | decision layer reads a broken distribution |
| Official scorer contract | `test_evaluator_agreement.py` | plans get rejected, or scored differently than expected |
| Plan validity | `test_planner.py` | **submission rejected outright** |
| Bundle is shippable | `test_bundle.py` | submission crashes in the competition image |
| Bundle needs nothing outside itself | `test_bundle_selfcontained.py` | submission crashes only on their machine |
| Determinism | `test_determinism.py` | results not reproducible |
| Cost arithmetic | `test_scoring.py` | our reimplementation drifted (dev-only path) |

---

## 2. Synthetic end-to-end (about 15 s, no data needed)

```bash
python -m batteryswap.cli dry-run --report --record
python -m batteryswap.cli cv
```

`dry-run` exercises the whole chain on generated data with a synthetic cost
model. `cv` runs GroupKFold by building and prints per-fold variance.

---

## 3. Against the official evaluator (needs the dataset)

```bash
python -m batteryswap.cli official                    # all 48 scenarios
python -m batteryswap.cli official --stride 8         # every 8th, much faster
python -m batteryswap.cli official --bundle submission  # also rebuild the bundle
```

Expected on the released train split:

```
no planning   3,324.7
ours          1,047.6   (-68.5%, wins 48/48)
```

Writes `reports/official_scores.csv` (per scenario) and
`artifacts/planner.pickle`.

### Tuning knobs

```bash
python -m batteryswap.cli official --q 0.12 --margin 0.50 --max-swaps 25 --batch 14
```

| Knob | Shipped | Meaning |
|---|---|---|
| `--q` | 0.12 | quantile of predicted EOL used as the swap day |
| `--margin` | 0.50 | skip a battery this far above the failure voltage |
| `--max-swaps` | 25 | hard cap on swaps per scenario |
| `--batch` | 14 | days within which same-building swaps merge onto one day |

Each was swept on all 48 scenarios; see `reports/full_validation.csv`.

---

## 4. The competition container (the real test)

Requires Docker running.

```bash
cd submission
docker build -t batteryswapai-2026-sub .
```

The base image is large; the first build takes 10–20 minutes.

```bash
docker run --rm -v /path/to/data/raw:/tmp/data batteryswapai-2026-sub \
  bash -c "/app/env/bin/python3 script.py && BATTERYSWAP_SUBMISSION_PATH=/app/submission.csv /app/env/bin/python3 -m batteryswap_public.metric"
```

On **Git Bash** prefix with `MSYS_NO_PATHCONV=1` and use a Windows-style host
path, or the mount is rewritten and the dataset is not found:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v 'D:\BatterySwapAIproject\data\raw:/tmp/data' batteryswapai-2026-sub \
  bash -c "/app/env/bin/python3 script.py && BATTERYSWAP_SUBMISSION_PATH=/app/submission.csv /app/env/bin/python3 -m batteryswap_public.metric"
```

Expected: `train_score.total_time = 1047.6282118055558`.

`public_score` and `private_score` come back as `inf` — those splits are not
in our copy of the dataset, and that is expected.

### If Docker will not start on Windows

Symptom: the GUI runs but `docker version` hangs and the WSL distro stays
`Stopped`. Cause seen here: orphaned socket files from an unclean shutdown
that Windows refuses to delete. Per-file deletion fails; renaming the parent
directory works.

```bash
powershell -Command "Get-Process 'Docker Desktop','com.docker.backend' | Stop-Process -Force"
mv "$LOCALAPPDATA/Docker/run" "$LOCALAPPDATA/Docker/run.stale"
mv "$LOCALAPPDATA/docker-secrets-engine" "$LOCALAPPDATA/docker-secrets-engine.stale"
```

Then start Docker Desktop and wait — it recreates both cleanly. Do not kill it
mid-boot; every socket a killed process leaves behind becomes undeletable
until reboot.

---

## 5. Submitting

`submission/` is a complete, self-contained repository. Push it to a
HuggingFace **model** repo:

```bash
cd submission
git init && git add -A && git commit -m "BatterySwapAI 2026 submission"
git remote add origin https://huggingface.co/<user>/<repo>
git push -u origin main
```

Then use *New submission* in the competition app.

---

## Known gaps

- Our score is a backtest on the **train** split. `public` and `private` are
  not in our copy, so leaderboard performance is an estimate.
- The EOL threshold (2.42 V) is inferred from 82 recorded labels, not given.
- B4 (Weibull AFT), CP-SAT and SAA are implemented but unused on the official
  path — cut for time, not for measured lack of benefit.
- `configs/unknowns.yaml` still has open entries; several are now answerable
  from the official source and simply have not been transcribed back.
