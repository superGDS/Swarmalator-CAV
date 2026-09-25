# Stage 2B report

Stage1/1B/1C/2A evidence and feedback packages remain unchanged. Stage2B writes a new v4 model and `outputs/stage2b`.

## Core scientific result

This finite deterministic matrix tests prediction, space-to-preparation feedback, partner action and non-reciprocal allocation after a common handoff repair. No arm is assumed to win. `z` is a bounded braking-preparation intensity, not an oscillator phase or merge progress.

Counts: {"public_checks": 4, "anchors": 20, "regression": 52, "main": 72, "validation_rows": 80, "mechanism": 8, "paired_rows": 108}

## Main regression

|method|n|geometry|mission-valid|full-window|all-run cost (m)|mission-valid cost (m)|full-valid cost (m)|mean candidates|mean compute (ms)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|P|18|12|12|12|153.016|90.270|90.270|48.8|111.95|
|S1|18|12|12|10|187.212|141.564|105.523|48.8|115.59|
|S2|18|12|11|11|164.416|111.193|111.193|48.7|116.30|
|S3|18|12|11|10|162.996|109.085|119.300|48.7|118.45|

Across the 72 main runs, 46 method-environment pairs are mission-valid and 43 are full-window-valid. The remaining rows are retained as failures or unresolved finite-search cases; the most frequent ledger causes are no merge completion and a small number of post-task h-margin violations.

All-run cost includes failures and is descriptive. Conditional means are labeled by their common validity denominator and are not a sole ranking.

## Paired contribution checks

`paired_results.csv` gives one row per environment and method pair; `paired_summary.csv` reports only common mission and common full subsets.
- P → S1: common mission n=12, Δ=51.294 m; common full n=10, Δ=48.059 m.
- P → S2: common mission n=11, Δ=16.726 m; common full n=11, Δ=16.726 m.
- P → S3: common mission n=11, Δ=14.618 m; common full n=10, Δ=15.997 m.
- S1 → S2: common mission n=11, Δ=-35.003 m; common full n=9, Δ=-31.553 m.
- S1 → S3: common mission n=11, Δ=-37.111 m; common full n=8, Δ=-37.482 m.
- S2 → S3: common mission n=11, Δ=-2.108 m; common full n=10, Δ=-2.281 m.

## Anchors and handoff

The five anchors are the first 20 main rows and are reused in the 72-run matrix. Handoff duration, completion h, relative speed/acceleration and rear braking demand are in `metrics.csv` and `handoff_metrics.csv`. `public_checks.csv` contains four diagnostic runs and is not silently added to the regression count.

## Mechanism interventions

S1 has effective gradient, partner and K/W fields equal to zero; raw finite-difference probes remain in separate audit columns. The S3 `symmetric_K` intervention reproduces S2 under the same initial controller state and acceptance rule on both mechanism anchors, so any normal S3 difference is attributable to allocation direction rather than a different total coupling budget.
- no_partner / collaborative / none / i0: mission=1, full=1, cost=56.635 m.
- no_partner / ample / none / i0: mission=1, full=1, cost=2.210 m.
- no_feedback / collaborative / none / i0: mission=1, full=1, cost=79.575 m.
- no_feedback / ample / none / i0: mission=1, full=1, cost=8.047 m.
- fixed_W / collaborative / none / i0: mission=1, full=1, cost=52.780 m.
- fixed_W / ample / none / i0: mission=1, full=1, cost=4.366 m.
- symmetric_K / collaborative / none / i0: mission=1, full=1, cost=57.956 m.
- symmetric_K / ample / none / i0: mission=1, full=1, cost=5.007 m.

## Public-layer findings

An empty actuator interval or ordered-chain intersection is an unresolved hard-constraint failure. It is logged separately from ordinary saturation. Reference continuity and physical error fields are separate in `validation.csv`.

## Limitations

This is a finite deterministic development/regression matrix, not a traffic simulator, distributed communication test, reachability proof or road-safety certification. Search failure is reported as unresolved, not as proof of physical infeasibility. Future F behavior is not read by any controller.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage2b --config configs/stage2b_config.json --out outputs/stage2b
```

## Figures
- `F:\Swarmalator–CAV\outputs\stage2b\figures\collaboration_handoff_recovery.png`
- `F:\Swarmalator–CAV\outputs\stage2b\figures\paired_same_service.png`
- `F:\Swarmalator–CAV\outputs\stage2b\figures\coupling_direction_allocation.png`