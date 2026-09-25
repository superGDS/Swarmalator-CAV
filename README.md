# Swarmalator–CAV Stage 1

This workspace contains a CPU-only four-vehicle merge prototype for testing a swarmalator-inspired space–process coordination hypothesis. M is the merging CAV, R is the CAV behind the target gap, F is the dynamic front boundary, and B is a closed-loop ordinary following vehicle behind R. The implementation compares:

* A: ordinary space gap/speed coordination;
* B: one-way internal process → space coordination;
* C: bidirectional symmetric coupling;
* D: bidirectional state-dependent non-reciprocal allocation with fixed pair gain sum;
* E: an independent rolling candidate timing baseline.

All five methods share the same observations, finite-body geometry, horizon, acceleration/jerk-limited executor, safety correction, lateral execution path, and success test. F receives a deterministic preparation disturbance in the disturbed condition. B follows R through a closed-loop IDM-like law. The full first-round matrix is 90 evaluations.

## Environment and commands

The project uses an isolated Python 3.12 environment. From PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1 --config configs/stage1_config.json --out outputs/stage1
```

The runner writes `outputs/stage1/trajectories.csv`, `events.csv`, `metrics.csv`, `vehicle_metrics.csv`, `run_log.txt`, and three PNG figures. The generated report is `reports/stage1_report.md`. A compact transfer package is created with:

```powershell
Compress-Archive -Path README.md,AGENTS.md,docs,configs,src,tests,outputs/stage1,reports -DestinationPath feedback/stage1_feedback.zip -Force
```

For the compact feedback package, include `outputs/stage1/metrics.csv`, `vehicle_metrics.csv`, `events.csv`, `key_trajectories.csv`, `key_events.csv`, `run_log.txt`, and `figures` rather than the 44 MB complete `trajectories.csv`:

```powershell
Compress-Archive -Path README.md,AGENTS.md,docs,configs,src,tests,reports,outputs/stage1/metrics.csv,outputs/stage1/vehicle_metrics.csv,outputs/stage1/events.csv,outputs/stage1/key_trajectories.csv,outputs/stage1/key_events.csv,outputs/stage1/run_log.txt,outputs/stage1/figures -DestinationPath feedback/stage1_feedback.zip -Force
```

The package excludes the virtual environment, caches, the complete trajectory table, and papers. The final zip is regenerated after the runner and tests finish.

## Scope and current status

This is a controlled prototype. It uses longitudinal point-mass dynamics and a continuous lateral path for M; it is not SUMO, CARLA, a bicycle model, MARL, a wireless network, or a road-safety guarantee. The candidate equations are treated as an independent project design. The RA-L swarmalator paper and the CAV opinion-dynamics paper are recorded in `references/literature_notes.md` with their full-text access limits.

## Stage route

Stage 1 establishes mechanism and implementation evidence. A later stage may add verified feasibility envelopes, message delay/loss, stronger timing optimization, and SUMO/CARLA validation. No such expansion is included in this run.
