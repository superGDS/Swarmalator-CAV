# Swarmalator–CAV

This repository is the reproducible snapshot of the four-vehicle Swarmalator–CAV study through **Stage 2B**. The current model and results are the finite, CPU-only closed-loop prototype in `src/swarmalator_cav` and `outputs/stage2b`. Stage 1, Stage 1B, Stage 1C and Stage 2A evidence is preserved alongside it; later outputs do not overwrite the earlier evidence.

The four longitudinal vehicles are M (the merging CAV), R (the vehicle behind the target gap), F (the dynamic front boundary), and B (the ordinary closed-loop follower behind R). The Stage 2B public arms are:

* **P**: finite prediction and candidate timing/space preparation;
* **S1**: the same prediction and execution path with the added gradient and partner terms disabled;
* **S2**: prediction plus task-potential feedback and symmetric partner action;
* **S3**: the same public base with state-dependent non-reciprocal allocation.

The study reports geometry completion, mission validity, full-window validity, all affected-vehicle costs, nominal/actuated/safety-projected/actual actions, handoff behavior, and unresolved finite-search failures. It does not assume that S3 wins, and it does not treat common prediction or handoff repairs as a S3-only contribution. Search failure is logged as unresolved rather than being called physical infeasibility.

## What is in the repository

* `src/swarmalator_cav/` — Stage 1 through Stage 2B simulators, controllers, predictors, independent checkers, and runners.
* `configs/` — the exact JSON configurations used for each stage.
* `outputs/stage1`, `outputs/stage1b`, `outputs/stage1c`, `outputs/stage1c_initial_audit`, `outputs/stage2a`, `outputs/stage2b` — CSV ledgers, complete/key trajectories, event logs, validation ledgers, figures, and run logs.
* `reports/` and `docs/` — model revisions, validation notes, and stage reports. `docs/model_v4.md` describes the current Stage 2B model.
* `feedback/` — compact transfer packages and manifests for every completed stage, including `feedback/stage2b_feedback.zip`.
* `references/literature_notes.md` — the literature record and access notes used in the study.
* `codex_swarmalator_*_prompt.md` — the stage-specific research instructions supplied for the project.

The complete trajectory tables are retained in the stage output directories. The compact feedback zips contain the files needed to reconstruct the reported checks and key trajectories without depending on the local virtual environment.

## Reproduce the current snapshot

The project uses Python, NumPy, Matplotlib, and pytest. The local `.venv`, Python caches, editor state, and transient logs are deliberately excluded from version control.

From PowerShell on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage2b --config configs/stage2b_config.json --out outputs/reproduce_stage2b
```

Use a new output directory such as `outputs/reproduce_stage2b` when checking the repository so the frozen `outputs/stage2b` evidence is not replaced. The Stage 2B runner performs four public diagnostics, five anchors reused in the matrix, the finite regression matrix, same-state and mechanism checks, figures, validation ledgers, and a compact feedback package. The exact counts and hashes are recorded in `outputs/stage2b/run_log.txt`, `outputs/stage2b/pilot_notes.json`, and `outputs/stage2b/frozen_evidence_hashes.json`.

The earlier runners can be reproduced in the same way with their matching configuration and a separate output directory:

```powershell
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1 --config configs/stage1_config.json --out outputs/reproduce_stage1
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1b --config configs/stage1b_config.json --out outputs/reproduce_stage1b
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1c --config configs/stage1c_config.json --out outputs/reproduce_stage1c
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage2a --config configs/stage2a_config.json --out outputs/reproduce_stage2a
```

## Scope and limits

This is a finite deterministic development and regression study with declared observations, finite-body geometry, bounded acceleration/jerk execution, safety projection, lateral merge progress, and a closed-loop IDM-like B follower. It is not SUMO, CARLA, a bicycle model, MARL training, a communication-delay/noise study, a reachability proof, or a road-safety certification. The candidate equations are treated as a research design and are checked for units, feedback direction, information dependency, saturation, and terminal behavior in the stage reports.

For an independent review, start with `reports/stage2b_report.md`, `reports/stage2b_validation.md`, `outputs/stage2b/paired_summary.csv`, `outputs/stage2b/mechanism.csv`, and `outputs/stage2b/figures/`, then use the earlier stage reports and frozen feedback hashes to trace the revisions.


