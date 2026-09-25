# Stage 2A report

Stage 1, Stage 1B and Stage 1C evidence remain frozen. Stage2A uses a new model version and outputs/stage2a.

## Direct answer

The finite-window controller generated a causal plan on the collaborative none/0 anchor for P, S1, S2 and S3; the independent task label for each is shown below. This is the first recovered collaborative task segment compared with oldC under the common Stage2A executor. Because S1–S3 use the same predictor and execution layer as P, any P versus oldC change is a prediction/reference-interface result. The internal-state and non-reciprocal increments are evaluated separately; no method is declared the winner.
The continuous reference check passed 90/90 main runs; the largest sampled reference acceleration was 3.563 m/s².

## Counts

Anchors: 25 runs. Regression: 65 runs. Main methods: 90 runs. Validation rows: 96. Ablations: 6 runs.

## Method comparison

| method | n | geometry | mission-valid | full-window | mission-valid denominator | all-run speed deficit (m) | valid-run speed deficit (m) | planner ms/decision | candidates/decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P | 18 | 16 | 16 | 7 | 16 | 276.990 | 252.668 | 144.55 | 29.7 |
| S1 | 18 | 16 | 15 | 11 | 15 | 235.808 | 214.170 | 120.67 | 30.1 |
| S2 | 18 | 16 | 15 | 11 | 15 | 204.582 | 177.575 | 104.61 | 30.5 |
| S3 | 18 | 16 | 15 | 12 | 15 | 196.769 | 168.522 | 106.17 | 30.4 |
| oldC | 18 | 3 | 1 | 1 | 1 | 51.435 | 4.915 | 0.00 | 0.0 |

The speed-deficit means use all 18 runs in the all-run column and only the independently mission-valid runs in the valid-run column. The phase-resolved all-run and valid-run values, including prepare, execute, success-release and failure-handling denominators, are in `outputs/stage2a/summary.csv`.

## Anchor evidence

Collaborative none/0 anchor labels:
- P: geometry=1, mission=1, full=0, outcome=completed
- S1: geometry=1, mission=1, full=1, outcome=completed
- S2: geometry=1, mission=1, full=1, outcome=completed
- S3: geometry=1, mission=1, full=1, outcome=completed
- oldC: geometry=0, mission=0, full=0, outcome=missed_window

The key planner trace, predicted completion and actual completion are in `outputs/stage2a/key_trajectories.csv`, `plan_events.csv` and `validation.csv`. The selected plan is generated from current observations; F prediction uses its current state and the declared nominal model, B responds with IDM, and the real run applies the prepare disturbance only after planning. No future disturbance, scenario identifier or witness answer is passed to the controller.

Geometry completion, mission validity and full-window validity are separate labels. In particular, a task can complete at 170 m with all task-segment constraints valid while a later post-release h violation makes the 16 s full-window label zero. That failure remains in the ledger.

## Contribution interpretation

P and S1 share the finite planner, reference family and common executor. S1 adds a bounded internal preparation intensity while disabling the new space-to-process feedback. S2 adds current-width feedback and symmetric partner action. S3 changes only the role allocation while preserving the total partner coupling budget. The controlled S2 partner-removal, fixed-nonzero-W and feedback-removal ablations are in `outputs/stage2a/ablations.csv`; their task labels and cost changes are not folded into the main method counts.

## Controlled ablations

These six runs keep the S2 planner, state and executor fixed while changing one declared term on collaborative none/0 and ample none/0:
- no_partner / collaborative: mission=1, total speed-deficit=66.753 m, prepare=0.765 m, execute=12.934 m, success-release=53.054 m
- no_partner / ample: mission=1, total speed-deficit=2.258 m, prepare=0.000 m, execute=0.000 m, success-release=2.258 m
- no_feedback / collaborative: mission=1, total speed-deficit=79.575 m, prepare=0.995 m, execute=18.041 m, success-release=60.538 m
- no_feedback / ample: mission=1, total speed-deficit=8.047 m, prepare=0.157 m, execute=2.385 m, success-release=5.505 m
- fixed_W / collaborative: mission=1, total speed-deficit=56.232 m, prepare=0.858 m, execute=14.243 m, success-release=41.131 m
- fixed_W / ample: mission=1, total speed-deficit=4.537 m, prepare=0.059 m, execute=1.132 m, success-release=3.345 m

The common predictor and executor are not credited to swarmalator coupling. A zero S2/S3 task increment over S1 is retained as a result; a lower cost with the same task count is reported as a cost effect, not proof of physical necessity.

## Public execution and validation basis

The checker records geometry completion, mission-valid completion and full-window validity. It rejects corrupted actions, boundary crossings, dynamic h violations, missing dynamics, body overlap, road exits and discontinuous progress. A saved nominal action is never substituted for the actual action. Reference continuity is reported separately in `validation.csv`; the largest jump and acceleration are directly logged per run.

## Limits

The 18 environments are a development/regression set used in earlier stages, not an unseen test set. A failed finite plan remains unresolved rather than being called physical infeasibility. Communication noise/delay, SUMO, CARLA, hardware and MARL remain outside Stage2A.

## Figures

* `F:\Swarmalator–CAV\outputs\stage2a\figures\plan_generation_anchor.png`
* `F:\Swarmalator–CAV\outputs\stage2a\figures\space_process_action_chain.png`
* `F:\Swarmalator–CAV\outputs\stage2a\figures\validity_cost_compute_tradeoff.png`

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage2a --config configs/stage2a_config.json --out outputs/stage2a
```
