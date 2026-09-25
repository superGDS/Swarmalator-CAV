# Swarmalator–CAV Stage 1B report

The corrected model ran the original 18 environments with five methods (**90
primary runs**), an 18-environment feasibility diagnostic, **36**
bounded intervention runs, and **56** same-snapshot controller
rows. Stage 1 outputs were not overwritten. The requested
`stage1_review_and_next_step.md` and independent review package were not found
in the workspace or Downloads; the findings reproduced in the Stage1B prompt
were treated as constraints.

## Actual changes

* All methods share the finite-body target gap. The 18 m/s non-empty-slot floor
  is 21.4 m; larger initial gaps are retained. Process q targets are separate.
* The old moving-reference speed surrogate is replaced by reference position,
  speed and acceleration feedforward plus error feedback. B/C/D use current
  qdot in the reference derivative without future information or an algebraic
  loop; E uses the same low-level tracker.
* Start feasibility checks the remaining fixed three-second maneuver using
  maximum braking and jerk. Boundaries are [55,170] m, completion 170 m,
  post-task marker 190 m, and road 450 m.
* Success release and failure handling are explicit. Post-success motion is
  common cruise/following; a started failure continues its lateral path under
  safety, while an unstarted failure remains on the ramp.
* Distance, speed-deficit units, role gap minima, nominal/safety/actual action,
  saturation, post-jerk checks and terminal logs use the corrected v1 fields.

## Paired results

| Method | v1 completion | v1 mean cost (m) | v1 mean min gap (m) | action saturation/run | rechecks/run | Stage1 completion | Stage1 mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 3/18 (16.7%) | 9.11 | 13.24 | 41.5 | 0.0 | 8/18 | 15.33 |
| B | 4/18 (22.2%) | 18.99 | 12.74 | 67.8 | 7.8 | 6/18 | 15.52 |
| C | 4/18 (22.2%) | 18.35 | 12.74 | 64.9 | 7.8 | 6/18 | 15.22 |
| D | 4/18 (22.2%) | 18.35 | 12.75 | 65.1 | 7.8 | 6/18 | 15.22 |
| E | 2/18 (11.1%) | 9.21 | 13.16 | 84.8 | 1.9 | 6/18 | 8.13 |

Old and new columns are separate model versions. Improvements from common basis
changes are not D-specific. Full v1 outcomes are `{'completed': 17, 'missed_window': 70, 'started_but_failed': 3}`. Aggregate hard body/road violations are `0`; remaining post-jerk rechecks are retained as diagnostics and are not declared physical infeasibility.

Success-set comparison against frozen Stage1:

A: old∩new=3, old_only=5, new_only=0
B: old∩new=4, old_only=2, new_only=0
C: old∩new=4, old_only=2, new_only=0
D: old∩new=4, old_only=2, new_only=0
E: old∩new=2, old_only=4, new_only=0

The frozen Stage1 E baseline had lower fixed-window cost than D in all 18
paired environments. Under the corrected v1 basis, E is lower than D in 13/18
pairs; this cost comparison is separate from completion and is not a claim that
E or D solves the same environments.

## Feasibility strata

Counts are **{'feasible_witness': 9, 'unresolved': 9}**. `feasible_witness` means either a saved successful
trajectory passed an independent replay or a bounded piecewise candidate passed
the independent discrete dynamics, action, finite-body occupancy and road
checks. `excluded_by_necessary_bound` uses only the
optimistic gap-opening/late-start outer bound. Everything else is
`unresolved`; controller failure is not labelled physical infeasibility. The
witness diagnostic may inspect the saved future trajectory and does not enter
online A–E.

## Mechanism and intervention check

Logs retain raw space/partner feedback, clipped q changes, K/W, phase, q
saturation, reference offset, nominal action, safety target and actual action.
The same-snapshot table isolates C's space, W and partner terms and D's
symmetric-K alternative. Primary mean raw space feedback is C=0.0331
s⁻¹ and D=0.0331 s⁻¹. These are activity measures, not traffic-value
claims; D is not presumed to win.

The active-phase gain-pair sum error is 0 in both C and D. D's mean allocation
is K_MR=0.04635 s⁻¹ and K_RM=0.05539 s⁻¹ (mean imbalance 0.00924 s⁻¹), so
non-reciprocity is present in the controller state while the paired outcome
increment remains unconfirmed.

The bounded paired intervention deltas were:

* C `no_W` versus same-environment full: Δcost=+1.078 m, Δcompletion=+0.000 (2/6 versus 2/6).
* C `no_partner` versus same-environment full: Δcost=+1.078 m, Δcompletion=+0.000 (2/6 versus 2/6).
* C `no_space` versus same-environment full: Δcost=+2.028 m, Δcompletion=+0.000 (2/6 versus 2/6).
* D `symmetric_k` versus same-environment full: Δcost=-0.016 m, Δcompletion=+0.000 (2/6 versus 2/6).

## Figures and main issue

1. `F:\Swarmalator–CAV\outputs\stage1b\figures\stage1b_startup_feasibility.png`: remaining distance versus optimistic minimum
   three-second travel, colored by feasibility label.
2. `F:\Swarmalator–CAV\outputs\stage1b\figures\stage1b_effective_q_feedback.png`: raw/effective q feedback and moving-reference /
   nominal/actual action for a real D trajectory.
3. `F:\Swarmalator–CAV\outputs\stage1b\figures\stage1b_cost_by_feasibility.png`: all-vehicle cost and completion within strata.

The main unresolved scientific issue is whether non-reciprocal allocation has a
repeatable net value after a separately verified feasible trajectory and delayed
or noisy information are introduced. This stage does not expand to those
experiments. Short-window failures remain a mixture of timing limits and
controller behavior, not a blanket infeasibility result.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1b --config configs/stage1b_config.json --out outputs/stage1b
```
