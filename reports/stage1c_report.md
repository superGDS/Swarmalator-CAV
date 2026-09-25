# Swarmalator–CAV Stage 1C report

Stage 1 and Stage 1B evidence remain untouched. Stage 1C ran the original 18
environments with methods A–E (**90 primary runs**), the independent
checker and bounded candidate search for all 18 environments, same-state
controller interventions (**60 rows**) plus 36
closed-loop ablation runs, and four-environment
opportunity diagnostics (**20 rows**).

## Model changes

* A completing step above 170 m is a failure; the old success-after-crossing
  outcome is removed.
* R release follows the actual occupied front (M or F), and strict witness h≥0
  is separate from the 0.15 m post-jerk tolerance.
* E recomputes the endpoint after smoothed target time and uses a current
  position reference. `Delta_new` is used consistently in the process gradient;
  ample therefore has zero process opening.
* One common geometry-derived finite-window opening term is applied to M and R
  for every method. It is not a D-specific term.
* Controller state before each compute is saved, so full same-snapshot replay is
  a true state clone rather than q-only reconstruction.

## Primary results

| Method | Corrected completions | Mean all-vehicle cost (m) | Mean witness-stratum cost (m) | Mean minimum gap (m) | Mean saturation count |
|---|---:|---:|---:|---:|---:|
| A | 3/18 | 27.12 | 5.48 | 11.69 | 108.0 |
| B | 3/18 | 23.91 | 15.88 | 12.92 | 109.9 |
| C | 3/18 | 19.43 | 15.73 | 13.47 | 101.6 |
| D | 3/18 | 19.44 | 15.70 | 13.47 | 101.4 |
| E | 3/18 | 34.03 | 5.65 | 11.48 | 115.9 |

The completion column is the corrected controller outcome; a controller
failure does not prove physical infeasibility. Feasibility strata are
**{'feasible_witness': 9, 'unresolved': 9}**. Stage 1B successful traces were rechecked without their
success labels (**17 rows**), and invalid examples remain
evidence of the prior boundary/checker problem.

Stage 1B reported 17/90 completions; Stage 1C reports 15/90 after the endpoint and execution corrections. The difference is retained as a boundary/model-version effect, not a method ranking across versions.

## Mechanism result

Collaborative online failures arise at the first lost opportunity: the initial
finite-body interval is negative, while the online pair commands and q gate do
not create a strong enough M/R opening before M reaches the completion boundary.
The bounded offline candidate can use a predeclared wait and roughly -2 m/s²
M/R action, so a candidate witness is not equivalent to online success. The
opening revision is common to A–E and its benefit is not attributed to D.

Same-state full replay maximum error is 0. Fixed nonzero W has mean absolute partner term 0.0044 s⁻¹, while deleting the partner term has 0.0000 s⁻¹. These are local intervention effects; D is not presumed to win.

The single revision was checked on the four key environments for all five
methods (40 targeted runs): 7/20 completed with the
common opening term and 5/20 without it. This is a targeted mechanism
comparison, not a new tuned matrix; the full 90-run Stage 1C table uses the
opening term for every method.

The full closed-loop ablation table keeps the same six configured intervention
environments. It is read after the same-state action table, so an instantaneous
term difference is not treated as a task-level gain. The complete rows are in
`outputs/stage1c/closed_loop_interventions.csv`.

* C `fixed_W`: Δ all-vehicle cost=-0.046 m, completions=2/6 versus full 2/6.
* C `no_partner`: Δ all-vehicle cost=+0.416 m, completions=2/6 versus full 2/6.
* C `no_space`: Δ all-vehicle cost=+9.731 m, completions=2/6 versus full 2/6.
* D `symmetric_k`: Δ all-vehicle cost=-0.014 m, completions=2/6 versus full 2/6.

The five prompt-priority candidates were rechecked explicitly in
`outputs/stage1c/known_candidate_recheck.csv`; the two saved ample traces pass,
the two collaborative i0 candidates pass to maneuver completion only, and the
collaborative i2 wait=2 s / -2 m/s² candidate is retained with its independent
failure reason rather than silently promoted to a witness.

## Figures and limits

* `F:\Swarmalator–CAV\outputs\stage1c\figures\task_window_space.png` shows the 170 m boundary and representative motion.
* `F:\Swarmalator–CAV\outputs\stage1c\figures\q_action_chain.png` separates q feedback from nominal/actual action and marks post-terminal release.
* `F:\Swarmalator–CAV\outputs\stage1c\figures\costs_all_vs_witness.png` compares all-sample and witness-stratum costs and completions.

The main remaining issue is the gap between a strict offline witness and an
online controller that discovers it from the current state. The common revision
does not establish a robust feasible envelope. Communication noise/delay,
SUMO/CARLA, MARL, and large parameter scans remain outside Stage 1C.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1c --config configs/stage1c_config.json --out outputs/stage1c
```

The feedback archive is `feedback/stage1c_feedback.zip`; Stage 1 and Stage 1B
archives remain unchanged.
