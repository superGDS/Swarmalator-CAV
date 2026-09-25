# Swarmalator–CAV Stage 1 report

This is a first reproducible CPU prototype, not a confirmation study or a road-safety claim. The deliverable contains 90 paired evaluations: 5 methods × 3 physical scenarios × 2 preparation-disturbance conditions × 3 initial-state variants. All methods use the same snapshot timing, finite-body geometry, longitudinal action limits, jerk-limited executor, lateral execution rule, safety checks, horizon, and four-vehicle cost accounting.

## Actual model and necessary corrections

The implemented state is `(s, y, v, a)` in SI units plus a task process `q_i ∈ [0,1]` for M and R. M's physical lateral progress is stored separately as `merge_progress`; q reaching 1 does not declare a merge success. The target gap is `G_R(q_R) = G0 + ΔG q_R`. The M reference is `c + (1−q_M)e_M0`, where `c` is the current finite-body slot midpoint and `e_M0` is captured at task start.

The code uses the candidate gradients after checking their signs. For R, `∂Φ_R/∂q_R = −(g_FR−G_R)ΔG/ell_g²`, so a deficient gap slows q_R. For M, `∂Φ_M/∂q_M = (s_M−s_ref)e_M0/ell_s²`; with M initially behind the slot (`e_M0 < 0`), a lagging spatial position slows q_M. The qdot sum is clipped to `[0, ν_max]`, frozen after a terminal task state, and logged by base, space, partner, and clipping components. `α = π/2` keeps `sin(α(q_j−q_i))` monotone over the finite q range. When `e_M0=0` or `ΔG=0`, the corresponding space feedback is exactly zero, which is treated as a meaningful ablation/degeneracy rather than hidden error.

The executor applies common acceleration and jerk limits, then predicts finite-body gaps for F–R, R–B, and M during lateral execution. It records nominal action, safety-adjusted target, actual action, jerk, and correction reasons. A task is completed only when M has physically traversed the lateral path and lies within the current finite-body slot. Failures remain in the tables.

## Paired results

| Method | Configuration | n | Completion | Mean all-vehicle speed-deficit integral (m·s) | Mean minimum net gap (m) | Mean safety corrections | Mean absolute jerk (m/s³) | Outcome counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A | A space-gap baseline | 18 | 44.4% | 15.33 | 10.70 | 20.4 | 0.310 | {'completed': 8, 'started_but_failed': 4, 'missed_window': 6} |
| B | B one-way process→space | 18 | 33.3% | 15.52 | 10.86 | 7.6 | 0.246 | {'completed': 6, 'started_but_failed': 6, 'missed_window': 6} |
| C | C symmetric bidirectional | 18 | 33.3% | 15.22 | 10.86 | 7.6 | 0.232 | {'completed': 6, 'started_but_failed': 6, 'missed_window': 6} |
| D | D non-reciprocal bidirectional | 18 | 33.3% | 15.22 | 10.86 | 7.3 | 0.232 | {'completed': 6, 'started_but_failed': 6, 'missed_window': 6} |
| E | E online timing baseline | 18 | 33.3% | 8.13 | 11.96 | 3.7 | 0.264 | {'completed': 6, 'started_but_failed': 6, 'missed_window': 6} |

Failure/outcome counts across all methods: `{'completed': 32, 'started_but_failed': 28, 'missed_window': 30}`.

The speed-deficit integral is a fixed-window proxy `∫ max(0, v_des−v) dt` summed over M/R/F/B; it is not claimed as travel-time delay. Per-vehicle values, nominal–actual action differences, jerk, and correction counts are in `vehicle_metrics.csv`.

## Mechanism check

The saved trajectories show whether the channels were numerically active. Mean absolute space-feedback rates were 0.0227 s⁻¹ for C and 0.0227 s⁻¹ for D. The mean absolute gain imbalance `|K_MR−K_RM|` was 0.0000 s⁻¹ for C and 0.0676 s⁻¹ for D, while each D run kept the pair sum at the configured `k_total` to the recorded tolerance (`pair_gain_total_error_max_s-1` in the metrics table). These are implementation/mechanism results; they do not by themselves establish a traffic benefit.

The non-reciprocal scatter uses the real per-run M and R speed-deficit integrals and safety-correction counts. The representative plot was selected from `D_collaborative_prepare_i1` and is not selected by outcome after looking at all results.

## Figures and data provenance

* `outputs\stage1\figures\stage1_representative_trajectory.png`: saved trajectory, actual/target gap, q, and nominal/actual action for one real run; dashed vertical lines are logged task events.
* `outputs\stage1\figures\stage1_method_outcomes.png`: completion and fixed-window speed-deficit cost from `metrics.csv`.
* `outputs\stage1\figures\stage1_nonreciprocal_tradeoff.png`: gain allocation and role cost from `metrics.csv` plus `vehicle_metrics.csv`.

The plotting code rereads those CSV files before rendering. `trajectories.csv`, `events.csv`, and `metrics.csv` are the authoritative run outputs; `run_log.txt` contains the invocation and test status.

## Literature boundary

No PDF was present in the workspace. The RA-L swarmalator paper was checked through the IEEE DOI/abstract and related public metadata, but its requested pages 2, 3, and 6–8 were not available for full-text verification. It supports the high-level distinction between space–phase planning and dynamics/constraint execution; its robot parameters and CBF details were not used as traffic calibration.

The CAV opinion-dynamics paper was checked through the ScienceDirect indexed abstract/full-text snippets and SSRN abstract. It is treated as a state-aware plan–preference–consensus–rolling-execution reference, not as a no-motion-feedback method and not as a direct reproduction. The present E baseline is an independent rolling candidate timing controller, not that paper's roundabout algorithm.

## Limitations and next step

This prototype has a single target gap, ideal instantaneous messages, deterministic F disturbance scripts, simplified longitudinal dynamics, and a continuous lateral path rather than a bicycle model. It does not model communication loss, SUMO/CARLA, human drivers, or a formal CBF/QP. Short-window failures are labelled constraint outcomes or feasibility-uncertain failures, not proofs of physical impossibility. Safety corrections can mask upper-layer differences, and the selected target gap/thresholds can make the scenario insensitive. The next most valuable experiment is a controlled feasibility envelope with a separately verified safe trajectory and a delayed/noisy message ablation; this would distinguish physical-window limits from controller-specific failure and test whether the observed non-reciprocity survives information imperfections.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.swarmalator_cav.run_stage1 --config configs/stage1_config.json --out outputs/stage1
```

