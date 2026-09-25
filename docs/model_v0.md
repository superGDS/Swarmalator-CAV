# Model v0: implemented equations and audit

## Scope

The simulator uses a single longitudinal road coordinate `s` (m), lateral center coordinate `y` (m), speed `v` (m/s), and acceleration `a` (m/s²). Acceleration is integrated with a common jerk bound (m/s³). Vehicle length is 4.8 m and width is 1.9 m. M starts in a ramp lane at `y=3.6 m` and follows a smooth linear lateral path to the mainline; its physical `merge_progress` is separate from its internal task process.

The configured merge zone is `55 ≤ s ≤ 170 m`, the task terminal position is `s=190 m`, and the longitudinal observation road bound is `−1 ≤ s ≤ 450 m`. The 16 s observation window continues all four vehicle dynamics after a task succeeds or fails so post-task external cost is not truncated.

The finite-body front/rear gap is

```text
g_FR = s_F − s_R − (L_F + L_R)/2.
```

At each snapshot, with dynamic distance margins `d = 2.0 + 0.35 v`, the slot bounds are

```text
lower = s_R + (L_R + L_M)/2 + d_RM
upper = s_F − (L_F + L_M)/2 − d_MF
c = (lower + upper)/2.
```

`lower > upper` is retained as an empty slot; the controller may still compute a reference, but the task cannot be declared successful.

## Internal process and feedback

For M and R, `q_i ∈ [0,1]` is a dimensionless task-preparation process. It is neither a yaw angle, a cooperative probability, nor the physical lateral completion fraction. The gap target is

```text
G_R(q_R) = G0 + ΔG q_R,
Φ_R = 0.5 ((g_FR − G_R)/ell_g)^2.
```

For M, at the first task snapshot `e_M0 = s_M − c` is captured and

```text
s_M_ref = c + (1 − q_M) e_M0,
Φ_M = 0.5 ((s_M − s_M_ref)/ell_s)^2.
```

The implemented gradients (holding the observed geometry terms fixed while differentiating the task reference) are

```text
∂Φ_R/∂q_R = −(g_FR − G_R) ΔG / ell_g²
∂Φ_M/∂q_M = (s_M − s_M_ref) e_M0 / ell_s².
```

The M and R nominal longitudinal actions track `s_M_ref` or an online timing target and a gap-dependent R speed reference. For C and D,

```text
qdot_i = clip(ν_i − κ_i ∂Φ_i/∂q_i
              + K_ij W_ij sin(α(q_j−q_i)), 0, ν_max).
```

The rates have units s⁻¹; `Φ`, q, and the sine are dimensionless; `κ`, `K`, and ν therefore have units s⁻¹. `α = π/2` keeps the coupling direction monotone over the finite q range. The process freezes when a task is terminal and is clipped at 1.0. The degenerate cases `e_M0=0` and `ΔG=0` yield zero for the corresponding feedback term by construction.

For C, `K_MR = K_RM = K_total/2`. For D, `K_MR` is a smoothed sigmoid allocation from interpretable current acceleration and distance/gap margins, `K_RM = K_total − K_MR`, and the pair sum is logged. The convention is `K_ij`: i responds to j. A larger `K_MR` means M follows the R process more strongly; it does not mean M has priority.

## Baselines and execution

A has no q. B integrates q at its base rate and lets q influence the spatial targets, but sets space and partner process terms to zero. E rolls a small candidate arrival-time search using only current M/R/F state and current gap geometry; it is an independent engineering timing baseline, not a reproduction of the opinion-dynamics paper.

All methods use the same jerk-limited executor, finite-body predicted-gap correction, lateral path, road bounds, and hard body-overlap test. B follows R with a closed-loop IDM-like law. F follows a desired-speed rule plus an explicit preparation disturbance. The simulator updates all vehicle actions from one snapshot before moving any vehicle.

## Literature boundary and modifications

The cited RA-L work was available only through DOI/abstract metadata in this workspace. It motivates separating a space–phase planner from dynamics/constraint execution, but no robot parameters or CBF equation is copied. The CAV opinion-dynamics work was available through public indexed snippets/abstract; it motivates state-aware proposal/preference/consensus followed by rolling motion execution. The present q, gap geometry, non-reciprocal allocation, executor, and four-vehicle merge task are this project's independent design.

The minimum modifications from the candidate prompt were: finite-body dynamic margins instead of bare scalar gaps; explicit holding of observed geometry in the q-gradient; `α=π/2` for monotone finite-range coupling; clipping and terminal freezing; a separate physical lateral progress variable; and a common safety/jerk layer whose corrections are logged rather than hidden.
