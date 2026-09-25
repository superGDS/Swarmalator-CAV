# Swarmalator–CAV Stage 2A model v3

Stage 2A uses a causal finite-window planner followed by a common jerk-limited
executor. The controlled vehicles are M (the merging vehicle) and R (the rear
vehicle in the target gap); F is the observed/predicted front boundary and B
follows R with the declared IDM model. A controller receives the current
four-vehicle state, the continuously propagated public reference, and the
current task phase. It never receives the disturbance name, scenario name,
initial-variant identifier, success label, or a future F script.

For a candidate absolute start time `t_s` and bounded role intensities
`z=(z_M,z_R)`, the predictor rolls the state through the preparation interval,
the fixed `merge_duration`, and `post_preview_s`. The internal state is a
bounded preparation intensity, not the physical lateral progress `q_M`. It is
mapped to a longitudinal spatial reference through the target accelerations
`a_target,M/R = -3 z_M/R`; the public reference then uses the same current
state feedback and jerk-limited acceleration update as execution. Thus z can
request a costly transient gap expansion or hold while the planner decides
when to start. The physical lateral progress still advances only during the
declared task execution.

Each candidate uses the vectorized form of the same executor used by the
simulation. It clips acceleration, velocity, and jerk; projects the ordered
F–M–R–B (or F–R–B before overlap) chain onto one-step dynamic margins; and
records empty action intersections instead of calling the projection a safety
proof. Candidate selection first filters valid finite-window task candidates,
then compares all-vehicle speed deficit and action effort. A failed bounded
search is reported as `finite_search_no_valid_plan`, which is unresolved and
is not a physical infeasibility claim.

P is this finite space–time planner without an internal state law. S1 keeps
the same planner and reference family but disables the new space-to-process
gradient and partner term. S2 adds a finite-difference gradient of the
predicted task potential together with symmetric partner action. S3 preserves
the same total partner budget but allocates it asymmetrically from the current
rear dynamic margin. These are causal state updates; the planner still owns
absolute timing and candidate feasibility. `oldC` is the Stage1C C coordination
law run through the common Stage2A reference and executor, so it is a cross
structure diagnostic rather than a reproduction of the old output.

The reference state stores `(s_ref, v_ref, a_ref)` and propagates acceleration
with the jerk bound. It is continuous through q completion and release; no
endpoint reset removes a derivative term. After completion, R follows the
current physical leader occupancy and the same safety layer remains active.
The independent checker reports three labels: geometric/event completion,
mission-valid completion through the declared maneuver, and full-window
validity through the observation horizon. Net body clearance and dynamic
margin `h` are logged separately, as are nominal, executable safety-target,
and actual actions.

This is a transparent engineering prototype inspired by planner–constraint
correction–tracking separations in cooperative CAV and swarmalator work. It is
not a reproduction of either paper, does not claim distributed communication,
continuous reachability, or road-safety certification, and keeps the limits of
the finite development matrix explicit.
