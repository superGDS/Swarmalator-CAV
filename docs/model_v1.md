# Swarmalator–CAV Stage 1B model v1

Stage 1B is a versioned companion to `simulation.py`; the Stage 1 runner and
`outputs/stage1/` are frozen. The state is `(s,y,v,a)` in SI units for M/R/F/B.
M follows a continuous three-second lateral path and `merge_progress` is
physical progress independent of q. F is the scripted environment and B uses
the same IDM response to R for every method.

The slot is finite-body: `lower = R+(L_R+L_M)/2+d_RM` and
`upper = F-(L_F+L_M)/2-d_MF`. Therefore `width = G_FR-L_M-d_RM-d_MF`,
and the common 18 m/s floor is `4.8+2*(2+0.35*18)=21.4 m`. Every method uses
`target_gap_physical_m=max(G0,21.4)`; process q targets are logged separately.

M uses reference position/speed/acceleration feedforward plus position/speed
feedback. B/C/D include the current same-snapshot qdot derivative in
`s_ref=c+(1-q_M)e_M0`, while D uses direction-aware action/time margins rather
than the old `a_max-abs(a_nom)` proxy. E's rolling candidate is converted into
the same one-step reference interface. A uniform 18 m/s reference with zero
position/speed error has zero command.

The v1 task boundaries are start zone [55,170] m, completion 170 m, the old
190 m field retained as a post-task marker, and a 450 m observation road. A
common three-second maximum-braking/jerk necessary startup check is separated
from q thresholds. Phases are PREPARE, EXECUTE, SUCCESS_RELEASE,
FAILURE_HANDLING, and TIMEOUT. Success uses normal cruise/following; an
execute failure continues the started lateral path under the same safety layer,
and an unstarted failure stays on the ramp.

The bounded post-jerk prediction is logged with a 0.15 m numerical tolerance;
hard body/road checks use actual states. Speed-deficit is named distance (m),
with the v0-compatible alias retained in new tables. Role minima are computed
from the actual FR/RB/MF/MR gaps involving each vehicle. Terminal rows retain
the last real diagnostics. This remains a deterministic discrete prototype,
not a continuous reachability proof or empirical traffic calibration.
