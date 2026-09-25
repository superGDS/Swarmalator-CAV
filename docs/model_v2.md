# Swarmalator–CAV Stage 1C model v2

Stage 1 and Stage 1B outputs are frozen. Stage 1C uses the same four point
vehicles M/R/F/B, SI state `(s, y, v, a)`, 0.05 s discrete execution, a
three-second continuous lateral path for M, and a 16 s observation window.
F remains the scripted environment and B remains a closed-loop IDM follower.

The finite-body slot is `lower = R+(L_R+L_M)/2+d_RM` and
`upper = F-(L_F+L_M)/2-d_MF`, with `width = upper-lower`. The common
non-empty 18 m/s floor is 21.4 m. The process target is
`G_R(q)=G0+Delta_new*q`, where `Delta_new=max(0,21.4-G0)`; the R gradient is
`-(g-G_R)*Delta_new/ell_g^2`. Ample has Delta_new=0, so it has no artificial
process opening.

Stage 1C corrects four implementation boundaries. A completing step with
`M.s>170 m` is rejected even if lateral progress reaches one in that same
step. After the task, R follows M when M is ahead and in main lane overlap,
otherwise F; B continues IDM. The E online target smooths time first,
recomputes the endpoint at that time, and sends current position plus planned
velocity/acceleration to the same tracker. Pre-compute controller state is
logged so same-snapshot interventions restore q, e_M0, c0, previous K, and
previous target time exactly.

One common revision is enabled for every method: when the current slot width is
negative, M and R receive `a_open=-clip(2*max(0,-width)/1.5^2,0,3) m/s^2`.
It is inactive for open slots and is not a D-specific advantage.

The executor applies common acceleration and jerk bounds, records nominal,
safety-target, actual actions and their differences, and keeps the 0.15 m
post-jerk recheck tolerance separate from strict witness h>=0. The independent
checker rebuilds dynamics, F/B rules, body/lateral constraints, strict dynamic
margins, continuity, endpoint and slot. It reports mission and full-window
results separately. A witness is evidence for this finite discrete model, not
a continuous reachability proof or road-safety claim.
