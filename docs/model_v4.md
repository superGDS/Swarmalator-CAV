# Swarmalator–CAV Stage 2B model v4

Stage2B keeps the Stage2A finite-window joint predictor but makes the public control arms and common execution path explicit. The state is `X_i=(s_i,v_i,a_i,y_i,q_i)` for M (merge), R (rear gap vehicle), F (front boundary) and B (rear follower). The bounded internal variable `z_i∈[0,1]` is preparation intensity; it is not phase or physical lateral progress. Its target acceleration is `a_target,i=-3 z_i` (m/s²).

A candidate absolute start and `(z_M,z_R)` are rolled through preparation, the 3 s lateral execution and a 1.25 s release preview. The observed Stage2A empty-action transient occurred about 0.65 s after completion; M braking release at the 2.5 m/s³ jerk cap takes about `|a|/J`, so the common handoff duration is fixed at 1.0 s and the extra preview is 0.25 s. This is a finite terminal condition, not a recursive-feasibility proof.

Prediction and execution call the same `control_command` interface. In PREPARE/EXECUTE it propagates `(s_ref,v_ref,a_ref)` with the declared jerk bound and tracks the current state. In SUCCESS_RELEASE/FAILURE_HANDLING it switches to current-occupancy `post_targets`; after the common 1.0 s successful handoff the logged control stage is `normal_follow`. No position or velocity reset is used. F uses the current declared disturbance input and B uses current-state IDM.

The action chain is `nominal → actuator interval → safety projection → actual`. An empty actuator interval (acceleration, jerk or speed limits) is distinct from an empty ordered-chain safety intersection. On either empty set the fallback is logged as invalid and is never called safe.

The four main arms are P, joint finite prediction and candidate selection without added state-law terms; S1, the same predictor/reference with gradient and partner terms zero; S2, S1 plus finite-difference task-potential gradient and symmetric partner action; and S3, S2 with the same total partner budget allocated from the current rear margin. The `symmetric_K` S3 intervention uses the 0.5/0.5 allocation as S2 and clones its decision under the same controller state. Planning is centralized over the four-vehicle snapshot; distributed communication is not implemented.

Geometry/event completion, mission validity through the first q=1 sample, and full-window validity through 16 s remain separate. The cost is fixed four-vehicle speed-deficit distance, with prepare, execute, handoff, normal-follow and failure-handling stages retained. Pairwise deltas use the same environment and the same valid-success denominator.
