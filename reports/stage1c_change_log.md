# Stage 1C change log

The workspace is not a Git repository, so this file records the version boundary
for review. Stage 1 and Stage 1B files and outputs were left unchanged. Stage 1C
is implemented only in `src/swarmalator_cav/stage1c_simulation.py`,
`src/swarmalator_cav/run_stage1c.py`, `configs/stage1c_config.json`,
`docs/model_v2.md`, `reports/stage1c_*`, `tests/test_stage1c.py`, and
`outputs/stage1c/`.

The corrections are: same-step completion rejection above 170 m; task-release
front selection for R; endpoint-consistent E timing reference; physical-target
consistent q gradient; pre-compute controller-state logging and clone replay;
strict independent mission/full-window checking; a common finite-window opening
reference; and a jerk/velocity-bounded one-step safety projection. The only
model revision is the geometry-derived opening reference, used by every method
and ablated in `outputs/stage1c/revision_comparison.csv`.

Validation at delivery: 17 pytest tests passed; 90 primary runs, 18 feasibility
rows, 5 prompt-priority candidate rechecks, 60 same-state rows, 36 closed-loop
ablation runs, and 40 targeted revision comparison runs completed. The light archive is
`feedback/stage1c_feedback.zip`.
