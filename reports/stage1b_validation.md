# Stage 1B validation record

The final validation command was:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

It completed with **8 passed**. The Stage 1B runner then completed **90 primary
runs**, **18 feasibility environments**, **36 bounded intervention runs**, and
**56 same-snapshot controller rows** using `.venv\Scripts\python.exe`.

Independent post-run checks found:

* 0 hard body-overlap or road-exit counts in the primary table;
* minimum observed finite-body net gap 3.690 m;
* 17 completed, 70 missed-window, and 3 started-but-failed outcomes;
* 9 `feasible_witness` and 9 `unresolved` feasibility labels, with no
  `excluded_by_necessary_bound` label in this matrix;
* all five method-specific replay spot checks passed the independent dynamics,
  action, jerk, finite-body and road checks;
* the feedback archive passed `zipfile.testzip()` and contains 29 light members.

The 458 aggregate post-jerk dynamic-gap recheck flags are retained in the
metrics and events. They reflect a bounded executor correction that remains
short of the dynamic margin under jerk limits; they were not recoded as hard
collisions or as physical infeasibility.
