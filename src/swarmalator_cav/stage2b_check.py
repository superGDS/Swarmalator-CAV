"""Stage2B validation labels and reference continuity checks."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from .stage1c_check import independent_trajectory_check
from .stage1c_simulation import SimConfig, SimulationResult


def _f(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def reference_continuity_check(result_or_rows: SimulationResult | Sequence[Mapping[str, object]],
                               cfg: SimConfig) -> dict:
    """Check the reference sent to M without accepting a hard q endpoint jump.

    The check is deliberately independent of the task success label.  It
    reports finite reference values, the largest sample-to-sample velocity
    jump, and whether the reference remains defined through the active task.
    The threshold is a diagnostic bound (15 m/s2 over one 0.05 s sample), not a
    claimed vehicle safety limit; the common actuator checker remains the
    authority for actual motion.
    """
    rows = result_or_rows.logs if isinstance(result_or_rows, SimulationResult) else list(result_or_rows)
    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("vehicle")) == "M":
            grouped[round(_f(row.get("t_s")), 6)].append(row)
    ordered = [grouped[t][0] for t in sorted(grouped)]
    active = [row for row in ordered if str(row.get("task_phase", "")) in ("PREPARE", "EXECUTE")]
    missing = [row for row in active if not all(math.isfinite(_f(row.get(k))) for k in ("s_ref_m", "s_ref_dot_mps", "s_ref_ddot_mps2"))]
    jumps = []
    accelerations = []
    for left, right in zip(active, active[1:]):
        dt = _f(right.get("t_s")) - _f(left.get("t_s"))
        if dt <= 0:
            continue
        dv = abs(_f(right.get("s_ref_dot_mps")) - _f(left.get("s_ref_dot_mps")))
        jumps.append(dv)
        accelerations.append(dv / dt)
    max_jump = max(jumps, default=0.0)
    max_accel = max(accelerations, default=0.0)
    # 15 m/s2 is deliberately much looser than the 2.5 m/s3 vehicle jerk cap;
    # it catches q=1 derivative resets while allowing current-boundary motion.
    ok = not missing and max_accel <= 15.0 + 1.0e-9
    terminal_dot = _f(active[-1].get("s_ref_dot_mps")) if active else float("nan")
    return {
        "reference_ok": int(ok),
        "active_reference_rows": len(active),
        "missing_reference_rows": len(missing),
        "max_reference_velocity_jump_mps": max_jump,
        "max_reference_acceleration_mps2": max_accel,
        "terminal_reference_velocity_mps": terminal_dot,
        "error": "missing_reference" if missing else ("reference_velocity_jump" if not ok else ""),
    }


def validate_stage2b(result: SimulationResult, cfg: SimConfig) -> dict:
    check = independent_trajectory_check(result, cfg)
    ref = reference_continuity_check(result, cfg)
    geometry_completed = any(str(event.get("event")) == "merge_completed" for event in result.events)
    out = dict(check)
    # Keep independent physical error codes intact; reference continuity is a
    # separate diagnostic and must not overwrite the checker error field.
    for key, value in ref.items():
        out["reference_" + key] = value
    out["geometry_completed"] = int(geometry_completed)
    out["mission_valid"] = int(check.get("mission_ok", 0) and ref.get("reference_ok", 0))
    out["full_window_valid"] = int(check.get("full_window_ok", 0) and ref.get("reference_ok", 0))
    out["combined_error"] = ";".join(v for v in (str(check.get("error", "")), str(ref.get("error", ""))) if v)
    return out


__all__ = ["reference_continuity_check", "validate_stage2b"]
