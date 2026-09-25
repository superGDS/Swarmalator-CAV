"""Independent trajectory checks for the Stage 1C four-vehicle model.

The runner contains reporting helpers for the reproducible experiment.  This
module is deliberately separate from it so that a saved trajectory can be
checked without trusting controller outcome counters, events, or feasibility
labels.  The checker uses only the saved states and actions together with the
common model functions.

The two acceptance layers are intentionally separate:

``mission_ok``
    The task was physically completed through its first ``merge_progress == 1``
    sample, with a valid activation, three-second lateral path, endpoint, slot,
    and all hard constraints through that sample.

``physical_constraint_ok_full``
    The entire available trace, including the post-task window, satisfies the
    structural, dynamics, rule, road/body, and hard ``h`` checks.  It does not
    require a task completion, which lets a safe incomplete run be reported as
    an unresolved task rather than as a physical failure.

``full_window_ok``
    Both layers pass and the trace reaches ``cfg.horizon``.  A truncated trace
    is never accepted as a full-window result.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .stage1c_simulation import (
    POST_PHASES,
    VEHICLE_IDS,
    SimConfig,
    SimulationResult,
    VehicleState,
    build_scenario,
    dynamic_gap,
    environment_accel,
    execute_acceleration,
    geometry,
    idm_accel,
    make_vehicles,
)


# The trajectory writer rounds times to six decimal places.  State values are
# written as regular Python floats, so the tolerances below are intentionally
# small relative to the 0.05 s and metre-scale model.
_TIME_TOL = 2.0e-6
_STATE_TOL = 3.0e-5
_BOUND_TOL = 1.0e-7
_PATH_TOL = 3.0e-5
_PROGRESS_TOL = 3.0e-4


def _float(value: object, default: float = float("nan")) -> float:
    """Parse a saved numeric field and reject blank/non-finite values."""

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite(value: object) -> bool:
    return math.isfinite(_float(value))


def _same_time(left: float, right: float, tol: float = _TIME_TOL) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tol


def _state(vid: str, row: Mapping[str, object]) -> VehicleState:
    """Build a state while retaining NaNs for the validation layer."""

    return VehicleState(
        vid,
        _float(row.get("s_m")),
        _float(row.get("y_m")),
        _float(row.get("v_mps")),
        _float(row.get("a_mps2")),
        merge_progress=_float(row.get("merge_progress")),
    )


def _rows_and_metadata(
    result_or_rows: SimulationResult | Sequence[Mapping[str, object]],
    scenario_name: Optional[str],
    disturbance: Optional[str],
    init_variant: Optional[int],
) -> Tuple[List[Mapping[str, object]], Optional[str], Optional[str], Optional[int]]:
    """Extract logs and metadata without reading outcome counters or events."""

    if isinstance(result_or_rows, SimulationResult):
        rows = list(result_or_rows.logs)
        if scenario_name is None:
            scenario_name = result_or_rows.scenario
        if disturbance is None:
            disturbance = result_or_rows.disturbance
        if init_variant is None:
            init_variant = int(result_or_rows.init_variant)
    else:
        # Keeping this duck-typed fallback makes the checker useful with small
        # fixture objects while the public sequence API remains unchanged.
        if hasattr(result_or_rows, "logs") and not isinstance(result_or_rows, (list, tuple)):
            rows = list(getattr(result_or_rows, "logs"))
            if scenario_name is None:
                scenario_name = getattr(result_or_rows, "scenario", None)
            if disturbance is None:
                disturbance = getattr(result_or_rows, "disturbance", None)
            if init_variant is None and getattr(result_or_rows, "init_variant", None) is not None:
                init_variant = int(getattr(result_or_rows, "init_variant"))
        else:
            rows = list(result_or_rows)

    if rows:
        first = rows[0]
        if scenario_name is None and first.get("scenario") not in (None, ""):
            scenario_name = str(first.get("scenario"))
        if disturbance is None and first.get("disturbance") not in (None, ""):
            disturbance = str(first.get("disturbance"))
        if init_variant is None and first.get("init_variant") not in (None, ""):
            raw_variant = _float(first.get("init_variant"), 0.0)
            init_variant = int(raw_variant) if math.isfinite(raw_variant) else 0

    return rows, scenario_name or "ample", disturbance or "none", 0 if init_variant is None else int(init_variant)


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    issues: List[Dict[str, object]],
) -> Tuple[Dict[float, Dict[str, Mapping[str, object]]], List[float], List[float]]:
    """Group rows while preserving duplicate/unknown vehicle evidence."""

    raw: MutableMapping[float, List[Mapping[str, object]]] = defaultdict(list)
    bad_time_count = 0
    for index, row in enumerate(rows):
        t = _float(row.get("t_s"))
        if not math.isfinite(t):
            bad_time_count += 1
            issues.append({"code": "invalid_time", "time": None, "kind": "structure"})
            continue
        raw[round(t, 6)].append(row)

    grouped: Dict[float, Dict[str, Mapping[str, object]]] = {}
    duplicate_times: List[float] = []
    for t in sorted(raw):
        entries = raw[t]
        by_vehicle: Dict[str, Mapping[str, object]] = {}
        duplicate_vids: List[str] = []
        unknown_vids: List[str] = []
        for row in entries:
            vid = str(row.get("vehicle", ""))
            if vid not in VEHICLE_IDS:
                unknown_vids.append(vid)
                continue
            if vid in by_vehicle:
                duplicate_vids.append(vid)
                continue
            by_vehicle[vid] = row
        if duplicate_vids:
            duplicate_times.append(t)
            for vid in sorted(set(duplicate_vids)):
                issues.append({"code": f"duplicate_vehicle_{vid}", "time": t, "kind": "structure"})
        if unknown_vids:
            issues.append({"code": "unknown_vehicle", "time": t, "kind": "structure"})
        if len(entries) != len(VEHICLE_IDS) or set(by_vehicle) != set(VEHICLE_IDS):
            issues.append({"code": "vehicle_set", "time": t, "kind": "structure"})
        grouped[t] = by_vehicle

    if bad_time_count:
        duplicate_times = duplicate_times  # Keep the local useful for debuggers.
    return grouped, sorted(raw), duplicate_times


def _net_gap(leader: VehicleState, follower: VehicleState) -> float:
    return leader.s - follower.s - (leader.length + follower.length) / 2.0


def _h_values(snapshot: Mapping[str, VehicleState], cfg: SimConfig, active: bool) -> Dict[str, float]:
    """Dynamic margin h using the follower velocity in ``snapshot``."""

    values = {
        "h_FR": _net_gap(snapshot["F"], snapshot["R"]) - dynamic_gap(snapshot["R"], cfg),
        "h_RB": _net_gap(snapshot["R"], snapshot["B"]) - dynamic_gap(snapshot["B"], cfg),
    }
    if active:
        values.update({
            "h_FM": _net_gap(snapshot["F"], snapshot["M"]) - dynamic_gap(snapshot["M"], cfg),
            "h_MR": _net_gap(snapshot["M"], snapshot["R"]) - dynamic_gap(snapshot["R"], cfg),
        })
    return values


def _active_corridor(phase: str, progress: float) -> bool:
    """Whether M's reserved corridor is physically occupied at a sample."""

    # EXECUTE at p = 0 is deliberately active.  SUCCESS_RELEASE is retained
    # through the completion sample and the post-task observation window.  A
    # failed partial merge remains active while its physical body is in motion.
    return phase in ("EXECUTE", "SUCCESS_RELEASE") or progress > _PROGRESS_TOL


def _phase_key(phase: str) -> str:
    return {
        "PREPARE": "prepare",
        "EXECUTE": "execute",
        "SUCCESS_RELEASE": "success_release",
        "FAILURE_HANDLING": "failure_handling",
        "TIMEOUT": "timeout",
    }.get(phase, "unknown")


def _record(
    issues: List[Dict[str, object]],
    code: str,
    time_s: Optional[float],
    kind: str,
) -> None:
    """Append an issue once per code/time pair."""

    for old in issues:
        if old.get("code") == code and _same_time(_float(old.get("time")), _float(time_s)):
            return
    issues.append({"code": code, "time": time_s, "kind": kind})


def _issue_codes(
    issues: Iterable[Mapping[str, object]],
    completion_time: Optional[float],
    *,
    full: bool,
) -> List[Mapping[str, object]]:
    """Select issues belonging to the mission prefix or full trace."""

    if full:
        return list(issues)
    out: List[Mapping[str, object]] = []
    for issue in issues:
        t = _float(issue.get("time"))
        if completion_time is None or not math.isfinite(t) or t <= completion_time + _TIME_TOL:
            out.append(issue)
    return out


def _first_issue(issues: Sequence[Mapping[str, object]]) -> Tuple[str, object]:
    if not issues:
        return "", ""
    ordered = sorted(
        enumerate(issues),
        key=lambda item: (float("inf") if not math.isfinite(_float(item[1].get("time"))) else _float(item[1].get("time")), item[0]),
    )
    issue = ordered[0][1]
    return str(issue.get("code", "")), issue.get("time", "")


def _record_h(
    issues: List[Dict[str, object]],
    values: Mapping[str, float],
    time_s: float,
    phase: str,
    minima: MutableMapping[str, float],
) -> int:
    count = 0
    key = _phase_key(phase)
    for name, value in values.items():
        if not math.isfinite(value):
            _record(issues, f"h_invalid_{name}", time_s, "safety")
            count += 1
            continue
        minima[key] = min(minima.get(key, float("inf")), value)
        if value < -_BOUND_TOL:
            _record(issues, f"{name}_negative", time_s, "safety")
            count += 1
    return count


def _snapshot_valid(group: Mapping[str, Mapping[str, object]]) -> bool:
    return set(group) == set(VEHICLE_IDS)


def independent_trajectory_check(
    result_or_rows: SimulationResult | Sequence[Mapping[str, object]],
    cfg: SimConfig,
    scenario_name: Optional[str] = None,
    disturbance: Optional[str] = None,
    init_variant: Optional[int] = None,
) -> Dict[str, object]:
    """Independently validate one saved trajectory.

    ``result_or_rows`` may be a :class:`SimulationResult` or a sequence of log
    mappings.  Metadata arguments override fields in the result/rows.  The
    function never reads ``SimulationResult.metrics['success']`` or its events
    when deciding completion; completion is the first physical sample whose M
    progress reaches one.
    """

    rows, scenario_name, disturbance, init_variant = _rows_and_metadata(
        result_or_rows, scenario_name, disturbance, init_variant
    )
    issues: List[Dict[str, object]] = []
    grouped, times, duplicate_times = _group_rows(rows, issues)

    base: Dict[str, object] = {
        "scenario": scenario_name,
        "disturbance": disturbance,
        "init_variant": init_variant,
        "mission_ok": 0,
        "full_window_ok": 0,
        "physical_constraint_ok_full": 0,
        "full_window_evaluated": 0,
        "full_window_coverage_ok": 0,
        "trace_start_s": "",
        "trace_end_s": "",
        "required_horizon_s": float(cfg.horizon),
        "completion_time_s": "",
        "completion_s_m": "",
        "activation_time_s": "",
        "activation_s_m": "",
        "activation_progress": "",
        "merge_duration_s": "",
        "merge_duration_expected_s": float(cfg.merge_duration),
        "merge_duration_ok": 0,
        "start_zone_ok": 0,
        "activation_slot_ok": 0,
        "endpoint_ok": 0,
        "slot_ok": 0,
        "initial_state_ok": 0,
        "dynamics_ok": 0,
        "f_rule_ok": 0,
        "b_rule_ok": 0,
        "progress_ok": 0,
        "continuity_ok": 0,
        "road_ok": 0,
        "body_overlap_ok": 0,
        "h_violation_count": 0,
        "h_violation_count_mission": 0,
        "h_violation_count_full": 0,
        "structural_error_count": 0,
        "mission_error_count": 0,
        "full_error_count": 0,
        "min_h_mission_m": float("nan"),
        "min_h_full_m": float("nan"),
        "min_h_prepare_m": float("nan"),
        "min_h_execute_m": float("nan"),
        "min_h_success_release_m": float("nan"),
        "min_h_failure_handling_m": float("nan"),
        "min_h_by_phase": {},
        "first_error": "",
        "first_error_time_s": "",
        "first_mission_error": "",
        "first_mission_error_time_s": "",
        "first_full_error": "",
        "first_full_error_time_s": "",
        "error": "",
        "errors": [],
        "missing_full_window": 0,
        "duplicate_times": duplicate_times,
    }

    if not rows or not times:
        _record(issues, "empty_trace", None, "structure")
        base.update({
            "structural_error_count": 1,
            "mission_error_count": 1,
            "full_error_count": 1,
            "first_error": "empty_trace",
            "first_mission_error": "empty_trace",
            "first_full_error": "empty_trace",
            "error": "empty_trace",
            "errors": ["empty_trace"],
        })
        return base

    base["trace_start_s"] = times[0]
    base["trace_end_s"] = times[-1]
    sc = build_scenario(str(scenario_name), int(init_variant))
    expected_initial = make_vehicles(sc, cfg)

    if not _same_time(times[0], 0.0):
        _record(issues, "initial_time", times[0], "structure")
    if times[0] < -_TIME_TOL:
        _record(issues, "negative_time", times[0], "structure")

    # A temporal grid check catches truncation inside the mission and avoids
    # silently treating a sparse offline trace as a full discrete trajectory.
    for t0, t1 in zip(times, times[1:]):
        delta = t1 - t0
        if delta > cfg.dt + _TIME_TOL:
            _record(issues, "time_gap", t0, "structure")
        elif delta < cfg.dt - _TIME_TOL:
            _record(issues, "time_step", t0, "structure")

    initial_ok = True
    first_group = grouped[times[0]]
    if set(first_group) != set(VEHICLE_IDS):
        initial_ok = False
        _record(issues, "initial_vehicle_set", times[0], "structure")
    for vid in VEHICLE_IDS:
        row = first_group.get(vid)
        if row is None:
            continue
        state = _state(vid, row)
        expected = expected_initial[vid]
        for field, actual, wanted in (
            ("s_m", state.s, expected.s),
            ("y_m", state.y, expected.y),
            ("v_mps", state.v, expected.v),
            ("a_mps2", state.a, expected.a),
            ("merge_progress", state.merge_progress, 0.0),
        ):
            if not math.isfinite(actual) or abs(actual - wanted) > _STATE_TOL:
                initial_ok = False
                _record(issues, f"initial_{field}_{vid}", times[0], "structure")
    base["initial_state_ok"] = int(initial_ok)

    # Completion is inferred solely from the first physical progress sample.
    completion_time: Optional[float] = None
    completion_group: Optional[Dict[str, Mapping[str, object]]] = None
    completion_index: Optional[int] = None
    for index, t in enumerate(times):
        group = grouped[t]
        row = group.get("M")
        p = _float(row.get("merge_progress")) if row is not None else float("nan")
        if math.isfinite(p) and p >= 1.0 - _PROGRESS_TOL:
            completion_time, completion_group, completion_index = t, group, index
            break

    activation_time: Optional[float] = None
    activation_group: Optional[Dict[str, Mapping[str, object]]] = None
    activation_index: Optional[int] = None
    for index, t in enumerate(times):
        group = grouped[t]
        row = group.get("M")
        if row is None:
            continue
        phase = str(row.get("task_phase", "PREPARE"))
        p = _float(row.get("merge_progress"))
        if phase == "EXECUTE" or (math.isfinite(p) and p > _PROGRESS_TOL):
            activation_time, activation_group, activation_index = t, group, index
            break

    if completion_time is not None:
        m_completion = completion_group.get("M") if completion_group else None
        completion_state = _state("M", m_completion) if m_completion is not None else None
        base["completion_time_s"] = completion_time
        if completion_state is not None and math.isfinite(completion_state.s):
            base["completion_s_m"] = completion_state.s
        if activation_time is None:
            _record(issues, "completion_without_activation", completion_time, "task")
        elif completion_time < activation_time - _TIME_TOL:
            _record(issues, "completion_before_activation", completion_time, "task")
        if m_completion is not None:
            phase_completion = str(m_completion.get("task_phase", ""))
            if phase_completion not in ("EXECUTE", "SUCCESS_RELEASE"):
                _record(issues, "completion_after_failed_start", completion_time, "task")
        if completion_state is not None and completion_state.s > sc.completion_s + _BOUND_TOL:
            _record(issues, "endpoint_crossed_completion", completion_time, "task")
    else:
        _record(issues, "no_merge_completion", None, "task")

    if activation_time is not None and activation_group is not None:
        m_activation_row = activation_group.get("M")
        if m_activation_row is not None:
            m_activation = _state("M", m_activation_row)
            base["activation_time_s"] = activation_time
            base["activation_s_m"] = m_activation.s
            base["activation_progress"] = m_activation.merge_progress
            start_zone_ok = (
                math.isfinite(m_activation.s)
                and sc.merge_zone[0] - _BOUND_TOL <= m_activation.s <= sc.completion_s + _BOUND_TOL
            )
            base["start_zone_ok"] = int(start_zone_ok)
            if not start_zone_ok:
                _record(issues, "activation_outside_start_zone", activation_time, "task")
            g_activation = geometry(
                {vid: _state(vid, activation_group[vid]) for vid in VEHICLE_IDS}
                if _snapshot_valid(activation_group)
                else {},
                sc,
                cfg,
            ) if _snapshot_valid(activation_group) else None
            if g_activation is None:
                _record(issues, "activation_vehicle_set", activation_time, "task")
            else:
                slot_ok = (
                    g_activation["interval_width"] >= -_BOUND_TOL
                    and g_activation["lower"] - _BOUND_TOL <= m_activation.s <= g_activation["upper"] + _BOUND_TOL
                )
                base["activation_slot_ok"] = int(slot_ok)
                if not slot_ok:
                    _record(issues, "activation_outside_slot", activation_time, "task")
            if not math.isfinite(m_activation.merge_progress) or m_activation.merge_progress > _PROGRESS_TOL:
                _record(issues, "activation_progress_nonzero", activation_time, "task")

    if completion_time is not None and completion_index is not None:
        if activation_time is not None:
            duration = completion_time - activation_time
            base["merge_duration_s"] = duration
            duration_ok = abs(duration - cfg.merge_duration) <= max(_TIME_TOL, cfg.dt * 0.51)
            base["merge_duration_ok"] = int(duration_ok)
            if not duration_ok:
                _record(issues, "merge_duration", completion_time, "task")
        # Progress after a failed start cannot be promoted to a completion.
        for t in times[: completion_index + 1]:
            group = grouped[t]
            row = group.get("M")
            if row is None:
                continue
            p = _float(row.get("merge_progress"))
            m_state = _state("M", row)
            if math.isfinite(m_state.s) and m_state.s > sc.completion_s + _BOUND_TOL and t <= completion_time + _TIME_TOL:
                _record(issues, "endpoint_crossed_before_progress_one", t, "task")
            if t > (activation_time if activation_time is not None else -float("inf")) and p >= 1.0 - _PROGRESS_TOL and str(row.get("task_phase", "")) not in ("EXECUTE", "SUCCESS_RELEASE"):
                _record(issues, "repeated_failed_start_completion", t, "task")

    # Sample-level physical checks.  The active corridor starts at EXECUTE at
    # p = 0 and remains active through release; PREPARE only has FR/RB h.
    phase_minima: Dict[str, float] = {}
    min_h_full = float("inf")
    min_h_mission = float("inf")
    h_count = 0
    h_count_full = 0
    h_count_mission = 0
    road_ok = True
    body_ok = True
    progress_ok = True
    canonical_times: List[float] = []
    canonical_snapshots: Dict[float, Dict[str, VehicleState]] = {}
    canonical_phases: Dict[float, str] = {}
    for t in times:
        group = grouped[t]
        if not _snapshot_valid(group):
            continue
        snap = {vid: _state(vid, group[vid]) for vid in VEHICLE_IDS}
        canonical_times.append(t)
        canonical_snapshots[t] = snap
        phase = str(group["M"].get("task_phase", "PREPARE"))
        canonical_phases[t] = phase
        p = snap["M"].merge_progress
        if not math.isfinite(p) or p < -_PROGRESS_TOL or p > 1.0 + _PROGRESS_TOL:
            progress_ok = False
            _record(issues, "progress_range", t, "structure")
        if math.isfinite(p):
            expected_y = cfg.lane_ramp + (cfg.lane_main - cfg.lane_ramp) * min(max(p, 0.0), 1.0)
            if not math.isfinite(snap["M"].y) or abs(snap["M"].y - expected_y) > _PATH_TOL:
                progress_ok = False
                _record(issues, "lateral_path", t, "structure")
        active = _active_corridor(phase, p)
        h = _h_values(snap, cfg, active)
        this_h_count = _record_h(issues, h, t, phase, phase_minima)
        h_count += this_h_count
        h_count_full += this_h_count
        if completion_time is None or t <= completion_time + _TIME_TOL:
            h_count_mission += this_h_count
        if h:
            hmin = min(h.values())
            min_h_full = min(min_h_full, hmin)
            if completion_time is None or t <= completion_time + _TIME_TOL:
                min_h_mission = min(min_h_mission, hmin)

        # Longitudinal/lateral body overlap is checked for every pair.  M is
        # naturally inactive in PREPARE because its y is still on the ramp.
        for first, second in combinations(VEHICLE_IDS, 2):
            one, two = snap[first], snap[second]
            if not all(math.isfinite(x) for x in (one.s, one.y, two.s, two.y)):
                continue
            if abs(one.y - two.y) <= (one.width + two.width) / 2.0 + _BOUND_TOL and abs(one.s - two.s) < (one.length + two.length) / 2.0 - _BOUND_TOL:
                body_ok = False
                _record(issues, f"body_overlap_{first}{second}", t, "safety")
        for vid, state in snap.items():
            if not all(math.isfinite(x) for x in (state.s, state.y, state.v, state.a)):
                road_ok = False
                _record(issues, f"nonfinite_state_{vid}", t, "structure")
                continue
            if state.s < -1.0 - _BOUND_TOL or state.s > sc.road_max + _BOUND_TOL:
                road_ok = False
                _record(issues, f"road_exit_{vid}", t, "safety")
            if state.y < -0.5 - _BOUND_TOL or state.y > cfg.lane_ramp + 0.5 + _BOUND_TOL:
                road_ok = False
                _record(issues, f"lateral_exit_{vid}", t, "safety")
            if state.v < cfg.v_min - _BOUND_TOL or state.v > cfg.v_max + _BOUND_TOL:
                _record(issues, f"speed_bound_{vid}", t, "dynamics")

    # Explicit progress/lateral continuity and exact three-second task timing.
    continuity_ok = True
    if activation_index is not None:
        end_index = completion_index if completion_index is not None else len(times) - 1
        for index in range(activation_index, min(end_index, len(times) - 1) + 1):
            t = times[index]
            group = grouped[t]
            row = group.get("M")
            if row is None:
                continuity_ok = False
                _record(issues, "merge_vehicle_set", t, "structure")
                continue
            p = _float(row.get("merge_progress"))
            if not math.isfinite(p):
                continuity_ok = False
                continue
            if index > activation_index:
                previous_group = grouped[times[index - 1]]
                previous_row = previous_group.get("M")
                previous_p = _float(previous_row.get("merge_progress")) if previous_row else float("nan")
                delta_p = p - previous_p
                expected_delta = (times[index] - times[index - 1]) / max(cfg.merge_duration, cfg.dt)
                if not math.isfinite(previous_p) or delta_p < -_PROGRESS_TOL or delta_p > expected_delta + _PROGRESS_TOL:
                    continuity_ok = False
                    _record(issues, "progress_discontinuity", t, "task")
            if completion_time is not None and t <= completion_time + _TIME_TOL:
                expected_p = min(max((t - activation_time) / max(cfg.merge_duration, cfg.dt), 0.0), 1.0) if activation_time is not None else p
                if abs(p - expected_p) > _PROGRESS_TOL:
                    continuity_ok = False
                    _record(issues, "progress_timing", t, "task")
            phase = str(row.get("task_phase", "PREPARE"))
            if index > activation_index and completion_time is not None and t < completion_time - _TIME_TOL and phase not in ("EXECUTE",):
                continuity_ok = False
                _record(issues, "release_before_completion", t, "task")
    base["continuity_ok"] = int(continuity_ok)
    base["progress_ok"] = int(progress_ok)

    # Dynamics and F/B rule checks are transition checks.  F is always the
    # prescribed environment acceleration and its actual action is exactly the
    # jerk-limited execution of that prescription.  B's nominal action is IDM
    # against the current R; its actual action may be safety projected.
    dynamics_ok = True
    f_rule_ok = True
    b_rule_ok = True
    for t0, t1 in zip(canonical_times, canonical_times[1:]):
        if not _same_time(t1 - t0, cfg.dt, max(_TIME_TOL, cfg.dt * 0.001)):
            # The temporal structural issue was already recorded above.
            continue
        cur = grouped[t0]
        nxt = grouped[t1]
        for vid in VEHICLE_IDS:
            row0, row1 = cur[vid], nxt[vid]
            state0, state1 = canonical_snapshots[t0][vid], canonical_snapshots[t1][vid]
            actual = _float(row0.get("a_actual_mps2"))
            if not math.isfinite(actual):
                dynamics_ok = False
                _record(issues, f"missing_actual_action_{vid}", t0, "dynamics")
                continue
            if actual < cfg.a_min - _BOUND_TOL or actual > cfg.a_max + _BOUND_TOL:
                dynamics_ok = False
                _record(issues, f"accel_bound_{vid}", t0, "dynamics")
            if state0.v < cfg.v_min - _BOUND_TOL or state0.v > cfg.v_max + _BOUND_TOL:
                dynamics_ok = False
                _record(issues, f"speed_bound_{vid}", t0, "dynamics")
            predicted_s = state0.s + state0.v * cfg.dt + 0.5 * actual * cfg.dt * cfg.dt
            predicted_v = min(cfg.v_max, max(cfg.v_min, state0.v + actual * cfg.dt))
            if not math.isfinite(state1.s) or abs(state1.s - predicted_s) > _STATE_TOL:
                dynamics_ok = False
                _record(issues, f"dynamics_s_{vid}", t0, "dynamics")
            if not math.isfinite(state1.v) or abs(state1.v - predicted_v) > _STATE_TOL:
                dynamics_ok = False
                _record(issues, f"dynamics_v_{vid}", t0, "dynamics")
            if not math.isfinite(state1.a) or abs(state1.a - actual) > _STATE_TOL:
                dynamics_ok = False
                _record(issues, f"dynamics_a_{vid}", t0, "dynamics")
            jerk = (actual - state0.a) / cfg.dt
            if abs(jerk) > cfg.jerk_max + _BOUND_TOL:
                dynamics_ok = False
                _record(issues, f"jerk_bound_{vid}", t0, "dynamics")
            saved_jerk = _float(row0.get("jerk_mps3"))
            if math.isfinite(saved_jerk) and abs(saved_jerk - jerk) > _STATE_TOL:
                dynamics_ok = False
                _record(issues, f"saved_jerk_{vid}", t0, "dynamics")

        # F is not a controller-controlled vehicle.  A safety projection may
        # alter M/R/B, but F must retain its scripted nominal and jerk action.
        f_row = cur["F"]
        f_state = canonical_snapshots[t0]["F"]
        try:
            expected_f = environment_accel(f_state, t0, sc, str(disturbance), cfg)
        except (TypeError, ValueError):
            expected_f = float("nan")
        f_nom = _float(f_row.get("a_nom_mps2"))
        f_actual = _float(f_row.get("a_actual_mps2"))
        if not math.isfinite(expected_f) or not math.isfinite(f_nom) or abs(f_nom - expected_f) > _STATE_TOL:
            f_rule_ok = False
            _record(issues, "F_nominal_script", t0, "rule")
        if math.isfinite(expected_f) and math.isfinite(f_state.a):
            expected_f_actual = execute_acceleration(f_state.a, expected_f, cfg)[0]
            if not math.isfinite(f_actual) or abs(f_actual - expected_f_actual) > _STATE_TOL:
                f_rule_ok = False
                _record(issues, "F_actual_script", t0, "rule")
        # The safety target is the executable, jerk-limited action target. It
        # must agree with the actual F action; comparing it to the raw
        # environment prescription rejects otherwise valid first steps.
        f_target = _float(f_row.get("a_safety_target_mps2"))
        if math.isfinite(f_target) and math.isfinite(f_actual) and abs(f_target - f_actual) > _STATE_TOL:
            f_rule_ok = False
            _record(issues, "F_safety_target_actual_mismatch", t0, "rule")

        b_row = cur["B"]
        b_state, r_state = canonical_snapshots[t0]["B"], canonical_snapshots[t0]["R"]
        try:
            expected_b = idm_accel(b_state, r_state, cfg)
        except (TypeError, ValueError):
            expected_b = float("nan")
        b_nom = _float(b_row.get("a_nom_mps2"))
        if not math.isfinite(expected_b) or not math.isfinite(b_nom) or abs(b_nom - expected_b) > _STATE_TOL:
            b_rule_ok = False
            _record(issues, "B_idm_rule", t0, "rule")

        # Evaluate h after this action using the actual next speed, rather than
        # the current follower speed.  This catches a margin lost inside a
        # discrete step even if a tampered next row happens to hide it.
        next_snap: Dict[str, VehicleState] = {}
        for vid in VEHICLE_IDS:
            source = canonical_snapshots[t0][vid]
            actual = _float(cur[vid].get("a_actual_mps2"))
            if not math.isfinite(actual):
                continue
            next_snap[vid] = VehicleState(
                vid,
                source.s + source.v * cfg.dt + 0.5 * actual * cfg.dt * cfg.dt,
                canonical_snapshots[t1][vid].y,
                min(cfg.v_max, max(cfg.v_min, source.v + actual * cfg.dt)),
                actual,
                length=source.length,
                width=source.width,
                merge_progress=canonical_snapshots[t1][vid].merge_progress,
            )
        if set(next_snap) == set(VEHICLE_IDS):
            next_phase = canonical_phases[t1]
            next_active = _active_corridor(next_phase, next_snap["M"].merge_progress)
            h_next = _h_values(next_snap, cfg, next_active)
            count_next = _record_h(issues, h_next, t0, next_phase, phase_minima)
            h_count += count_next
            h_count_full += count_next
            if completion_time is None or t1 <= completion_time + _TIME_TOL:
                h_count_mission += count_next
            if h_next:
                next_min = min(h_next.values())
                min_h_full = min(min_h_full, next_min)
                if completion_time is None or t1 <= completion_time + _TIME_TOL:
                    min_h_mission = min(min_h_mission, next_min)

    base["dynamics_ok"] = int(dynamics_ok)
    base["f_rule_ok"] = int(f_rule_ok)
    base["b_rule_ok"] = int(b_rule_ok)
    base["road_ok"] = int(road_ok)
    base["body_overlap_ok"] = int(body_ok)
    base["h_violation_count"] = h_count
    base["h_violation_count_full"] = h_count_full
    base["h_violation_count_mission"] = h_count_mission
    base["min_h_full_m"] = min_h_full if math.isfinite(min_h_full) else float("nan")
    base["min_h_mission_m"] = min_h_mission if math.isfinite(min_h_mission) else float("nan")
    for key in ("prepare", "execute", "success_release", "failure_handling"):
        value = phase_minima.get(key, float("nan"))
        base[f"min_h_{key}_m"] = value
    base["min_h_by_phase"] = {
        key: phase_minima.get(key, float("nan"))
        for key in ("prepare", "execute", "success_release", "failure_handling")
    }

    # Endpoint and slot are checked against the completion sample, not a
    # success event.  A sample at p = 1 after the deadline is already a failed
    # start and remains rejected even if its geometry happens to fit.
    if completion_group is not None:
        if _snapshot_valid(completion_group):
            end_snapshot = {vid: _state(vid, completion_group[vid]) for vid in VEHICLE_IDS}
            end_m = end_snapshot["M"]
            endpoint_ok = math.isfinite(end_m.s) and end_m.s <= sc.completion_s + _BOUND_TOL
            base["endpoint_ok"] = int(endpoint_ok)
            if not endpoint_ok:
                _record(issues, "endpoint_crossed_completion", completion_time, "task")
            g_end = geometry(end_snapshot, sc, cfg)
            slot_ok = (
                g_end["interval_width"] >= -_BOUND_TOL
                and g_end["lower"] - _BOUND_TOL <= end_m.s <= g_end["upper"] + _BOUND_TOL
            )
            base["slot_ok"] = int(slot_ok)
            if not slot_ok:
                _record(issues, "endpoint_outside_slot", completion_time, "task")
        else:
            _record(issues, "completion_vehicle_set", completion_time, "task")

    # A failed-start path that crosses the endpoint before p = 1 is explicitly
    # distinct from a later repeated p = 1 sample.  Both prevent mission pass.
    if completion_time is not None and activation_time is not None:
        for t in times:
            if t <= activation_time + _TIME_TOL or t > completion_time + _TIME_TOL:
                continue
            row = grouped[t].get("M")
            if row is None:
                continue
            m = _state("M", row)
            if math.isfinite(m.s) and m.s > sc.completion_s + _BOUND_TOL and m.merge_progress < 1.0 - _PROGRESS_TOL:
                _record(issues, "failed_start_before_completion", t, "task")

    # Full-window coverage is a separate structural fact.  We still evaluate
    # every available sample above, but no missing tail can be called full.
    reaches_horizon = times[-1] >= cfg.horizon - _TIME_TOL
    no_extra_time = times[-1] <= cfg.horizon + _TIME_TOL
    coverage_ok = bool(reaches_horizon and no_extra_time and _same_time(times[0], 0.0))
    if not coverage_ok:
        _record(issues, "missing_full_window", times[-1], "full_window")
    base["full_window_coverage_ok"] = int(coverage_ok)
    base["full_window_evaluated"] = int(coverage_ok)
    base["missing_full_window"] = int(not coverage_ok)

    # Classify issues after completion has been inferred.  Task issues are
    # mission failures, while physical/rule/dynamics issues after completion
    # only invalidate the full observation window.
    mission_issues = _issue_codes(issues, completion_time, full=False)
    full_issues = list(issues)
    physical_full_issues = [issue for issue in full_issues if issue.get("kind") not in ("task", "full_window")]
    if not coverage_ok:
        # The full physical result is explicitly not evaluated without the
        # requested observation window, even if all available rows are clean.
        physical_full_issues = list(physical_full_issues) + [{"code": "missing_full_window", "time": times[-1], "kind": "full_window"}]

    endpoint_and_duration_ok = bool(
        completion_time is not None
        and activation_time is not None
        and base["start_zone_ok"]
        and base["activation_slot_ok"]
        and base["merge_duration_ok"]
        and base["endpoint_ok"]
        and base["slot_ok"]
    )
    # ``mission_issues`` includes h/rule/dynamics errors through completion;
    # the explicit condition prevents a completion from passing with no
    # physical samples or an invalid state set.
    mission_ok = bool(endpoint_and_duration_ok and not mission_issues)
    physical_full_ok = bool(coverage_ok and not physical_full_issues)
    full_ok = bool(mission_ok and physical_full_ok and not full_issues)

    first_all, first_all_time = _first_issue(full_issues)
    first_mission, first_mission_time = _first_issue(mission_issues)
    first_full, first_full_time = _first_issue(full_issues)
    display_errors = [str(issue.get("code", "")) for issue in mission_issues + [x for x in full_issues if x not in mission_issues]]
    # Preserve order but avoid a long repeated string in CSV/report output.
    unique_errors: List[str] = []
    for code in display_errors:
        if code and code not in unique_errors:
            unique_errors.append(code)

    base.update({
        "mission_ok": int(mission_ok),
        "physical_constraint_ok_full": int(physical_full_ok),
        "full_window_ok": int(full_ok),
        "structural_error_count": sum(issue.get("kind") == "structure" for issue in full_issues),
        "mission_error_count": len(mission_issues),
        "full_error_count": len(full_issues),
        "first_error": first_all,
        "first_error_time_s": first_all_time,
        "first_mission_error": first_mission,
        "first_mission_error_time_s": first_mission_time,
        "first_full_error": first_full,
        "first_full_error_time_s": first_full_time,
        "error": ";".join(unique_errors[:20]),
        "errors": unique_errors,
    })
    return base


__all__ = ["independent_trajectory_check"]
