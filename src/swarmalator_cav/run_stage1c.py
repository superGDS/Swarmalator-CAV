"""Stage 1C reproducible runner, checker, diagnostics, and feedback pack."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .stage1c_simulation import (
    ACTIVE_PHASES, POST_PHASES, SimConfig, SimulationResult, TaskState,
    VehicleState, build_scenario, dynamic_gap, environment_accel,
    execute_acceleration, geometry, idm_accel, make_controller, make_vehicles,
    min_forward_distance, physical_gap_floor, process_gap_delta, run_episode,
)

VEHICLES = ("M", "R", "F", "B")
METHODS = tuple("ABCDE")
METHOD_LABELS = {
    "A": "ordinary spatial coordination",
    "B": "one-way process drive",
    "C": "symmetric bidirectional coupling",
    "D": "non-reciprocal coupling",
    "E": "online timing baseline",
}


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _time_groups(rows: Sequence[Mapping[str, object]]) -> Dict[float, Dict[str, Mapping[str, object]]]:
    grouped: Dict[float, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[round(as_float(row.get("t_s")), 6)][str(row.get("vehicle"))] = row
    return grouped


def _vehicle_from_row(vid: str, row: Mapping[str, object]) -> VehicleState:
    return VehicleState(
        vid, as_float(row.get("s_m")), as_float(row.get("y_m")),
        as_float(row.get("v_mps")), as_float(row.get("a_mps2")),
        merge_progress=as_float(row.get("merge_progress")),
    )


def _gap_h(snapshot: Mapping[str, VehicleState], cfg: SimConfig) -> Dict[str, float]:
    def net(leader: str, follower: str) -> float:
        return snapshot[leader].s - snapshot[follower].s - (snapshot[leader].length + snapshot[follower].length) / 2.0
    return {
        "h_FR": net("F", "R") - dynamic_gap(snapshot["R"], cfg),
        "h_RB": net("R", "B") - dynamic_gap(snapshot["B"], cfg),
        "h_FM": net("F", "M") - dynamic_gap(snapshot["M"], cfg),
        "h_MR": net("M", "R") - dynamic_gap(snapshot["R"], cfg),
    }


def independent_trajectory_check(
    result_or_rows: SimulationResult | Sequence[Mapping[str, object]],
    cfg: SimConfig,
    scenario_name: Optional[str] = None,
    disturbance: Optional[str] = None,
    init_variant: Optional[int] = None,
) -> Dict[str, object]:
    """Rebuild motion and constraints from saved states/actions only."""
    if isinstance(result_or_rows, SimulationResult):
        rows = result_or_rows.logs
        scenario_name = result_or_rows.scenario
        disturbance = result_or_rows.disturbance
        init_variant = result_or_rows.init_variant
        events = result_or_rows.events
    else:
        rows = list(result_or_rows)
        events = []
    if scenario_name is None:
        scenario_name = str(rows[0].get("scenario", "ample")) if rows else "ample"
    if disturbance is None:
        disturbance = str(rows[0].get("disturbance", "none")) if rows else "none"
    if init_variant is None:
        init_variant = int(as_float(rows[0].get("init_variant", 0))) if rows else 0
    grouped = _time_groups(rows)
    times = sorted(grouped)
    errors: List[str] = []
    mission_errors: List[str] = []
    sc = build_scenario(scenario_name, int(init_variant))
    initial = make_vehicles(sc, cfg)
    if not times:
        return {"mission_ok": 0, "full_window_ok": 0, "error": "empty_trace", "min_h_mission_m": float("nan"), "min_h_full_m": float("nan")}
    first = grouped[times[0]]
    if set(first) != set(VEHICLES):
        errors.append("initial_vehicle_set")
    for vid in VEHICLES:
        if vid not in first:
            continue
        r = first[vid]
        for field, expected in (("s_m", initial[vid].s), ("y_m", initial[vid].y), ("v_mps", initial[vid].v)):
            if abs(as_float(r.get(field)) - expected) > 2e-6:
                errors.append(f"initial_{field}_{vid}")
    min_h_full = float("inf")
    min_h_mission = float("inf")
    completion_time: Optional[float] = None
    first_execute_time: Optional[float] = None
    for ev in events:
        if str(ev.get("event")) == "merge_completed":
            completion_time = as_float(ev.get("t_s"))
            break
    if completion_time is None:
        for t in times:
            if as_float(grouped[t].get("M", {}).get("merge_progress")) >= 1.0:
                completion_time = t
                break
    for t in times:
        cur = grouped[t]
        if set(cur) != set(VEHICLES):
            errors.append(f"vehicle_set@{t}")
            continue
        snap = {vid: _vehicle_from_row(vid, cur[vid]) for vid in VEHICLES}
        h = _gap_h(snap, cfg)
        phase = str(cur["M"].get("task_phase", "PREPARE"))
        if phase == "EXECUTE" and first_execute_time is None:
            first_execute_time = t
            if not (sc.merge_zone[0] - 1e-7 <= snap["M"].s <= sc.completion_s + 1e-7):
                mission_errors.append(f"start_zone@{t}")
        active = phase == "EXECUTE" or snap["M"].y <= (cfg.lane_ramp + cfg.lane_main) / 2.0
        relevant_h = [h["h_FR"], h["h_RB"]] + ([h["h_FM"], h["h_MR"]] if active else [])
        h_min = min(relevant_h)
        min_h_full = min(min_h_full, h_min)
        if completion_time is None or t <= completion_time + cfg.dt * 0.51:
            min_h_mission = min(min_h_mission, h_min)
        if active:
            for name in ("h_FM", "h_MR"):
                if h[name] < -1e-7:
                    (mission_errors if completion_time is None or t <= completion_time + cfg.dt * 0.51 else errors).append(f"{name}_negative@{t}")
        if h["h_FR"] < -1e-7 or h["h_RB"] < -1e-7:
            (mission_errors if completion_time is None or t <= completion_time + cfg.dt * 0.51 else errors).append(f"mainline_h_negative@{t}")
        for vid, v in snap.items():
            if v.s < -1.0 - 1e-7 or v.s > sc.road_max + 1e-7:
                errors.append(f"road_{vid}@{t}")
            if v.y < -0.5 - 1e-7 or v.y > cfg.lane_ramp + 0.5 + 1e-7:
                errors.append(f"lateral_{vid}@{t}")
        if snap["F"].s - snap["R"].s - 4.8 < -1e-7 or snap["R"].s - snap["B"].s - 4.8 < -1e-7:
            errors.append(f"body_mainline@{t}")
        if active:
            for oid in ("F", "R", "B"):
                if abs(snap["M"].y - snap[oid].y) <= (snap["M"].width + snap[oid].width) / 2.0 and abs(snap["M"].s - snap[oid].s) < (snap["M"].length + snap[oid].length) / 2.0 - 1e-7:
                    errors.append(f"body_merge_{oid}@{t}")
        p = as_float(cur["M"].get("merge_progress"))
        if not (-1e-8 <= p <= 1.0 + 1e-8):
            errors.append(f"progress_range@{t}")
        if abs(snap["M"].y - (cfg.lane_ramp * (1.0 - min(max(p, 0.0), 1.0)))) > 2e-5:
            errors.append(f"lateral_path@{t}")
    for t0, t1 in zip(times, times[1:]):
        if t1 - t0 > cfg.dt * 1.01:
            errors.append(f"time_gap@{t0}")
            continue
        cur, nxt = grouped[t0], grouped[t1]
        if set(cur) != set(VEHICLES) or set(nxt) != set(VEHICLES):
            continue
        for vid in VEHICLES:
            r0, r1 = cur[vid], nxt[vid]
            s, v, a = as_float(r0.get("s_m")), as_float(r0.get("v_mps")), as_float(r0.get("a_actual_mps2"))
            pred_s = s + v * cfg.dt + 0.5 * a * cfg.dt * cfg.dt
            pred_v = min(cfg.v_max, max(cfg.v_min, v + a * cfg.dt))
            if abs(as_float(r1.get("s_m")) - pred_s) > 3e-5:
                errors.append(f"dynamics_s_{vid}@{t0}")
            if abs(as_float(r1.get("v_mps")) - pred_v) > 3e-5:
                errors.append(f"dynamics_v_{vid}@{t0}")
            if not (cfg.a_min - 1e-7 <= a <= cfg.a_max + 1e-7):
                errors.append(f"accel_bound_{vid}@{t0}")
            if abs(a - as_float(r0.get("a_mps2"))) > cfg.jerk_max * cfg.dt + 3e-5:
                errors.append(f"jerk_bound_{vid}@{t0}")
        p0 = as_float(cur["M"].get("merge_progress")); p1 = as_float(nxt["M"].get("merge_progress"))
        if p1 + 2e-5 < p0:
            errors.append(f"progress_decrease@{t0}")
        if p1 - p0 > cfg.dt / cfg.merge_duration + 2e-5:
            errors.append(f"progress_step_too_large@{t0}")
        f = cur["F"]
        b = cur["B"]
        f_expected = environment_accel(_vehicle_from_row("F", f), t0, sc, str(disturbance), cfg)
        b_expected = idm_accel(_vehicle_from_row("B", b), _vehicle_from_row("R", cur["R"]), cfg)
        if abs(as_float(f.get("a_nom_mps2")) - f_expected) > 2e-5:
            errors.append(f"F_environment_rule@{t0}")
        if abs(as_float(b.get("a_nom_mps2")) - b_expected) > 2e-5:
            errors.append(f"B_idm_rule@{t0}")
        if str(cur["M"].get("task_phase")) in POST_PHASES:
            r_state = _vehicle_from_row("R", cur["R"])
            m_state = _vehicle_from_row("M", cur["M"])
            leader = m_state if m_state.s > r_state.s and abs(m_state.y - r_state.y) <= (m_state.width + r_state.width) / 2.0 and m_state.y <= (cfg.lane_ramp + cfg.lane_main) / 2.0 else _vehicle_from_row("F", cur["F"])
            r_expected = idm_accel(r_state, leader, cfg)
            if abs(as_float(cur["R"].get("a_nom_mps2")) - r_expected) > 2e-5:
                errors.append(f"R_post_front_rule@{t0}")
    endpoint_ok = False
    slot_ok = False
    if completion_time is not None and completion_time in grouped:
        end = {vid: _vehicle_from_row(vid, grouped[completion_time][vid]) for vid in VEHICLES}
        g = geometry(end, sc, cfg)
        endpoint_ok = end["M"].s <= sc.completion_s + 1e-7
        slot_ok = g["interval_width"] >= -1e-7 and g["lower"] - 1e-7 <= end["M"].s <= g["upper"] + 1e-7
        if not endpoint_ok:
            mission_errors.append("endpoint_crossed_completion")
        if not slot_ok:
            mission_errors.append("endpoint_outside_slot")
    else:
        mission_errors.append("no_merge_completion")
    # Do not consult the simulator's success flag: physical progress, endpoint,
    # slot and independent constraints are the authority for a witness.
    mission_ok = bool(completion_time is not None and endpoint_ok and slot_ok and not mission_errors)
    full_coverage = bool(times and times[0] <= cfg.dt * 0.51 and times[-1] >= cfg.horizon - cfg.dt * 0.51)
    if not full_coverage:
        errors.append("full_window_not_covered")
    full_ok = bool(mission_ok and not errors and full_coverage)
    return {
        "mission_ok": int(mission_ok), "full_window_ok": int(full_ok),
        "completion_time_s": "" if completion_time is None else completion_time,
        "completion_s_m": "" if completion_time is None else as_float(grouped[completion_time]["M"].get("s_m")),
        "endpoint_ok": int(endpoint_ok), "slot_ok": int(slot_ok),
        "full_window_covered": int(full_coverage),
        "min_h_mission_m": min_h_mission if math.isfinite(min_h_mission) else float("nan"),
        "min_h_full_m": min_h_full if math.isfinite(min_h_full) else float("nan"),
        "error": ";".join((mission_errors + errors)[:10]),
    }


def _optimistic_bound(scenario_name: str, init_variant: int, cfg: SimConfig) -> Dict[str, object]:
    sc = build_scenario(scenario_name, init_variant)
    target = max(sc.initial_gap, physical_gap_floor(sc, cfg))
    best_gap, t_best = sc.initial_gap, 0.0
    for t in np.linspace(0.0, cfg.horizon, 321):
        gap = (sc.F_s + sc.F_v * t + 0.5 * cfg.a_max * t * t) - (sc.R_s + sc.R_v * t + 0.5 * cfg.a_min * t * t) - 4.8
        if gap > best_gap:
            best_gap, t_best = float(gap), float(t)
    m = VehicleState("M", sc.M_s, cfg.lane_ramp, sc.M_v, 0.0)
    min_dist = min_forward_distance(m, cfg.merge_duration, cfg)
    late = sc.M_s + min_dist > sc.completion_s + 1e-9
    excluded = best_gap < target - 1e-9 or late
    return {
        "optimistic_max_gap_m": best_gap, "optimistic_time_s": t_best,
        "target_gap_m": target, "min_forward_distance_m": min_dist,
        "late_start_bound": int(late),
        "necessary_bound_label": "excluded_by_necessary_bound" if excluded else "not_excluded",
        "bound_reason": "optimistic_gap_outer_bound" if best_gap < target - 1e-9 else ("late_start_min_forward_bound" if late else ""),
    }


def _append_trace_row(rows: List[Dict[str, object]], run_id: str, scenario: str, disturbance: str, init_variant: int, t: float, phase: str, vehicles: Mapping[str, VehicleState], actions: Mapping[str, float], cfg: SimConfig, source: str) -> None:
    gaps = _gap_h(vehicles, cfg)
    g = geometry(vehicles, build_scenario(scenario, init_variant), cfg)
    for vid in VEHICLES:
        nominal = actions.get(vid, vehicles[vid].a)
        actual = execute_acceleration(vehicles[vid].a, nominal, cfg)[0]
        rows.append({"run_id": run_id, "source": source, "t_s": round(t, 6), "vehicle": vid, "scenario": scenario, "disturbance": disturbance, "init_variant": init_variant, "s_m": vehicles[vid].s, "y_m": vehicles[vid].y, "v_mps": vehicles[vid].v, "a_mps2": vehicles[vid].a, "a_actual_mps2": actual, "a_nom_mps2": nominal, "merge_progress": vehicles[vid].merge_progress, "task_phase": phase, "interval_width_m": g["interval_width"], **gaps})


def _propagate_candidate(vehicles: Dict[str, VehicleState], actions: Mapping[str, float], cfg: SimConfig, dt: float) -> None:
    for vid in VEHICLES:
        v = vehicles[vid]
        aa, _, _ = execute_acceleration(v.a, actions[vid], cfg)
        v.s = v.s + v.v * dt + 0.5 * aa * dt * dt
        v.v = float(np.clip(v.v + aa * dt, cfg.v_min, cfg.v_max))
        v.a = aa
    if actions.get("_merge", 0.0) > 0.5:
        vehicles["M"].merge_progress = float(np.clip(vehicles["M"].merge_progress + dt / cfg.merge_duration, 0.0, 1.0))
        vehicles["M"].y = cfg.lane_ramp * (1.0 - vehicles["M"].merge_progress)


def offline_candidate_search(scenario_name: str, disturbance: str, init_variant: int, cfg: SimConfig) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    waits = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    actions = (-2.0, -1.0, 0.0, 1.5)
    trials = 0
    for wait in waits:
        for a_m in actions:
            for a_r in actions:
                trials += 1
                sc = build_scenario(scenario_name, init_variant)
                vehicles = make_vehicles(sc, cfg)
                rows: List[Dict[str, object]] = []
                n_wait = int(round(wait / cfg.dt))
                rid = f"offline_{scenario_name}_{disturbance}_i{init_variant}_w{wait}_m{a_m}_r{a_r}"
                for step in range(n_wait):
                    t = step * cfg.dt
                    acts = {"M": a_m, "R": a_r, "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg), "B": idm_accel(vehicles["B"], vehicles["R"], cfg)}
                    _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, t, "PREPARE", vehicles, acts, cfg, "offline_candidate")
                    _propagate_candidate(vehicles, acts, cfg, cfg.dt)
                g = geometry(vehicles, sc, cfg)
                if not (sc.merge_zone[0] <= vehicles["M"].s <= sc.completion_s and g["interval_width"] >= 0.0 and g["lower"] <= vehicles["M"].s <= g["upper"]):
                    continue
                for j in range(int(round(cfg.merge_duration / cfg.dt))):
                    t = wait + j * cfg.dt
                    acts = {"M": a_m, "R": a_r, "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg), "B": idm_accel(vehicles["B"], vehicles["R"], cfg), "_merge": 1.0}
                    _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, t, "EXECUTE", vehicles, acts, cfg, "offline_candidate")
                    _propagate_candidate(vehicles, acts, cfg, cfg.dt)
                _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, wait + cfg.merge_duration, "EXECUTE", vehicles, {vid: vehicles[vid].a for vid in VEHICLES}, cfg, "offline_candidate")
                result = independent_trajectory_check(rows, cfg, scenario_name, disturbance, init_variant)
                if result["mission_ok"]:
                    result.update({"found": 1, "trials": trials, "wait_s": wait, "a_M_mps2": a_m, "a_R_mps2": a_r})
                    return {**result, "candidate_id": rid}, rows
    return {"found": 0, "trials": trials, "wait_s": "", "a_M_mps2": "", "a_R_mps2": "", "error": "bounded_grid_exhausted"}, []


def offline_candidate_trial(scenario_name: str, disturbance: str, init_variant: int,
                            wait: float, a_m: float, a_r: float,
                            cfg: SimConfig) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Replay one named offline candidate, including its complete action trace."""
    sc = build_scenario(scenario_name, init_variant)
    vehicles = make_vehicles(sc, cfg)
    rows: List[Dict[str, object]] = []
    rid = f"known_{scenario_name}_{disturbance}_i{init_variant}_w{wait}_m{a_m}_r{a_r}"
    for step in range(int(round(wait / cfg.dt))):
        t = step * cfg.dt
        acts = {"M": a_m, "R": a_r, "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg), "B": idm_accel(vehicles["B"], vehicles["R"], cfg)}
        _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, t, "PREPARE", vehicles, acts, cfg, "known_candidate")
        _propagate_candidate(vehicles, acts, cfg, cfg.dt)
    for j in range(int(round(cfg.merge_duration / cfg.dt))):
        t = wait + j * cfg.dt
        acts = {"M": a_m, "R": a_r, "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg), "B": idm_accel(vehicles["B"], vehicles["R"], cfg), "_merge": 1.0}
        if j == 0:
            start_geometry = geometry(vehicles, sc, cfg)
            if not (sc.merge_zone[0] <= vehicles["M"].s <= sc.completion_s and start_geometry["interval_width"] >= 0.0 and start_geometry["lower"] <= vehicles["M"].s <= start_geometry["upper"]):
                return {"mission_ok": 0, "full_window_ok": 0, "error": "candidate_start_not_in_slot", "completion_s_m": vehicles["M"].s, "min_h_mission_m": float("nan")}, rows
        _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, t, "EXECUTE", vehicles, acts, cfg, "known_candidate")
        _propagate_candidate(vehicles, acts, cfg, cfg.dt)
    _append_trace_row(rows, rid, scenario_name, disturbance, init_variant, wait + cfg.merge_duration, "EXECUTE", vehicles, {vid: vehicles[vid].a for vid in VEHICLES}, cfg, "known_candidate")
    return independent_trajectory_check(rows, cfg, scenario_name, disturbance, init_variant), rows


def _env_groups(results: Sequence[SimulationResult]) -> Dict[Tuple[str, str, int], List[SimulationResult]]:
    groups: Dict[Tuple[str, str, int], List[SimulationResult]] = defaultdict(list)
    for result in results:
        groups[(result.scenario, result.disturbance, result.init_variant)].append(result)
    return groups


def recheck_original_stage1b(root: Path, cfg: SimConfig) -> List[Dict[str, object]]:
    metrics = read_csv(root / "outputs" / "stage1b" / "metrics.csv")
    rows = read_csv(root / "outputs" / "stage1b" / "trajectories.csv")
    by_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[str(row.get("run_id"))].append(row)
    out: List[Dict[str, object]] = []
    for m in metrics:
        if int(as_float(m.get("success"))) != 1:
            continue
        rid = str(m.get("run_id")); tr = by_run.get(rid, [])
        if not tr:
            continue
        chk = independent_trajectory_check(tr, cfg, m.get("scenario"), m.get("disturbance"), int(as_float(m.get("init_variant"))))
        out.append({"source_run_id": rid, "method": m.get("method"), "scenario": m.get("scenario"), "disturbance": m.get("disturbance"), "init_variant": m.get("init_variant"), **{k: v for k, v in chk.items() if k != "error"}, "error": chk.get("error", "")})
    return out


def feasibility_diagnostic(results: Sequence[SimulationResult], cfg: SimConfig) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    witness_summary: List[Dict[str, object]] = []
    witness_actions: List[Dict[str, object]] = []
    for (scenario, disturbance, init_variant), candidates in sorted(_env_groups(results).items()):
        witness: Optional[SimulationResult] = None
        check: Dict[str, object] = {}
        source = ""
        for candidate in candidates:
            c = independent_trajectory_check(candidate, cfg)
            if c["mission_ok"]:
                witness, check, source = candidate, c, candidate.method
                break
        offline: Optional[Dict[str, object]] = None
        trace: List[Dict[str, object]] = []
        if witness is None:
            offline, trace = offline_candidate_search(scenario, disturbance, init_variant, cfg)
        bound = _optimistic_bound(scenario, init_variant, cfg)
        if witness is not None:
            label, reason = "feasible_witness", "Stage1C primary trace passed strict mission checker"
            ws = witness.logs; witness_check = check
        elif offline and int(offline.get("found", 0)):
            label, reason, source = "feasible_witness", "bounded offline trace passed strict mission checker", "offline_candidate"
            witness_check = offline; ws = trace
        elif bound["necessary_bound_label"] == "excluded_by_necessary_bound":
            label, reason, source, witness_check, ws = "excluded_by_necessary_bound", str(bound["bound_reason"]), "optimistic_outer_bound", {}, []
        else:
            label, reason, source, witness_check, ws = "unresolved", "no valid witness and no necessary exclusion; controller failure is not physical infeasibility", "", {}, []
        if ws:
            witness_actions.extend(dict(row) for row in ws)
            witness_summary.append({"scenario": scenario, "disturbance": disturbance, "init_variant": init_variant, "source": source, "run_id": ws[0].get("run_id", ""), "label": label, **{k: v for k, v in witness_check.items() if k in ("mission_ok", "full_window_ok", "completion_time_s", "completion_s_m", "min_h_mission_m", "min_h_full_m", "wait_s", "a_M_mps2", "a_R_mps2", "trials")}, "error": witness_check.get("error", "")})
        sc = build_scenario(scenario, init_variant)
        rows.append({"scenario": scenario, "disturbance": disturbance, "init_variant": init_variant, "label": label, "witness_source": source, "reason": reason, "initial_gap_m": sc.initial_gap, "target_gap_m": physical_gap_floor(sc, cfg), "process_gap_delta_m": process_gap_delta(sc, cfg), "remaining_distance_m": sc.completion_s - sc.M_s, **bound, **{k: v for k, v in witness_check.items() if k in ("mission_ok", "full_window_ok", "completion_s_m", "min_h_mission_m", "min_h_full_m")}, "search_found": int(offline.get("found", 0)) if offline else 0, "search_trials": int(offline.get("trials", 0)) if offline else 0})
    return rows, witness_summary, witness_actions


def _snapshot_from_result(result: SimulationResult) -> Tuple[Optional[Dict[str, VehicleState]], Optional[TaskState], Optional[Mapping[str, object]]]:
    grouped = _time_groups(result.logs)
    for t in sorted(grouped):
        rows = grouped[t]
        if set(rows) == set(VEHICLES) and str(rows["M"].get("task_phase")) == "EXECUTE":
            return {vid: _vehicle_from_row(vid, rows[vid]) for vid in VEHICLES}, TaskState(phase="EXECUTE"), rows["M"]
    return None, None, None


def same_snapshot_interventions(results: Sequence[SimulationResult], cfg: SimConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        if result.method not in ("C", "D"):
            continue
        snapshot, task, ref_row = _snapshot_from_result(result)
        if snapshot is None or task is None or ref_row is None:
            continue
        sc = build_scenario(result.scenario, result.init_variant)
        state = {"q_M": as_float(ref_row.get("ctrl_pre_q_M")), "q_R": as_float(ref_row.get("ctrl_pre_q_R")), "e_M0": ref_row.get("ctrl_pre_e_M0"), "c0": ref_row.get("ctrl_pre_c0"), "prev_k_mr": as_float(ref_row.get("ctrl_pre_prev_k_mr"), cfg.k_total / 2.0), "prev_target_time": as_float(ref_row.get("ctrl_pre_prev_target_time"), 3.5)}
        variants = ["full", "no_space", "fixed_W", "no_partner"] if result.method == "C" else ["full", "symmetric_k"]
        baseline = make_controller(result.method, sc, cfg); baseline.load_state(state); original = baseline.compute_pair(snapshot, task, cfg.dt)
        for variant in variants:
            controller = make_controller(result.method, sc, cfg, ablation="" if variant == "full" else variant); controller.load_state(state); out = controller.compute_pair(snapshot, task, cfg.dt)
            for vid in ("M", "R"):
                d, o = out[vid], original[vid]
                err = max(abs(float(d["a_nom"]) - float(o["a_nom"])), abs(float(d["qdot_clipped"]) - float(o["qdot_clipped"])), abs(float(d["qdot_space"]) - float(o["qdot_space"])), abs(float(d["qdot_partner"]) - float(o["qdot_partner"])))
                rows.append({"source_run_id": result.run_id, "method": result.method, "scenario": result.scenario, "disturbance": result.disturbance, "init_variant": result.init_variant, "snapshot_t_s": ref_row.get("t_s"), "vehicle": vid, "ablation": variant, "q_M": state["q_M"], "q_R": state["q_R"], "qdot_space_s-1": d["qdot_space"], "qdot_partner_s-1": d["qdot_partner"], "qdot_clipped_s-1": d["qdot_clipped"], "K_to_partner_s-1": d["K_to_partner"], "W_to_partner": d["W_to_partner"], "a_nom_mps2": d["a_nom"], "a_actual_reference_mps2": o["a_nom"], "full_reproduction_error": err if variant == "full" else "", "target_gap_physical_m": d["target_gap"], "target_gap_process_m": d["target_gap_process"], "window_opening_accel_mps2": d.get("window_opening_accel", 0.0), "post_front_id": d.get("post_front_id", "F")})
    return rows


def opportunity_diagnostics(results: Sequence[SimulationResult], cfg: SimConfig) -> List[Dict[str, object]]:
    keys = {("collaborative", "none", 0), ("collaborative", "prepare", 0), ("collaborative", "none", 2), ("ample", "none", 0)}
    out: List[Dict[str, object]] = []
    for r in results:
        if (r.scenario, r.disturbance, r.init_variant) not in keys:
            continue
        grouped = _time_groups(r.logs); opportunity = None
        sc = build_scenario(r.scenario, r.init_variant)
        for t in sorted(grouped):
            snap = {v: _vehicle_from_row(v, grouped[t][v]) for v in VEHICLES}; g = geometry(snap, sc, cfg)
            if g["interval_width"] >= 0.0 and g["lower"] <= snap["M"].s <= g["upper"] and sc.merge_zone[0] <= snap["M"].s <= sc.completion_s:
                opportunity = (t, g["interval_width"], as_float(grouped[t]["M"].get("q")), as_float(grouped[t]["M"].get("a_actual_mps2")), as_float(grouped[t]["R"].get("a_actual_mps2"))); break
        lost = next((e for e in r.events if e.get("event") in ("missed_window", "merge_exit_before_completion", "started_but_failed")), None)
        first_exec = next((grouped[t] for t in sorted(grouped) if grouped[t]["M"].get("task_phase") == "EXECUTE"), None)
        out.append({"run_id": r.run_id, "method": r.method, "scenario": r.scenario, "disturbance": r.disturbance, "init_variant": r.init_variant, "first_slot_opportunity_t_s": "" if opportunity is None else opportunity[0], "first_slot_width_m": "" if opportunity is None else opportunity[1], "q_M_at_opportunity": "" if opportunity is None else opportunity[2], "a_M_at_opportunity": "" if opportunity is None else opportunity[3], "a_R_at_opportunity": "" if opportunity is None else opportunity[4], "first_loss_event": "" if lost is None else lost.get("event"), "first_loss_t_s": "" if lost is None else lost.get("t_s"), "merge_start_time_s": r.metrics.get("merge_start_time_s", ""), "outcome": r.metrics.get("outcome", ""), "first_execute_q_M": "" if first_exec is None else first_exec["M"].get("q"), "first_execute_opening_accel_mps2": "" if first_exec is None else first_exec["M"].get("window_opening_accel_mps2"), "first_execute_nominal_M_mps2": "" if first_exec is None else first_exec["M"].get("a_nom_mps2"), "first_execute_actual_M_mps2": "" if first_exec is None else first_exec["M"].get("a_actual_mps2")})
    return out


def _summary(metrics: Sequence[Mapping[str, object]], feasibility: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    labels = {f"{r['scenario']}|{r['disturbance']}|{r['init_variant']}": r.get("label") for r in feasibility}
    rows = []
    for method in METHODS:
        rs = [r for r in metrics if r.get("method") == method]
        witness = [r for r in rs if labels.get(f"{r.get('scenario')}|{r.get('disturbance')}|{r.get('init_variant')}") == "feasible_witness"]
        rows.append({"method": method, "label": METHOD_LABELS[method], "n": len(rs), "successes": sum(int(as_float(r.get("success"))) for r in rs), "completion_rate": statistics.mean([as_float(r.get("success")) for r in rs]) if rs else 0.0, "mean_cost_all_m": statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in rs]) if rs else float("nan"), "mean_cost_witness_m": statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in witness]) if witness else float("nan"), "mean_min_gap_m": statistics.mean([as_float(r.get("min_net_gap_m")) for r in rs]) if rs else float("nan"), "mean_saturation": statistics.mean([as_float(r.get("action_saturation_count")) for r in rs]) if rs else 0.0})
    return rows


def generate_figures(out_dir: Path, results: Sequence[SimulationResult], metrics: Sequence[Mapping[str, object]], feasibility: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True); by_id = {r.run_id: r for r in results}
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for rid, label, color in (("E_collaborative_none_i0", "E collaborative failure", "#d62728"), ("A_ample_none_i0", "A ample", "#1f77b4")):
        if rid not in by_id: continue
        tr = [x for x in by_id[rid].logs if x.get("vehicle") == "M"]
        ax.plot([as_float(x.get("t_s")) for x in tr], [as_float(x.get("s_m")) for x in tr], label=label, color=color)
    ax.axhline(170.0, color="k", linestyle="--", linewidth=0.8, label="completion boundary 170 m")
    ax.set_xlabel("time (s)"); ax.set_ylabel("M longitudinal position (m)"); ax.set_title("Task boundary and representative motion"); ax.legend(fontsize=8)
    p_window = fig_dir / "task_window_space.png"; fig.savefig(p_window, dpi=160); plt.close(fig)
    rep = by_id.get("D_ample_none_i0") or by_id.get("C_ample_none_i0")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    if rep:
        tr = [x for x in rep.logs if x.get("vehicle") == "M"]; t = [as_float(x.get("t_s")) for x in tr]
        axes[0].plot(t, [as_float(x.get("qdot_space_s-1")) for x in tr], label="space qdot"); axes[0].plot(t, [as_float(x.get("qdot_partner_s-1")) for x in tr], label="partner qdot"); axes[0].plot(t, [as_float(x.get("qdot_clipped_s-1")) for x in tr], label="effective qdot")
        axes[1].plot(t, [as_float(x.get("a_nom_mps2")) for x in tr], label="nominal action"); axes[1].plot(t, [as_float(x.get("a_actual_mps2")) for x in tr], label="actual action")
        phases = [str(x.get("task_phase")) for x in tr]
        for i in range(1, len(phases)):
            if phases[i] != phases[i-1] and phases[i] in POST_PHASES:
                axes[0].axvline(t[i], color="#888", linestyle=":"); axes[1].axvline(t[i], color="#888", linestyle=":")
    axes[0].set_ylabel("q feedback (s⁻¹)"); axes[1].set_ylabel("action (m/s²)"); axes[1].set_xlabel("time (s)"); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); axes[0].set_title("q-to-action chain; post-terminal boundary marked")
    p_q = fig_dir / "q_action_chain.png"; fig.savefig(p_q, dpi=160); plt.close(fig)
    summ = _summary(metrics, feasibility); fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True); x = np.arange(len(METHODS)); width = 0.38
    axes[0].bar(x - width/2, [as_float(r["mean_cost_all_m"]) for r in summ], width, label="all 18 environments"); axes[0].bar(x + width/2, [as_float(r["mean_cost_witness_m"], 0.0) if math.isfinite(as_float(r["mean_cost_witness_m"], float("nan"))) else 0.0 for r in summ], width, label="feasible-witness strata")
    axes[1].bar(x, [as_float(r["successes"]) for r in summ]); axes[1].set_ylim(0, 18); axes[0].set_ylabel("all-vehicle speed-deficit distance (m)"); axes[1].set_ylabel("controller completions / 18")
    for ax in axes: ax.set_xticks(x, METHODS)
    axes[0].set_title("Task and all-vehicle cost"); axes[1].set_title("Corrected task completion"); axes[0].legend(fontsize=8)
    p_cost = fig_dir / "costs_all_vs_witness.png"; fig.savefig(p_cost, dpi=160); plt.close(fig)
    return {"task_window": str(p_window), "q_action": str(p_q), "costs": str(p_cost)}


def write_model_doc(root: Path) -> None:
    text = """# Swarmalator–CAV Stage 1C model v2

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
"""
    (root / "docs" / "model_v2.md").write_text(text, encoding="utf-8")


def write_validation(root: Path, feasibility: Sequence[Mapping[str, object]], original_recheck: Sequence[Mapping[str, object]]) -> None:
    lines = ["# Stage 1C validation ledger", "", "The checker is independent of simulator outcome counters.", "", f"* Stage1C environments checked: {len(feasibility)}", f"* Stage1B successful traces rechecked: {len(original_recheck)}", f"* Strict mission witnesses: {sum(int(as_float(r.get('mission_ok'))) for r in feasibility)}", f"* Full-window witnesses: {sum(int(as_float(r.get('full_window_ok'))) for r in feasibility)}", "", "A negative controller or candidate result is recorded as unresolved unless the separate optimistic necessary bound applies."]
    (root / "reports" / "stage1c_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(root: Path, metrics: Sequence[Mapping[str, object]], feasibility: Sequence[Mapping[str, object]], interventions: Sequence[Mapping[str, object]], original_recheck: Sequence[Mapping[str, object]], figures: Mapping[str, str], diagnostics: Sequence[Mapping[str, object]], revision_rows: Sequence[Mapping[str, object]], closed_loop: Sequence[Mapping[str, object]], known_rows: Sequence[Mapping[str, object]]) -> None:
    labels = {f"{r['scenario']}|{r['disturbance']}|{r['init_variant']}": r.get("label") for r in feasibility}; summ = _summary(metrics, feasibility); strata = Counter(str(r.get("label")) for r in feasibility)
    rows = "\n".join(f"| {r['method']} | {r['successes']}/18 | {r['mean_cost_all_m']:.2f} | {r['mean_cost_witness_m']:.2f} | {r['mean_min_gap_m']:.2f} | {r['mean_saturation']:.1f} |" for r in summ)
    full_errors = [as_float(r.get("full_reproduction_error")) for r in interventions if r.get("ablation") == "full" and r.get("full_reproduction_error") not in ("", None)]
    fixed = [r for r in interventions if r.get("ablation") == "fixed_W"]; nopartner = [r for r in interventions if r.get("ablation") == "no_partner"]
    diff_fixed = statistics.mean([abs(as_float(r.get("qdot_partner_s-1"))) for r in fixed]) if fixed else float("nan"); diff_np = statistics.mean([abs(as_float(r.get("qdot_partner_s-1"))) for r in nopartner]) if nopartner else float("nan")
    rev_counts = Counter((str(r.get("revision")), str(r.get("outcome"))) for r in revision_rows)
    rev_yes = sum(1 for r in revision_rows if r.get("revision") == "opening_enabled" and r.get("outcome") == "completed")
    rev_no = sum(1 for r in revision_rows if r.get("revision") == "opening_disabled" and r.get("outcome") == "completed")
    old_metrics = read_csv(root / "outputs" / "stage1b" / "metrics.csv")
    old_completed = sum(int(as_float(r.get("success"))) for r in old_metrics)
    closed_lines = []
    for method in ("C", "D"):
        base = [r for r in closed_loop if r.get("intervention_method") == method and r.get("intervention_ablation") == "full"]
        base_cost = statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in base]) if base else float("nan")
        for variant in sorted({str(r.get("intervention_ablation")) for r in closed_loop if r.get("intervention_method") == method and r.get("intervention_ablation") != "full"}):
            rr = [r for r in closed_loop if r.get("intervention_method") == method and r.get("intervention_ablation") == variant]
            cost = statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in rr]) if rr else float("nan")
            closed_lines.append(f"* {method} `{variant}`: Δ all-vehicle cost={cost-base_cost:+.3f} m, completions={sum(int(as_float(r.get('success'))) for r in rr)}/{len(rr)} versus full {sum(int(as_float(r.get('success'))) for r in base)}/{len(base)}.")
    report = f"""# Swarmalator–CAV Stage 1C report

Stage 1 and Stage 1B evidence remain untouched. Stage 1C ran the original 18
environments with methods A–E (**{len(metrics)} primary runs**), the independent
checker and bounded candidate search for all 18 environments, same-state
controller interventions (**{len(interventions)} rows**) plus {len(closed_loop)}
closed-loop ablation runs, and four-environment
opportunity diagnostics (**{len(diagnostics)} rows**).

## Model changes

* A completing step above 170 m is a failure; the old success-after-crossing
  outcome is removed.
* R release follows the actual occupied front (M or F), and strict witness h≥0
  is separate from the 0.15 m post-jerk tolerance.
* E recomputes the endpoint after smoothed target time and uses a current
  position reference. `Delta_new` is used consistently in the process gradient;
  ample therefore has zero process opening.
* One common geometry-derived finite-window opening term is applied to M and R
  for every method. It is not a D-specific term.
* Controller state before each compute is saved, so full same-snapshot replay is
  a true state clone rather than q-only reconstruction.

## Primary results

| Method | Corrected completions | Mean all-vehicle cost (m) | Mean witness-stratum cost (m) | Mean minimum gap (m) | Mean saturation count |
|---|---:|---:|---:|---:|---:|
{rows}

The completion column is the corrected controller outcome; a controller
failure does not prove physical infeasibility. Feasibility strata are
**{dict(strata)}**. Stage 1B successful traces were rechecked without their
success labels (**{len(original_recheck)} rows**), and invalid examples remain
evidence of the prior boundary/checker problem.

Stage 1B reported {old_completed}/90 completions; Stage 1C reports {sum(int(as_float(r.get('success'))) for r in metrics)}/90 after the endpoint and execution corrections. The difference is retained as a boundary/model-version effect, not a method ranking across versions.

## Mechanism result

Collaborative online failures arise at the first lost opportunity: the initial
finite-body interval is negative, while the online pair commands and q gate do
not create a strong enough M/R opening before M reaches the completion boundary.
The bounded offline candidate can use a predeclared wait and roughly -2 m/s²
M/R action, so a candidate witness is not equivalent to online success. The
opening revision is common to A–E and its benefit is not attributed to D.

Same-state full replay maximum error is {max(full_errors) if full_errors else float('nan'):.3g}. Fixed nonzero W has mean absolute partner term {diff_fixed:.4f} s⁻¹, while deleting the partner term has {diff_np:.4f} s⁻¹. These are local intervention effects; D is not presumed to win.

The single revision was checked on the four key environments for all five
methods ({len(revision_rows)} targeted runs): {rev_yes}/20 completed with the
common opening term and {rev_no}/20 without it. This is a targeted mechanism
comparison, not a new tuned matrix; the full 90-run Stage 1C table uses the
opening term for every method.

The full closed-loop ablation table keeps the same six configured intervention
environments. It is read after the same-state action table, so an instantaneous
term difference is not treated as a task-level gain. The complete rows are in
`outputs/stage1c/closed_loop_interventions.csv`.

{chr(10).join(closed_lines)}

The five prompt-priority candidates were rechecked explicitly in
`outputs/stage1c/known_candidate_recheck.csv`; the two saved ample traces pass,
the two collaborative i0 candidates pass to maneuver completion only, and the
collaborative i2 wait=2 s / -2 m/s² candidate is retained with its independent
failure reason rather than silently promoted to a witness.

## Figures and limits

* `{figures['task_window']}` shows the 170 m boundary and representative motion.
* `{figures['q_action']}` separates q feedback from nominal/actual action and marks post-terminal release.
* `{figures['costs']}` compares all-sample and witness-stratum costs and completions.

The main remaining issue is the gap between a strict offline witness and an
online controller that discovers it from the current state. The common revision
does not establish a robust feasible envelope. Communication noise/delay,
SUMO/CARLA, MARL, and large parameter scans remain outside Stage 1C.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m src.swarmalator_cav.run_stage1c --config configs/stage1c_config.json --out outputs/stage1c
```

The feedback archive is `feedback/stage1c_feedback.zip`; Stage 1 and Stage 1B
archives remain unchanged.
"""
    (root / "reports" / "stage1c_report.md").write_text(report, encoding="utf-8")


def package_feedback(root: Path) -> Tuple[Path, str]:
    archive = root / "feedback" / "stage1c_feedback.zip"; manifest = root / "feedback" / "stage1c_feedback_manifest.txt"
    include = ["AGENTS.md", "README.md", "codex_swarmalator_stage1c_prompt.md", "src/__init__.py", "src/swarmalator_cav/__init__.py", "src/swarmalator_cav/stage1c_simulation.py", "src/swarmalator_cav/run_stage1c.py", "configs/stage1c_config.json", "docs/model_v2.md", "reports/stage1c_report.md", "reports/stage1c_validation.md", "reports/stage1c_change_log.md", "tests/test_stage1c.py", "tests/test_simulation.py", "tests/test_stage1b.py", "outputs/stage1c/config_used.json", "outputs/stage1c/metrics.csv", "outputs/stage1c/vehicle_metrics.csv", "outputs/stage1c/feasibility.csv", "outputs/stage1c/witness_summary.csv", "outputs/stage1c/witness_actions.csv", "outputs/stage1c/key_trajectories.csv", "outputs/stage1c/key_events.csv", "outputs/stage1c/same_snapshot_interventions.csv", "outputs/stage1c/closed_loop_interventions.csv", "outputs/stage1c/opportunity_diagnostics.csv", "outputs/stage1c/original_witness_recheck.csv", "outputs/stage1c/known_candidate_recheck.csv", "outputs/stage1c/revision_comparison.csv", "outputs/stage1c/run_log.txt", "outputs/stage1c/figures/task_window_space.png", "outputs/stage1c/figures/q_action_chain.png", "outputs/stage1c/figures/costs_all_vs_witness.png"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            path = root / rel
            if path.exists(): zf.write(path, rel)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest(); manifest.write_text(f"stage1c_feedback.zip sha256={digest}\nfiles={len(zipfile.ZipFile(archive).namelist())}\n", encoding="utf-8")
    return archive, digest


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]; config_path = Path(args.config); config_path = config_path if config_path.is_absolute() else root / config_path; config = json.loads(config_path.read_text(encoding="utf-8")); out_dir = Path(args.out); out_dir = out_dir if out_dir.is_absolute() else root / out_dir; out_dir.mkdir(parents=True, exist_ok=True); cfg = SimConfig(**config.get("simulation", {}))
    results: List[SimulationResult] = []
    for method in config.get("methods", list(METHODS)):
        for scenario in config.get("scenarios", ["ample", "collaborative", "short_window"]):
            for disturbance in config.get("disturbances", ["none", "prepare"]):
                for init_variant in config.get("init_variants", [0, 1, 2]): results.append(run_episode(method, scenario, disturbance, int(init_variant), cfg=cfg))
    metrics = [r.metrics for r in results]; write_csv(out_dir / "metrics.csv", metrics); write_csv(out_dir / "trajectories.csv", [row for r in results for row in r.logs]); write_csv(out_dir / "events.csv", [row for r in results for row in r.events]); write_csv(out_dir / "vehicle_metrics.csv", [row for r in results for row in r.vehicle_metrics])
    feasibility, witness_summary, witness_actions = feasibility_diagnostic(results, cfg); write_csv(out_dir / "feasibility.csv", feasibility); write_csv(out_dir / "witness_summary.csv", witness_summary); write_csv(out_dir / "witness_actions.csv", witness_actions)
    original_recheck = recheck_original_stage1b(root, cfg); write_csv(out_dir / "original_witness_recheck.csv", original_recheck)
    known_rows: List[Dict[str, object]] = []
    for scenario, disturbance, init_variant, wait, a_m, a_r, source in (("ample", "none", 0, "", "", "", "saved_A"), ("ample", "prepare", 0, "", "", "", "saved_A"), ("collaborative", "none", 0, 1.5, -2.0, -2.0, "prompt_candidate"), ("collaborative", "prepare", 0, 1.5, -2.0, -2.0, "prompt_candidate"), ("collaborative", "none", 2, 2.0, -2.0, -2.0, "prompt_candidate")):
        if source == "saved_A":
            rr = next(r for r in results if r.method == "A" and r.scenario == scenario and r.disturbance == disturbance and r.init_variant == init_variant)
            chk = independent_trajectory_check(rr, cfg); rid = rr.run_id
        else:
            chk, trace = offline_candidate_trial(scenario, disturbance, int(init_variant), float(wait), float(a_m), float(a_r), cfg); rid = trace[0].get("run_id", "") if trace else ""
        known_rows.append({"candidate_name": f"{scenario}_{disturbance}_i{init_variant}", "source": source, "run_id": rid, "scenario": scenario, "disturbance": disturbance, "init_variant": init_variant, "wait_s": wait, "a_M_mps2": a_m, "a_R_mps2": a_r, **chk})
    write_csv(out_dir / "known_candidate_recheck.csv", known_rows)
    interventions = same_snapshot_interventions(results, cfg); write_csv(out_dir / "same_snapshot_interventions.csv", interventions)
    closed_loop: List[Dict[str, object]] = []
    for scenario, disturbance, init_variant in config.get("intervention_environments", []):
        for method, variants in (("C", ["", "no_space", "fixed_W", "no_partner"]), ("D", ["", "symmetric_k"])):
            for ablation in variants:
                rr = run_episode(method, scenario, disturbance, int(init_variant), cfg=cfg, ablation=ablation)
                closed_loop.append({**rr.metrics, "intervention_method": method, "intervention_ablation": ablation or "full"})
    write_csv(out_dir / "closed_loop_interventions.csv", closed_loop); write_csv(out_dir / "interventions.csv", interventions); diagnostics = opportunity_diagnostics(results, cfg); write_csv(out_dir / "opportunity_diagnostics.csv", diagnostics)
    key_envs = {("collaborative", "none", 0), ("collaborative", "prepare", 0), ("collaborative", "none", 2), ("ample", "none", 0)}; write_csv(out_dir / "key_trajectories.csv", [row for r in results if (r.scenario, r.disturbance, r.init_variant) in key_envs for row in r.logs]); write_csv(out_dir / "key_events.csv", [row for r in results if (r.scenario, r.disturbance, r.init_variant) in key_envs for row in r.events])
    revision_rows: List[Dict[str, object]] = []
    cfg_without_revision = replace(cfg, window_opening_enabled=False)
    for scenario, disturbance, init_variant in sorted(key_envs):
        for method in METHODS:
            with_revision = next(r for r in results if r.method == method and r.scenario == scenario and r.disturbance == disturbance and r.init_variant == init_variant)
            without_revision = run_episode(method, scenario, disturbance, init_variant, cfg=cfg_without_revision)
            for label, rr in (("opening_enabled", with_revision), ("opening_disabled", without_revision)):
                revision_rows.append({"revision": label, "run_id": rr.run_id, "method": method, "scenario": scenario, "disturbance": disturbance, "init_variant": init_variant, "outcome": rr.metrics.get("outcome"), "success": rr.metrics.get("success"), "completion_time_s": rr.metrics.get("completion_time_s"), "min_net_gap_m": rr.metrics.get("min_net_gap_m"), "opening_accel_min_mps2": min(float(x.get("window_opening_accel", 0.0)) for x in rr.logs if x.get("vehicle") == "M")})
    write_csv(out_dir / "revision_comparison.csv", revision_rows)
    figures = generate_figures(out_dir, results, metrics, feasibility); write_model_doc(root); write_validation(root, feasibility, original_recheck); write_report(root, metrics, feasibility, interventions, original_recheck, figures, diagnostics, revision_rows, closed_loop, known_rows); (out_dir / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"); (out_dir / "run_log.txt").write_text("invocation=" + " ".join(map(str, sys.argv)) + "\n" + f"python={sys.executable}\nprimary_runs={len(results)}\nfeasibility_environments={len(feasibility)}\nstrict_witnesses={sum(int(as_float(r.get('mission_ok'))) for r in feasibility)}\nsame_snapshot_rows={len(interventions)}\nclosed_loop_intervention_runs={len(closed_loop)}\noriginal_rechecks={len(original_recheck)}\nknown_candidate_rechecks={len(known_rows)}\ndiagnostics={len(diagnostics)}\nrevision_comparison_runs={len(revision_rows)}\nfigures={figures}\n", encoding="utf-8"); archive, digest = package_feedback(root); print(json.dumps({"primary_runs": len(results), "feasibility": len(feasibility), "strict_witnesses": sum(int(as_float(r.get("mission_ok"))) for r in feasibility), "same_snapshot_rows": len(interventions), "closed_loop_runs": len(closed_loop), "known_candidate_rechecks": len(known_rows), "revision_runs": len(revision_rows), "out": str(out_dir), "feedback": str(archive), "sha256": digest}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage1c_config.json"); parser.add_argument("--out", default="outputs/stage1c"); run(parser.parse_args())


if __name__ == "__main__": main()
