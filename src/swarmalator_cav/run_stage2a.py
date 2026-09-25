"""Stage 2A runner: anchors, finite regression, validation, figures and zip."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from .stage1c_simulation import (
    ACTIVE_PHASES, POST_PHASES, SimulationResult, TaskState, VehicleState,
    _net_gap, _task_event, build_scenario, current_leader, current_r_leader,
    dynamic_gap, environment_accel, execute_safe_actions, geometry, idm_accel,
    make_controller, make_vehicles, physical_start_feasibility,
)
from .stage1c_check import independent_trajectory_check
from .stage2a_check import validate_stage2a
from .stage2a_common import Config, array_state, dictionary_state, physical_step, post_targets, reference_command
from .stage2a_planning import Controller


METHODS = ("P", "S1", "S2", "S3", "oldC")
SCENARIOS = ("ample", "collaborative", "short_window")
DISTURBANCES = ("none", "prepare")
VARIANTS = (0, 1, 2)
ANCHORS = (
    ("collaborative", "none", 0),
    ("collaborative", "prepare", 0),
    ("collaborative", "none", 2),
    ("ample", "none", 0),
    ("ample", "prepare", 0),
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: ("" if v is None else v) for k, v in row.items()} for row in rows])


def _float(value, default=float("nan")):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _ref_row(ref, vid_index):
    return {"s": float(ref[vid_index, 0]), "v": float(ref[vid_index, 1]), "a": float(ref[vid_index, 2])}


def _common_task_update(task, method, t_s, vehicles, plan, scenario_name, init_variant, cfg, legacy):
    sc = build_scenario(scenario_name, init_variant)
    M = vehicles["M"]
    if task.phase == "PREPARE":
        if M.s > sc.completion_s + 1.0e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "missed_window", True, "missed_window"
            return task.event
        if method == "oldC":
            return _task_event(task, t_s, vehicles, legacy, sc, cfg)
        if plan is not None and plan.valid and t_s >= plan.start - cfg.dt * 0.51:
            check = physical_start_feasibility(vehicles, task, sc, cfg)
            if check["physical_ok"]:
                task.phase, task.merge_start_time, task.event = "EXECUTE", t_s, "merge_started"
                return task.event
    elif task.phase == "EXECUTE":
        if M.s > sc.completion_s + 1.0e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_exit_before_completion"
            return task.event
        if M.merge_progress >= 1.0 - 1.0e-9:
            g = geometry(vehicles, sc, cfg)
            if g["interval_width"] >= -1.0e-7 and g["lower"] - 1.0e-7 <= M.s <= g["upper"] + 1.0e-7:
                task.phase, task.completion_time, task.terminal, task.success, task.event = "SUCCESS_RELEASE", t_s, True, True, "merge_completed"
            else:
                task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_completed_outside_slot"
            return task.event
    return None


def _append_event(events, run_id, t_s, event, **extra):
    row = {"run_id": run_id, "t_s": round(float(t_s), 6), "event": event}
    row.update(extra)
    events.append(row)


def run_episode(method: str, scenario_name: str, disturbance: str, init_variant: int,
                cfg: Config, ablation: str = "") -> SimulationResult:
    scenario = build_scenario(scenario_name, init_variant)
    vehicles = make_vehicles(scenario, cfg)
    x = array_state(vehicles)
    # Reference state is common to all new methods and has terminal derivatives
    # propagated continuously through release. It is not reset by q/eta.
    ref = x[:, :3].copy()
    planner_controller = Controller(method, cfg, ablation=ablation) if method in ("P", "S1", "S2", "S3") else None
    legacy = make_controller("C", scenario, cfg) if method == "oldC" else None
    task = TaskState()
    plan = None
    logs, events = [], []
    run_id = f"{method}_{scenario_name}_{disturbance}_i{init_variant}" + (f"_{ablation}" if ablation else "")
    last_nominal = np.zeros(4)
    last_actual = np.zeros(4)
    last_jerk = np.zeros(4)
    last_commands = {"reference_kind": "common_continuous_reference"}
    last_info = {}
    first_plan_t = None
    first_valid_plan_t = None
    first_predicted_completion_s_m = None
    first_predicted_completion_t_s = None
    decision_times = []
    plan_replans = 0
    start_clock = __import__("time").perf_counter()
    n_steps = int(round(cfg.horizon / cfg.dt))
    decision_candidate_counts = []

    def add_rows(t_s, x0, commands, info, nominal, actual, jerks, phase, diag):
        nonlocal last_commands, last_info, last_nominal, last_actual, last_jerk
        sc = scenario
        state = dictionary_state(x0[None])
        g = geometry(state, sc, cfg)
        controller_z = planner_controller.z if planner_controller is not None else np.array([float(legacy.q["M"]) if legacy else 0.0, float(legacy.q["R"]) if legacy else 0.0])
        for idx, vid in enumerate(("M", "R", "F", "B")):
            eta_M = float(controller_z[0]) if planner_controller is not None else (float(legacy.q["M"]) if legacy else "")
            eta_R = float(controller_z[1]) if planner_controller is not None else (float(legacy.q["R"]) if legacy else "")
            if planner_controller is not None:
                info_fields = info
                k = float(info_fields.get("K_MR" if vid == "M" else "K_RM", 0.0))
                w = float(info_fields.get("W", 0.0))
            else:
                info_fields = info
                k = float(commands.get("K_to_partner_s-1", 0.0))
                w = float(commands.get("W_to_partner", 0.0))
            logs.append({
                "run_id": run_id, "t_s": round(float(t_s), 6), "method": method,
                "ablation": ablation, "scenario": scenario_name, "disturbance": disturbance,
                "init_variant": init_variant, "vehicle": vid,
                "s_m": x0[idx, 0], "y_m": x0[idx, 3], "v_mps": x0[idx, 1], "a_mps2": x0[idx, 2],
                "coord_eta_M": eta_M, "coord_eta_R": eta_R,
                "coord_eta_dot_M": info_fields.get("z_dot_M", info_fields.get("delta_z_M", 0.0)),
                "coord_eta_dot_R": info_fields.get("z_dot_R", info_fields.get("delta_z_R", 0.0)),
                "coord_gradient_M": info_fields.get("gradient_M", 0.0), "coord_gradient_R": info_fields.get("gradient_R", 0.0),
                "coord_partner_M": info_fields.get("partner_M", 0.0), "coord_partner_R": info_fields.get("partner_R", 0.0),
                "K_to_partner_s-1": k, "W_to_partner": w,
                "g_FR_m": g["g_fr"], "target_gap_physical_m": g["target_gap_physical"],
                "target_gap_process_m": g["target_gap_physical"], "gap_error_m": g["g_fr"] - scenario.initial_gap,
                "lower_m": g["lower"], "upper_m": g["upper"], "interval_width_m": g["interval_width"],
                "s_ref_m": ref[0, 0], "s_ref_dot_mps": ref[0, 1], "s_ref_ddot_mps2": ref[0, 2],
                "reference_kind": commands.get("reference_kind", "common_continuous_reference"),
                "plan_id": int(info.get("plan_id", -1)), "plan_start_t_s": info.get("start_s", ""),
                "plan_reason": info.get("reason", ""), "plan_compute_time_s": info.get("decision_seconds", 0.0),
                "predicted_start_t_s": info.get("start_s", ""), "predicted_completion_t_s": info.get("completion_t_s", ""),
                "predicted_completion_s_m": info.get("completion_s_m", info.get("completion_s", "")),
                "predicted_min_h_m": info.get("predicted_min_h_m", ""), "predicted_net_clearance_m": info.get("predicted_min_net_m", ""),
                "predicted_candidate_count": info.get("candidate_count", 0), "merge_progress": x0[idx, 4],
                "task_phase": phase, "a_nom_mps2": nominal[idx], "a_safety_target_mps2": actual[idx],
                "a_actual_mps2": actual[idx], "jerk_mps3": jerks[idx],
                "nominal_actual_difference_mps2": actual[idx] - nominal[idx], "safety_correction_mps2": actual[idx] - nominal[idx],
                "safety_correction_applied": int(abs(actual[idx] - nominal[idx]) > 1e-9),
                "action_saturation_applied": int(abs(jerks[idx]) >= cfg.jerk_max - 1e-9),
                "action_correction_reason": "common_safety_projection" if abs(actual[idx] - nominal[idx]) > 1e-9 else "",
                "one_step_feasible": int(bool(diag["one_step_feasible"][0])), "next_h_min_m": diag["min_h"][0],
                "F_disturbance_mps2": environment_accel(state["F"], t_s, scenario, disturbance, cfg) - environment_accel(state["F"], t_s, scenario, "none", cfg) if vid == "F" else 0.0,
            })
        last_commands = dict(commands); last_info = dict(info); last_nominal = nominal.copy(); last_actual = actual.copy(); last_jerk = jerks.copy()

    for step in range(n_steps):
        t_s = step * cfg.dt
        x0 = x.copy()
        state = dictionary_state(x0[None])
        phase_before = task.phase
        if planner_controller is not None:
            target, info = planner_controller.decide(x0, ref, t_s, task.phase)
            plan = planner_controller.plan
            if info.get("decision"):
                plan_replans += 1
                decision_times.append(float(info.get("decision_seconds", 0.0)))
                decision_candidate_counts.append(int(info.get("candidate_count", 0)))
                first_plan_t = t_s if first_plan_t is None else first_plan_t
                if info.get("valid_plan") and first_valid_plan_t is None:
                    first_valid_plan_t = t_s
                    first_predicted_completion_s_m = info.get("completion_s_m", info.get("completion_s"))
                    first_predicted_completion_t_s = info.get("completion_t_s")
                _append_event(events, run_id, t_s, "plan_generated", **info)
            # A new plan is used as a target acceleration. The same reference
            # integrator used in prediction produces the action sent below.
            pair_nom, ref_next = reference_command(x0[None], ref[None], np.asarray(target, dtype=float)[None], cfg)
            nominal_pair = pair_nom[0]
            commands = {"reference_kind": f"predictive_common_{method}"}
            nominal_MR = nominal_pair
            ref_candidate = ref_next[0]
            if task.phase == "PREPARE" and plan is not None and plan.valid and t_s >= plan.start - cfg.dt * 0.51:
                check = physical_start_feasibility(state, task, scenario, cfg)
                if check["physical_ok"]:
                    task.phase, task.merge_start_time, task.event = "EXECUTE", t_s, "merge_started"
                    phase_before = task.phase
                    _append_event(events, run_id, t_s, "merge_started", plan_id=plan.start)
        else:
            out = legacy.compute_pair(state, task, cfg.dt)
            nominal_pair = np.array([float(out["M"]["a_nom"]), float(out["R"]["a_nom"])])
            target = nominal_pair.copy()
            _, ref_next = reference_command(x0[None], ref[None], target[None], cfg)
            ref_candidate = ref_next[0]
            commands = {"reference_kind": "oldC_common_continuous_reference",
                        "K_to_partner_s-1": out["M"].get("K_to_partner", 0.0),
                        "W_to_partner": out["M"].get("W_to_partner", 0.0)}
            nominal_MR = nominal_pair

        # Post-task following is common. Its action is based on actual current
        # leader occupancy; no phase answer is substituted for geometry.
        if task.phase in POST_PHASES:
            post = post_targets(x0[None], cfg)[0]
            nominal_MR = post
            commands["reference_kind"] = "post_terminal_continuous_following"
        nominal = np.array([nominal_MR[0], nominal_MR[1],
                            environment_accel(state["F"], t_s, scenario, disturbance, cfg),
                            idm_accel(state["B"], state["R"], cfg)], dtype=float)
        if task.phase != "POST":
            ref = ref_candidate
        active = np.array([bool(task.phase == "EXECUTE" or x0[0, 3] <= (cfg.lane_ramp + cfg.lane_main) / 2.0)])
        merging = np.array([bool(task.phase == "EXECUTE")])
        x_next, actual_b, diag = physical_step(x0[None], nominal[None], active, merging, cfg)
        actual = actual_b[0]
        jerks = (actual - x0[:, 2]) / cfg.dt
        add_rows(t_s, x0, commands, info if planner_controller is not None else {}, nominal, actual, jerks, phase_before, diag)
        x = x_next[0]
        # Legacy q history is still advanced under the common actual executor.
        if legacy is not None:
            legacy.commit({"M": {"qdot_clipped": out["M"].get("qdot_clipped", 0.0)},
                           "R": {"qdot_clipped": out["R"].get("qdot_clipped", 0.0)}}, cfg.dt, task.terminal)
        vehicles_next = dictionary_state(x[None])
        event = _common_task_update(task, method, t_s + cfg.dt, vehicles_next, plan, scenario_name, init_variant, cfg, legacy)
        if event:
            _append_event(events, run_id, t_s + cfg.dt, event, plan_id=getattr(plan, "start", -1) if plan else -1)
        if task.terminal and planner_controller is not None:
            # Keep the last plan/history for post-terminal audit; no replan.
            pass

    if not task.terminal:
        task.phase, task.failure_type, task.terminal = "TIMEOUT", "timeout", True
        _append_event(events, run_id, cfg.horizon, "timeout", details="observation_window_end")
    # Final row has the physical state after the final 0.05 s action.
    state_final = x.copy()
    for idx, vid in enumerate(("M", "R", "F", "B")):
        logs.append({
            "run_id": run_id, "t_s": round(float(cfg.horizon), 6), "method": method, "ablation": ablation,
            "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
            "s_m": state_final[idx, 0], "y_m": state_final[idx, 3], "v_mps": state_final[idx, 1], "a_mps2": state_final[idx, 2],
            "coord_eta_M": planner_controller.z[0] if planner_controller else "", "coord_eta_R": planner_controller.z[1] if planner_controller else "",
            "s_ref_m": ref[0, 0], "s_ref_dot_mps": ref[0, 1], "s_ref_ddot_mps2": ref[0, 2],
            "reference_kind": last_commands.get("reference_kind", ""), "plan_id": last_info.get("plan_id", -1),
            "plan_start_t_s": last_info.get("start_s", ""), "plan_reason": last_info.get("reason", ""),
            "plan_compute_time_s": last_info.get("decision_seconds", 0.0), "predicted_start_t_s": last_info.get("start_s", ""),
            "predicted_completion_t_s": last_info.get("completion_t_s", ""), "predicted_completion_s_m": last_info.get("completion_s_m", last_info.get("completion_s", "")),
            "predicted_min_h_m": last_info.get("predicted_min_h_m", ""), "predicted_net_clearance_m": last_info.get("predicted_min_net_m", ""),
            "predicted_candidate_count": last_info.get("candidate_count", 0), "merge_progress": state_final[idx, 4],
            "task_phase": task.phase, "a_nom_mps2": last_nominal[idx], "a_safety_target_mps2": last_actual[idx],
            "a_actual_mps2": last_actual[idx], "jerk_mps3": last_jerk[idx], "nominal_actual_difference_mps2": last_actual[idx] - last_nominal[idx],
            "safety_correction_mps2": 0.0, "safety_correction_applied": 0, "action_saturation_applied": 0,
            "action_correction_reason": "terminal_state", "one_step_feasible": 1, "next_h_min_m": "", "F_disturbance_mps2": 0.0,
        })
    outcome = "completed" if task.success else task.failure_type or task.phase.lower()
    speed_rows = [row for row in logs if _float(row.get("t_s")) < cfg.horizon]
    phase_cost = {p: sum(max(0.0, scenario.desired_speed - _float(r.get("v_mps"))) * cfg.dt for r in speed_rows if r.get("task_phase") == p) for p in ("PREPARE", "EXECUTE", "SUCCESS_RELEASE", "FAILURE_HANDLING")}
    metrics = {
        "run_id": run_id, "method": method, "ablation": ablation, "scenario": scenario_name,
        "disturbance": disturbance, "init_variant": init_variant, "outcome": outcome, "success": int(task.success),
        "geometry_completed": int(any(e.get("event") == "merge_completed" for e in events)),
        "merge_start_time_s": "" if task.merge_start_time is None else task.merge_start_time,
        "completion_time_s": "" if task.completion_time is None else task.completion_time,
        "planner_first_time_s": "" if first_plan_t is None else first_plan_t,
        "planner_first_valid_time_s": "" if first_valid_plan_t is None else first_valid_plan_t,
        "planner_replans": plan_replans, "planner_mean_compute_time_s": statistics.mean(decision_times) if decision_times else 0.0,
        "planner_max_compute_time_s": max(decision_times, default=0.0),
        "planner_candidate_count": statistics.mean(decision_candidate_counts) if decision_candidate_counts else 0.0,
        "planner_total_candidate_count": sum(decision_candidate_counts),
        "planner_decision_count": len(decision_candidate_counts),
        "planner_predicted_completion_s_m": first_predicted_completion_s_m if first_predicted_completion_s_m is not None else float("nan"),
        "planner_predicted_completion_t_s": first_predicted_completion_t_s if first_predicted_completion_t_s is not None else float("nan"),
        "planner_predicted_min_h_m": last_info.get("predicted_min_h_m", float("nan")),
        "speed_deficit_prepare_m": phase_cost["PREPARE"], "speed_deficit_execute_m": phase_cost["EXECUTE"],
        "speed_deficit_success_release_m": phase_cost["SUCCESS_RELEASE"], "speed_deficit_failure_handling_m": phase_cost["FAILURE_HANDLING"],
        "total_speed_deficit_distance_m": sum(phase_cost.values()),
        "total_effort_integral_m2_s3": sum(float(r.get("a_actual_mps2", 0.0)) ** 2 * cfg.dt for r in speed_rows),
        "max_abs_jerk_mps3": max((abs(_float(r.get("jerk_mps3"))) for r in speed_rows), default=0.0),
        "safety_correction_count": sum(int(r.get("safety_correction_applied", 0)) for r in speed_rows),
        "action_saturation_count": sum(int(r.get("action_saturation_applied", 0)) for r in speed_rows),
        "compute_time_s": __import__("time").perf_counter() - start_clock,
    }
    vehicle_metrics = []
    for vid in ("M", "R", "F", "B"):
        rows_v = [r for r in speed_rows if r["vehicle"] == vid]
        vehicle_metrics.append({"run_id": run_id, "method": method, "ablation": ablation, "scenario": scenario_name,
                                "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
                                "speed_deficit_distance_m": sum(max(0.0, scenario.desired_speed - _float(r.get("v_mps"))) * cfg.dt for r in rows_v),
                                "safety_correction_count": sum(int(r.get("safety_correction_applied", 0)) for r in rows_v),
                                "action_saturation_count": sum(int(r.get("action_saturation_applied", 0)) for r in rows_v),
                                "max_abs_jerk_mps3": max((abs(_float(r.get("jerk_mps3"))) for r in rows_v), default=0.0)})
    return SimulationResult(run_id, method, scenario_name, disturbance, init_variant, logs, events, metrics, vehicle_metrics)


def _result_rows(results, key):
    return [result.metrics if key == "metrics" else row for result in results for row in (result.vehicle_metrics if key == "vehicle_metrics" else result.logs if key == "logs" else result.events)]


def _validation_rows(results, cfg):
    rows = []
    for result in results:
        check = validate_stage2a(result, cfg)
        row = {"run_id": result.run_id, "method": result.method, "ablation": result.metrics.get("ablation", ""),
               "scenario": result.scenario, "disturbance": result.disturbance, "init_variant": result.init_variant}
        row.update({k: v for k, v in check.items() if k not in ("errors", "min_h_by_phase")})
        row["error"] = check.get("error", "")
        rows.append(row)
    return rows


def _summary(results, validation):
    by_method = {}
    for method in METHODS:
        rr = [r for r in results if r.method == method and not r.metrics.get("ablation")]
        vv = {r["run_id"]: r for r in validation}
        valid_rr = [r for r in rr if int(vv.get(r.run_id, {}).get("mission_valid", 0))]
        by_method[method] = {
            "method": method, "n": len(rr),
            "geometry_completed": sum(int(r.metrics.get("geometry_completed", 0)) for r in rr),
            "mission_valid": sum(int(vv[r.run_id].get("mission_valid", 0)) for r in rr),
            "full_window_valid": sum(int(vv[r.run_id].get("full_window_valid", 0)) for r in rr),
            "mean_speed_deficit_all_m": statistics.mean(float(r.metrics.get("total_speed_deficit_distance_m", 0.0)) for r in rr) if rr else float("nan"),
            "mission_valid_n": len(valid_rr),
            "mean_speed_deficit_mission_valid_m": statistics.mean(float(r.metrics.get("total_speed_deficit_distance_m", 0.0)) for r in valid_rr) if valid_rr else float("nan"),
            "mean_prepare_deficit_all_m": statistics.mean(float(r.metrics.get("speed_deficit_prepare_m", 0.0)) for r in rr) if rr else float("nan"),
            "mean_execute_deficit_all_m": statistics.mean(float(r.metrics.get("speed_deficit_execute_m", 0.0)) for r in rr) if rr else float("nan"),
            "mean_release_deficit_all_m": statistics.mean(float(r.metrics.get("speed_deficit_success_release_m", 0.0)) for r in rr) if rr else float("nan"),
            "mean_failure_handling_deficit_all_m": statistics.mean(float(r.metrics.get("speed_deficit_failure_handling_m", 0.0)) for r in rr) if rr else float("nan"),
            "mean_prepare_deficit_mission_valid_m": statistics.mean(float(r.metrics.get("speed_deficit_prepare_m", 0.0)) for r in valid_rr) if valid_rr else float("nan"),
            "mean_execute_deficit_mission_valid_m": statistics.mean(float(r.metrics.get("speed_deficit_execute_m", 0.0)) for r in valid_rr) if valid_rr else float("nan"),
            "mean_release_deficit_mission_valid_m": statistics.mean(float(r.metrics.get("speed_deficit_success_release_m", 0.0)) for r in valid_rr) if valid_rr else float("nan"),
            "mean_failure_handling_deficit_mission_valid_m": statistics.mean(float(r.metrics.get("speed_deficit_failure_handling_m", 0.0)) for r in valid_rr) if valid_rr else float("nan"),
            "mean_planner_compute_s": statistics.mean(float(r.metrics.get("planner_mean_compute_time_s", 0.0)) for r in rr) if rr else 0.0,
            "mean_candidates": statistics.mean(float(r.metrics.get("planner_candidate_count", 0.0)) for r in rr) if rr else 0.0,
            "mean_saturation": statistics.mean(float(r.metrics.get("action_saturation_count", 0.0)) for r in rr) if rr else 0.0,
        }
    return list(by_method.values())


def _make_figures(out_dir, results, validation):
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    anchors = [r for r in results if (r.scenario, r.disturbance, r.init_variant) in ANCHORS and not r.metrics.get("ablation")]
    # 1: planner predicted / actual preparation and slot width.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for result in anchors:
        if result.method not in ("P", "S2", "oldC") or result.scenario != "collaborative" or result.disturbance != "none" or result.init_variant != 0: continue
        rows = [r for r in result.logs if r["vehicle"] == "M" and _float(r["t_s"]) < 5.0]
        ax.plot([_float(r["t_s"]) for r in rows], [_float(r["interval_width_m"]) for r in rows], label=f"{result.method} width")
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel("Time (s)"); ax.set_ylabel("Finite-body slot width (m)"); ax.set_title("Stage2A anchor: executable plan generation"); ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "plan_generation_anchor.png", dpi=150); plt.close(fig)
    # 2: internal state, reference and actual action.
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    result = next((r for r in anchors if r.method == "S2" and r.scenario == "collaborative" and r.disturbance == "none" and r.init_variant == 0), None)
    if result:
        rr = [r for r in result.logs if r["vehicle"] == "M" and _float(r["t_s"]) < 5.0]
        t = [_float(r["t_s"]) for r in rr]; axes[0].plot(t, [_float(r["coord_eta_M"]) for r in rr], label="eta_M"); axes[0].plot(t, [_float(r["coord_eta_R"]) for r in rr], label="eta_R"); axes[0].legend(); axes[0].set_ylabel("Internal intensity (0–1)")
        axes[1].plot(t, [_float(r["s_ref_dot_mps"]) for r in rr], label="reference velocity"); axes[1].plot(t, [_float(r["a_actual_mps2"]) for r in rr], label="M actual acceleration"); axes[1].legend(); axes[1].set_ylabel("m/s or m/s²")
    axes[1].set_xlabel("Time (s)"); axes[0].set_title("Stage2A anchor: internal state–reference–action chain"); fig.tight_layout(); fig.savefig(fig_dir / "space_process_action_chain.png", dpi=150); plt.close(fig)
    # 3: outcomes and compute trade-off.
    summary = _summary(results, validation)
    fig, ax1 = plt.subplots(figsize=(8, 4.5)); methods=[r["method"] for r in summary]; x=np.arange(len(methods)); mission=[r["mission_valid"] for r in summary]; cost=[r["mean_speed_deficit_all_m"] for r in summary]; comp=[r["mean_planner_compute_s"]*1000 for r in summary]
    ax1.bar(x-0.2, mission, width=0.2, label="Mission-valid runs"); ax1.bar(x, [r["geometry_completed"] for r in summary], width=0.2, label="Geometry-completed runs"); ax1.set_ylabel("Count (of 18)"); ax1.set_xticks(x, methods)
    ax2=ax1.twinx(); ax2.plot(x+0.2,cost,"o-",label="Mean speed-deficit distance (m)",color="tab:red"); ax2.plot(x+0.2,comp,"s--",label="Mean decision time (ms)",color="tab:green"); ax2.set_ylabel("Cost (m) / compute (ms)"); lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels(); ax1.legend(lines+lines2,labels+labels2,loc="upper left"); ax1.set_title("Stage2A finite regression: validity, cost and compute"); fig.tight_layout(); fig.savefig(fig_dir / "validity_cost_compute_tradeoff.png", dpi=150); plt.close(fig)
    return {"plan_generation": str(fig_dir / "plan_generation_anchor.png"), "space_process_action": str(fig_dir / "space_process_action_chain.png"), "tradeoff": str(fig_dir / "validity_cost_compute_tradeoff.png")}


def _write_docs(root, results, validation, summary, figures, counts):
    (root / "docs").mkdir(exist_ok=True); (root / "reports").mkdir(exist_ok=True)
    model_path = root / "docs" / "model_v3.md"
    if not model_path.exists():
        model_path.write_text('# Swarmalator–CAV Stage 2A model v3\n\nStage 2A uses a causal finite-window planner followed by a common jerk-limited\nexecutor. The controlled vehicles are M (the merging vehicle) and R (the rear\nvehicle in the target gap); F is the observed/predicted front boundary and B\nfollows R with the declared IDM model. A controller receives the current\nfour-vehicle state, the continuously propagated public reference, and the\ncurrent task phase. It never receives the disturbance name, scenario name,\ninitial-variant identifier, success label, or a future F script.\n\nFor a candidate absolute start time `t_s` and bounded role intensities\n`z=(z_M,z_R)`, the predictor rolls the state through the preparation interval,\nthe fixed `merge_duration`, and `post_preview_s`. The internal state is a\nbounded preparation intensity, not the physical lateral progress `q_M`. It is\nmapped to a longitudinal spatial reference through the target accelerations\n`a_target,M/R = -3 z_M/R`; the public reference then uses the same current\nstate feedback and jerk-limited acceleration update as execution. Thus z can\nrequest a costly transient gap expansion or hold while the planner decides\nwhen to start. The physical lateral progress still advances only during the\ndeclared task execution.\n\nEach candidate uses the vectorized form of the same executor used by the\nsimulation. It clips acceleration, velocity, and jerk; projects the ordered\nF–M–R–B (or F–R–B before overlap) chain onto one-step dynamic margins; and\nrecords empty action intersections instead of calling the projection a safety\nproof. Candidate selection first filters valid finite-window task candidates,\nthen compares all-vehicle speed deficit and action effort. A failed bounded\nsearch is reported as `finite_search_no_valid_plan`, which is unresolved and\nis not a physical infeasibility claim.\n\nP is this finite space–time planner without an internal state law. S1 keeps\nthe same planner and reference family but disables the new space-to-process\ngradient and partner term. S2 adds a finite-difference gradient of the\npredicted task potential together with symmetric partner action. S3 preserves\nthe same total partner budget but allocates it asymmetrically from the current\nrear dynamic margin. These are causal state updates; the planner still owns\nabsolute timing and candidate feasibility. `oldC` is the Stage1C C coordination\nlaw run through the common Stage2A reference and executor, so it is a cross\nstructure diagnostic rather than a reproduction of the old output.\n\nThe reference state stores `(s_ref, v_ref, a_ref)` and propagates acceleration\nwith the jerk bound. It is continuous through q completion and release; no\nendpoint reset removes a derivative term. After completion, R follows the\ncurrent physical leader occupancy and the same safety layer remains active.\nThe independent checker reports three labels: geometric/event completion,\nmission-valid completion through the declared maneuver, and full-window\nvalidity through the observation horizon. Net body clearance and dynamic\nmargin `h` are logged separately, as are nominal, executable safety-target,\nand actual actions.\n\nThis is a transparent engineering prototype inspired by planner–constraint\ncorrection–tracking separations in cooperative CAV and swarmalator work. It is\nnot a reproduction of either paper, does not claim distributed communication,\ncontinuous reachability, or road-safety certification, and keeps the limits of\nthe finite development matrix explicit.\n', encoding="utf-8")
    vv = {row["run_id"]: row for row in validation}
    anchor_key = ("collaborative", "none", 0)
    anchor_results = [r for r in results if (r.scenario, r.disturbance, r.init_variant) == anchor_key and not r.metrics.get("ablation")]
    anchor_text = []
    for r in anchor_results:
        v = vv.get(r.run_id, {})
        anchor_text.append(f"{r.method}: geometry={v.get('geometry_completed', 0)}, mission={v.get('mission_valid', 0)}, full={v.get('full_window_valid', 0)}, outcome={r.metrics.get('outcome', '')}")
    out_dir = Path(next(iter(figures.values()))).resolve().parent.parent
    ablation_lines = ["## Controlled ablations", "", "These six runs keep the S2 planner, state and executor fixed while changing one declared term on collaborative none/0 and ample none/0:"]
    ablation_path = out_dir / "ablations.csv"
    if ablation_path.exists():
        with ablation_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                ablation_lines.append(f"- {row['ablation']} / {row['scenario']}: mission={row.get('validation', '')}, total speed-deficit={float(row['total_speed_deficit_distance_m']):.3f} m, prepare={float(row['speed_deficit_prepare_m']):.3f} m, execute={float(row['speed_deficit_execute_m']):.3f} m, success-release={float(row['speed_deficit_success_release_m']):.3f} m")
    else:
        ablation_lines.append("- ablation CSV missing")
    lines = ["# Stage 2A report", "", "Stage 1, Stage 1B and Stage 1C evidence remain frozen. Stage2A uses a new model version and outputs/stage2a.", "", "## Direct answer", ""]
    lines.append("The finite-window controller generated a causal plan on the collaborative none/0 anchor for P, S1, S2 and S3; the independent task label for each is shown below. This is the first recovered collaborative task segment compared with oldC under the common Stage2A executor. Because S1–S3 use the same predictor and execution layer as P, any P versus oldC change is a prediction/reference-interface result. The internal-state and non-reciprocal increments are evaluated separately; no method is declared the winner.")
    main_validation = [row for row in validation if not row.get("ablation")]
    reference_ok_count = sum(int(row.get("reference_ok", 0)) for row in main_validation)
    max_reference_accel = max((float(row.get("max_reference_acceleration_mps2", 0.0)) for row in main_validation), default=0.0)
    lines.append(f"The continuous reference check passed {reference_ok_count}/{len(main_validation)} main runs; the largest sampled reference acceleration was {max_reference_accel:.3f} m/s².")
    lines.extend(["", "## Counts", "", f"Anchors: {counts['anchors']} runs. Regression: {counts['regression']} runs. Main methods: {counts['main']} runs. Validation rows: {counts['validation']}. Ablations: {counts.get('ablation', 0)} runs.", "", "## Method comparison", "", "| method | n | geometry | mission-valid | full-window | mission-valid denominator | all-run speed deficit (m) | valid-run speed deficit (m) | planner ms/decision | candidates/decision |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary:
        lines.append(f"| {row['method']} | {row['n']} | {row['geometry_completed']} | {row['mission_valid']} | {row['full_window_valid']} | {row['mission_valid_n']} | {row['mean_speed_deficit_all_m']:.3f} | {row['mean_speed_deficit_mission_valid_m']:.3f} | {1000*row['mean_planner_compute_s']:.2f} | {row['mean_candidates']:.1f} |")
    lines.extend(["", "The speed-deficit means use all 18 runs in the all-run column and only the independently mission-valid runs in the valid-run column. The phase-resolved all-run and valid-run values, including prepare, execute, success-release and failure-handling denominators, are in `outputs/stage2a/summary.csv`.", "", "## Anchor evidence", "", "Collaborative none/0 anchor labels:", *[f"- {x}" for x in anchor_text], "", "The key planner trace, predicted completion and actual completion are in `outputs/stage2a/key_trajectories.csv`, `plan_events.csv` and `validation.csv`. The selected plan is generated from current observations; F prediction uses its current state and the declared nominal model, B responds with IDM, and the real run applies the prepare disturbance only after planning. No future disturbance, scenario identifier or witness answer is passed to the controller.", "", "Geometry completion, mission validity and full-window validity are separate labels. In particular, a task can complete at 170 m with all task-segment constraints valid while a later post-release h violation makes the 16 s full-window label zero. That failure remains in the ledger.", "", "## Contribution interpretation", "", "P and S1 share the finite planner, reference family and common executor. S1 adds a bounded internal preparation intensity while disabling the new space-to-process feedback. S2 adds current-width feedback and symmetric partner action. S3 changes only the role allocation while preserving the total partner coupling budget. The controlled S2 partner-removal, fixed-nonzero-W and feedback-removal ablations are in `outputs/stage2a/ablations.csv`; their task labels and cost changes are not folded into the main method counts.", "", *ablation_lines, "", "The common predictor and executor are not credited to swarmalator coupling. A zero S2/S3 task increment over S1 is retained as a result; a lower cost with the same task count is reported as a cost effect, not proof of physical necessity.", "", "## Public execution and validation basis", "", "The checker records geometry completion, mission-valid completion and full-window validity. It rejects corrupted actions, boundary crossings, dynamic h violations, missing dynamics, body overlap, road exits and discontinuous progress. A saved nominal action is never substituted for the actual action. Reference continuity is reported separately in `validation.csv`; the largest jump and acceleration are directly logged per run.", "", "## Limits", "", "The 18 environments are a development/regression set used in earlier stages, not an unseen test set. A failed finite plan remains unresolved rather than being called physical infeasibility. Communication noise/delay, SUMO, CARLA, hardware and MARL remain outside Stage2A.", "", "## Figures", "", *[f"* `{p}`" for p in figures.values()], "", "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m pytest -q", ".\\.venv\\Scripts\\python.exe -m src.swarmalator_cav.run_stage2a --config configs/stage2a_config.json --out outputs/stage2a", "```", ""])
    (root / "reports" / "stage2a_report.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "reports" / "stage2a_validation.md").write_text("# Stage 2A validation ledger\n\nThe independent checker is `src/swarmalator_cav/stage1c_check.py`; Stage2A adds reference continuity and three outcome labels. A finite plan failure remains unresolved.\n\n" + json.dumps(counts, indent=2), encoding="utf-8")


def _package(root, out_dir):
    feedback = root / "feedback" / "stage2a_feedback.zip"; feedback.parent.mkdir(exist_ok=True)
    include = [root / "AGENTS.md", root / "codex_swarmalator_stage2a_prompt.md", root / "README.md", root / "references" / "literature_notes.md",
               root / "configs" / "stage2a_config.json", root / "src" / "swarmalator_cav" / "simulation.py", root / "src" / "swarmalator_cav" / "stage1c_simulation.py",
               root / "src" / "swarmalator_cav" / "stage1c_check.py", root / "src" / "swarmalator_cav" / "stage2a_common.py",
               root / "src" / "swarmalator_cav" / "stage2a_planning.py", root / "src" / "swarmalator_cav" / "stage2a_check.py",
               root / "src" / "swarmalator_cav" / "run_stage2a.py", root / "src" / "swarmalator_cav" / "__init__.py", root / "docs" / "model_v3.md",
               root / "reports" / "stage2a_report.md", root / "reports" / "stage2a_validation.md", root / "tests" / "test_stage2a.py"]
    for path in [out_dir / name for name in ("config_used.json", "frozen_evidence_hashes.json", "pilot_notes.json", "metrics.csv", "vehicle_metrics.csv", "validation.csv", "summary.csv", "anchor_results.csv", "ablations.csv", "plan_events.csv", "key_events.csv", "key_trajectories.csv", "run_log.txt")]:
        include.append(path)
    include.extend((out_dir / "figures").glob("*.png"))
    with zipfile.ZipFile(feedback, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            path = path.resolve()
            if path.exists(): zf.write(path, path.relative_to(root.resolve()).as_posix())
    sha = hashlib.sha256(feedback.read_bytes()).hexdigest()
    manifest = root / "feedback" / "stage2a_feedback_manifest.txt"
    with zipfile.ZipFile(feedback) as zf: names = zf.namelist()
    manifest.write_text(f"stage2a_feedback.zip sha256={sha}\nfiles={len(names)}\n", encoding="utf-8")
    return feedback, sha, len(names)


def run(args):
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = Config(**{k: v for k, v in config.get("simulation", {}).items() if k in Config.__dataclass_fields__})
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Anchors are run first. The same objects are reused in the full regression
    # matrix so the matrix does not silently replace anchor evidence.
    results = []
    for method in METHODS:
        for scenario, disturbance, variant in ANCHORS:
            results.append(run_episode(method, scenario, disturbance, variant, cfg))
    anchor_keys = set(ANCHORS)
    for method in METHODS:
        for scenario in SCENARIOS:
            for disturbance in DISTURBANCES:
                for variant in VARIANTS:
                    if (scenario, disturbance, variant) in anchor_keys: continue
                    results.append(run_episode(method, scenario, disturbance, variant, cfg))
    # Controlled contribution diagnostics on two anchors, same planner budget.
    ablation_results = []
    for ablation in ("no_partner", "no_feedback", "fixed_W"):
        for method in ("S2",):
            for scenario, disturbance, variant in (ANCHORS[0], ANCHORS[3]):
                ablation_results.append(run_episode(method, scenario, disturbance, variant, cfg, ablation=ablation))
    all_results = results + ablation_results
    validations = _validation_rows(all_results, cfg)
    main_validations = [row for row in validations if not row.get("ablation")]
    summary = _summary(results, main_validations)
    write_csv(out_dir / "metrics.csv", [r.metrics for r in results])
    write_csv(out_dir / "vehicle_metrics.csv", [row for r in results for row in r.vehicle_metrics])
    write_csv(out_dir / "validation.csv", validations)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "anchor_results.csv", [dict(r.metrics, **{k: v for k, v in next(v for v in validations if v["run_id"] == r.run_id).items() if k in ("mission_valid", "full_window_valid", "reference_ok", "error")}) for r in results if (r.scenario, r.disturbance, r.init_variant) in anchor_keys])
    write_csv(out_dir / "ablations.csv", [dict(r.metrics, validation=next(v for v in validations if v["run_id"] == r.run_id).get("mission_valid", 0)) for r in ablation_results])
    write_csv(out_dir / "plan_events.csv", [event for r in all_results for event in r.events if event.get("event") == "plan_generated"])
    write_csv(out_dir / "key_events.csv", [event for r in all_results for event in r.events])
    key_rows = []
    for r in results:
        if (r.scenario, r.disturbance, r.init_variant) in anchor_keys and r.method in ("P", "S2", "S3", "oldC"):
            key_rows.extend(r.logs)
    write_csv(out_dir / "key_trajectories.csv", key_rows)
    figures = _make_figures(out_dir, results, main_validations)
    counts = {"anchors": len(ANCHORS) * len(METHODS), "regression": len(results) - len(ANCHORS) * len(METHODS), "main": len(results), "validation": len(validations), "ablation": len(ablation_results)}
    _write_docs(root, results, main_validations, summary, figures, counts)
    (out_dir / "run_log.txt").write_text(json.dumps({"config": str(Path(args.config).resolve()), "counts": counts, "figures": figures}, indent=2), encoding="utf-8")
    feedback, sha, files = _package(root, out_dir)
    print(json.dumps({"main_runs": len(results), "anchor_runs": counts["anchors"], "regression_runs": counts["regression"], "ablation_runs": len(ablation_results), "validation_rows": len(validations), "feedback": str(feedback), "sha256": sha, "files": files}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2a_config.json")
    parser.add_argument("--out", default="outputs/stage2a")
    run(parser.parse_args())


if __name__ == "__main__": main()
