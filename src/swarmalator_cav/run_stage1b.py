"""Run the bounded Stage 1B matrix, diagnostics, figures, and reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .stage1b_simulation import (
    SimConfig, SimulationResult, TaskState, VehicleState, build_scenario,
    dynamic_gap, environment_accel, execute_acceleration, geometry, idm_accel,
    make_controller, make_vehicles, min_forward_distance, physical_gap_floor,
    run_episode,
)


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
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def independent_replay_check(result: SimulationResult, cfg: SimConfig) -> Tuple[bool, str]:
    """Recheck saved states/actions without using the simulator counters."""
    by_time: Dict[float, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in result.logs:
        by_time[round(as_float(row.get("t_s")), 6)][str(row["vehicle"])] = row
    times = sorted(by_time)
    errors: List[str] = []
    for t in times:
        if set(by_time[t]) != {"M", "R", "F", "B"}:
            errors.append(f"missing_vehicle@{t}")
    for t0, t1 in zip(times, times[1:]):
        if t1 - t0 > cfg.dt * 1.01:
            continue
        cur, nxt = by_time[t0], by_time[t1]
        for vid in ("M", "R", "F", "B"):
            r0, r1 = cur[vid], nxt[vid]
            s = as_float(r0["s_m"]); v = as_float(r0["v_mps"]); a = as_float(r0["a_actual_mps2"])
            pred_s = s + v * cfg.dt + 0.5 * a * cfg.dt * cfg.dt
            pred_v = min(cfg.v_max, max(cfg.v_min, v + a * cfg.dt))
            if abs(as_float(r1["s_m"]) - pred_s) > 2e-7:
                errors.append(f"dynamics_s_{vid}@{t0}")
            if abs(as_float(r1["v_mps"]) - pred_v) > 2e-7:
                errors.append(f"dynamics_v_{vid}@{t0}")
            if not (cfg.a_min - 1e-8 <= a <= cfg.a_max + 1e-8):
                errors.append(f"accel_bound_{vid}@{t0}")
            prev_a = as_float(r0["a_mps2"])
            if abs(a - prev_a) > cfg.jerk_max * cfg.dt + 2e-7:
                errors.append(f"jerk_bound_{vid}@{t0}")
            y = as_float(r0["y_m"])
            if not (-0.5 <= y <= cfg.lane_ramp + 0.5):
                errors.append(f"lateral_bound_{vid}@{t0}")
        def net(leader: str, follower: str) -> float:
            return as_float(cur[leader]["s_m"]) - as_float(cur[follower]["s_m"]) - 4.8
        if net("F", "R") < -1e-6 or net("R", "B") < -1e-6:
            errors.append(f"body_overlap@{t0}")
        m = cur["M"]
        if as_float(m.get("merge_progress", 0.0)) > 0.0 or as_float(m["y_m"]) <= cfg.lane_ramp / 2:
            for oid in ("F", "R", "B"):
                if abs(as_float(m["y_m"]) - as_float(cur[oid]["y_m"])) <= 1.9 and abs(as_float(m["s_m"]) - as_float(cur[oid]["s_m"])) < 4.8:
                    errors.append(f"merge_body_overlap@{t0}")
    return not errors, ";".join(errors[:6])


def _env_groups(results: Sequence[SimulationResult]) -> Dict[Tuple[str, str, int], List[SimulationResult]]:
    groups: Dict[Tuple[str, str, int], List[SimulationResult]] = defaultdict(list)
    for result in results:
        groups[(result.scenario, result.disturbance, result.init_variant)].append(result)
    return groups


def _optimistic_bound(scenario_name: str, init_variant: int, cfg: SimConfig) -> Dict[str, object]:
    sc = build_scenario(scenario_name, init_variant)
    target = max(sc.initial_gap, physical_gap_floor(sc, cfg))
    # An outer gap-opening trajectory: F uses a_max and R uses a_min.  It is
    # deliberately optimistic and is used only for exclusions.
    best_gap, t_best = sc.initial_gap, 0.0
    for t in np.linspace(0.0, cfg.horizon, 321):
        gap = (sc.F_s + sc.F_v * t + 0.5 * cfg.a_max * t * t) - (sc.R_s + sc.R_v * t + 0.5 * cfg.a_min * t * t) - 4.8
        if gap > best_gap:
            best_gap, t_best = float(gap), float(t)
    m = VehicleState("M", sc.M_s, cfg.lane_ramp, sc.M_v, 0.0)
    min_dist = min_forward_distance(m, cfg.merge_duration, cfg)
    late = sc.M_s + min_dist > sc.completion_s + 1e-9
    excluded = best_gap < target - 1e-9 or late
    return {"necessary_bound_label": "excluded_by_necessary_bound" if excluded else "not_excluded",
            "optimistic_max_gap_m": best_gap, "optimistic_time_s": t_best,
            "target_gap_m": target, "min_forward_distance_m": min_dist,
            "late_start_bound": int(late),
            "bound_reason": "optimistic_gap_outer_bound" if best_gap < target - 1e-9 else ("late_start_min_forward_bound" if late else "")}


def _bounded_piecewise_search(scenario_name: str, disturbance: str, init_variant: int,
                              cfg: SimConfig) -> Dict[str, object]:
    """Small, transparent offline witness search for unresolved environments.

    It searches five preparation waits and three constant M/R acceleration
    candidates. F follows the known environment rule and B follows its closed
    loop IDM rule. The candidate is accepted only when every discrete step of
    the three-second lateral path satisfies body and dynamic-gap checks and
    the endpoint is in the finite-body slot. This is a diagnostic witness, not
    an online controller and not a continuous-time certificate.
    """
    sc = build_scenario(scenario_name, init_variant)
    waits = (0.0, 0.5, 1.0, 1.5, 2.0)
    actions = (-2.0, 0.0, 1.5)
    trials = 0
    for wait in waits:
        for a_m_cmd in actions:
            for a_r_cmd in actions:
                trials += 1
                vehicles = make_vehicles(sc, cfg)
                ok = True
                # Preparation is propagated with the same F/B rules and a
                # bounded constant candidate for M/R.
                n_wait = int(round(wait / cfg.dt))
                for step in range(n_wait):
                    t = step * cfg.dt
                    acts = {"M": a_m_cmd, "R": a_r_cmd,
                            "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg),
                            "B": idm_accel(vehicles["B"], vehicles["R"], cfg)}
                    for vid in ("M", "R", "F", "B"):
                        v = vehicles[vid]; aa, _, _ = execute_acceleration(v.a, acts[vid], cfg); v.s += v.v * cfg.dt + 0.5 * aa * cfg.dt * cfg.dt; v.v = max(cfg.v_min, min(cfg.v_max, v.v + aa * cfg.dt)); v.a = aa
                g = geometry(vehicles, sc, cfg)
                if not (sc.merge_zone[0] <= vehicles["M"].s <= sc.merge_zone[1] and g["interval_width"] >= 0.0 and g["lower"] <= vehicles["M"].s <= g["upper"]):
                    continue
                # Execute a fixed three-second path under the same finite-body
                # constraints. q is intentionally absent from this search.
                for j in range(int(round(cfg.merge_duration / cfg.dt))):
                    t = wait + j * cfg.dt
                    acts = {"M": a_m_cmd, "R": a_r_cmd,
                            "F": environment_accel(vehicles["F"], t, sc, disturbance, cfg),
                            "B": idm_accel(vehicles["B"], vehicles["R"], cfg)}
                    for vid in ("M", "R", "F", "B"):
                        v = vehicles[vid]; aa, _, _ = execute_acceleration(v.a, acts[vid], cfg); v.s += v.v * cfg.dt + 0.5 * aa * cfg.dt * cfg.dt; v.v = max(cfg.v_min, min(cfg.v_max, v.v + aa * cfg.dt)); v.a = aa
                    vehicles["M"].merge_progress = min(1.0, vehicles["M"].merge_progress + cfg.dt / cfg.merge_duration)
                    vehicles["M"].y = cfg.lane_ramp * (1.0 - vehicles["M"].merge_progress)
                    if (vehicles["F"].s - vehicles["R"].s - 4.8 < dynamic_gap(vehicles["R"], cfg) or vehicles["R"].s - vehicles["B"].s - 4.8 < dynamic_gap(vehicles["B"], cfg)):
                        ok = False; break
                    if vehicles["F"].s - vehicles["M"].s - 4.8 < dynamic_gap(vehicles["M"], cfg) or vehicles["M"].s - vehicles["R"].s - 4.8 < dynamic_gap(vehicles["R"], cfg):
                        ok = False; break
                if ok:
                    g_end = geometry(vehicles, sc, cfg)
                    if g_end["interval_width"] >= 0.0 and g_end["lower"] <= vehicles["M"].s <= g_end["upper"]:
                        return {"found": 1, "trials": trials, "wait_s": wait, "a_M_mps2": a_m_cmd, "a_R_mps2": a_r_cmd,
                                "search_note": "bounded piecewise candidate passed discrete occupancy and endpoint checks"}
    return {"found": 0, "trials": trials, "wait_s": "", "a_M_mps2": "", "a_R_mps2": "",
            "search_note": "bounded piecewise candidates exhausted; no witness claim"}


def feasibility_diagnostic(results: Sequence[SimulationResult], cfg: SimConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (scenario, disturbance, init_variant), candidates in sorted(_env_groups(results).items()):
        witness = None; replay_note = ""
        for candidate in candidates:
            if int(candidate.metrics.get("success", 0)):
                ok, note = independent_replay_check(candidate, cfg)
                if ok:
                    witness, replay_note = candidate, "independent_replay_ok"; break
                replay_note = note
        bound = _optimistic_bound(scenario, init_variant, cfg)
        search = {"found": 0, "trials": 0, "wait_s": "", "a_M_mps2": "", "a_R_mps2": "", "search_note": "not needed after replay witness"}
        if witness is None and bound["necessary_bound_label"] != "excluded_by_necessary_bound":
            search = _bounded_piecewise_search(scenario, disturbance, init_variant, cfg)
        if witness is not None:
            label, source, reason = "feasible_witness", witness.method, "saved successful discrete trajectory replay-checked independently"
        elif search["found"]:
            label, source, reason = "feasible_witness", "bounded_piecewise_search", search["search_note"]
        elif bound["necessary_bound_label"] == "excluded_by_necessary_bound":
            label, source, reason = "excluded_by_necessary_bound", "optimistic_outer_bound", str(bound["bound_reason"])
        else:
            label, source, reason = "unresolved", "", "no witness and no valid exclusion; controller failure is not physical infeasibility"
        sc = build_scenario(scenario, init_variant)
        m = VehicleState("M", sc.M_s, cfg.lane_ramp, sc.M_v, 0.0)
        rows.append({"scenario": scenario, "disturbance": disturbance, "init_variant": init_variant,
                     "label": label, "witness_method": source, "replay_note": replay_note, "reason": reason,
                     "initial_gap_m": sc.initial_gap, "target_gap_m": physical_gap_floor(sc, cfg),
                     "remaining_distance_m": sc.completion_s - m.s,
                     "min_forward_distance_m": min_forward_distance(m, cfg.merge_duration, cfg), **bound})
        rows[-1].update(search)
    return rows


def _snapshot_from_result(result: SimulationResult) -> Tuple[Optional[Dict[str, VehicleState]], Optional[TaskState]]:
    by_time: Dict[float, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in result.logs:
        by_time[round(as_float(row["t_s"]), 6)][str(row["vehicle"])] = row
    selected = None
    for t in sorted(by_time):
        if all(by_time[t][v].get("task_phase") == "EXECUTE" for v in ("M", "R", "F", "B")):
            selected = by_time[t]; break
    if selected is None:
        return None, None
    snapshot = {}
    for vid, row in selected.items():
        snapshot[vid] = VehicleState(vid, as_float(row["s_m"]), as_float(row["y_m"]), as_float(row["v_mps"]), as_float(row["a_mps2"]), merge_progress=as_float(row.get("merge_progress", 0.0)))
    return snapshot, TaskState(phase="EXECUTE")


def same_snapshot_interventions(results: Sequence[SimulationResult], cfg: SimConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        if result.method not in ("C", "D"):
            continue
        snapshot, task = _snapshot_from_result(result)
        if snapshot is None or task is None:
            continue
        variants = ["full", "no_space", "no_partner", "no_W"] if result.method == "C" else ["full", "symmetric_k", "no_space"]
        sc = build_scenario(result.scenario, result.init_variant)
        qlogs = {v: next((r for r in result.logs if r["vehicle"] == v and r.get("task_phase") == "EXECUTE"), None) for v in ("M", "R")}
        for variant in variants:
            c = make_controller(result.method, sc, cfg, ablation="" if variant == "full" else variant)
            for vid in ("M", "R"):
                if qlogs[vid] is not None:
                    c.q[vid] = as_float(qlogs[vid].get("q", 0.0)); c.e_M0 = as_float(qlogs[vid].get("e_M0", 0.0))
            out = c.compute_pair(snapshot, task, cfg.dt)
            for vid in ("M", "R"):
                d = out[vid]
                rows.append({"source_run_id": result.run_id, "method": result.method, "scenario": result.scenario,
                             "disturbance": result.disturbance, "init_variant": result.init_variant,
                             "snapshot_phase": task.phase, "ablation": variant, "vehicle": vid,
                             "qdot_space_s-1": d["qdot_space"], "qdot_partner_s-1": d["qdot_partner"],
                             "qdot_clipped_s-1": d["qdot_clipped"], "K_to_partner_s-1": d["K_to_partner"],
                             "W_to_partner": d["W_to_partner"], "a_nom_mps2": d["a_nom"], "s_ref_m": d["s_ref"],
                             "target_gap_physical_m": d["target_gap"]})
    return rows


def generate_figures(out_dir: Path, metrics: Sequence[Mapping[str, object]], feasibility: Sequence[Mapping[str, object]],
                     key_trajectories: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"feasible_witness": "#2ca02c", "excluded_by_necessary_bound": "#d62728", "unresolved": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, color in colors.items():
        xs = [as_float(r["remaining_distance_m"]) for r in feasibility if r["label"] == label]
        ys = [as_float(r["min_forward_distance_m"]) for r in feasibility if r["label"] == label]
        if xs: ax.scatter(xs, ys, color=color, label=label, s=45)
    xmax = max([as_float(r["remaining_distance_m"]) for r in feasibility] + [1.0])
    ax.plot([0, xmax], [0, xmax], "k--", linewidth=0.8, label="equal")
    ax.set_xlabel("Remaining distance to completion (m)"); ax.set_ylabel("3 s optimistic minimum forward travel (m)")
    ax.set_title("Startup timing bound and diagnostic strata"); ax.legend(fontsize=8)
    p1 = fig_dir / "stage1b_startup_feasibility.png"; fig.savefig(p1, dpi=160); plt.close(fig)

    by_run: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in key_trajectories:
        if row.get("vehicle") == "M": by_run[str(row["run_id"])].append(row)
    rep = next((v for k, v in by_run.items() if k.startswith("D_ample_none_i0")), next(iter(by_run.values()), []))
    rep = sorted(rep, key=lambda r: as_float(r["t_s"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    if rep:
        t = [as_float(r["t_s"]) for r in rep]
        axes[0].plot(t, [as_float(r["qdot_space_s-1"]) for r in rep], label="raw space")
        axes[0].plot(t, [as_float(r["qdot_partner_s-1"]) for r in rep], label="raw partner")
        axes[0].plot(t, [as_float(r["qdot_clipped_s-1"]) for r in rep], label="effective clipped qdot")
        axes[1].plot(t, [as_float(r["s_ref_m"], float("nan")) for r in rep], label="moving reference")
        axes[1].plot(t, [as_float(r["a_nom_mps2"]) for r in rep], label="nominal action")
        axes[1].plot(t, [as_float(r["a_actual_mps2"]) for r in rep], label="actual action")
        axes[1].set_xlabel("time (s)")
    axes[0].set_ylabel("q feedback (s$^{-1}$)"); axes[1].set_ylabel("reference / action")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); axes[0].set_title("Effective q feedback and spatial action effect")
    p2 = fig_dir / "stage1b_effective_q_feedback.png"; fig.savefig(p2, dpi=160); plt.close(fig)

    strata = {f"{r['scenario']}|{r['disturbance']}|{r['init_variant']}": r["label"] for r in feasibility}
    methods = list("ABCDE"); labels = ["feasible_witness", "unresolved", "excluded_by_necessary_bound"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    width = 0.15; x = np.arange(len(methods))
    for j, label in enumerate(labels):
        costs, success = [], []
        for method in methods:
            rs = [r for r in metrics if r.get("method") == method and strata.get(f"{r.get('scenario')}|{r.get('disturbance')}|{r.get('init_variant')}") == label]
            costs.append(statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in rs]) if rs else 0.0)
            success.append(statistics.mean([as_float(r.get("success")) for r in rs]) if rs else 0.0)
        axes[0].bar(x + (j - 1) * width, costs, width, label=label); axes[1].bar(x + (j - 1) * width, success, width, label=label)
    axes[0].set_xticks(x, methods); axes[1].set_xticks(x, methods); axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("All-vehicle speed-deficit distance (m)"); axes[1].set_ylabel("Completion fraction")
    axes[0].set_title("Cost within feasibility strata"); axes[1].set_title("Success within feasibility strata")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    p3 = fig_dir / "stage1b_cost_by_feasibility.png"; fig.savefig(p3, dpi=160); plt.close(fig)
    return {"startup": str(p1), "q_feedback": str(p2), "strata": str(p3)}


def summary(metrics: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for method in "ABCDE":
        rs = [r for r in metrics if r.get("method") == method]
        rows.append({"method": method, "label": METHOD_LABELS[method], "n": len(rs),
                     "successes": sum(int(as_float(r.get("success"))) for r in rs),
                     "completion_rate": statistics.mean([as_float(r.get("success")) for r in rs]) if rs else 0.0,
                     "mean_cost_m": statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in rs]) if rs else float("nan"),
                     "mean_min_gap_m": statistics.mean([as_float(r.get("min_net_gap_m")) for r in rs]) if rs else float("nan"),
                     "mean_action_saturation": statistics.mean([as_float(r.get("action_saturation_count")) for r in rs]) if rs else 0.0,
                     "mean_recheck": statistics.mean([as_float(r.get("safety_recheck_violation_count")) for r in rs]) if rs else 0.0})
    return rows


def write_model_doc(root: Path) -> None:
    text = """# Swarmalator–CAV Stage 1B model v1

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
"""
    (root / "docs" / "model_v1.md").write_text(text, encoding="utf-8")


def write_report(root: Path, metrics: Sequence[Mapping[str, object]], feasibility: Sequence[Mapping[str, object]],
                 interventions: Sequence[Mapping[str, object]], snapshots: Sequence[Mapping[str, object]], figures: Mapping[str, str]) -> None:
    old_path = root / "outputs" / "stage1" / "metrics.csv"
    old = read_csv(old_path) if old_path.exists() else []
    old_by = {m: [r for r in old if r.get("method") == m] for m in "ABCDE"}
    lines = []
    for s in summary(metrics):
        o = old_by[s["method"]]
        old_cost = statistics.mean([as_float(r.get("total_speed_deficit_integral_m_s")) for r in o]) if o else float("nan")
        lines.append(f"| {s['method']} | {s['successes']}/18 ({100*s['completion_rate']:.1f}%) | {s['mean_cost_m']:.2f} | {s['mean_min_gap_m']:.2f} | {s['mean_action_saturation']:.1f} | {s['mean_recheck']:.1f} | {sum(int(as_float(r.get('success'))) for r in o)}/18 | {old_cost:.2f} |")
    strata = Counter(r["label"] for r in feasibility)
    matched = []
    for method in "ABCDE":
        old_s = {r["run_id"] for r in old_by[method] if int(as_float(r.get("success")))}
        new_s = {r["run_id"] for r in metrics if r.get("method") == method and int(as_float(r.get("success")))}
        matched.append(f"{method}: old∩new={len(old_s & new_s)}, old_only={len(old_s-new_s)}, new_only={len(new_s-old_s)}")
    c_space = statistics.mean([as_float(r.get("space_feedback_abs_mean_s-1")) for r in metrics if r.get("method") == "C"])
    d_space = statistics.mean([as_float(r.get("space_feedback_abs_mean_s-1")) for r in metrics if r.get("method") == "D"])
    intervention_lines = []
    for method in ("C", "D"):
        full = [r for r in interventions if r.get("method") == method and not r.get("ablation")]
        full_cost = statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in full]) if full else float("nan")
        full_success = statistics.mean([as_float(r.get("success")) for r in full]) if full else float("nan")
        variants = sorted({str(r.get("ablation")) for r in interventions if r.get("method") == method and r.get("ablation")})
        for variant in variants:
            rs = [r for r in interventions if r.get("method") == method and r.get("ablation") == variant]
            cost = statistics.mean([as_float(r.get("total_speed_deficit_distance_m")) for r in rs]) if rs else float("nan")
            success = statistics.mean([as_float(r.get("success")) for r in rs]) if rs else float("nan")
            intervention_lines.append(f"* {method} `{variant}` versus same-environment full: Δcost={cost-full_cost:+.3f} m, Δcompletion={success-full_success:+.3f} ({sum(int(as_float(r.get('success'))) for r in rs)}/{len(rs)} versus {sum(int(as_float(r.get('success'))) for r in full)}/{len(full)}).")
    report = f"""# Swarmalator–CAV Stage 1B report

The corrected model ran the original 18 environments with five methods (**90
primary runs**), an 18-environment feasibility diagnostic, **{len(interventions)}**
bounded intervention runs, and **{len(snapshots)}** same-snapshot controller
rows. Stage 1 outputs were not overwritten. The requested
`stage1_review_and_next_step.md` and independent review package were not found
in the workspace or Downloads; the findings reproduced in the Stage1B prompt
were treated as constraints.

## Actual changes

* All methods share the finite-body target gap. The 18 m/s non-empty-slot floor
  is 21.4 m; larger initial gaps are retained. Process q targets are separate.
* The old moving-reference speed surrogate is replaced by reference position,
  speed and acceleration feedforward plus error feedback. B/C/D use current
  qdot in the reference derivative without future information or an algebraic
  loop; E uses the same low-level tracker.
* Start feasibility checks the remaining fixed three-second maneuver using
  maximum braking and jerk. Boundaries are [55,170] m, completion 170 m,
  post-task marker 190 m, and road 450 m.
* Success release and failure handling are explicit. Post-success motion is
  common cruise/following; a started failure continues its lateral path under
  safety, while an unstarted failure remains on the ramp.
* Distance, speed-deficit units, role gap minima, nominal/safety/actual action,
  saturation, post-jerk checks and terminal logs use the corrected v1 fields.

## Paired results

| Method | v1 completion | v1 mean cost (m) | v1 mean min gap (m) | action saturation/run | rechecks/run | Stage1 completion | Stage1 mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Old and new columns are separate model versions. Improvements from common basis
changes are not D-specific. Full v1 outcomes are `{dict(Counter(r.get('outcome','') for r in metrics))}`. Aggregate hard body/road violations are `{sum(int(as_float(r.get('hard_violation_count'))) for r in metrics)}`; remaining post-jerk rechecks are retained as diagnostics and are not declared physical infeasibility.

Success-set comparison against frozen Stage1:

{chr(10).join(matched)}

## Feasibility strata

Counts are **{dict(strata)}**. `feasible_witness` means either a saved successful
trajectory passed an independent replay or a bounded piecewise candidate passed
the independent discrete dynamics, action, finite-body occupancy and road
checks. `excluded_by_necessary_bound` uses only the
optimistic gap-opening/late-start outer bound. Everything else is
`unresolved`; controller failure is not labelled physical infeasibility. The
witness diagnostic may inspect the saved future trajectory and does not enter
online A–E.

## Mechanism and intervention check

Logs retain raw space/partner feedback, clipped q changes, K/W, phase, q
saturation, reference offset, nominal action, safety target and actual action.
The same-snapshot table isolates C's space, W and partner terms and D's
symmetric-K alternative. Primary mean raw space feedback is C={c_space:.4f}
s⁻¹ and D={d_space:.4f} s⁻¹. These are activity measures, not traffic-value
claims; D is not presumed to win.

The bounded paired intervention deltas were:

{chr(10).join(intervention_lines)}

## Figures and main issue

1. `{figures['startup']}`: remaining distance versus optimistic minimum
   three-second travel, colored by feasibility label.
2. `{figures['q_feedback']}`: raw/effective q feedback and moving-reference /
   nominal/actual action for a real D trajectory.
3. `{figures['strata']}`: all-vehicle cost and completion within strata.

The main unresolved scientific issue is whether non-reciprocal allocation has a
repeatable net value after a separately verified feasible trajectory and delayed
or noisy information are introduced. This stage does not expand to those
experiments. Short-window failures remain a mixture of timing limits and
controller behavior, not a blanket infeasibility result.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m src.swarmalator_cav.run_stage1b --config configs/stage1b_config.json --out outputs/stage1b
```
"""
    (root / "reports" / "stage1b_report.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config); config_path = config_path if config_path.is_absolute() else root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out); out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(**config.get("simulation", {}))
    results: List[SimulationResult] = []
    for method in config.get("methods", list("ABCDE")):
        for scenario in config.get("scenarios", ["ample", "collaborative", "short_window"]):
            for disturbance in config.get("disturbances", ["none", "prepare"]):
                for init_variant in config.get("init_variants", [0, 1, 2]):
                    results.append(run_episode(method, scenario, disturbance, int(init_variant), cfg=cfg))
    metrics = [r.metrics for r in results]
    trajectories = [row for r in results for row in r.logs]
    events = [row for r in results for row in r.events]
    vehicle_metrics = [row for r in results for row in r.vehicle_metrics]
    write_csv(out_dir / "metrics.csv", metrics); write_csv(out_dir / "trajectories.csv", trajectories)
    write_csv(out_dir / "events.csv", events); write_csv(out_dir / "vehicle_metrics.csv", vehicle_metrics)
    preferred = next((r for r in results if r.method == "D" and r.scenario == "ample" and r.disturbance == "none" and r.init_variant == 0), results[0])
    key_ids = {preferred.run_id, "A_ample_none_i0", "E_collaborative_none_i0"}
    write_csv(out_dir / "key_trajectories.csv", [r for r in trajectories if r["run_id"] in key_ids])
    write_csv(out_dir / "key_events.csv", [r for r in events if r["run_id"] in key_ids])
    feasibility = feasibility_diagnostic(results, cfg); write_csv(out_dir / "feasibility.csv", feasibility)
    intervention_results: List[SimulationResult] = []
    for scenario, disturbance, init_variant in config.get("intervention_environments", []):
        for method, variants in (("C", ["", "no_space", "no_W", "no_partner"]), ("D", ["", "symmetric_k"])):
            for ablation in variants:
                intervention_results.append(run_episode(method, scenario, disturbance, int(init_variant), cfg=cfg, ablation=ablation))
    interventions = [r.metrics for r in intervention_results]; write_csv(out_dir / "interventions.csv", interventions)
    snapshots = same_snapshot_interventions(results, cfg); write_csv(out_dir / "same_snapshot_interventions.csv", snapshots)
    figures = generate_figures(out_dir, metrics, feasibility, [r for r in trajectories if r["run_id"] in key_ids])
    write_model_doc(root); write_report(root, metrics, feasibility, interventions, snapshots, figures)
    (out_dir / "config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "run_log.txt").write_text("invocation=" + " ".join(map(str, sys.argv)) + "\n" +
        f"python={sys.executable}\nprimary_runs={len(results)}\nfeasibility_environments={len(feasibility)}\nintervention_runs={len(intervention_results)}\nsame_snapshot_rows={len(snapshots)}\nfigures={figures}\n", encoding="utf-8")
    print(json.dumps({"primary_runs": len(results), "feasibility": len(feasibility), "intervention_runs": len(intervention_results), "out": str(out_dir), "figures": figures}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage1b_config.json"); parser.add_argument("--out", default="outputs/stage1b")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
