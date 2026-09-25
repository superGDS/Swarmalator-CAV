"""Stage 2A predictive coordination model.

This module keeps the Stage 1C physical model as a common execution layer and
adds a small, transparent online planning layer.  The planner only receives a
current four-vehicle snapshot and a route/task description.  It predicts F by
the current-state environment model and B by IDM; the realized F disturbance
is used only by the simulator after the plan has been selected.

The five Stage 2A structures are deliberately separated:

``P``
    finite-window joint spatial/temporal planning without an internal state;
``S1``
    the same planner plus a smooth internal preparation state driving the
    spatial reference, with space-to-process feedback disabled;
``S2``
    S1 plus symmetric space/process and partner coupling;
``S3``
    S2 with a fixed non-reciprocal allocation of the same total coupling;
``oldC``
    the Stage 1C C coordination law under the Stage 2A common executor.

All actions are subsequently passed through the same jerk, velocity and
one-step chain safety projection.  The planner is a bounded candidate search,
not a continuous reachability or safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stage1c_simulation import (
    ACTIVE_PHASES,
    POST_PHASES,
    SimConfig,
    SimulationResult,
    TaskState,
    VehicleState,
    _clip,
    _lateral_overlap,
    _net_gap,
    _task_event,
    build_scenario,
    current_leader,
    current_r_leader,
    dynamic_gap,
    environment_accel,
    execute_safe_actions,
    geometry,
    idm_accel,
    make_controller,
    make_vehicles,
    physical_start_feasibility,
)


VEHICLES = ("M", "R", "F", "B")
STAGE2_METHODS = ("P", "S1", "S2", "S3", "oldC")


@dataclass
class Stage2AConfig(SimConfig):
    """Common simulation settings plus the finite planning budget."""

    planner_horizon_s: float = 7.0
    # The prediction uses the same 0.05 s physical step as execution.  A
    # caller may choose a coarser planning refresh, but the propagation step
    # itself stays aligned with the checked vehicle dynamics.
    planner_dt: float = 0.05
    planner_refresh_s: float = 0.50
    planner_wait_values: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    planner_action_values: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0)
    planner_max_candidates: int = 160
    planner_action_effort_weight: float = 0.015
    planner_wait_weight: float = 0.02
    coordinator_rate: float = 0.22
    coordinator_feedback_gain: float = 0.18
    coordinator_partner_gain: float = 0.16
    coordinator_reference_kp: float = 0.20
    coordinator_reference_kv: float = 0.30
    coordinator_action_limit: float = 0.85


@dataclass
class Stage2APlan:
    """A bounded plan generated from one legal current snapshot."""

    plan_id: int
    created_t_s: float
    valid: bool
    wait_s: float
    a_M_mps2: float
    a_R_mps2: float
    predicted_start_s: float = float("nan")
    predicted_start_t_s: float = float("nan")
    predicted_completion_s: float = float("nan")
    predicted_completion_t_s: float = float("nan")
    predicted_min_h_m: float = float("nan")
    predicted_net_clearance_m: float = float("nan")
    predicted_speed_cost_m: float = float("nan")
    candidate_count: int = 0
    compute_time_s: float = 0.0
    reason: str = ""

    def command(self, t_s: float) -> Dict[str, float]:
        """Return the planned M/R target at an absolute simulation time."""
        if not self.valid:
            return {"M": 0.0, "R": 0.0}
        # ``wait_s`` is the planned preparation duration, not a delay before
        # applying the preparation action.  Applying the action throughout
        # preparation is what lets the finite-window plan create the slot that
        # it later uses; resetting it to zero would recreate Stage1C's missed
        # online opportunity.
        return {"M": self.a_M_mps2, "R": self.a_R_mps2}


@dataclass
class CoordinationState:
    """Mutable state for S1--S3, saved in every relevant trajectory row."""

    eta_M: float = 0.0
    eta_R: float = 0.0
    e_M0: Optional[float] = None
    plan_id: int = -1
    last_eta_dot_M: float = 0.0
    last_eta_dot_R: float = 0.0

    def state_dict(self) -> Dict[str, object]:
        return {
            "eta_M": self.eta_M,
            "eta_R": self.eta_R,
            "e_M0": self.e_M0,
            "plan_id": self.plan_id,
            "last_eta_dot_M": self.last_eta_dot_M,
            "last_eta_dot_R": self.last_eta_dot_R,
        }


def _smoothstep(x: float) -> float:
    x = _clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoothstep_prime(x: float) -> float:
    x = _clip(x, 0.0, 1.0)
    return 6.0 * x * (1.0 - x)


def _smoothstep_second(x: float) -> float:
    x = _clip(x, 0.0, 1.0)
    return 6.0 - 12.0 * x


def _predicted_h(snapshot: Mapping[str, VehicleState], cfg: SimConfig,
                 active: bool) -> Dict[str, float]:
    def h(leader: str, follower: str) -> float:
        return _net_gap(snapshot[leader], snapshot[follower]) - dynamic_gap(snapshot[follower], cfg)

    out = {"FR": h("F", "R"), "RB": h("R", "B")}
    if active:
        out.update({"FM": h("F", "M"), "MR": h("M", "R")})
    return out


def _advance_snapshot(snapshot: Mapping[str, VehicleState], actual: Mapping[str, float],
                      cfg: SimConfig, active: bool) -> Dict[str, VehicleState]:
    out = {vid: state.copy() for vid, state in snapshot.items()}
    for vid, state in out.items():
        state.s = state.s + state.v * cfg.planner_dt + 0.5 * actual[vid] * cfg.planner_dt ** 2
        state.v = _clip(state.v + actual[vid] * cfg.planner_dt, cfg.v_min, cfg.v_max)
        state.a = actual[vid]
    if active:
        out["M"].merge_progress = _clip(
            out["M"].merge_progress + cfg.planner_dt / max(cfg.merge_duration, cfg.planner_dt),
            0.0, 1.0,
        )
        out["M"].y = cfg.lane_ramp + (cfg.lane_main - cfg.lane_ramp) * out["M"].merge_progress
    return out


def _candidate_rollout(snapshot: Mapping[str, VehicleState], scenario_name: str,
                       init_variant: int, cfg: Stage2AConfig, wait_s: float,
                       a_M: float, a_R: float) -> Dict[str, object]:
    """Predict one candidate without knowing the future environment script."""
    scenario = build_scenario(scenario_name, init_variant)
    states = {vid: state.copy() for vid, state in snapshot.items()}
    started = False
    start_tau: Optional[float] = None
    completion_tau: Optional[float] = None
    completion_s = float("nan")
    min_h = float("inf")
    min_clearance = float("inf")
    speed_cost = 0.0
    n_steps = int(math.ceil(cfg.planner_horizon_s / cfg.planner_dt))

    for index in range(n_steps):
        tau = index * cfg.planner_dt
        target_M = a_M
        target_R = a_R
        desired = {
            "M": target_M,
            "R": target_R,
            # Only the current-state environment model is used.  There is no
            # access to the future ``disturbance`` argument here.
            "F": environment_accel(states["F"], 0.0, scenario, "none", cfg),
            "B": idm_accel(states["B"], states["R"], cfg),
        }
        active = started
        actual, diag = execute_safe_actions(states, desired, active, cfg)
        if not int(diag["one_step_feasible"]):
            return {"valid": False, "reason": "predicted_action_interval_empty", "min_h": min_h,
                    "min_clearance": min_clearance, "speed_cost": speed_cost}
        states = _advance_snapshot(states, actual, cfg, active)
        h_values = _predicted_h(states, cfg, started)
        if h_values:
            min_h = min(min_h, min(h_values.values()))
        for key in ("FR", "RB"):
            min_clearance = min(min_clearance, _net_gap(states["F" if key == "FR" else "R"],
                                                        states["R" if key == "FR" else "B"]))
        speed_cost += sum(max(0.0, scenario.desired_speed - states[vid].v) * cfg.planner_dt
                          for vid in VEHICLES)
        if min(h_values.values()) < -1.0e-7:
            return {"valid": False, "reason": "predicted_dynamic_margin_negative", "min_h": min_h,
                    "min_clearance": min_clearance, "speed_cost": speed_cost}
        if states["M"].s > scenario.completion_s + 1.0e-7:
            return {"valid": False, "reason": "predicted_completion_boundary", "min_h": min_h,
                    "min_clearance": min_clearance, "speed_cost": speed_cost}

        if not started and tau + cfg.planner_dt >= wait_s - 1.0e-9:
            geom = geometry(states, scenario, cfg)
            if (scenario.merge_zone[0] - 1.0e-7 <= states["M"].s <= scenario.completion_s + 1.0e-7
                    and geom["interval_width"] >= -1.0e-7
                    and geom["lower"] - 1.0e-7 <= states["M"].s <= geom["upper"] + 1.0e-7):
                started = True
                start_tau = tau + cfg.planner_dt
                states["M"].merge_progress = 0.0
                states["M"].y = cfg.lane_ramp

        if started and states["M"].merge_progress >= 1.0 - 1.0e-9:
            geom = geometry(states, scenario, cfg)
            if (states["M"].s <= scenario.completion_s + 1.0e-7
                    and geom["interval_width"] >= -1.0e-7
                    and geom["lower"] - 1.0e-7 <= states["M"].s <= geom["upper"] + 1.0e-7):
                completion_tau = tau + cfg.planner_dt
                completion_s = states["M"].s
                break
            return {"valid": False, "reason": "predicted_endpoint_outside_slot", "min_h": min_h,
                    "min_clearance": min_clearance, "speed_cost": speed_cost}

    if start_tau is None:
        return {"valid": False, "reason": "no_predicted_slot_start", "min_h": min_h,
                "min_clearance": min_clearance, "speed_cost": speed_cost}
    if completion_tau is None:
        return {"valid": False, "reason": "no_predicted_completion", "min_h": min_h,
                "min_clearance": min_clearance, "speed_cost": speed_cost}
    return {"valid": True, "reason": "predicted_joint_witness", "min_h": min_h,
            "min_clearance": min_clearance, "speed_cost": speed_cost,
            "start_tau": start_tau, "completion_tau": completion_tau,
            "completion_s": completion_s}


class JointPredictivePlanner:
    """Finite candidate planner with a fixed, disclosed search budget."""

    def __init__(self, cfg: Stage2AConfig):
        self.cfg = cfg
        self.next_plan_id = 1

    def plan(self, snapshot: Mapping[str, VehicleState], scenario_name: str,
             created_t_s: float, init_variant: int = 0) -> Stage2APlan:
        started_clock = time.perf_counter()
        candidates = list(itertools.product(self.cfg.planner_wait_values,
                                             self.cfg.planner_action_values,
                                             self.cfg.planner_action_values))
        candidates = candidates[: self.cfg.planner_max_candidates]
        valid: List[Tuple[Tuple[float, float, float, float], float, float, float, Dict[str, object]]] = []
        last_reason = "no_candidate"
        for wait_s, a_M, a_R in candidates:
            result = _candidate_rollout(snapshot, scenario_name, init_variant, self.cfg, wait_s, a_M, a_R)
            if not result.get("valid"):
                last_reason = str(result.get("reason", last_reason))
                continue
            start_tau = float(result["start_tau"])
            completion_tau = float(result["completion_tau"])
            speed_cost = float(result["speed_cost"])
            effort = (a_M * a_M + a_R * a_R) * max(completion_tau - wait_s, 0.0)
            score = (completion_tau + self.cfg.planner_wait_weight * wait_s
                     + self.cfg.planner_action_effort_weight * effort
                     + 0.001 * speed_cost)
            valid.append(((score, completion_tau, wait_s, effort), wait_s, a_M, a_R, result))
        elapsed = time.perf_counter() - started_clock
        plan_id = self.next_plan_id
        self.next_plan_id += 1
        if not valid:
            return Stage2APlan(plan_id, created_t_s, False, 0.0, 0.0, 0.0,
                               candidate_count=len(candidates), compute_time_s=elapsed,
                               reason=last_reason)
        valid.sort(key=lambda row: row[0])
        _, wait_s, a_M, a_R, best = valid[0]
        return Stage2APlan(
            plan_id=plan_id, created_t_s=created_t_s, valid=True,
            wait_s=float(wait_s), a_M_mps2=float(a_M), a_R_mps2=float(a_R),
            predicted_start_s=float(best["start_tau"]),
            predicted_start_t_s=created_t_s + float(best["start_tau"]),
            predicted_completion_s=float(best["completion_s"]),
            predicted_completion_t_s=created_t_s + float(best["completion_tau"]),
            predicted_min_h_m=float(best["min_h"]),
            predicted_net_clearance_m=float(best["min_clearance"]),
            predicted_speed_cost_m=float(best["speed_cost"]),
            candidate_count=len(candidates), compute_time_s=elapsed,
            reason=str(best["reason"]),
        )


class SmoothCoordinator:
    """S1--S3 internal state and smooth reference family."""

    def __init__(self, method: str, scenario_name: str, cfg: Stage2AConfig, init_variant: int = 0):
        if method not in ("S1", "S2", "S3"):
            raise ValueError(method)
        self.method, self.scenario_name, self.init_variant, self.cfg = method, scenario_name, init_variant, cfg
        self.state = CoordinationState()

    def state_dict(self) -> Dict[str, object]:
        return self.state.state_dict()

    def _coupling(self, snapshot: Mapping[str, VehicleState], eta_M: float, eta_R: float,
                  phase: str) -> Tuple[float, float, float, float, float, float]:
        if phase not in ACTIVE_PHASES:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        sc = build_scenario(self.scenario_name, self.init_variant)
        geom = geometry(snapshot, sc, self.cfg)
        # Negative width creates process pressure, but only S2/S3 close this
        # space-to-process path.  S1 is a true forward-only process drive.
        space_to_process = 0.0 if self.method == "S1" else self.cfg.coordinator_feedback_gain * math.tanh(-geom["interval_width"] / 5.0)
        partner_phase = math.sin(0.5 * math.pi * (eta_R - eta_M))
        if self.method == "S3":
            k_MR, k_RM = 0.22, 0.10
        else:
            k_MR = k_RM = self.cfg.coordinator_partner_gain
        partner_M = k_MR * partner_phase
        partner_R = k_RM * (-partner_phase)
        return space_to_process, partner_M, partner_R, k_MR, k_RM, geom["interval_width"]

    def compute(self, snapshot: Mapping[str, VehicleState], task: TaskState,
                base_cmd: Mapping[str, float], plan: Optional[Stage2APlan],
                t_s: float) -> Dict[str, object]:
        sc = build_scenario(self.scenario_name, self.init_variant)
        geom = geometry(snapshot, sc, self.cfg)
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        if self.state.e_M0 is None:
            self.state.e_M0 = M.s - geom["c"]
        eta_M, eta_R = self.state.eta_M, self.state.eta_R
        space_to_process, partner_M, partner_R, k_MR, k_RM, width = self._coupling(
            snapshot, eta_M, eta_R, task.phase
        )
        base_rate = self.cfg.coordinator_rate if task.phase in ACTIVE_PHASES else 0.0
        # S1 advances from task/plan progress only. S2/S3 also respond to the
        # currently observed finite-body width, never to a future script.
        eta_dot_M = base_rate + space_to_process + partner_M
        eta_dot_R = base_rate + space_to_process + partner_R
        eta_dot_M = _clip(eta_dot_M, 0.0, 0.8) if eta_M < 1.0 else 0.0
        eta_dot_R = _clip(eta_dot_R, 0.0, 0.8) if eta_R < 1.0 else 0.0
        smooth = _smoothstep(eta_M)
        smooth_p = _smoothstep_prime(eta_M)
        smooth_pp = _smoothstep_second(eta_M)
        e0 = float(self.state.e_M0)
        c_dot = 0.5 * (R.v + F.v + self.cfg.time_headway * (R.a - M.a))
        s_ref = geom["c"] + (1.0 - smooth) * e0
        s_ref_dot = c_dot - e0 * smooth_p * eta_dot_M
        s_ref_ddot = -e0 * smooth_pp * eta_dot_M * eta_dot_M
        ref_adjust = _clip(
            self.cfg.coordinator_reference_kp * (s_ref - M.s)
            + self.cfg.coordinator_reference_kv * (s_ref_dot - M.v),
            -self.cfg.coordinator_action_limit, self.cfg.coordinator_action_limit,
        )
        a_M = float(base_cmd["M"] + ref_adjust)
        gap_error = geom["g_fr"] - sc.initial_gap
        # R receives a common current-state gap correction. The independent
        # partner term is logged separately and enters eta, not hidden in K.
        a_R = float(base_cmd["R"] + _clip(0.06 * gap_error, -0.35, 0.35))
        return {
            "a_nom_M": a_M, "a_nom_R": a_R,
            "eta_M": eta_M, "eta_R": eta_R,
            "eta_dot_M": eta_dot_M, "eta_dot_R": eta_dot_R,
            "eta_dot_space": space_to_process,
            "eta_dot_partner_M": partner_M, "eta_dot_partner_R": partner_R,
            "K_MR": k_MR, "K_RM": k_RM,
            "W_MR": 1.0 if self.method in ("S2", "S3") else 0.0,
            "W_RM": 1.0 if self.method in ("S2", "S3") else 0.0,
            "s_ref": s_ref, "s_ref_dot": s_ref_dot, "s_ref_ddot": s_ref_ddot,
            "reference_kind": f"smooth_internal_{self.method}",
            "interval_width": width, "gap_error": gap_error,
            "plan_id": plan.plan_id if plan is not None else -1,
            "plan_start_t_s": plan.predicted_start_t_s if plan is not None else float("nan"),
            "plan_reason": plan.reason if plan is not None else "",
            "plan_compute_time_s": plan.compute_time_s if plan is not None else 0.0,
            "target_time": "", "target_endpoint": "",
            "window_opening_accel": 0.0,
            "post_front_id": "F", "target_gap": sc.initial_gap,
            "target_gap_process": sc.initial_gap,
            "grad_phi_space": 0.0, "e_M0": e0,
            "t_s": t_s,
        }

    def commit(self, output: Mapping[str, object], dt: float, terminal: bool) -> None:
        if terminal:
            return
        self.state.eta_M = _clip(self.state.eta_M + float(output["eta_dot_M"]) * dt, 0.0, 1.0)
        self.state.eta_R = _clip(self.state.eta_R + float(output["eta_dot_R"]) * dt, 0.0, 1.0)
        self.state.last_eta_dot_M = float(output["eta_dot_M"])
        self.state.last_eta_dot_R = float(output["eta_dot_R"])
        self.state.plan_id = int(output.get("plan_id", -1))


def _post_commands(snapshot: Mapping[str, VehicleState], task: TaskState,
                   scenario_name: str, init_variant: int, cfg: Stage2AConfig) -> Dict[str, object]:
    sc = build_scenario(scenario_name, init_variant)
    M, R = snapshot["M"], snapshot["R"]
    leader_M = current_leader(snapshot, "M")
    leader_R = current_r_leader(snapshot, task, cfg)
    a_M = idm_accel(M, leader_M, cfg) if leader_M is not None else 0.85 * (sc.desired_speed - M.v)
    return {
        "a_nom_M": a_M, "a_nom_R": idm_accel(R, leader_R, cfg),
        "s_ref": "", "s_ref_dot": "", "s_ref_ddot": "",
        "reference_kind": "post_terminal_following", "interval_width": geometry(snapshot, sc, cfg)["interval_width"],
        "eta_M": "", "eta_R": "", "eta_dot_M": 0.0, "eta_dot_R": 0.0,
        "eta_dot_space": 0.0, "eta_dot_partner_M": 0.0, "eta_dot_partner_R": 0.0,
        "K_MR": 0.0, "K_RM": 0.0, "W_MR": 0.0, "W_RM": 0.0,
        "gap_error": 0.0, "plan_id": -1, "plan_start_t_s": "", "plan_reason": "",
        "plan_compute_time_s": 0.0, "target_time": "", "target_endpoint": "",
        "window_opening_accel": 0.0, "post_front_id": leader_R.vid,
        "target_gap": sc.initial_gap, "target_gap_process": sc.initial_gap,
        "grad_phi_space": 0.0, "e_M0": "",
    }


def _planner_commands(snapshot: Mapping[str, VehicleState], task: TaskState,
                      scenario_name: str, init_variant: int, cfg: Stage2AConfig, plan: Optional[Stage2APlan],
                      t_s: float) -> Dict[str, object]:
    sc = build_scenario(scenario_name, init_variant)
    geom = geometry(snapshot, sc, cfg)
    c_dot = 0.5 * (snapshot["R"].v + snapshot["F"].v
                   + cfg.time_headway * (snapshot["R"].a - snapshot["M"].a))
    if plan is None:
        cmd = {"M": 0.0, "R": 0.0}
        plan_id, start_t, reason, compute = -1, float("nan"), "no_plan", 0.0
    else:
        cmd = plan.command(t_s)
        plan_id, start_t, reason, compute = plan.plan_id, plan.predicted_start_t_s, plan.reason, plan.compute_time_s
    return {
        "a_nom_M": cmd["M"], "a_nom_R": cmd["R"],
        "s_ref": geom["c"], "s_ref_dot": c_dot, "s_ref_ddot": 0.0,
        "reference_kind": "predictive_joint_plan", "interval_width": geom["interval_width"],
        "eta_M": "", "eta_R": "", "eta_dot_M": 0.0, "eta_dot_R": 0.0,
        "eta_dot_space": 0.0, "eta_dot_partner_M": 0.0, "eta_dot_partner_R": 0.0,
        "K_MR": 0.0, "K_RM": 0.0, "W_MR": 0.0, "W_RM": 0.0,
        "gap_error": geom["g_fr"] - sc.initial_gap, "plan_id": plan_id,
        "plan_start_t_s": start_t, "plan_reason": reason,
        "plan_compute_time_s": compute, "target_time": "", "target_endpoint": "",
        "window_opening_accel": 0.0, "post_front_id": "F",
        "target_gap": sc.initial_gap, "target_gap_process": sc.initial_gap,
        "grad_phi_space": 0.0, "e_M0": "",
    }


def _set_task_after_step(task: TaskState, method: str, t_s: float,
                         snapshot: Mapping[str, VehicleState], plan: Optional[Stage2APlan],
                         scenario_name: str, init_variant: int, cfg: Stage2AConfig,
                         controller: object) -> Optional[str]:
    """Apply the common task gate after one physical state transition."""
    sc = build_scenario(scenario_name, init_variant)
    M = snapshot["M"]
    if task.phase == "PREPARE":
        if M.s > sc.completion_s + 1.0e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "missed_window", True, "missed_window"
            return task.event
        if method == "oldC":
            return _task_event(task, t_s, snapshot, controller, sc, cfg)
        if plan is not None and plan.valid and t_s >= plan.predicted_start_t_s - cfg.dt * 0.51:
            check = physical_start_feasibility(snapshot, task, sc, cfg)
            if check["physical_ok"]:
                task.phase, task.merge_start_time, task.event = "EXECUTE", t_s, "merge_started"
                return task.event
    elif task.phase == "EXECUTE":
        if M.s > sc.completion_s + 1.0e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_exit_before_completion"
            return task.event
        if M.merge_progress >= 1.0 - 1.0e-9:
            geom = geometry(snapshot, sc, cfg)
            if geom["interval_width"] >= -1.0e-7 and geom["lower"] - 1.0e-7 <= M.s <= geom["upper"] + 1.0e-7:
                task.phase, task.completion_time, task.terminal, task.success, task.event = "SUCCESS_RELEASE", t_s, True, True, "merge_completed"
            else:
                task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_completed_outside_slot"
            return task.event
    return None


def run_stage2a_episode(method: str, scenario_name: str, disturbance: str,
                        init_variant: int, cfg: Optional[Stage2AConfig] = None) -> SimulationResult:
    """Run one Stage 2A closed-loop episode with common execution."""
    if method not in STAGE2_METHODS:
        raise ValueError(method)
    cfg = cfg or Stage2AConfig()
    scenario = build_scenario(scenario_name, init_variant)
    vehicles = make_vehicles(scenario, cfg)
    planner = JointPredictivePlanner(cfg) if method != "oldC" else None
    coordinator = SmoothCoordinator(method, scenario_name, cfg, init_variant) if method in ("S1", "S2", "S3") else None
    legacy = make_controller("C", scenario, cfg) if method == "oldC" else None
    task = TaskState()
    plan: Optional[Stage2APlan] = None
    plan_replans = 0
    logs: List[Dict[str, object]] = []
    events: List[Dict[str, object]] = []
    run_id = f"{method}_{scenario_name}_{disturbance}_i{init_variant}"
    last_outputs: Dict[str, object] = {}
    last_actual = {vid: 0.0 for vid in VEHICLES}
    last_jerk = {vid: 0.0 for vid in VEHICLES}
    first_plan_time: Optional[float] = None
    first_plan_success_time: Optional[float] = None
    start_clock = time.perf_counter()

    def write_row(t_s: float, snapshot: Mapping[str, VehicleState], phase: str,
                  commands: Mapping[str, object], nominal: Mapping[str, float],
                  actual: Mapping[str, float], jerks: Mapping[str, float],
                  reasons: Mapping[str, Sequence[str]], safe_diag: Mapping[str, object]) -> None:
        plan_data = plan
        for vid in VEHICLES:
            if vid in ("M", "R"):
                d = commands
                q = d.get("eta_M" if vid == "M" else "eta_R", "")
            else:
                d = commands
                q = ""
            logs.append({
                "run_id": run_id, "t_s": round(t_s, 6), "method": method, "ablation": "",
                "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant,
                "vehicle": vid, "s_m": snapshot[vid].s, "y_m": snapshot[vid].y,
                "v_mps": snapshot[vid].v, "a_mps2": snapshot[vid].a,
                "q": q, "coord_eta_M": commands.get("eta_M", ""), "coord_eta_R": commands.get("eta_R", ""),
                "coord_eta_dot_M": commands.get("eta_dot_M", 0.0), "coord_eta_dot_R": commands.get("eta_dot_R", 0.0),
                "coord_eta_dot_space": commands.get("eta_dot_space", 0.0),
                "coord_eta_dot_partner_M": commands.get("eta_dot_partner_M", 0.0),
                "coord_eta_dot_partner_R": commands.get("eta_dot_partner_R", 0.0),
                "K_to_partner_s-1": commands.get("K_MR" if vid == "M" else "K_RM", 0.0),
                "W_to_partner": commands.get("W_MR" if vid == "M" else "W_RM", 0.0),
                "g_FR_m": geometry(snapshot, scenario, cfg)["g_fr"],
                "target_gap_physical_m": commands.get("target_gap", scenario.initial_gap),
                "target_gap_process_m": commands.get("target_gap_process", scenario.initial_gap),
                "gap_error_m": commands.get("gap_error", 0.0),
                "lower_m": geometry(snapshot, scenario, cfg)["lower"], "upper_m": geometry(snapshot, scenario, cfg)["upper"],
                "interval_width_m": geometry(snapshot, scenario, cfg)["interval_width"],
                "s_ref_m": commands.get("s_ref", "") if vid == "M" else "",
                "s_ref_dot_mps": commands.get("s_ref_dot", "") if vid == "M" else "",
                "s_ref_ddot_mps2": commands.get("s_ref_ddot", "") if vid == "M" else "",
                "reference_kind": commands.get("reference_kind", ""),
                "plan_id": commands.get("plan_id", -1), "plan_start_t_s": commands.get("plan_start_t_s", ""),
                "plan_reason": commands.get("plan_reason", ""), "plan_compute_time_s": commands.get("plan_compute_time_s", 0.0),
                "predicted_start_t_s": plan_data.predicted_start_t_s if plan_data else "",
                "predicted_completion_t_s": plan_data.predicted_completion_t_s if plan_data else "",
                "predicted_completion_s_m": plan_data.predicted_completion_s if plan_data else "",
                "predicted_min_h_m": plan_data.predicted_min_h_m if plan_data else "",
                "predicted_net_clearance_m": plan_data.predicted_net_clearance_m if plan_data else "",
                "predicted_candidate_count": plan_data.candidate_count if plan_data else 0,
                "merge_progress": snapshot[vid].merge_progress,
                "task_phase": phase, "a_nom_mps2": nominal[vid],
                "a_safety_target_mps2": actual[vid], "a_actual_mps2": actual[vid],
                "jerk_mps3": jerks[vid], "nominal_actual_difference_mps2": actual[vid] - nominal[vid],
                "safety_correction_mps2": actual[vid] - nominal[vid],
                "safety_correction_applied": int(abs(actual[vid] - nominal[vid]) > 1.0e-9),
                "action_saturation_applied": int(bool(reasons.get(vid))),
                "action_correction_reason": ";".join(reasons.get(vid, [])),
                "one_step_feasible": safe_diag.get("one_step_feasible", 0),
                "next_h_min_m": safe_diag.get("next_h_min", float("nan")),
                "F_disturbance_mps2": environment_accel(snapshot["F"], t_s, scenario, disturbance, cfg) - environment_accel(snapshot["F"], t_s, scenario, "none", cfg) if vid == "F" else 0.0,
            })

    n_steps = int(round(cfg.horizon / cfg.dt))
    for step in range(n_steps):
        t_s = step * cfg.dt
        snapshot = {vid: vehicles[vid].copy() for vid in VEHICLES}
        phase_before = task.phase

        # A plan is created once from the current state and retained through
        # its absolute start time. Replanning is allowed only after the plan's
        # predicted start has passed without a physically valid start.
        if method != "oldC" and task.phase == "PREPARE":
            should_plan = plan is None
            if plan is not None and (not plan.valid or t_s > plan.predicted_start_t_s + cfg.planner_refresh_s):
                should_plan = True
            if should_plan:
                assert planner is not None
                plan = planner.plan(snapshot, scenario_name, t_s, init_variant)
                plan_replans += 1
                first_plan_time = t_s if first_plan_time is None else first_plan_time
                events.append({"run_id": run_id, "t_s": t_s, "event": "plan_generated",
                               "details": plan.reason, "plan_id": plan.plan_id,
                               "candidate_count": plan.candidate_count,
                               "plan_valid": int(plan.valid), "wait_s": plan.wait_s,
                               "a_M_mps2": plan.a_M_mps2, "a_R_mps2": plan.a_R_mps2,
                               "predicted_start_t_s": plan.predicted_start_t_s,
                               "predicted_completion_t_s": plan.predicted_completion_t_s,
                               "predicted_min_h_m": plan.predicted_min_h_m,
                               "compute_time_s": plan.compute_time_s})
                if plan.valid and first_plan_success_time is None:
                    first_plan_success_time = t_s

        if task.phase in POST_PHASES:
            commands = _post_commands(snapshot, task, scenario_name, init_variant, cfg)
        elif method == "oldC":
            assert legacy is not None
            out = legacy.compute_pair(snapshot, task, cfg.dt)
            commands = {
                "a_nom_M": float(out["M"]["a_nom"]), "a_nom_R": float(out["R"]["a_nom"]),
                "eta_M": out["M"].get("q", ""), "eta_R": out["R"].get("q", ""),
                "eta_dot_M": out["M"].get("qdot_clipped", 0.0), "eta_dot_R": out["R"].get("qdot_clipped", 0.0),
                "eta_dot_space": out["M"].get("qdot_space", 0.0),
                "eta_dot_partner_M": out["M"].get("qdot_partner", 0.0), "eta_dot_partner_R": out["R"].get("qdot_partner", 0.0),
                "K_MR": out["M"].get("K_to_partner", 0.0), "K_RM": out["R"].get("K_to_partner", 0.0),
                "W_MR": out["M"].get("W_to_partner", 0.0), "W_RM": out["R"].get("W_to_partner", 0.0),
                "s_ref": out["M"].get("s_ref", ""), "s_ref_dot": out["M"].get("s_ref_dot", ""),
                "s_ref_ddot": out["M"].get("s_ref_ddot", ""), "reference_kind": out["M"].get("reference_kind", "oldC"),
                "interval_width": out["M"].get("interval_width", geometry(snapshot, scenario, cfg)["interval_width"]),
                "gap_error": out["M"].get("gap_error", 0.0), "plan_id": -1, "plan_start_t_s": "",
                "plan_reason": "oldC_no_predictive_plan", "plan_compute_time_s": 0.0,
                "target_time": out["M"].get("target_time", ""), "target_endpoint": out["M"].get("target_endpoint", ""),
                "window_opening_accel": out["M"].get("window_opening_accel", 0.0),
                "post_front_id": out["R"].get("post_front_id", "F"),
                "target_gap": out["R"].get("target_gap", scenario.initial_gap),
                "target_gap_process": out["R"].get("target_gap_process", scenario.initial_gap),
                "grad_phi_space": out["R"].get("grad_phi_space", 0.0), "e_M0": out["M"].get("e_M0", ""),
            }
        elif coordinator is not None:
            base = _planner_commands(snapshot, task, scenario_name, init_variant, cfg, plan, t_s)
            commands = coordinator.compute(snapshot, task, {"M": base["a_nom_M"], "R": base["a_nom_R"]}, plan, t_s)
        else:
            commands = _planner_commands(snapshot, task, scenario_name, init_variant, cfg, plan, t_s)

        # F and B are common environment/follower controls; M/R are generated
        # by the selected method and all four pass the same executor.
        a_nom = {
            "M": float(commands["a_nom_M"]), "R": float(commands["a_nom_R"]),
            "F": environment_accel(snapshot["F"], t_s, scenario, disturbance, cfg),
            "B": idm_accel(snapshot["B"], snapshot["R"], cfg),
        }
        active_merge = task.phase == "EXECUTE" or snapshot["M"].y <= 0.5 * (cfg.lane_ramp + cfg.lane_main)
        actual, safe_diag = execute_safe_actions(snapshot, a_nom, active_merge, cfg)
        jerks: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {vid: [] for vid in VEHICLES}
        for vid in VEHICLES:
            delta = actual[vid] - a_nom[vid]
            if abs(delta) > 1.0e-9:
                reasons[vid].append("common_safety_projection")
            previous = snapshot[vid].a
            jerks[vid] = (actual[vid] - previous) / cfg.dt
            if abs(jerks[vid]) > cfg.jerk_max + 1.0e-9:
                reasons[vid].append("jerk_limit_violation")

        if not int(safe_diag["one_step_feasible"]):
            events.append({"run_id": run_id, "t_s": t_s, "event": "action_interval_infeasible",
                           "details": str(safe_diag)})
        if float(safe_diag.get("next_h_min", 0.0)) < -cfg.safety_recheck_tolerance_m:
            events.append({"run_id": run_id, "t_s": t_s, "event": "safety_recheck",
                           "details": str(safe_diag.get("next_h", {}))})
        write_row(t_s, snapshot, phase_before, commands, a_nom, actual, jerks, reasons, safe_diag)
        last_outputs = commands
        last_actual = dict(actual)
        last_jerk = dict(jerks)

        # Execute one real physical step.
        for vid in VEHICLES:
            v = vehicles[vid]
            v.s = v.s + v.v * cfg.dt + 0.5 * actual[vid] * cfg.dt * cfg.dt
            v.v = _clip(v.v + actual[vid] * cfg.dt, cfg.v_min, cfg.v_max)
            v.a = actual[vid]
        if phase_before == "EXECUTE" or (task.phase == "FAILURE_HANDLING" and vehicles["M"].merge_progress > 0.0):
            vehicles["M"].merge_progress = _clip(vehicles["M"].merge_progress + cfg.dt / cfg.merge_duration, 0.0, 1.0)
        vehicles["M"].y = cfg.lane_ramp + (cfg.lane_main - cfg.lane_ramp) * vehicles["M"].merge_progress
        if coordinator is not None:
            coordinator.commit(commands, cfg.dt, task.terminal)
        if legacy is not None:
            # Match the old controller's internal q update under the shared
            # executor; task terminal state is handled by the common layer.
            legacy.commit({"M": {"qdot_clipped": commands.get("eta_dot_M", 0.0)},
                           "R": {"qdot_clipped": commands.get("eta_dot_R", 0.0)}}, cfg.dt, task.terminal)

        event = _set_task_after_step(task, method, t_s + cfg.dt, vehicles, plan, scenario_name, init_variant, cfg, legacy)
        if event:
            events.append({"run_id": run_id, "t_s": t_s + cfg.dt, "event": event,
                           "details": task.failure_type or "", "plan_id": plan.plan_id if plan else -1})
            if event == "merge_started" and first_plan_success_time is None:
                first_plan_success_time = t_s + cfg.dt

    if not task.terminal:
        task.phase, task.failure_type, task.terminal = "TIMEOUT", "timeout", True
        events.append({"run_id": run_id, "t_s": cfg.horizon, "event": "timeout", "details": "observation_window_end"})

    # Append one final state row per vehicle.  It carries the last real action
    # and lets the independent checker evaluate the complete window.
    for vid in VEHICLES:
        logs.append({
            "run_id": run_id, "t_s": round(cfg.horizon, 6), "method": method, "ablation": "",
            "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant,
            "vehicle": vid, "s_m": vehicles[vid].s, "y_m": vehicles[vid].y, "v_mps": vehicles[vid].v,
            "a_mps2": vehicles[vid].a, "q": commands.get("eta_M" if vid == "M" else "eta_R", "") if last_outputs else "",
            "coord_eta_M": last_outputs.get("eta_M", ""), "coord_eta_R": last_outputs.get("eta_R", ""),
            "coord_eta_dot_M": last_outputs.get("eta_dot_M", 0.0), "coord_eta_dot_R": last_outputs.get("eta_dot_R", 0.0),
            "coord_eta_dot_space": last_outputs.get("eta_dot_space", 0.0),
            "coord_eta_dot_partner_M": last_outputs.get("eta_dot_partner_M", 0.0),
            "coord_eta_dot_partner_R": last_outputs.get("eta_dot_partner_R", 0.0),
            "K_to_partner_s-1": last_outputs.get("K_MR" if vid == "M" else "K_RM", 0.0),
            "W_to_partner": last_outputs.get("W_MR" if vid == "M" else "W_RM", 0.0),
            "g_FR_m": _net_gap(vehicles["F"], vehicles["R"]),
            "target_gap_physical_m": last_outputs.get("target_gap", scenario.initial_gap),
            "target_gap_process_m": last_outputs.get("target_gap_process", scenario.initial_gap),
            "gap_error_m": last_outputs.get("gap_error", 0.0),
            "lower_m": geometry(vehicles, scenario, cfg)["lower"], "upper_m": geometry(vehicles, scenario, cfg)["upper"],
            "interval_width_m": geometry(vehicles, scenario, cfg)["interval_width"],
            "s_ref_m": last_outputs.get("s_ref", "") if vid == "M" else "",
            "s_ref_dot_mps": last_outputs.get("s_ref_dot", "") if vid == "M" else "",
            "s_ref_ddot_mps2": last_outputs.get("s_ref_ddot", "") if vid == "M" else "",
            "reference_kind": last_outputs.get("reference_kind", ""),
            "plan_id": last_outputs.get("plan_id", -1), "plan_start_t_s": last_outputs.get("plan_start_t_s", ""),
            "plan_reason": last_outputs.get("plan_reason", ""), "plan_compute_time_s": last_outputs.get("plan_compute_time_s", 0.0),
            "predicted_start_t_s": plan.predicted_start_t_s if plan else "",
            "predicted_completion_t_s": plan.predicted_completion_t_s if plan else "",
            "predicted_completion_s_m": plan.predicted_completion_s if plan else "",
            "predicted_min_h_m": plan.predicted_min_h_m if plan else "",
            "predicted_net_clearance_m": plan.predicted_net_clearance_m if plan else "",
            "predicted_candidate_count": plan.candidate_count if plan else 0,
            "merge_progress": vehicles[vid].merge_progress, "task_phase": task.phase,
            "a_nom_mps2": last_outputs.get("a_nom_M" if vid == "M" else "a_nom_R", last_actual[vid]) if vid in ("M", "R") else last_actual[vid],
            "a_safety_target_mps2": last_actual[vid], "a_actual_mps2": last_actual[vid],
            "jerk_mps3": last_jerk[vid], "nominal_actual_difference_mps2": last_actual[vid] - (last_outputs.get("a_nom_M" if vid == "M" else "a_nom_R", last_actual[vid]) if vid in ("M", "R") else last_actual[vid]),
            "safety_correction_mps2": 0.0, "safety_correction_applied": 0,
            "action_saturation_applied": 0, "action_correction_reason": "terminal_state",
            "one_step_feasible": 1, "next_h_min_m": "",
            "F_disturbance_mps2": 0.0,
        })

    outcome = "completed" if task.success else task.failure_type or task.phase.lower()
    compute_time = time.perf_counter() - start_clock
    metrics = {
        "run_id": run_id, "method": method, "scenario": scenario_name,
        "disturbance": disturbance, "init_variant": init_variant,
        "outcome": outcome, "success": int(task.success), "geometry_completed": int(any(e.get("event") == "merge_completed" for e in events)),
        "merge_start_time_s": "" if task.merge_start_time is None else task.merge_start_time,
        "completion_time_s": "" if task.completion_time is None else task.completion_time,
        "planner_first_time_s": "" if first_plan_time is None else first_plan_time,
        "planner_first_valid_time_s": "" if first_plan_success_time is None else first_plan_success_time,
        "planner_replans": plan_replans, "planner_candidate_count": plan.candidate_count if plan else 0,
        "planner_compute_time_s": plan.compute_time_s if plan else 0.0,
        "predicted_completion_s_m": plan.predicted_completion_s if plan else float("nan"),
        "predicted_min_h_m": plan.predicted_min_h_m if plan else float("nan"),
        "predicted_net_clearance_m": plan.predicted_net_clearance_m if plan else float("nan"),
        "total_speed_deficit_distance_m": sum(max(0.0, scenario.desired_speed - r["v_mps"]) * cfg.dt for r in logs if r.get("t_s") < cfg.horizon),
        "total_effort_integral_m2_s3": sum(float(r.get("a_actual_mps2", 0.0)) ** 2 * cfg.dt for r in logs if r.get("t_s") < cfg.horizon),
        "min_net_clearance_m": min((_net_gap(vehicles["F"], vehicles["R"]), _net_gap(vehicles["R"], vehicles["B"])), default=float("nan")),
        "max_abs_jerk_mps3": max((abs(float(r.get("jerk_mps3", 0.0))) for r in logs if r.get("t_s") < cfg.horizon), default=0.0),
        "safety_correction_count": sum(1 for r in logs if r.get("safety_correction_applied") and r.get("t_s") < cfg.horizon),
        "action_saturation_count": sum(1 for r in logs if r.get("action_saturation_applied") and r.get("t_s") < cfg.horizon),
        "compute_time_s": compute_time,
    }
    vehicle_metrics: List[Dict[str, object]] = []
    for vid in VEHICLES:
        rows = [r for r in logs if r.get("vehicle") == vid and r.get("t_s") < cfg.horizon]
        vehicle_metrics.append({
            "run_id": run_id, "method": method, "scenario": scenario_name,
            "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
            "speed_deficit_distance_m": sum(max(0.0, scenario.desired_speed - float(r["v_mps"])) * cfg.dt for r in rows),
            "action_saturation_count": sum(int(r.get("action_saturation_applied", 0)) for r in rows),
            "safety_correction_count": sum(int(r.get("safety_correction_applied", 0)) for r in rows),
            "max_abs_jerk_mps3": max((abs(float(r.get("jerk_mps3", 0.0))) for r in rows), default=0.0),
        })
    return SimulationResult(run_id, method, scenario_name, disturbance, init_variant, logs, events, metrics, vehicle_metrics)


def run_stage2a_matrix(methods: Iterable[str], scenarios: Iterable[str], disturbances: Iterable[str],
                       init_variants: Iterable[int], cfg: Optional[Stage2AConfig] = None) -> List[SimulationResult]:
    return [run_stage2a_episode(method, scenario, disturbance, init_variant, cfg)
            for method in methods for scenario in scenarios for disturbance in disturbances for init_variant in init_variants]


__all__ = [
    "Stage2AConfig", "Stage2APlan", "CoordinationState", "JointPredictivePlanner",
    "SmoothCoordinator", "run_stage2a_episode", "run_stage2a_matrix", "STAGE2_METHODS",
]
