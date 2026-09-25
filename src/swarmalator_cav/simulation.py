"""Small, deterministic four-vehicle merge simulation for Stage 1.

The implementation intentionally keeps the traffic model narrow: longitudinal
point-mass dynamics with finite body dimensions and a continuous lateral path
for M.  The purpose is to test the information flow in the proposed
space-process coupling, not to claim a complete vehicle or road model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


VEHICLE_IDS = ("M", "R", "F", "B")


@dataclass
class VehicleState:
    vid: str
    s: float
    y: float
    v: float
    a: float
    length: float = 4.8
    width: float = 1.9
    merge_progress: float = 0.0

    def copy(self) -> "VehicleState":
        return replace(self)


@dataclass
class Scenario:
    name: str
    F_s: float
    R_s: float
    B_s: float
    M_s: float
    F_v: float
    R_v: float
    B_v: float
    M_v: float
    delta_g: float
    desired_speed: float = 18.0
    merge_zone: Tuple[float, float] = (55.0, 95.0)
    terminal_s: float = 120.0
    road_max: float = 180.0

    @property
    def initial_gap(self) -> float:
        return self.F_s - self.R_s - 4.8


@dataclass
class SimConfig:
    dt: float = 0.05
    horizon: float = 16.0
    lane_main: float = 0.0
    lane_ramp: float = 3.6
    lane_width: float = 3.6
    min_base_gap: float = 2.0
    time_headway: float = 0.35
    merge_duration: float = 3.0
    merge_ready_threshold: float = 0.45
    a_min: float = -4.0
    a_max: float = 2.5
    jerk_max: float = 2.5
    v_min: float = 0.0
    v_max: float = 28.0
    ell_s: float = 24.0
    ell_g: float = 10.0
    nu_M: float = 0.42
    nu_R: float = 0.36
    nu_max: float = 0.8
    kappa_M: float = 0.55
    kappa_R: float = 0.50
    k_total: float = 0.34
    alpha: float = math.pi / 2.0
    phase_gain_smoothing: float = 0.2
    online_t_min: float = 1.5
    online_t_max: float = 5.5
    online_t_step: float = 0.5
    online_t_smoothing: float = 0.25
    observation_dt: float = 0.05


@dataclass
class TaskState:
    phase: str = "PREPARE"
    failure_type: str = ""
    merge_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    terminal: bool = False
    event: str = ""


@dataclass
class SimulationResult:
    run_id: str
    method: str
    scenario: str
    disturbance: str
    init_variant: int
    logs: List[Dict[str, object]]
    events: List[Dict[str, object]]
    metrics: Dict[str, object]
    vehicle_metrics: List[Dict[str, object]]


def build_scenario(name: str, init_variant: int = 0) -> Scenario:
    """Return one of three deliberately different physical merge settings.

    The variants modify initial positions and speeds, rather than only a random
    seed.  The short-window case is labelled as a constraint case; a failure is
    not automatically interpreted as proof of physical infeasibility.
    """

    base = {
        "ample": dict(F_s=100.0, R_s=65.0, B_s=38.0, M_s=74.0, F_v=18.5, R_v=18.0, B_v=18.0, M_v=22.0, delta_g=4.0),
        "collaborative": dict(F_s=94.0, R_s=68.0, B_s=42.0, M_s=76.0, F_v=18.0, R_v=18.0, B_v=18.0, M_v=21.5, delta_g=8.0),
        "short_window": dict(F_s=90.0, R_s=72.0, B_s=48.0, M_s=78.0, F_v=18.0, R_v=18.5, B_v=18.0, M_v=21.0, delta_g=7.0),
    }
    if name not in base:
        raise ValueError(f"unknown scenario: {name}")
    p = dict(base[name])
    if init_variant == 1:
        p["M_s"] -= 3.0
        p["M_v"] -= 1.7
        p["R_v"] += 0.2
    elif init_variant == 2:
        p["F_s"] -= 1.5
        p["R_s"] += 1.5
        p["B_s"] += 1.0
        p["M_s"] -= 1.0
        p["R_v"] -= 0.6
        p["B_v"] -= 0.3
    elif init_variant != 0:
        raise ValueError(f"unknown initial-state variant: {init_variant}")
    return Scenario(name=name, merge_zone=(55.0, 170.0), terminal_s=190.0, road_max=450.0, **p)


def make_vehicles(scenario: Scenario, cfg: SimConfig) -> Dict[str, VehicleState]:
    return {
        "M": VehicleState("M", scenario.M_s, cfg.lane_ramp, scenario.M_v, 0.0),
        "R": VehicleState("R", scenario.R_s, cfg.lane_main, scenario.R_v, 0.0),
        "F": VehicleState("F", scenario.F_s, cfg.lane_main, scenario.F_v, 0.0),
        "B": VehicleState("B", scenario.B_s, cfg.lane_main, scenario.B_v, 0.0),
    }


def dynamic_gap(follower: VehicleState, cfg: SimConfig) -> float:
    return cfg.min_base_gap + cfg.time_headway * max(follower.v, 0.0)


def geometry(snapshot: Mapping[str, VehicleState], scenario: Scenario, cfg: SimConfig) -> Dict[str, float]:
    M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
    g_fr = F.s - R.s - (F.length + R.length) / 2.0
    d_rm = dynamic_gap(R, cfg)
    d_mf = dynamic_gap(M, cfg)
    lower = R.s + (R.length + M.length) / 2.0 + d_rm
    upper = F.s - (F.length + M.length) / 2.0 - d_mf
    return {
        "g_fr": g_fr,
        "d_rm": d_rm,
        "d_mf": d_mf,
        "lower": lower,
        "upper": upper,
        "c": 0.5 * (lower + upper),
        "interval_width": upper - lower,
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _sigmoid(x: float) -> float:
    x = _clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-x))


class PairController:
    """Shared synchronized controller for M/R.

    ``compute_pair`` is the same-snapshot implementation of the requested
    ``step(local_observation, received_messages, task_state, dt)`` interface.
    The public ``step`` wrapper accepts a local observation containing the
    complete current snapshot and returns the requested role's output.  The
    simulation uses ``compute_pair`` so M and R cannot observe different
    instants.
    """

    def __init__(self, method: str, scenario: Scenario, cfg: SimConfig):
        self.method = method
        self.scenario = scenario
        self.cfg = cfg
        self.q: Dict[str, float] = {"M": 0.0, "R": 0.0}
        self.e_M0: Optional[float] = None
        self.c0: Optional[float] = None
        self.prev_k_mr = cfg.k_total / 2.0
        self.prev_target_time = 3.5

    def step(self, local_observation: Mapping[str, object], received_messages: Mapping[str, object], task_state: TaskState, dt: float) -> Dict[str, object]:
        snapshot = local_observation["snapshot"]
        role = str(local_observation["role"])
        return self.compute_pair(snapshot, task_state, dt)[role]

    def _target_gap(self, q_r: float, mode: Optional[str] = None) -> float:
        if mode in ("A", "E"):
            return self.scenario.initial_gap + 0.55 * self.scenario.delta_g
        return self.scenario.initial_gap + self.scenario.delta_g * q_r

    def _online_target(self, snapshot: Mapping[str, VehicleState], target_gap: float) -> Tuple[float, float, float]:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        best = None
        for T in np.arange(self.cfg.online_t_min, self.cfg.online_t_max + 0.01, self.cfg.online_t_step):
            fp = F.s + F.v * T
            rp = R.s + R.v * T
            dp_mf = self.cfg.min_base_gap + self.cfg.time_headway * max(M.v, 0.0)
            dp_rm = self.cfg.min_base_gap + self.cfg.time_headway * max(R.v, 0.0)
            lo = rp + (R.length + M.length) / 2.0 + dp_rm
            hi = fp - (F.length + M.length) / 2.0 - dp_mf
            c = 0.5 * (lo + hi)
            a_req = 2.0 * (c - M.s - M.v * T) / (T * T)
            gap_pred = fp - rp - (F.length + R.length) / 2.0
            cost = (a_req / self.cfg.a_max) ** 2 + 0.35 * ((gap_pred - target_gap) / self.cfg.ell_g) ** 2 + 0.025 * T
            if hi < lo:
                cost += 1.5 * (lo - hi) / self.cfg.ell_g
            candidate = (cost, T, c, a_req)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, target_T, target_c, a_req = best
        self.prev_target_time = (1.0 - self.cfg.online_t_smoothing) * self.prev_target_time + self.cfg.online_t_smoothing * target_T
        return self.prev_target_time, target_c, a_req

    def _capability_scores(self, snapshot: Mapping[str, VehicleState], geom: Mapping[str, float], a_m: float, a_r: float) -> Tuple[float, float]:
        M, R = snapshot["M"], snapshot["R"]
        accel_m = _clip((self.cfg.a_max - abs(a_m)) / self.cfg.a_max, -1.0, 1.0)
        accel_r = _clip((self.cfg.a_max - abs(a_r)) / self.cfg.a_max, -1.0, 1.0)
        dist_m = _clip((self.scenario.merge_zone[1] - M.s) / (self.scenario.merge_zone[1] - self.scenario.merge_zone[0]), -1.0, 1.0)
        gap_margin = _clip((geom["g_fr"] - self._target_gap(self.q["R"])) / self.cfg.ell_g, -1.0, 1.0)
        pressure = _clip((self.cfg.min_base_gap + self.cfg.time_headway * R.v - geom["g_fr"]) / self.cfg.ell_g, -1.0, 1.0)
        z_m = 0.60 * accel_m + 0.40 * dist_m
        z_r = 0.55 * accel_r + 0.30 * gap_margin - 0.35 * pressure
        return z_m, z_r

    def compute_pair(self, snapshot: Mapping[str, VehicleState], task: TaskState, dt: float) -> Dict[str, Dict[str, object]]:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        geom = geometry(snapshot, self.scenario, self.cfg)
        if self.e_M0 is None:
            self.c0 = geom["c"]
            self.e_M0 = M.s - self.c0
        assert self.e_M0 is not None and self.c0 is not None

        target_gap = self._target_gap(self.q["R"], self.method)
        if self.method == "E":
            target_time, c_online, a_req = self._online_target(snapshot, target_gap)
            s_ref = c_online
            v_ref_m = _clip((s_ref - M.s) / max(target_time, 0.5), 17.0, 23.0)
            a_nom_m = 0.65 * a_req + 0.55 * (v_ref_m - M.v)
        else:
            if self.method == "A":
                s_ref = geom["c"]
            else:
                s_ref = geom["c"] + (1.0 - self.q["M"]) * self.e_M0
            v_ref_m = _clip((s_ref - M.s) / 2.4, 17.0, 23.0)
            a_nom_m = 0.25 * (s_ref - M.s) + 0.60 * (v_ref_m - M.v)
            target_time = float("nan")

        gap_error = geom["g_fr"] - target_gap
        v_ref_r = _clip(F.v + 0.36 * gap_error, 9.0, 23.0)
        a_nom_r = 1.0 * (v_ref_r - R.v) + 0.08 * gap_error

        grad_m = (M.s - s_ref) * self.e_M0 / (self.cfg.ell_s * self.cfg.ell_s)
        grad_r = -gap_error * self.scenario.delta_g / (self.cfg.ell_g * self.cfg.ell_g)
        if self.method in ("A", "E"):
            qdot_m = 0.0
            qdot_r = 0.0
            space_m = 0.0
            space_r = 0.0
            partner_m = 0.0
            partner_r = 0.0
            k_mr = k_rm = 0.0
            w_mr = w_rm = 0.0
            z_m = z_r = 0.0
        else:
            space_m = -self.cfg.kappa_M * grad_m if self.method in ("C", "D") else 0.0
            space_r = -self.cfg.kappa_R * grad_r if self.method in ("C", "D") else 0.0
            if self.method == "B":
                k_mr = k_rm = 0.0
                w_mr = w_rm = 0.0
                partner_m = partner_r = 0.0
                z_m = z_r = 0.0
            elif self.method == "C":
                k_mr = k_rm = self.cfg.k_total / 2.0
                w_mr = w_rm = _clip(0.15 + 0.85 * abs(gap_error) / self.cfg.ell_g, 0.0, 1.0)
                partner_m = k_mr * w_mr * math.sin(self.cfg.alpha * (self.q["R"] - self.q["M"]))
                partner_r = k_rm * w_rm * math.sin(self.cfg.alpha * (self.q["M"] - self.q["R"]))
                z_m = z_r = 0.0
            else:
                z_m, z_r = self._capability_scores(snapshot, geom, a_nom_m, a_nom_r)
                raw_k_mr = self.cfg.k_total * _sigmoid(z_m - z_r)
                k_mr = (1.0 - self.cfg.phase_gain_smoothing) * self.prev_k_mr + self.cfg.phase_gain_smoothing * raw_k_mr
                k_rm = self.cfg.k_total - k_mr
                self.prev_k_mr = k_mr
                w_mr = w_rm = _clip(0.15 + 0.85 * abs(gap_error) / self.cfg.ell_g, 0.0, 1.0)
                partner_m = k_mr * w_mr * math.sin(self.cfg.alpha * (self.q["R"] - self.q["M"]))
                partner_r = k_rm * w_rm * math.sin(self.cfg.alpha * (self.q["M"] - self.q["R"]))

            qdot_m = self.cfg.nu_M + space_m + partner_m
            qdot_r = self.cfg.nu_R + space_r + partner_r

        if task.terminal:
            qdot_m = qdot_r = 0.0
        qdot_m_clipped = _clip(qdot_m, 0.0, self.cfg.nu_max) if self.q["M"] < 1.0 else 0.0
        qdot_r_clipped = _clip(qdot_r, 0.0, self.cfg.nu_max) if self.q["R"] < 1.0 else 0.0

        def out(role: str, a_nom: float, qdot: float, qdot_clipped: float, space: float, partner: float, k_partner: float, w_partner: float, z_self: float, z_other: float) -> Dict[str, object]:
            return {
                "a_nom": float(a_nom),
                "q": float(self.q[role]),
                "qdot_nominal": float(qdot),
                "qdot_base": float(self.cfg.nu_M if role == "M" else self.cfg.nu_R) if self.method not in ("A", "E") else 0.0,
                "qdot_space": float(space),
                "qdot_partner": float(partner),
                "qdot_clipped": float(qdot_clipped),
                "K_to_partner": float(k_partner),
                "W_to_partner": float(w_partner),
                "z_self": float(z_self),
                "z_other": float(z_other),
                "target_gap": float(target_gap),
                "gap_error": float(gap_error),
                "s_ref": float(s_ref),
                "target_time": float(target_time) if math.isfinite(target_time) else "",
                "grad_phi_space": float(grad_m if role == "M" else grad_r),
                "e_M0": float(self.e_M0),
                "interval_width": float(geom["interval_width"]),
            }

        return {
            "M": out("M", a_nom_m, qdot_m, qdot_m_clipped, space_m, partner_m, k_mr, w_mr, z_m, z_r),
            "R": out("R", a_nom_r, qdot_r, qdot_r_clipped, space_r, partner_r, k_rm, w_rm, z_r, z_m),
        }

    def commit(self, outputs: Mapping[str, Mapping[str, object]], dt: float, terminal: bool) -> None:
        if self.method in ("A", "E") or terminal:
            return
        for role in ("M", "R"):
            self.q[role] = _clip(self.q[role] + float(outputs[role]["qdot_clipped"]) * dt, 0.0, 1.0)


def make_controller(method: str, scenario: Scenario, cfg: SimConfig) -> PairController:
    if method not in ("A", "B", "C", "D", "E"):
        raise ValueError(f"unknown method {method}")
    return PairController(method, scenario, cfg)


def disturbance_accel(t: float, disturbance: str) -> float:
    if disturbance == "none":
        return 0.0
    if disturbance == "prepare":
        if 1.25 <= t < 2.35:
            return -1.25
        if 2.35 <= t < 3.20:
            return 0.75
        return 0.0
    raise ValueError(f"unknown disturbance {disturbance}")


def environment_accel(vehicle: VehicleState, t: float, scenario: Scenario, disturbance: str, cfg: SimConfig) -> float:
    return _clip(0.85 * (scenario.desired_speed - vehicle.v) + disturbance_accel(t, disturbance), cfg.a_min, cfg.a_max)


def idm_accel(follower: VehicleState, leader: VehicleState, cfg: SimConfig) -> float:
    v0 = 20.0
    a0 = 1.3
    b0 = 2.2
    gap = leader.s - follower.s - (leader.length + follower.length) / 2.0
    gap = max(gap, 0.1)
    dv = follower.v - leader.v
    s_star = cfg.min_base_gap + cfg.time_headway * follower.v + follower.v * dv / (2.0 * math.sqrt(a0 * b0))
    return _clip(a0 * (1.0 - (follower.v / v0) ** 4 - (max(s_star, 0.1) / gap) ** 2), cfg.a_min, cfg.a_max)


def _lateral_overlap(v1: VehicleState, v2: VehicleState) -> bool:
    return abs(v1.y - v2.y) <= (v1.width + v2.width) / 2.0


def _net_gap(leader: VehicleState, follower: VehicleState) -> float:
    return leader.s - follower.s - (leader.length + follower.length) / 2.0


def safety_adjustments(snapshot: Mapping[str, VehicleState], desired: Mapping[str, float], task: TaskState, scenario: Scenario, cfg: SimConfig) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    adjusted = {k: _clip(float(v), cfg.a_min, cfg.a_max) for k, v in desired.items()}
    reasons: Dict[str, List[str]] = {k: [] for k in VEHICLE_IDS}

    def predicted(vehicle: VehicleState, accel: float) -> float:
        return vehicle.s + vehicle.v * cfg.dt + 0.5 * accel * cfg.dt * cfg.dt

    # Mainline car-following safety is shared by all methods.
    for leader_id, follower_id in (("F", "R"), ("R", "B")):
        leader, follower = snapshot[leader_id], snapshot[follower_id]
        gap_pred = predicted(leader, adjusted[leader_id]) - predicted(follower, adjusted[follower_id]) - (leader.length + follower.length) / 2.0
        required = dynamic_gap(follower, cfg)
        if gap_pred < required:
            correction = 2.0 * (required - gap_pred) / (cfg.dt * cfg.dt)
            adjusted[follower_id] -= correction
            reasons[follower_id].append("mainline_gap")

    # During lateral execution, M must remain behind F and ahead of R.
    if task.phase == "EXECUTE" or snapshot["M"].merge_progress > 0.0:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        m_pred = predicted(M, adjusted["M"])
        r_pred = predicted(R, adjusted["R"])
        f_pred = predicted(F, adjusted["F"])
        m_f_gap = f_pred - m_pred - (F.length + M.length) / 2.0
        if m_f_gap < dynamic_gap(M, cfg):
            adjusted["M"] -= 2.0 * (dynamic_gap(M, cfg) - m_f_gap) / (cfg.dt * cfg.dt)
            reasons["M"].append("merge_front_gap")
        # Preserve whichever longitudinal order actually exists.  Before a
        # merge M is normally behind R; after a successful merge M is the
        # leader and R must not overtake it.  The earlier candidate expression
        # treated M as leader in both cases and could create a post-success
        # R/M overlap, so the follower is selected from the snapshot order.
        if m_pred >= r_pred:
            m_r_gap = m_pred - r_pred - (R.length + M.length) / 2.0
            if m_r_gap < dynamic_gap(R, cfg):
                adjusted["R"] -= 2.0 * (dynamic_gap(R, cfg) - m_r_gap) / (cfg.dt * cfg.dt)
                reasons["R"].append("merge_rear_gap")
        else:
            r_m_gap = r_pred - m_pred - (R.length + M.length) / 2.0
            if r_m_gap < dynamic_gap(M, cfg):
                adjusted["M"] -= 2.0 * (dynamic_gap(M, cfg) - r_m_gap) / (cfg.dt * cfg.dt)
                reasons["M"].append("merge_rear_gap")

    for k in VEHICLE_IDS:
        adjusted[k] = _clip(adjusted[k], cfg.a_min, cfg.a_max)
    return adjusted, reasons


def execute_acceleration(previous_a: float, target_a: float, cfg: SimConfig) -> Tuple[float, float, List[str]]:
    target_a = _clip(target_a, cfg.a_min, cfg.a_max)
    delta = target_a - previous_a
    max_delta = cfg.jerk_max * cfg.dt
    limited_delta = _clip(delta, -max_delta, max_delta)
    actual = _clip(previous_a + limited_delta, cfg.a_min, cfg.a_max)
    reasons: List[str] = []
    if abs(limited_delta - delta) > 1e-9:
        reasons.append("jerk_limit")
    if abs(actual - (previous_a + limited_delta)) > 1e-9:
        reasons.append("accel_limit")
    return actual, (actual - previous_a) / cfg.dt, reasons


def hard_violations(vehicles: Mapping[str, VehicleState], task: TaskState, scenario: Scenario, cfg: SimConfig) -> List[str]:
    violations: List[str] = []
    if _net_gap(vehicles["F"], vehicles["R"]) < 0.0:
        violations.append("FR_body_overlap")
    if _net_gap(vehicles["R"], vehicles["B"]) < 0.0:
        violations.append("RB_body_overlap")
    M = vehicles["M"]
    if M.merge_progress > 0.0 or M.y <= (cfg.lane_ramp + cfg.lane_main) / 2.0:
        for other_id in ("F", "R", "B"):
            other = vehicles[other_id]
            if _lateral_overlap(M, other) and abs(M.s - other.s) < (M.length + other.length) / 2.0:
                violations.append(f"M{other_id}_body_overlap")
    for vid, v in vehicles.items():
        if v.s < -1.0 or v.s > scenario.road_max:
            violations.append(f"{vid}_road_exit")
        if v.y < -0.5 or v.y > cfg.lane_ramp + 0.5:
            violations.append(f"{vid}_lateral_exit")
    return violations


def _task_event(task: TaskState, t: float, vehicles: Mapping[str, VehicleState], controller: PairController, scenario: Scenario, cfg: SimConfig) -> Optional[str]:
    """Advance task phase after a synchronized state update."""
    M, R, F = vehicles["M"], vehicles["R"], vehicles["F"]
    geom = geometry(vehicles, scenario, cfg)
    if task.terminal:
        return None
    if task.phase == "PREPARE":
        if M.s > scenario.merge_zone[1] + 3.0:
            task.phase = "FAILED"
            task.failure_type = "missed_window"
            task.terminal = True
            task.event = "missed_window"
            return task.event
        q_ready = controller.method in ("A", "E") or (
            controller.q["M"] >= cfg.merge_ready_threshold
            and controller.q["R"] >= cfg.merge_ready_threshold
        )
        if (
            scenario.merge_zone[0] <= M.s <= scenario.merge_zone[1]
            and q_ready
            and geom["interval_width"] >= 0.0
            and geom["lower"] - 1.5 <= M.s <= geom["upper"]
        ):
            task.phase = "EXECUTE"
            task.merge_start_time = t
            task.event = "merge_started"
            return task.event
    elif task.phase == "EXECUTE":
        if M.s > scenario.merge_zone[1] + 3.0 and M.merge_progress < 1.0:
            task.phase = "FAILED"
            task.failure_type = "started_but_failed"
            task.terminal = True
            task.event = "merge_exit_before_completion"
            return task.event
        if M.merge_progress >= 1.0:
            safe = geom["lower"] <= M.s <= geom["upper"] and geom["interval_width"] >= 0.0
            if safe:
                task.phase = "SUCCESS"
                task.failure_type = ""
                task.completion_time = t
                task.terminal = True
                task.event = "merge_completed"
            else:
                task.phase = "FAILED"
                task.failure_type = "started_but_failed"
                task.terminal = True
                task.event = "merge_completed_outside_slot"
            return task.event
    return None


def _vehicle_metric_accumulator() -> Dict[str, Dict[str, float]]:
    return {
        vid: {
            "speed_deficit_integral_m_s": 0.0,
            "accel_squared_integral": 0.0,
            "abs_jerk_integral": 0.0,
            "abs_nominal_actual_difference_integral": 0.0,
            "safety_correction_count": 0.0,
            "safety_correction_abs_integral": 0.0,
            "distance_m": 0.0,
            "min_net_gap_m": float("inf"),
        }
        for vid in VEHICLE_IDS
    }


def run_episode(method: str, scenario_name: str, disturbance: str, init_variant: int, cfg: Optional[SimConfig] = None, seed: int = 0) -> SimulationResult:
    del seed  # Explicitly deterministic in Stage 1; the state variants carry the perturbations.
    cfg = cfg or SimConfig()
    scenario = build_scenario(scenario_name, init_variant)
    vehicles = make_vehicles(scenario, cfg)
    controller = make_controller(method, scenario, cfg)
    task = TaskState()
    run_id = f"{method}_{scenario_name}_{disturbance}_i{init_variant}"
    logs: List[Dict[str, object]] = []
    events: List[Dict[str, object]] = []
    accum = _vehicle_metric_accumulator()
    min_gap = float("inf")
    hard_violation_count = 0
    road_violation_count = 0
    safety_abs_values: List[float] = []
    all_jerks: List[float] = []
    k_total_errors: List[float] = []
    feedback_abs: List[float] = []
    start_clock = time.perf_counter()
    n_steps = int(round(cfg.horizon / cfg.dt))

    for step in range(n_steps):
        t = step * cfg.dt
        snapshot = {vid: vehicles[vid].copy() for vid in VEHICLE_IDS}
        outputs = controller.compute_pair(snapshot, task, cfg.dt)
        a_nom: Dict[str, float] = {
            "M": float(outputs["M"]["a_nom"]),
            "R": float(outputs["R"]["a_nom"]),
            "F": environment_accel(snapshot["F"], t, scenario, disturbance, cfg),
            "B": idm_accel(snapshot["B"], snapshot["R"], cfg),
        }
        adjusted, safety_reasons = safety_adjustments(snapshot, a_nom, task, scenario, cfg)
        actual_a: Dict[str, float] = {}
        jerk: Dict[str, float] = {}
        exec_reasons: Dict[str, List[str]] = {}
        for vid in VEHICLE_IDS:
            actual_a[vid], jerk[vid], exec_reasons[vid] = execute_acceleration(snapshot[vid].a, adjusted[vid], cfg)
            all_jerks.append(abs(jerk[vid]))

        geom = geometry(snapshot, scenario, cfg)
        q_diag = {"M": outputs["M"], "R": outputs["R"]}
        for vid in VEHICLE_IDS:
            if vid in ("M", "R"):
                d = q_diag[vid]
            else:
                d = {
                    "q": 0.0,
                    "qdot_nominal": 0.0,
                    "qdot_base": 0.0,
                    "qdot_space": 0.0,
                    "qdot_partner": 0.0,
                    "qdot_clipped": 0.0,
                    "K_to_partner": 0.0,
                    "W_to_partner": 0.0,
                    "z_self": 0.0,
                    "z_other": 0.0,
                    "target_gap": outputs["R"]["target_gap"],
                    "gap_error": outputs["R"]["gap_error"],
                    "s_ref": "",
                    "target_time": "",
                    "grad_phi_space": 0.0,
                    "e_M0": outputs["M"]["e_M0"],
                    "interval_width": geom["interval_width"],
                }
            safety_delta = adjusted[vid] - a_nom[vid]
            reasons = safety_reasons[vid] + exec_reasons[vid]
            log_row = {
                "run_id": run_id,
                "t_s": round(t, 6),
                "method": method,
                "scenario": scenario_name,
                "disturbance": disturbance,
                "init_variant": init_variant,
                "vehicle": vid,
                "s_m": snapshot[vid].s,
                "y_m": snapshot[vid].y,
                "v_mps": snapshot[vid].v,
                "a_mps2": snapshot[vid].a,
                "q": d["q"],
                "qdot_nominal_s-1": d["qdot_nominal"],
                "qdot_base_s-1": d["qdot_base"],
                "qdot_space_s-1": d["qdot_space"],
                "qdot_partner_s-1": d["qdot_partner"],
                "qdot_clipped_s-1": d["qdot_clipped"],
                "K_to_partner_s-1": d["K_to_partner"],
                "W_to_partner": d["W_to_partner"],
                "z_self": d["z_self"],
                "z_other": d["z_other"],
                "g_FR_m": geom["g_fr"],
                "target_gap_m": d["target_gap"],
                "gap_error_m": d["gap_error"],
                "lower_m": geom["lower"],
                "upper_m": geom["upper"],
                "interval_width_m": geom["interval_width"],
                "s_ref_m": d["s_ref"],
                "target_time_s": d["target_time"],
                "merge_progress": snapshot[vid].merge_progress,
                "task_phase": task.phase,
                "a_nom_mps2": a_nom[vid],
                "a_safety_target_mps2": adjusted[vid],
                "a_actual_mps2": actual_a[vid],
                "jerk_mps3": jerk[vid],
                "nominal_actual_difference_mps2": actual_a[vid] - a_nom[vid],
                "safety_correction_mps2": safety_delta,
                "action_correction_reason": ";".join(reasons),
                "F_disturbance_mps2": disturbance_accel(t, disturbance) if vid == "F" else 0.0,
            }
            logs.append(log_row)

            ref_v = scenario.desired_speed
            accum[vid]["speed_deficit_integral_m_s"] += max(0.0, ref_v - snapshot[vid].v) * cfg.dt
            accum[vid]["accel_squared_integral"] += actual_a[vid] ** 2 * cfg.dt
            accum[vid]["abs_jerk_integral"] += abs(jerk[vid]) * cfg.dt
            accum[vid]["abs_nominal_actual_difference_integral"] += abs(actual_a[vid] - a_nom[vid]) * cfg.dt
            if safety_reasons[vid]:
                accum[vid]["safety_correction_count"] += 1.0
                accum[vid]["safety_correction_abs_integral"] += abs(safety_delta) * cfg.dt
                safety_abs_values.append(abs(safety_delta))

        k_total_errors.append(abs((float(outputs["M"]["K_to_partner"]) + float(outputs["R"]["K_to_partner"])) - cfg.k_total) if method in ("C", "D") else 0.0)
        feedback_abs.extend([abs(float(outputs["M"]["qdot_space"])), abs(float(outputs["R"]["qdot_space"]))])

        # Integrate all vehicles only after every method has seen the same snapshot.
        for vid in VEHICLE_IDS:
            v = vehicles[vid]
            v.s = v.s + v.v * cfg.dt + 0.5 * actual_a[vid] * cfg.dt * cfg.dt
            v.v = _clip(v.v + actual_a[vid] * cfg.dt, cfg.v_min, cfg.v_max)
            v.a = actual_a[vid]
        if task.phase == "EXECUTE":
            vehicles["M"].merge_progress = _clip(vehicles["M"].merge_progress + cfg.dt / cfg.merge_duration, 0.0, 1.0)
        vehicles["M"].y = cfg.lane_ramp + (cfg.lane_main - cfg.lane_ramp) * vehicles["M"].merge_progress

        controller.commit(outputs, cfg.dt, task.terminal)
        violations = hard_violations(vehicles, task, scenario, cfg)
        if violations:
            hard_violation_count += len(violations)
            road_violation_count += sum("road_exit" in x or "lateral_exit" in x for x in violations)
            if not task.terminal:
                task.phase = "FAILED"
                task.failure_type = "physical_constraint_violation"
                task.terminal = True
                task.event = "physical_constraint_violation"
                events.append({"run_id": run_id, "t_s": t + cfg.dt, "event": task.event, "details": ";".join(violations)})
        event = _task_event(task, t + cfg.dt, vehicles, controller, scenario, cfg)
        if event:
            events.append({"run_id": run_id, "t_s": t + cfg.dt, "event": event, "details": task.failure_type or ""})

        current_pairs = [(_net_gap(vehicles["F"], vehicles["R"])), (_net_gap(vehicles["R"], vehicles["B"]))]
        if vehicles["M"].merge_progress > 0.0:
            for other in ("F", "R", "B"):
                if _lateral_overlap(vehicles["M"], vehicles[other]):
                    current_pairs.append(abs(vehicles["M"].s - vehicles[other].s) - (vehicles["M"].length + vehicles[other].length) / 2.0)
        min_gap = min(min_gap, min(current_pairs))

    # A final state row is included so that the trajectory spans the complete window.
    final_t = cfg.horizon
    for vid in VEHICLE_IDS:
        v = vehicles[vid]
        logs.append({
            "run_id": run_id, "t_s": round(final_t, 6), "method": method, "scenario": scenario_name,
            "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
            "s_m": v.s, "y_m": v.y, "v_mps": v.v, "a_mps2": v.a, "q": controller.q.get(vid, 0.0),
            "qdot_nominal_s-1": 0.0, "qdot_base_s-1": 0.0, "qdot_space_s-1": 0.0, "qdot_partner_s-1": 0.0,
            "qdot_clipped_s-1": 0.0, "K_to_partner_s-1": 0.0, "W_to_partner": 0.0, "z_self": 0.0, "z_other": 0.0,
            "g_FR_m": _net_gap(vehicles["F"], vehicles["R"]), "target_gap_m": scenario.initial_gap + 0.55 * scenario.delta_g,
            "gap_error_m": 0.0, "lower_m": "", "upper_m": "", "interval_width_m": "", "s_ref_m": "", "target_time_s": "",
            "merge_progress": v.merge_progress, "task_phase": task.phase, "a_nom_mps2": 0.0, "a_safety_target_mps2": 0.0,
            "a_actual_mps2": v.a, "jerk_mps3": 0.0, "nominal_actual_difference_mps2": 0.0,
            "safety_correction_mps2": 0.0, "action_correction_reason": "", "F_disturbance_mps2": 0.0,
        })

    if not task.terminal:
        task.phase = "TIMEOUT"
        task.failure_type = "timeout"
        task.terminal = True
        events.append({"run_id": run_id, "t_s": cfg.horizon, "event": "timeout", "details": "observation_window_end"})
    if task.phase == "SUCCESS":
        outcome = "completed"
    elif task.failure_type:
        outcome = task.failure_type
    else:
        outcome = task.phase.lower()

    # Recompute min gaps and role-level statistics from the actual state traces.
    for row in logs:
        vid = str(row["vehicle"])
        if vid in accum:
            accum[vid]["distance_m"] = float(accum[vid]["distance_m"]) + (float(row["v_mps"]) * cfg.dt if float(row["t_s"]) < cfg.horizon else 0.0)
        if vid == "R":
            accum["R"]["min_net_gap_m"] = min(float(accum["R"]["min_net_gap_m"]), float(row["g_FR_m"]))
    for vid in VEHICLE_IDS:
        if not math.isfinite(accum[vid]["min_net_gap_m"]):
            accum[vid]["min_net_gap_m"] = min_gap

    k_mean = float(np.mean(k_total_errors)) if k_total_errors else 0.0
    feedback_mean = float(np.mean(feedback_abs)) if feedback_abs else 0.0
    compute_time = time.perf_counter() - start_clock
    metrics: Dict[str, object] = {
        "run_id": run_id,
        "method": method,
        "scenario": scenario_name,
        "disturbance": disturbance,
        "init_variant": init_variant,
        "outcome": outcome,
        "success": int(outcome == "completed"),
        "task_phase": task.phase,
        "failure_type": task.failure_type,
        "merge_start_time_s": "" if task.merge_start_time is None else task.merge_start_time,
        "completion_time_s": "" if task.completion_time is None else task.completion_time,
        "min_net_gap_m": min_gap,
        "hard_violation_count": hard_violation_count,
        "road_violation_count": road_violation_count,
        "safety_correction_count": int(sum(v["safety_correction_count"] for v in accum.values())),
        "safety_correction_abs_mean_mps2": float(np.mean(safety_abs_values)) if safety_abs_values else 0.0,
        "max_abs_jerk_mps3": max(all_jerks) if all_jerks else 0.0,
        "mean_abs_jerk_mps3": float(np.mean(all_jerks)) if all_jerks else 0.0,
        "compute_time_s": compute_time,
        "pair_gain_total_error_max_s-1": float(max(k_total_errors)) if k_total_errors else 0.0,
        "space_feedback_abs_mean_s-1": feedback_mean,
        "K_MR_mean_s-1": float(np.mean([float(r["K_to_partner_s-1"]) for r in logs if r["vehicle"] == "M"])) if method in ("C", "D") else 0.0,
        "K_RM_mean_s-1": float(np.mean([float(r["K_to_partner_s-1"]) for r in logs if r["vehicle"] == "R"])) if method in ("C", "D") else 0.0,
        "total_speed_deficit_integral_m_s": float(sum(v["speed_deficit_integral_m_s"] for v in accum.values())),
        "total_effort_integral_m2_s3": float(sum(v["accel_squared_integral"] for v in accum.values())),
    }
    vehicle_metrics: List[Dict[str, object]] = []
    for vid in VEHICLE_IDS:
        row = {"run_id": run_id, "method": method, "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid}
        row.update(accum[vid])
        vehicle_metrics.append(row)
    return SimulationResult(run_id, method, scenario_name, disturbance, init_variant, logs, events, metrics, vehicle_metrics)


def run_matrix(methods: Iterable[str], scenarios: Iterable[str], disturbances: Iterable[str], init_variants: Iterable[int], cfg: Optional[SimConfig] = None) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    for method in methods:
        for scenario in scenarios:
            for disturbance in disturbances:
                for init_variant in init_variants:
                    results.append(run_episode(method, scenario, disturbance, init_variant, cfg=cfg))
    return results
