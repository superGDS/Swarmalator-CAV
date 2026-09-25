"""Stage 1C four-vehicle merge model.

This module is deliberately separate from :mod:`simulation`.  The Stage 1
outputs and runner are frozen evidence; Stage 1C uses the same deterministic
scenario matrix but a corrected common physical basis.  The model is still a
small longitudinal point-mass prototype with a continuous lateral path for M,
so its results are implementation and mechanism evidence rather than a road
safety claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


VEHICLE_IDS = ("M", "R", "F", "B")
ACTIVE_PHASES = ("PREPARE", "EXECUTE")
POST_PHASES = ("SUCCESS_RELEASE", "FAILURE_HANDLING")


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
    # The task boundary is explicit in v1.  The old +3 m tolerance is not used.
    merge_zone: Tuple[float, float] = (55.0, 170.0)
    completion_s: float = 170.0
    terminal_s: float = 190.0
    road_max: float = 450.0

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
    # A zero slot buffer is intentional: 21.4 m is the non-empty finite-body
    # interval at 18 m/s for the v0 vehicle dimensions and dynamic margins.
    slot_buffer_m: float = 0.0
    safety_recheck_tolerance_m: float = 0.15
    # One common Stage 1C revision: finite-window opening feed-forward derived
    # from the current slot deficit and one fixed preview horizon.  It is
    # applied to M and R for every controller and is inactive for open slots.
    window_opening_enabled: bool = True
    window_opening_gain: float = 2.0
    window_opening_time_s: float = 1.5
    window_opening_max_mps2: float = 3.0
    fixed_partner_weight: float = 0.20


@dataclass
class TaskState:
    phase: str = "PREPARE"
    failure_type: str = ""
    merge_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    terminal: bool = False
    success: bool = False
    event: str = ""
    startup_check_recorded: bool = False


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
    base = {
        "ample": dict(F_s=100.0, R_s=65.0, B_s=38.0, M_s=74.0, F_v=18.5, R_v=18.0, B_v=18.0, M_v=22.0, delta_g=4.0),
        "collaborative": dict(F_s=94.0, R_s=68.0, B_s=42.0, M_s=76.0, F_v=18.0, R_v=18.0, B_v=18.0, M_v=21.5, delta_g=8.0),
        "short_window": dict(F_s=90.0, R_s=72.0, B_s=48.0, M_s=78.0, F_v=18.0, R_v=18.5, B_v=18.0, M_v=21.0, delta_g=7.0),
    }
    if name not in base:
        raise ValueError(f"unknown scenario {name}")
    p = dict(base[name])
    if init_variant == 1:
        p.update(M_s=p["M_s"] - 3.0, M_v=p["M_v"] - 1.7, R_v=p["R_v"] + 0.2)
    elif init_variant == 2:
        p.update(F_s=p["F_s"] - 1.5, R_s=p["R_s"] + 1.5, B_s=p["B_s"] + 1.0,
                 M_s=p["M_s"] - 1.0, R_v=p["R_v"] - 0.6, B_v=p["B_v"] - 0.3)
    elif init_variant != 0:
        raise ValueError(f"unknown initial-state variant {init_variant}")
    return Scenario(name=name, merge_zone=(55.0, 170.0), completion_s=170.0,
                    terminal_s=190.0, road_max=450.0, **p)


def make_vehicles(scenario: Scenario, cfg: SimConfig) -> Dict[str, VehicleState]:
    return {
        "M": VehicleState("M", scenario.M_s, cfg.lane_ramp, scenario.M_v, 0.0),
        "R": VehicleState("R", scenario.R_s, cfg.lane_main, scenario.R_v, 0.0),
        "F": VehicleState("F", scenario.F_s, cfg.lane_main, scenario.F_v, 0.0),
        "B": VehicleState("B", scenario.B_s, cfg.lane_main, scenario.B_v, 0.0),
    }


def dynamic_gap(follower: VehicleState, cfg: SimConfig) -> float:
    return cfg.min_base_gap + cfg.time_headway * max(follower.v, 0.0)


def physical_gap_floor(scenario: Scenario, cfg: SimConfig) -> float:
    """Finite-body non-empty slot requirement at the common design speed.

    From ``upper-lower = g_FR - L_M - d_RM - d_MF`` and
    ``d = base + headway*v``, the v0 values give 4.8+2*(2+0.35*18)=21.4 m.
    The same formula is used for every method and is logged per run.
    """
    d = cfg.min_base_gap + cfg.time_headway * scenario.desired_speed
    return 4.8 + 2.0 * d + cfg.slot_buffer_m


def physical_target_gap(scenario: Scenario, cfg: SimConfig) -> float:
    # A gap already larger than the physical floor is not artificially closed.
    return max(scenario.initial_gap, physical_gap_floor(scenario, cfg))


def process_gap_delta(scenario: Scenario, cfg: SimConfig) -> float:
    """Return the q-dependent process-gap span on the corrected basis."""
    return max(0.0, physical_target_gap(scenario, cfg) - scenario.initial_gap)


def process_gap_gradient(gap_error: float, scenario: Scenario, cfg: SimConfig,
                         q: float = 0.0) -> float:
    """Derivative of the process potential with respect to R's q.

    The derivative uses the same physical target that is actually sent to the
    low-level tracker.  In ample environments the target is independent of q,
    so this function is exactly zero rather than carrying the old scenario
    ``delta_g`` surrogate.
    """
    del q  # the span is affine in q on the supported interval
    return -float(gap_error) * process_gap_delta(scenario, cfg) / (cfg.ell_g * cfg.ell_g)


def geometry(snapshot: Mapping[str, VehicleState], scenario: Scenario, cfg: SimConfig) -> Dict[str, float]:
    M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
    g_fr = F.s - R.s - (F.length + R.length) / 2.0
    d_rm = dynamic_gap(R, cfg)
    d_mf = dynamic_gap(M, cfg)
    lower = R.s + (R.length + M.length) / 2.0 + d_rm
    upper = F.s - (F.length + M.length) / 2.0 - d_mf
    return {"g_fr": g_fr, "d_rm": d_rm, "d_mf": d_mf,
            "lower": lower, "upper": upper, "c": 0.5 * (lower + upper),
            "interval_width": upper - lower,
            "target_gap_physical": physical_target_gap(scenario, cfg),
            "target_gap_floor": physical_gap_floor(scenario, cfg)}


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-_clip(x, -30.0, 30.0)))


def _moving_reference_control(s: float, v: float, s_ref: float, v_ref: float,
                             a_ref: float, kp: float = 0.25, kv: float = 0.60) -> float:
    """Reference acceleration with explicit feedforward and SI-consistent errors."""
    return float(a_ref + kp * (s_ref - s) + kv * (v_ref - v))


def cfg_midline(cfg: SimConfig) -> float:
    return 0.5 * (cfg.lane_ramp + cfg.lane_main)


class PairController:
    """Same-snapshot controller for M/R with a common Stage 1C interface."""

    def __init__(self, method: str, scenario: Scenario, cfg: SimConfig, ablation: str = ""):
        if method not in ("A", "B", "C", "D", "E"):
            raise ValueError(f"unknown method {method}")
        self.method, self.scenario, self.cfg, self.ablation = method, scenario, cfg, ablation
        self.q: Dict[str, float] = {"M": 0.0, "R": 0.0}
        self.e_M0: Optional[float] = None
        self.c0: Optional[float] = None
        self.prev_k_mr = cfg.k_total / 2.0
        self.prev_target_time = 3.5
        self.last_target_endpoint: Optional[float] = None

    def state_dict(self) -> Dict[str, object]:
        """Return all mutable controller state needed for a same-snapshot replay."""
        return {
            "q_M": float(self.q["M"]), "q_R": float(self.q["R"]),
            "e_M0": self.e_M0, "c0": self.c0,
            "prev_k_mr": float(self.prev_k_mr),
            "prev_target_time": float(self.prev_target_time),
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        """Restore :meth:`state_dict` without resetting history."""
        self.q["M"] = _clip(float(state.get("q_M", 0.0)), 0.0, 1.0)
        self.q["R"] = _clip(float(state.get("q_R", 0.0)), 0.0, 1.0)
        self.e_M0 = None if state.get("e_M0") in (None, "") else float(state["e_M0"])
        self.c0 = None if state.get("c0") in (None, "") else float(state["c0"])
        self.prev_k_mr = float(state.get("prev_k_mr", self.cfg.k_total / 2.0))
        self.prev_target_time = float(state.get("prev_target_time", 3.5))

    def step(self, local_observation: Mapping[str, object], received_messages: Mapping[str, object],
             task_state: TaskState, dt: float) -> Dict[str, object]:
        del received_messages, dt
        return self.compute_pair(local_observation["snapshot"], task_state, self.cfg.dt)[str(local_observation["role"])]

    def _target_gap_process(self, q_r: float) -> float:
        final = physical_target_gap(self.scenario, self.cfg)
        delta = process_gap_delta(self.scenario, self.cfg)
        return self.scenario.initial_gap + delta * _clip(q_r, 0.0, 1.0)

    def _endpoint_candidate(self, snapshot: Mapping[str, VehicleState], target_T: float,
                            target_gap: float) -> Tuple[float, float]:
        """Recompute the endpoint using the *smoothed* time target."""
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        T = max(float(target_T), 0.25)
        fp, rp = F.s + F.v * T, R.s + R.v * T
        d_mf = self.cfg.min_base_gap + self.cfg.time_headway * max(M.v, 0.0)
        d_rm = self.cfg.min_base_gap + self.cfg.time_headway * max(R.v, 0.0)
        lo = rp + (R.length + M.length) / 2.0 + d_rm
        hi = fp - (F.length + M.length) / 2.0 - d_mf
        c = 0.5 * (lo + hi)
        a_req = 2.0 * (c - M.s - M.v * T) / (T * T)
        return float(c), float(a_req)

    def _online_target(self, snapshot: Mapping[str, VehicleState], target_gap: float) -> Tuple[float, float, float]:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        best = None
        for T in np.arange(self.cfg.online_t_min, self.cfg.online_t_max + 0.01, self.cfg.online_t_step):
            fp, rp = F.s + F.v * T, R.s + R.v * T
            d_mf = self.cfg.min_base_gap + self.cfg.time_headway * max(M.v, 0.0)
            d_rm = self.cfg.min_base_gap + self.cfg.time_headway * max(R.v, 0.0)
            lo = rp + (R.length + M.length) / 2.0 + d_rm
            hi = fp - (F.length + M.length) / 2.0 - d_mf
            c = 0.5 * (lo + hi)
            a_req = 2.0 * (c - M.s - M.v * T) / (T * T)
            gap_pred = fp - rp - (F.length + R.length) / 2.0
            cost = (a_req / max(self.cfg.a_max, 0.1)) ** 2 + 0.35 * ((gap_pred - target_gap) / self.cfg.ell_g) ** 2 + 0.025 * T
            if hi < lo:
                cost += 1.5 * (lo - hi) / self.cfg.ell_g
            candidate = (float(cost), float(T), float(c))
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, target_T, _ = best
        self.prev_target_time = (1.0 - self.cfg.online_t_smoothing) * self.prev_target_time + self.cfg.online_t_smoothing * target_T
        target_c, a_req = self._endpoint_candidate(snapshot, self.prev_target_time, target_gap)
        return self.prev_target_time, target_c, float(a_req)

    def _directional_capability(self, snapshot: Mapping[str, VehicleState], geom: Mapping[str, float],
                                ref_error: float, gap_error: float) -> Tuple[float, float]:
        """Direction-aware action/time margin using only the current snapshot.

        It replaces the old ``a_max-abs(a_nom)`` proxy.  No future state or
        low-level action is read, so qdot and the acceleration command form no
        algebraic loop.
        """
        M, R = snapshot["M"], snapshot["R"]
        def directional_margin(v: VehicleState, error: float) -> float:
            if error >= 0.0:
                return _clip((self.cfg.a_max - v.a) / (self.cfg.a_max - 0.0), -1.0, 1.0)
            return _clip((v.a - self.cfg.a_min) / (0.0 - self.cfg.a_min), -1.0, 1.0)
        time_margin = _clip((self.scenario.completion_s - M.s) /
                            max(self.scenario.completion_s - self.scenario.merge_zone[0], 1.0), -1.0, 1.0)
        gap_margin = _clip(gap_error / self.cfg.ell_g, -1.0, 1.0)
        pressure = _clip((dynamic_gap(R, self.cfg) - geom["g_fr"]) / self.cfg.ell_g, -1.0, 1.0)
        z_m = 0.55 * directional_margin(M, ref_error) + 0.45 * time_margin
        z_r = 0.55 * directional_margin(R, -gap_error) + 0.30 * gap_margin - 0.35 * pressure
        return z_m, z_r

    def _reference(self, snapshot: Mapping[str, VehicleState], geom: Mapping[str, float],
                   qdot_m_est: float, task: TaskState) -> Tuple[float, float, float, str]:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        # Derivative of the finite-body midpoint uses only current velocities
        # and recorded accelerations; qdot is the same-snapshot estimate.
        c_dot = 0.5 * (R.v + F.v + self.cfg.time_headway * (R.a - M.a))
        c_ddot = 0.0
        if self.method == "E" and task.phase in ACTIVE_PHASES:
            target_T, target_c, a_req = self._online_target(snapshot, physical_target_gap(self.scenario, self.cfg))
            # The endpoint is a planning diagnostic.  The low-level tracker
            # receives the current position and planned velocity/acceleration;
            # a matched 18 m/s state therefore has a zero command.
            s_ref = M.s
            c_dot = M.v
            c_ddot = a_req
            label = "rolling_candidate"
            self.last_target_endpoint = target_c
        elif self.method == "A" or task.phase in POST_PHASES or task.phase == "TIMEOUT":
            s_ref = geom["c"]
            label = "slot_midpoint"
            self.last_target_endpoint = None
        else:
            s_ref = geom["c"] + (1.0 - self.q["M"]) * float(self.e_M0 or 0.0)
            label = "phase_offset_slot"
            self.last_target_endpoint = None
        s_ref_dot = c_dot - qdot_m_est * float(self.e_M0 or 0.0) if label == "phase_offset_slot" else c_dot
        return float(s_ref), float(s_ref_dot), float(c_ddot), label

    def compute_pair(self, snapshot: Mapping[str, VehicleState], task: TaskState, dt: float) -> Dict[str, Dict[str, object]]:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        geom = geometry(snapshot, self.scenario, self.cfg)
        if self.e_M0 is None:
            self.c0, self.e_M0 = geom["c"], M.s - geom["c"]
        assert self.e_M0 is not None
        target_physical = physical_target_gap(self.scenario, self.cfg)
        target_process = self._target_gap_process(self.q["R"])
        target_gap = target_physical if self.method in ("A", "E") or task.phase in POST_PHASES else target_process
        gap_error = geom["g_fr"] - target_gap

        # First snapshot reference is used for the spatial gradient and the
        # direction-only D allocation.  The final reference includes qdot.
        s_ref0 = geom["c"] if self.method in ("A", "E") else geom["c"] + (1.0 - self.q["M"]) * self.e_M0
        grad_m = (M.s - s_ref0) * self.e_M0 / (self.cfg.ell_s * self.cfg.ell_s)
        grad_r = process_gap_gradient(gap_error, self.scenario, self.cfg, self.q["R"])
        space_m = -self.cfg.kappa_M * grad_m if self.method in ("C", "D") and self.ablation != "no_space" else 0.0
        space_r = -self.cfg.kappa_R * grad_r if self.method in ("C", "D") and self.ablation != "no_space" else 0.0
        k_mr = k_rm = w_mr = w_rm = partner_m = partner_r = 0.0
        z_m = z_r = 0.0
        if self.method in ("C", "D") and task.phase in ACTIVE_PHASES:
            w = _clip(0.15 + 0.85 * abs(gap_error) / self.cfg.ell_g, 0.0, 1.0)
            if self.ablation == "no_W":
                w = 0.0
            elif self.ablation == "fixed_W":
                w = _clip(self.cfg.fixed_partner_weight, 0.0, 1.0)
            w_mr = w_rm = w
            if self.method == "D" and self.ablation == "symmetric_k":
                k_mr = k_rm = self.cfg.k_total / 2.0
            elif self.method == "D":
                z_m, z_r = self._directional_capability(snapshot, geom, s_ref0 - M.s, gap_error)
                raw_k_mr = self.cfg.k_total * _sigmoid(z_m - z_r)
                k_mr = (1.0 - self.cfg.phase_gain_smoothing) * self.prev_k_mr + self.cfg.phase_gain_smoothing * raw_k_mr
                k_rm = self.cfg.k_total - k_mr
                self.prev_k_mr = k_mr
            else:
                k_mr = k_rm = self.cfg.k_total / 2.0
            if self.ablation != "no_partner":
                partner_m = k_mr * w_mr * math.sin(self.cfg.alpha * (self.q["R"] - self.q["M"]))
                partner_r = k_rm * w_rm * math.sin(self.cfg.alpha * (self.q["M"] - self.q["R"]))
        qdot_m = qdot_r = 0.0
        if self.method in ("B", "C", "D") and task.phase in ACTIVE_PHASES:
            qdot_m = self.cfg.nu_M + space_m + partner_m
            qdot_r = self.cfg.nu_R + space_r + partner_r
        qdot_m_clip = _clip(qdot_m, 0.0, self.cfg.nu_max) if self.q["M"] < 1.0 else 0.0
        qdot_r_clip = _clip(qdot_r, 0.0, self.cfg.nu_max) if self.q["R"] < 1.0 else 0.0

        s_ref, s_ref_dot, s_ref_ddot, reference_kind = self._reference(snapshot, geom, qdot_m_clip, task)
        opening_accel = 0.0
        if self.cfg.window_opening_enabled and task.phase in ACTIVE_PHASES and geom["interval_width"] < 0.0:
            deficit = -float(geom["interval_width"])
            horizon = max(self.cfg.window_opening_time_s, self.cfg.dt)
            opening_accel = -_clip(self.cfg.window_opening_gain * deficit / (horizon * horizon),
                                   0.0, self.cfg.window_opening_max_mps2)
        if task.phase in POST_PHASES:
            # Normal following/cruise is deliberately independent of the old
            # task target; the lateral path remains continuous below.
            m_front = current_leader(snapshot, "M")
            a_nom_m = (idm_accel(M, m_front, self.cfg) if m_front is not None
                       else 0.85 * (self.scenario.desired_speed - M.v))
            reference_kind = "post_terminal_cruise"
            s_ref_log: object = ""
        else:
            a_nom_m = _moving_reference_control(M.s, M.v, s_ref, s_ref_dot, s_ref_ddot)
            s_ref_log = s_ref
        if task.phase in POST_PHASES:
            r_leader = current_r_leader(snapshot, task, self.cfg)
            post_front_id = r_leader.vid
            a_nom_r = idm_accel(R, r_leader, self.cfg)
        else:
            post_front_id = "F"
            v_ref_r = _clip(F.v + 0.36 * gap_error, 9.0, 23.0)
            a_nom_r = 1.0 * (v_ref_r - R.v) + 0.08 * gap_error
        if task.phase in ACTIVE_PHASES:
            a_nom_m += opening_accel
            a_nom_r += opening_accel

        def out(role: str, a_nom: float, qdot: float, qclip: float, space: float, partner: float,
                k: float, w: float, zself: float, zother: float) -> Dict[str, object]:
            return {
                "a_nom": float(a_nom), "q": float(self.q[role]),
                "qdot_nominal": float(qdot),
                "qdot_base": float(self.cfg.nu_M if role == "M" else self.cfg.nu_R) if self.method in ("B", "C", "D") and task.phase in ACTIVE_PHASES else 0.0,
                "qdot_space": float(space), "qdot_partner": float(partner), "qdot_clipped": float(qclip),
                "K_to_partner": float(k), "W_to_partner": float(w), "z_self": float(zself), "z_other": float(zother),
                "target_gap": float(target_physical), "target_gap_process": float(target_process),
                "gap_error": float(gap_error), "s_ref": s_ref_log, "s_ref_dot": float(s_ref_dot),
                "s_ref_ddot": float(s_ref_ddot), "reference_kind": reference_kind,
                "target_time": float(self.prev_target_time) if self.method == "E" else "",
                "target_endpoint": (float(self.last_target_endpoint) if self.last_target_endpoint is not None else ""),
                "window_opening_accel": float(opening_accel), "post_front_id": post_front_id,
                "grad_phi_space": float(grad_m if role == "M" else grad_r),
                "e_M0": float(self.e_M0), "interval_width": float(geom["interval_width"]),
                "target_gap_floor": float(geom["target_gap_floor"]),
            }
        return {"M": out("M", a_nom_m, qdot_m, qdot_m_clip, space_m, partner_m, k_mr, w_mr, z_m, z_r),
                "R": out("R", a_nom_r, qdot_r, qdot_r_clip, space_r, partner_r, k_rm, w_rm, z_r, z_m)}

    def commit(self, outputs: Mapping[str, Mapping[str, object]], dt: float, terminal: bool) -> None:
        if self.method in ("A", "E") or terminal:
            return
        for role in ("M", "R"):
            self.q[role] = _clip(self.q[role] + float(outputs[role]["qdot_clipped"]) * dt, 0.0, 1.0)


def make_controller(method: str, scenario: Scenario, cfg: SimConfig, ablation: str = "") -> PairController:
    return PairController(method, scenario, cfg, ablation=ablation)


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
    v0, a0, b0 = 20.0, 1.3, 2.2
    gap = max(leader.s - follower.s - (leader.length + follower.length) / 2.0, 0.1)
    dv = follower.v - leader.v
    s_star = cfg.min_base_gap + cfg.time_headway * follower.v + follower.v * dv / (2.0 * math.sqrt(a0 * b0))
    return _clip(a0 * (1.0 - (follower.v / v0) ** 4 - (max(s_star, 0.1) / gap) ** 2), cfg.a_min, cfg.a_max)


def _lateral_overlap(v1: VehicleState, v2: VehicleState) -> bool:
    return abs(v1.y - v2.y) <= (v1.width + v2.width) / 2.0


def _net_gap(leader: VehicleState, follower: VehicleState) -> float:
    return leader.s - follower.s - (leader.length + follower.length) / 2.0


def min_forward_distance(vehicle: VehicleState, duration: float, cfg: SimConfig, step: float = 0.01) -> float:
    """Optimistic minimum forward travel under max braking and jerk bounds."""
    x, v, a = vehicle.s, vehicle.v, vehicle.a
    n = int(math.ceil(duration / step))
    h = duration / n
    for _ in range(n):
        delta = _clip(cfg.a_min - a, -cfg.jerk_max * h, cfg.jerk_max * h)
        aa = _clip(a + delta, cfg.a_min, cfg.a_max)
        x += max(v, 0.0) * h + 0.5 * aa * h * h
        v = _clip(v + aa * h, cfg.v_min, cfg.v_max)
        a = aa
    return float(x - vehicle.s)


def physical_start_feasibility(snapshot: Mapping[str, VehicleState], task: TaskState,
                                scenario: Scenario, cfg: SimConfig) -> Dict[str, object]:
    M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
    geom = geometry(snapshot, scenario, cfg)
    rem = scenario.completion_s - M.s
    min_dist = min_forward_distance(M, cfg.merge_duration, cfg)
    necessary_bound_ok = min_dist <= max(rem, 0.0) + 1e-9
    slot_ok = geom["interval_width"] >= 0.0 and geom["lower"] <= M.s <= geom["upper"]
    in_zone = scenario.merge_zone[0] <= M.s <= scenario.merge_zone[1]
    # This is a deliberately conservative same-snapshot occupancy diagnostic,
    # not a sufficient feasibility certificate.  It checks the current-speed
    # forecast over the fixed lateral duration; the actual executor and safety
    # recheck remain the authority during the run.
    occupancy_widths: List[float] = []
    occupancy_front: List[float] = []
    occupancy_rear: List[float] = []
    for tau in np.linspace(0.0, cfg.merge_duration, 31):
        fp = F.s + F.v * tau
        rp = R.s + R.v * tau
        mp = M.s + M.v * tau
        rm = cfg.min_base_gap + cfg.time_headway * max(R.v, 0.0)
        mf = cfg.min_base_gap + cfg.time_headway * max(M.v, 0.0)
        occupancy_widths.append((fp - rp - 4.8) - M.length - rm - mf)
        occupancy_front.append(fp - mp - (F.length + M.length) / 2.0 - mf)
        occupancy_rear.append(mp - rp - (M.length + R.length) / 2.0 - rm)
    return {
        "in_start_zone": bool(in_zone), "slot_ok": bool(slot_ok),
        "necessary_bound_ok": bool(necessary_bound_ok),
        "remaining_distance_m": float(rem), "min_forward_distance_m": float(min_dist),
        "interval_width_m": float(geom["interval_width"]),
        "current_speed_forecast_min_slot_width_m": float(min(occupancy_widths)),
        "current_speed_forecast_min_front_margin_m": float(min(occupancy_front)),
        "current_speed_forecast_min_rear_margin_m": float(min(occupancy_rear)),
        "physical_ok": bool(in_zone and slot_ok and necessary_bound_ok and not task.terminal),
    }


def _predicted_state(v: VehicleState, accel: float, cfg: SimConfig) -> Tuple[float, float]:
    return (v.s + v.v * cfg.dt + 0.5 * accel * cfg.dt * cfg.dt,
            _clip(v.v + accel * cfg.dt, cfg.v_min, cfg.v_max))


def _predicted_safety_violations(snapshot: Mapping[str, VehicleState], actual: Mapping[str, float],
                                 active_merge: bool, cfg: SimConfig) -> List[str]:
    p = {vid: _predicted_state(snapshot[vid], actual[vid], cfg)[0] for vid in VEHICLE_IDS}
    out: List[str] = []
    def gap(leader: str, follower: str) -> float:
        return p[leader] - p[follower] - (snapshot[leader].length + snapshot[follower].length) / 2.0
    tol = cfg.safety_recheck_tolerance_m
    if gap("F", "R") < dynamic_gap(snapshot["R"], cfg) - tol: out.append("FR_predicted_gap")
    if gap("R", "B") < dynamic_gap(snapshot["B"], cfg) - tol: out.append("RB_predicted_gap")
    if active_merge:
        if gap("F", "M") < dynamic_gap(snapshot["M"], cfg) - tol: out.append("FM_predicted_gap")
        if gap("M", "R") < dynamic_gap(snapshot["R"], cfg) - tol: out.append("MR_predicted_gap")
    return out


def current_r_leader(snapshot: Mapping[str, VehicleState], task: TaskState, cfg: SimConfig) -> VehicleState:
    """Return the physical front vehicle R must follow after the task."""
    M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
    if task.phase in POST_PHASES and M.s > R.s and _lateral_overlap(M, R):
        return M
    return F


def current_leader(snapshot: Mapping[str, VehicleState], follower_id: str) -> Optional[VehicleState]:
    follower = snapshot[follower_id]
    candidates = [v for key, v in snapshot.items() if key != follower_id
                  and v.s > follower.s and _lateral_overlap(v, follower)]
    return min(candidates, key=lambda v: v.s) if candidates else None


def feasible_action_interval(v: VehicleState, cfg: SimConfig) -> Tuple[float, float]:
    return (max(cfg.a_min, v.a - cfg.jerk_max * cfg.dt, (cfg.v_min - v.v) / cfg.dt),
            min(cfg.a_max, v.a + cfg.jerk_max * cfg.dt, (cfg.v_max - v.v) / cfg.dt))


def execute_safe_actions(snapshot: Mapping[str, VehicleState], desired: Mapping[str, float],
                         active_merge: bool, cfg: SimConfig) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Exact one-step interval feasibility for the longitudinal leader chain.

    h_next = h + (v_L-v_f)dt + dt^2 a_L/2
             - (dt^2/2 + headway*dt) a_f.
    F's jerk-executed script is fixed. Backward propagation tightens feasible
    leader lower bounds; a forward pass chooses the closest nominal action in
    each interval. This is a feasible chain projection, not a global L2 QP.
    B remains an IDM follower with only the same safety actuator correction.
    """
    interval = {k: feasible_action_interval(v, cfg) for k, v in snapshot.items()}
    lo = {k: ab[0] for k, ab in interval.items()}
    hi = {k: ab[1] for k, ab in interval.items()}
    c, d = 0.5 * cfg.dt**2, 0.5 * cfg.dt**2 + cfg.time_headway * cfg.dt
    chain = ["F", "M", "R", "B"] if active_merge else ["F", "R", "B"]
    fixed_f = _clip(desired["F"], lo["F"], hi["F"])
    base = {}
    for lead, follow in zip(chain, chain[1:]):
        L, f = snapshot[lead], snapshot[follow]
        base[(lead, follow)] = _net_gap(L, f) - dynamic_gap(f, cfg) + (L.v - f.v) * cfg.dt
    feasible = True
    for lead, follow in reversed(list(zip(chain, chain[1:]))):
        lo[lead] = max(lo[lead], (d * lo[follow] - base[(lead, follow)]) / c)
        if lo[lead] > hi[lead] + 1e-9:
            feasible = False
    if lo["F"] > fixed_f + 1e-9:
        feasible = False
    actual = {"F": fixed_f}
    if not active_merge:
        actual["M"] = _clip(desired["M"], *interval["M"])
    for lead, follow in zip(chain, chain[1:]):
        upper = min(interval[follow][1], (base[(lead, follow)] + c * actual[lead]) / d)
        lower = lo[follow] if feasible else interval[follow][0]
        if upper < lower - 1e-9:
            actual[follow] = interval[follow][0]
        else:
            actual[follow] = _clip(desired[follow], lower, max(lower, upper))
    margins = {f"{a}{b}": base[(a, b)] + c * actual[a] - d * actual[b]
               for a, b in zip(chain, chain[1:])}
    return actual, {"one_step_feasible": int(feasible), "next_h_min": min(margins.values()),
                    "next_h": margins, "action_intervals": interval}


def occupancy_plan_check(snapshot: Mapping[str, VehicleState], controller: PairController,
                         scenario: Scenario, cfg: SimConfig) -> Dict[str, object]:
    """Bounded 3 s proposed rollout; uses no future disturbance information."""
    v = {k: x.copy() for k, x in snapshot.items()}
    clone = make_controller(controller.method, scenario, cfg, controller.ablation)
    clone.load_state(controller.state_dict())
    task = TaskState(phase="EXECUTE")
    h_min = float("inf")
    for _ in range(int(round(cfg.merge_duration / cfg.dt))):
        out = clone.compute_pair(v, task, cfg.dt)
        nominal = {"M": float(out["M"]["a_nom"]), "R": float(out["R"]["a_nom"]),
                   "F": environment_accel(v["F"], 0.0, scenario, "none", cfg),
                   "B": idm_accel(v["B"], v["R"], cfg)}
        actual, diag = execute_safe_actions(v, nominal, True, cfg)
        h_min = min(h_min, float(diag["next_h_min"]))
        if not diag["one_step_feasible"]:
            return {"plan_ok": False, "reason": "predicted_action_interval_empty", "min_h": h_min}
        for key, x in v.items():
            x.s += x.v * cfg.dt + 0.5 * actual[key] * cfg.dt**2
            x.v = _clip(x.v + actual[key] * cfg.dt, cfg.v_min, cfg.v_max)
            x.a = actual[key]
        clone.commit(out, cfg.dt, False)
        if v["M"].s > scenario.completion_s + 1e-9:
            return {"plan_ok": False, "reason": "predicted_completion_boundary", "min_h": h_min}
    return {"plan_ok": True, "reason": "proposed_3s_rollout", "min_h": h_min}


def safety_adjustments(snapshot: Mapping[str, VehicleState], desired: Mapping[str, float], task: TaskState,
                       scenario: Scenario, cfg: SimConfig) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    del scenario
    adjusted = {k: _clip(float(v), cfg.a_min, cfg.a_max) for k, v in desired.items()}
    reasons: Dict[str, List[str]] = {k: [] for k in VEHICLE_IDS}
    def predicted(v: VehicleState, accel: float) -> float:
        return v.s + v.v * cfg.dt + 0.5 * accel * cfg.dt * cfg.dt
    r_leader_id = current_r_leader(snapshot, task, cfg).vid
    first_pair = (r_leader_id, "R") if r_leader_id == "M" else ("F", "R")
    for leader_id, follower_id in (first_pair, ("R", "B")):
        leader, follower = snapshot[leader_id], snapshot[follower_id]
        gap = predicted(leader, adjusted[leader_id]) - predicted(follower, adjusted[follower_id]) - (leader.length + follower.length) / 2.0
        required = dynamic_gap(follower, cfg)
        if gap < required:
            adjusted[follower_id] -= 2.0 * (required - gap) / (cfg.dt * cfg.dt)
            reasons[follower_id].append("mainline_gap")
    active = task.phase == "EXECUTE" or snapshot["M"].y <= cfg_midline(cfg)
    if active:
        M, R, F = snapshot["M"], snapshot["R"], snapshot["F"]
        mf = predicted(F, adjusted["F"]) - predicted(M, adjusted["M"]) - (F.length + M.length) / 2.0
        if mf < dynamic_gap(M, cfg):
            adjusted["M"] -= 2.0 * (dynamic_gap(M, cfg) - mf) / (cfg.dt * cfg.dt)
            reasons["M"].append("merge_front_gap")
        mr = predicted(M, adjusted["M"]) - predicted(R, adjusted["R"]) - (M.length + R.length) / 2.0
        if mr < dynamic_gap(R, cfg):
            adjusted["R"] -= 2.0 * (dynamic_gap(R, cfg) - mr) / (cfg.dt * cfg.dt)
            reasons["R"].append("merge_rear_gap")
    for k in VEHICLE_IDS:
        adjusted[k] = _clip(adjusted[k], cfg.a_min, cfg.a_max)
    return adjusted, reasons


def execute_acceleration(previous_a: float, target_a: float, cfg: SimConfig) -> Tuple[float, float, List[str]]:
    target = _clip(target_a, cfg.a_min, cfg.a_max)
    delta = target - previous_a
    max_delta = cfg.jerk_max * cfg.dt
    limited_delta = _clip(delta, -max_delta, max_delta)
    actual = _clip(previous_a + limited_delta, cfg.a_min, cfg.a_max)
    reasons: List[str] = []
    if abs(limited_delta - delta) > 1e-9: reasons.append("jerk_limit")
    if abs(actual - (previous_a + limited_delta)) > 1e-9: reasons.append("accel_limit")
    return actual, (actual - previous_a) / cfg.dt, reasons


def hard_violations(vehicles: Mapping[str, VehicleState], task: TaskState,
                    scenario: Scenario, cfg: SimConfig) -> List[str]:
    del task
    violations: List[str] = []
    if _net_gap(vehicles["F"], vehicles["R"]) < 0.0: violations.append("FR_body_overlap")
    if _net_gap(vehicles["R"], vehicles["B"]) < 0.0: violations.append("RB_body_overlap")
    M = vehicles["M"]
    if M.merge_progress > 0.0 or M.y <= (cfg.lane_ramp + cfg.lane_main) / 2.0:
        for oid in ("F", "R", "B"):
            other = vehicles[oid]
            if _lateral_overlap(M, other) and abs(M.s - other.s) < (M.length + other.length) / 2.0:
                violations.append(f"M{oid}_body_overlap")
    for vid, v in vehicles.items():
        if v.s < -1.0 or v.s > scenario.road_max: violations.append(f"{vid}_road_exit")
        if v.y < -0.5 or v.y > cfg.lane_ramp + 0.5: violations.append(f"{vid}_lateral_exit")
    return violations


def _task_event(task: TaskState, t: float, vehicles: Mapping[str, VehicleState], controller: PairController,
                scenario: Scenario, cfg: SimConfig) -> Optional[str]:
    M = vehicles["M"]
    if task.phase == "PREPARE":
        check = physical_start_feasibility(vehicles, task, scenario, cfg)
        q_ready = controller.method in ("A", "E") or (controller.q["M"] >= cfg.merge_ready_threshold and controller.q["R"] >= cfg.merge_ready_threshold)
        if M.s > scenario.completion_s + 1e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "missed_window", True, "missed_window"
            return task.event
        if check["physical_ok"] and q_ready:
            task.phase, task.merge_start_time, task.event = "EXECUTE", t, "merge_started"
            return task.event
    elif task.phase == "EXECUTE":
        # The completing discrete step is part of the task boundary.  Crossing
        # 170 m is invalid even if the lateral progress reaches one in that
        # same step; this removes the old success-after-crossing bug.
        if M.s > scenario.completion_s + 1e-9:
            task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_exit_before_completion"
            return task.event
        if M.merge_progress >= 1.0:
            geom = geometry(vehicles, scenario, cfg)
            if geom["interval_width"] >= 0.0 and geom["lower"] <= M.s <= geom["upper"]:
                task.phase, task.completion_time, task.terminal, task.success, task.event = "SUCCESS_RELEASE", t, True, True, "merge_completed"
            else:
                task.phase, task.failure_type, task.terminal, task.event = "FAILURE_HANDLING", "started_but_failed", True, "merge_completed_outside_slot"
            return task.event
    return None


def _accumulator() -> Dict[str, Dict[str, float]]:
    phases = ("prepare", "execute", "success_release", "failure_handling")
    return {vid: {
        "speed_deficit_distance_m": 0.0, "speed_deficit_integral_m_s": 0.0,
        "accel_squared_integral": 0.0, "abs_jerk_integral": 0.0,
        "abs_nominal_actual_difference_integral": 0.0, "safety_correction_count": 0.0,
        "safety_correction_abs_integral": 0.0, "action_saturation_count": 0.0,
        "safety_recheck_violation_count": 0.0, "distance_m": 0.0,
        "min_net_gap_m": float("inf"), "min_FR_gap_m": float("inf"),
        "min_RB_gap_m": float("inf"), "min_MF_gap_m": float("inf"),
        "min_MR_gap_m": float("inf"),
        **{f"{p}_speed_deficit_distance_m": 0.0 for p in phases},
        **{f"{p}_distance_m": 0.0 for p in phases},
    } for vid in VEHICLE_IDS}


def _phase_key(phase: str) -> str:
    return {"PREPARE": "prepare", "EXECUTE": "execute", "SUCCESS_RELEASE": "success_release", "FAILURE_HANDLING": "failure_handling"}.get(phase, "failure_handling")


def _gap_snapshot(v: Mapping[str, VehicleState], active_merge: bool, cfg: SimConfig) -> Dict[str, float]:
    gaps = {"FR": _net_gap(v["F"], v["R"]), "RB": _net_gap(v["R"], v["B"])}
    if active_merge:
        gaps.update({"MF": _net_gap(v["F"], v["M"]), "MR": _net_gap(v["M"], v["R"])})
    return gaps


def _update_role_minima(accum: Dict[str, Dict[str, float]], gaps: Mapping[str, float], active_merge: bool) -> None:
    for vid in VEHICLE_IDS:
        candidates: List[float] = []
        if vid in ("F", "R"): candidates.append(gaps["FR"])
        if vid in ("R", "B"): candidates.append(gaps["RB"])
        if active_merge and vid in ("M", "F"): candidates.append(gaps["MF"])
        if active_merge and vid in ("M", "R"): candidates.append(gaps["MR"])
        if candidates:
            accum[vid]["min_net_gap_m"] = min(float(accum[vid]["min_net_gap_m"]), min(candidates))
        for key, gap in gaps.items():
            name = f"min_{key}_gap_m"
            for v in VEHICLE_IDS:
                if (key == "FR" and v in ("F", "R")) or (key == "RB" and v in ("R", "B")) or (key == "MF" and v in ("M", "F")) or (key == "MR" and v in ("M", "R")):
                    accum[v][name] = min(float(accum[v][name]), float(gap))


def run_episode(method: str, scenario_name: str, disturbance: str, init_variant: int,
                cfg: Optional[SimConfig] = None, seed: int = 0, ablation: str = "") -> SimulationResult:
    del seed
    cfg = cfg or SimConfig()
    scenario = build_scenario(scenario_name, init_variant)
    vehicles = make_vehicles(scenario, cfg)
    controller = make_controller(method, scenario, cfg, ablation=ablation)
    task = TaskState()
    run_id = f"{method}_{scenario_name}_{disturbance}_i{init_variant}" + (f"_{ablation}" if ablation else "")
    logs: List[Dict[str, object]] = []
    events: List[Dict[str, object]] = []
    accum = _accumulator()
    hard_count = road_count = safety_recheck_count = 0
    min_gap = float("inf")
    safety_abs: List[float] = []
    all_jerks: List[float] = []
    k_errors: List[float] = []
    space_abs: List[float] = []
    partner_abs: List[float] = []
    effective_q_abs: List[float] = []
    q_saturated = 0
    phase_feedback: Dict[str, List[float]] = {p: [] for p in ("PREPARE", "EXECUTE", "SUCCESS_RELEASE", "FAILURE_HANDLING")}
    target_effects: List[float] = []
    start_clock = time.perf_counter()
    n_steps = int(round(cfg.horizon / cfg.dt))
    last_diag: Dict[str, Mapping[str, object]] = {}
    last_actual: Dict[str, float] = {v: 0.0 for v in VEHICLE_IDS}
    last_jerk: Dict[str, float] = {v: 0.0 for v in VEHICLE_IDS}

    for step in range(n_steps):
        t = step * cfg.dt
        snapshot = {vid: vehicles[vid].copy() for vid in VEHICLE_IDS}
        if task.phase == "PREPARE" and not task.startup_check_recorded and scenario.merge_zone[0] <= snapshot["M"].s:
            check = physical_start_feasibility(snapshot, task, scenario, cfg)
            events.append({"run_id": run_id, "t_s": t, "event": "startup_physical_check", "details": str(check)})
            task.startup_check_recorded = True
        controller_state_pre = controller.state_dict()
        outputs = controller.compute_pair(snapshot, task, cfg.dt)
        geom = geometry(snapshot, scenario, cfg)
        a_nom = {"M": float(outputs["M"]["a_nom"]), "R": float(outputs["R"]["a_nom"]),
                 "F": environment_accel(snapshot["F"], t, scenario, disturbance, cfg),
                 "B": idm_accel(snapshot["B"], snapshot["R"], cfg)}
        active_merge = task.phase == "EXECUTE" or snapshot["M"].y <= cfg_midline(cfg)
        actual, safe_diag = execute_safe_actions(snapshot, a_nom, active_merge, cfg)
        adjusted = dict(actual)
        safety_reasons: Dict[str, List[str]] = {vid: [] for vid in VEHICLE_IDS}
        exec_reasons: Dict[str, List[str]] = {vid: [] for vid in VEHICLE_IDS}
        jerks: Dict[str, float] = {}
        for vid in VEHICLE_IDS:
            _, jerks[vid], bounded_reasons = execute_acceleration(snapshot[vid].a, actual[vid], cfg)
            exec_reasons[vid].extend(bounded_reasons)
            _, _, nominal_bounded_reasons = execute_acceleration(snapshot[vid].a, a_nom[vid], cfg)
            exec_reasons[vid].extend(f"nominal_{reason}" for reason in nominal_bounded_reasons)
            if abs(actual[vid] - a_nom[vid]) > 1e-9:
                safety_reasons[vid].append("joint_action_interval")
        predicted_violations = _predicted_safety_violations(snapshot, actual, active_merge, cfg)
        if not int(safe_diag["one_step_feasible"]):
            events.append({"run_id": run_id, "t_s": t, "event": "action_interval_infeasible", "details": str(safe_diag)})
        all_jerks.extend(abs(jerks[vid]) for vid in VEHICLE_IDS)
        if predicted_violations:
            safety_recheck_count += len(predicted_violations)
            events.append({"run_id": run_id, "t_s": t, "event": "safety_recheck", "details": ";".join(predicted_violations)})

        phase_now = task.phase
        q_diag = {"M": outputs["M"], "R": outputs["R"]}
        for vid in VEHICLE_IDS:
            d = q_diag[vid] if vid in ("M", "R") else {
                "q": 0.0, "qdot_nominal": 0.0, "qdot_base": 0.0, "qdot_space": 0.0,
                "qdot_partner": 0.0, "qdot_clipped": 0.0, "K_to_partner": 0.0,
                "W_to_partner": 0.0, "z_self": 0.0, "z_other": 0.0,
                "target_gap": outputs["R"]["target_gap"], "target_gap_process": outputs["R"]["target_gap_process"],
                "gap_error": outputs["R"]["gap_error"], "s_ref": "", "s_ref_dot": "", "s_ref_ddot": "",
                "reference_kind": outputs["R"]["reference_kind"], "target_time": "", "grad_phi_space": 0.0,
                "target_endpoint": outputs["R"].get("target_endpoint", ""),
                "window_opening_accel": outputs["R"].get("window_opening_accel", 0.0),
                "post_front_id": outputs["R"].get("post_front_id", "F"),
                "e_M0": outputs["M"]["e_M0"], "interval_width": geom["interval_width"], "target_gap_floor": geom["target_gap_floor"],
            }
            safety_delta = adjusted[vid] - a_nom[vid]
            reasons = list(safety_reasons[vid]) + list(exec_reasons[vid])
            logs.append({
                "run_id": run_id, "t_s": round(t, 6), "method": method, "ablation": ablation,
                "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
                "s_m": snapshot[vid].s, "y_m": snapshot[vid].y, "v_mps": snapshot[vid].v, "a_mps2": snapshot[vid].a,
                "q": d["q"], "qdot_nominal_s-1": d["qdot_nominal"], "qdot_base_s-1": d["qdot_base"],
                "qdot_space_s-1": d["qdot_space"], "qdot_partner_s-1": d["qdot_partner"], "qdot_clipped_s-1": d["qdot_clipped"],
                "K_to_partner_s-1": d["K_to_partner"], "W_to_partner": d["W_to_partner"], "z_self": d["z_self"], "z_other": d["z_other"],
                "g_FR_m": geom["g_fr"], "target_gap_physical_m": d["target_gap"], "target_gap_process_m": d["target_gap_process"],
                "gap_error_m": d["gap_error"], "lower_m": geom["lower"], "upper_m": geom["upper"], "interval_width_m": geom["interval_width"],
                "s_ref_m": d["s_ref"], "s_ref_dot_mps": d["s_ref_dot"], "s_ref_ddot_mps2": d["s_ref_ddot"],
                "reference_kind": d["reference_kind"], "target_time_s": d["target_time"], "merge_progress": snapshot[vid].merge_progress,
                "target_endpoint_m": d.get("target_endpoint", ""), "window_opening_accel_mps2": d.get("window_opening_accel", 0.0),
                "post_front_id": d.get("post_front_id", "F"),
                "ctrl_pre_q_M": controller_state_pre.get("q_M", 0.0), "ctrl_pre_q_R": controller_state_pre.get("q_R", 0.0),
                "ctrl_pre_e_M0": controller_state_pre.get("e_M0", ""), "ctrl_pre_c0": controller_state_pre.get("c0", ""),
                "ctrl_pre_prev_k_mr": controller_state_pre.get("prev_k_mr", cfg.k_total / 2.0),
                "ctrl_pre_prev_target_time": controller_state_pre.get("prev_target_time", 3.5),
                "task_phase": phase_now, "a_nom_mps2": a_nom[vid], "a_safety_target_mps2": adjusted[vid], "a_actual_mps2": actual[vid],
                "jerk_mps3": jerks[vid], "nominal_actual_difference_mps2": actual[vid] - a_nom[vid],
                "safety_correction_mps2": safety_delta, "safety_correction_applied": int(bool(safety_reasons[vid])),
                "action_saturation_applied": int(bool(exec_reasons[vid])), "action_correction_reason": ";".join(reasons),
                "F_disturbance_mps2": disturbance_accel(t, disturbance) if vid == "F" else 0.0,
            })
            last_diag[vid], last_actual[vid], last_jerk[vid] = d, actual[vid], jerks[vid]
            ref_v = scenario.desired_speed
            deficit = max(0.0, ref_v - snapshot[vid].v) * cfg.dt
            pkey = _phase_key(phase_now)
            accum[vid]["speed_deficit_distance_m"] += deficit
            accum[vid]["speed_deficit_integral_m_s"] += deficit  # v0-compatible alias; units are metres.
            accum[vid][f"{pkey}_speed_deficit_distance_m"] += deficit
            accum[vid]["accel_squared_integral"] += actual[vid] ** 2 * cfg.dt
            accum[vid]["abs_jerk_integral"] += abs(jerks[vid]) * cfg.dt
            accum[vid]["abs_nominal_actual_difference_integral"] += abs(actual[vid] - a_nom[vid]) * cfg.dt
            if safety_reasons[vid]:
                accum[vid]["safety_correction_count"] += 1.0
                accum[vid]["safety_correction_abs_integral"] += abs(safety_delta) * cfg.dt
                safety_abs.append(abs(safety_delta))
            if exec_reasons[vid]: accum[vid]["action_saturation_count"] += 1.0
            if predicted_violations: accum[vid]["safety_recheck_violation_count"] += len(predicted_violations)
            if vid in ("M", "R"):
                phase_feedback[phase_now if phase_now in phase_feedback else "FAILURE_HANDLING"].append(abs(float(d["qdot_space"])) + abs(float(d["qdot_partner"])))
                space_abs.append(abs(float(d["qdot_space"])))
                partner_abs.append(abs(float(d["qdot_partner"])))
                effective_q_abs.append(abs(float(d["qdot_clipped"])))
            target_effects.append(abs(float(d["s_ref"]) - geom["c"]) if isinstance(d["s_ref"], (float, int)) else 0.0)
        if method in ("C", "D") and phase_now in ACTIVE_PHASES:
            k_errors.append(abs(float(outputs["M"]["K_to_partner"]) + float(outputs["R"]["K_to_partner"]) - cfg.k_total))
        q_saturated += int(float(outputs["M"]["qdot_nominal"]) != float(outputs["M"]["qdot_clipped"]) or float(outputs["R"]["qdot_nominal"]) != float(outputs["R"]["qdot_clipped"]))

        phase_before = task.phase
        old_s = {v: vehicles[v].s for v in VEHICLE_IDS}
        for vid in VEHICLE_IDS:
            v = vehicles[vid]
            v.s = v.s + v.v * cfg.dt + 0.5 * actual[vid] * cfg.dt * cfg.dt
            v.v = _clip(v.v + actual[vid] * cfg.dt, cfg.v_min, cfg.v_max)
            v.a = actual[vid]
            dist = v.s - old_s[vid]
            accum[vid]["distance_m"] += dist
            accum[vid][f"{_phase_key(phase_before)}_distance_m"] += dist
        if phase_before == "EXECUTE" or (task.phase == "FAILURE_HANDLING" and vehicles["M"].merge_progress > 0.0):
            vehicles["M"].merge_progress = _clip(vehicles["M"].merge_progress + cfg.dt / cfg.merge_duration, 0.0, 1.0)
        vehicles["M"].y = cfg.lane_ramp + (cfg.lane_main - cfg.lane_ramp) * vehicles["M"].merge_progress
        controller.commit(outputs, cfg.dt, task.terminal)

        violations = hard_violations(vehicles, task, scenario, cfg)
        if violations:
            hard_count += len(violations); road_count += sum("road_exit" in x or "lateral_exit" in x for x in violations)
            if not task.terminal or task.success:
                task.phase, task.failure_type, task.terminal, task.success = "FAILURE_HANDLING", "physical_constraint_violation", True, False
                events.append({"run_id": run_id, "t_s": t + cfg.dt, "event": "physical_constraint_violation", "details": ";".join(violations)})
        event = _task_event(task, t + cfg.dt, vehicles, controller, scenario, cfg)
        if event:
            events.append({"run_id": run_id, "t_s": t + cfg.dt, "event": event, "details": task.failure_type or ""})
        gaps = _gap_snapshot(vehicles, vehicles["M"].merge_progress > 0.0, cfg)
        _update_role_minima(accum, gaps, vehicles["M"].merge_progress > 0.0)
        min_gap = min(min_gap, min(gaps.values()))

    if not task.terminal:
        task.phase, task.failure_type, task.terminal = "TIMEOUT", "timeout", True
        events.append({"run_id": run_id, "t_s": cfg.horizon, "event": "timeout", "details": "observation_window_end"})
    # The terminal row carries the last real command/diagnostics and the actual
    # final state.  It is not a synthetic zero-control row.
    for vid in VEHICLE_IDS:
        v, d = vehicles[vid], last_diag.get(vid, {})
        logs.append({"run_id": run_id, "t_s": round(cfg.horizon, 6), "method": method, "ablation": ablation,
                     "scenario": scenario_name, "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid,
                     "s_m": v.s, "y_m": v.y, "v_mps": v.v, "a_mps2": v.a, "q": controller.q.get(vid, 0.0),
                     "qdot_nominal_s-1": d.get("qdot_nominal", 0.0), "qdot_base_s-1": d.get("qdot_base", 0.0),
                     "qdot_space_s-1": d.get("qdot_space", 0.0), "qdot_partner_s-1": d.get("qdot_partner", 0.0),
                     "qdot_clipped_s-1": d.get("qdot_clipped", 0.0), "K_to_partner_s-1": d.get("K_to_partner", 0.0),
                     "W_to_partner": d.get("W_to_partner", 0.0), "z_self": d.get("z_self", 0.0), "z_other": d.get("z_other", 0.0),
                     "g_FR_m": _net_gap(vehicles["F"], vehicles["R"]), "target_gap_physical_m": d.get("target_gap", physical_target_gap(scenario, cfg)),
                     "target_gap_process_m": d.get("target_gap_process", physical_target_gap(scenario, cfg)), "gap_error_m": d.get("gap_error", 0.0),
                     "lower_m": geometry(vehicles, scenario, cfg)["lower"], "upper_m": geometry(vehicles, scenario, cfg)["upper"],
                     "interval_width_m": geometry(vehicles, scenario, cfg)["interval_width"], "s_ref_m": d.get("s_ref", ""),
                     "s_ref_dot_mps": d.get("s_ref_dot", ""), "s_ref_ddot_mps2": d.get("s_ref_ddot", ""),
                     "reference_kind": d.get("reference_kind", ""), "target_time_s": d.get("target_time", ""),
                     "target_endpoint_m": d.get("target_endpoint", ""), "window_opening_accel_mps2": d.get("window_opening_accel", 0.0),
                     "post_front_id": d.get("post_front_id", current_r_leader(vehicles, task, cfg).vid),
                     "ctrl_pre_q_M": d.get("ctrl_pre_q_M", controller.q.get("M", 0.0)), "ctrl_pre_q_R": d.get("ctrl_pre_q_R", controller.q.get("R", 0.0)),
                     "ctrl_pre_e_M0": d.get("ctrl_pre_e_M0", controller.e_M0 if controller.e_M0 is not None else ""),
                     "ctrl_pre_c0": d.get("ctrl_pre_c0", controller.c0 if controller.c0 is not None else ""),
                     "ctrl_pre_prev_k_mr": d.get("ctrl_pre_prev_k_mr", controller.prev_k_mr),
                     "ctrl_pre_prev_target_time": d.get("ctrl_pre_prev_target_time", controller.prev_target_time),
                     "merge_progress": v.merge_progress, "task_phase": task.phase, "a_nom_mps2": d.get("a_nom", v.a),
                     "a_safety_target_mps2": v.a, "a_actual_mps2": v.a, "jerk_mps3": last_jerk[vid],
                     "nominal_actual_difference_mps2": v.a - float(d.get("a_nom", v.a)), "safety_correction_mps2": 0.0,
                     "safety_correction_applied": 0, "action_saturation_applied": 0, "action_correction_reason": "terminal_state",
                     "F_disturbance_mps2": 0.0, "final_state": 1})

    outcome = "completed" if task.success else (task.failure_type or task.phase.lower())
    for vid in VEHICLE_IDS:
        if not math.isfinite(accum[vid]["min_net_gap_m"]): accum[vid]["min_net_gap_m"] = float("nan")
        for name in ("min_FR_gap_m", "min_RB_gap_m", "min_MF_gap_m", "min_MR_gap_m"):
            if not math.isfinite(accum[vid][name]): accum[vid][name] = float("nan")
    compute_time = time.perf_counter() - start_clock
    metrics: Dict[str, object] = {
        "run_id": run_id, "method": method, "ablation": ablation, "scenario": scenario_name,
        "disturbance": disturbance, "init_variant": init_variant, "outcome": outcome,
        "success": int(task.success), "task_phase": task.phase, "failure_type": task.failure_type,
        "merge_start_time_s": "" if task.merge_start_time is None else task.merge_start_time,
        "completion_time_s": "" if task.completion_time is None else task.completion_time,
        "physical_target_gap_m": physical_target_gap(scenario, cfg), "physical_gap_floor_m": physical_gap_floor(scenario, cfg),
        "min_net_gap_m": min_gap, "hard_violation_count": hard_count, "road_violation_count": road_count,
        "safety_recheck_violation_count": safety_recheck_count,
        "safety_correction_count": int(sum(a["safety_correction_count"] for a in accum.values())),
        "action_saturation_count": int(sum(a["action_saturation_count"] for a in accum.values())),
        "safety_correction_abs_mean_mps2": float(np.mean(safety_abs)) if safety_abs else 0.0,
        "max_abs_jerk_mps3": max(all_jerks) if all_jerks else 0.0,
        "mean_abs_jerk_mps3": float(np.mean(all_jerks)) if all_jerks else 0.0,
        "compute_time_s": compute_time, "pair_gain_total_error_max_s-1": max(k_errors) if k_errors else 0.0,
        "space_feedback_abs_mean_s-1": float(np.mean(space_abs)) if space_abs else 0.0,
        "partner_feedback_abs_mean_s-1": float(np.mean(partner_abs)) if partner_abs else 0.0,
        "effective_q_delta_abs_mean_s-1": float(np.mean(effective_q_abs)) if effective_q_abs else 0.0,
        "q_saturation_fraction": q_saturated / max(n_steps, 1),
        "space_feedback_prepare_mean_s-1": float(np.mean(phase_feedback["PREPARE"])) if phase_feedback["PREPARE"] else 0.0,
        "space_feedback_execute_mean_s-1": float(np.mean(phase_feedback["EXECUTE"])) if phase_feedback["EXECUTE"] else 0.0,
        "space_feedback_failure_handling_mean_s-1": float(np.mean(phase_feedback["FAILURE_HANDLING"])) if phase_feedback["FAILURE_HANDLING"] else 0.0,
        "mean_spatial_reference_offset_m": float(np.mean(target_effects)) if target_effects else 0.0,
        "K_MR_mean_s-1": float(np.mean([float(r["K_to_partner_s-1"]) for r in logs if r["vehicle"] == "M"])) if method in ("C", "D") else 0.0,
        "K_RM_mean_s-1": float(np.mean([float(r["K_to_partner_s-1"]) for r in logs if r["vehicle"] == "R"])) if method in ("C", "D") else 0.0,
        "total_speed_deficit_distance_m": float(sum(a["speed_deficit_distance_m"] for a in accum.values())),
        "total_speed_deficit_integral_m_s": float(sum(a["speed_deficit_distance_m"] for a in accum.values())),
        "total_effort_integral_m2_s3": float(sum(a["accel_squared_integral"] for a in accum.values())),
    }
    vehicle_metrics = []
    for vid in VEHICLE_IDS:
        row = {"run_id": run_id, "method": method, "ablation": ablation, "scenario": scenario_name,
               "disturbance": disturbance, "init_variant": init_variant, "vehicle": vid}
        row.update(accum[vid]); vehicle_metrics.append(row)
    return SimulationResult(run_id, method, scenario_name, disturbance, init_variant, logs, events, metrics, vehicle_metrics)


def run_matrix(methods: Iterable[str], scenarios: Iterable[str], disturbances: Iterable[str],
               init_variants: Iterable[int], cfg: Optional[SimConfig] = None, ablation: str = "") -> List[SimulationResult]:
    return [run_episode(m, s, d, i, cfg=cfg, ablation=ablation)
            for m in methods for s in scenarios for d in disturbances for i in init_variants]
