import copy
import math

from src.swarmalator_cav.stage1c_check import independent_trajectory_check
from src.swarmalator_cav.stage1c_simulation import SimConfig, build_scenario, make_vehicles
from src.swarmalator_cav.stage2a_check import reference_continuity_check, validate_stage2a
from src.swarmalator_cav.stage2a_common import Config, array_state
from src.swarmalator_cav.stage2a_planning import Controller, Predictor
from src.swarmalator_cav.run_stage2a import run_episode


def test_predictive_plan_from_current_observation_finds_collaborative_anchor():
    cfg = Config()
    scenario = build_scenario("collaborative", 0)
    x = array_state(make_vehicles(scenario, cfg))
    ref = x[:, :3].copy()
    controller = Controller("P", cfg)
    target, info = controller.decide(x, ref, 0.0, "PREPARE", force=True)
    assert info["valid_plan"] == 1
    assert info["candidate_count"] >= 16
    assert math.isfinite(float(info["completion_s"]))
    assert math.isfinite(float(info["completion_s_m"]))
    assert math.isfinite(float(info["completion_t_s"]))
    assert float(info["completion_s_m"]) > 150.0
    assert 0.0 < float(info["completion_t_s"]) < 16.0
    assert target.shape == (2,)
    assert controller.plan is not None and controller.plan.start >= 0.0


def test_plan_state_clone_reproduces_next_decision():
    cfg = Config()
    scenario = build_scenario("collaborative", 0)
    x = array_state(make_vehicles(scenario, cfg))
    ref = x[:, :3].copy()
    first = Controller("S2", cfg)
    first.decide(x, ref, 0.0, "PREPARE", force=True)
    state = first.state_dict()
    clone = Controller("S2", cfg)
    clone.load_state(state)
    a, ia = first.decide(x, ref, 0.5, "PREPARE", force=True)
    b, ib = clone.decide(x, ref, 0.5, "PREPARE", force=True)
    assert all(abs(float(x1) - float(x2)) < 1.0e-10 for x1, x2 in zip(a, b))
    assert ia["reason"] == ib["reason"]
    assert ia["candidate_count"] == ib["candidate_count"]


def test_reference_endpoint_is_continuous_for_s2():
    cfg = Config(horizon=6.0)
    result = run_episode("S2", "collaborative", "none", 0, cfg)
    continuity = reference_continuity_check(result, cfg)
    assert continuity["reference_ok"] == 1
    assert continuity["missing_reference_rows"] == 0
    assert continuity["max_reference_acceleration_mps2"] <= 15.0


def test_checker_rejects_corrupted_action_as_mission_invalid():
    cfg = Config(horizon=6.0)
    result = run_episode("P", "collaborative", "none", 0, cfg)
    rows = [dict(row) for row in result.logs]
    for row in rows:
        if row["vehicle"] == "M" and float(row["t_s"]) < 1.0:
            row["a_actual_mps2"] = 100.0
            break
    check = independent_trajectory_check(rows, cfg, "collaborative", "none", 0)
    assert check["mission_ok"] == 0
    assert "accel_bound_M" in check["error"] or "dynamics_s_M" in check["error"]


def test_anchor_is_mission_valid_and_has_separate_geometry_label():
    cfg = Config(horizon=6.0)
    result = run_episode("P", "collaborative", "none", 0, cfg)
    check = validate_stage2a(result, cfg)
    assert check["geometry_completed"] == 1
    assert check["mission_valid"] == 1
    # The common post-task follower is checked separately from task validity.
    # This anchor completes the declared maneuver validly; the diagnostic
    # full-window label must remain visible rather than being silently folded
    # into geometry completion.
    assert check["full_window_valid"] in (0, 1)
