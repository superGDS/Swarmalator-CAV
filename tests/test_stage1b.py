import math

from src.swarmalator_cav.stage1b_simulation import (
    SimConfig,
    TaskState,
    _moving_reference_control,
    build_scenario,
    geometry,
    make_controller,
    make_vehicles,
    min_forward_distance,
    physical_gap_floor,
    physical_start_feasibility,
    run_episode,
)


def test_uniform_moving_reference_has_zero_error_acceleration():
    # A reference travelling at 18 m/s with matching state must not be slowed
    # merely because the reference position is moving.
    a = _moving_reference_control(100.0, 18.0, 100.0, 18.0, 0.0)
    assert abs(a) < 1e-12


def test_common_target_gap_is_physical_and_shared():
    cfg = SimConfig()
    assert math.isclose(physical_gap_floor(build_scenario("short_window"), cfg), 21.4)
    for method in "ABCDE":
        sc = build_scenario("short_window")
        c = make_controller(method, sc, cfg)
        out = c.compute_pair(make_vehicles(sc, cfg), TaskState(), cfg.dt)
        assert math.isclose(float(out["M"]["target_gap"]), physical_gap_floor(sc, cfg))
        assert math.isclose(float(out["R"]["target_gap"]), physical_gap_floor(sc, cfg))


def test_start_check_exposes_remaining_distance_and_jerk_bound():
    cfg = SimConfig()
    sc = build_scenario("ample", 0)
    vehicles = make_vehicles(sc, cfg)
    check = physical_start_feasibility(vehicles, TaskState(), sc, cfg)
    assert check["remaining_distance_m"] == sc.completion_s - vehicles["M"].s
    assert check["min_forward_distance_m"] > 0.0
    assert math.isclose(check["min_forward_distance_m"], min_forward_distance(vehicles["M"], cfg.merge_duration, cfg))


def test_post_failure_lateral_path_is_continuous_and_logged():
    # This run deliberately stops the horizon shortly after start; it must not
    # insert a zero-control/frozen lateral state as the terminal row.
    result = run_episode("E", "ample", "none", 2, SimConfig(horizon=5.0))
    m_rows = [r for r in result.logs if r["vehicle"] == "M"]
    assert m_rows[-1]["final_state"] == 1
    assert float(m_rows[-1]["y_m"]) >= 0.0
    assert all(math.isfinite(float(r["a_actual_mps2"])) for r in m_rows)
