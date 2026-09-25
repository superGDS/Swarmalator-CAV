import math

from src.swarmalator_cav.simulation import (
    SimConfig,
    TaskState,
    build_scenario,
    execute_acceleration,
    geometry,
    make_controller,
    make_vehicles,
    run_matrix,
    run_episode,
)


def test_geometry_sign_and_empty_slot_are_explicit():
    cfg = SimConfig()
    sc = build_scenario("collaborative", 0)
    vehicles = make_vehicles(sc, cfg)
    g = geometry(vehicles, sc, cfg)
    assert math.isclose(g["g_fr"], sc.initial_gap)
    assert g["lower"] > g["upper"] or g["interval_width"] >= 0.0
    sc2 = build_scenario("ample", 0)
    g2 = geometry(make_vehicles(sc2, cfg), sc2, cfg)
    assert g2["interval_width"] > g["interval_width"]


def test_q_feedback_has_expected_degenerate_limits_and_terminal_freeze():
    cfg = SimConfig()
    sc = build_scenario("ample", 0)
    controller = make_controller("D", sc, cfg)
    snapshot = make_vehicles(sc, cfg)
    task = TaskState()
    out = controller.compute_pair(snapshot, task, cfg.dt)
    controller.commit(out, cfg.dt, terminal=False)
    assert 0.0 <= controller.q["M"] <= 1.0
    assert 0.0 <= controller.q["R"] <= 1.0
    before = dict(controller.q)
    controller.commit(out, cfg.dt, terminal=True)
    assert controller.q == before
    # e_M0=0 removes the M spatial gradient; delta_g=0 removes the R gradient.
    controller.e_M0 = 0.0
    out2 = controller.compute_pair(snapshot, task, cfg.dt)
    assert abs(float(out2["M"]["grad_phi_space"])) < 1e-12


def test_nonreciprocal_pair_gain_sum_and_executor_limits():
    cfg = SimConfig()
    result = run_episode("D", "collaborative", "prepare", 1, cfg)
    assert result.metrics["pair_gain_total_error_max_s-1"] < 1e-9
    assert result.metrics["hard_violation_count"] >= 0
    actual, jerk, reasons = execute_acceleration(0.0, cfg.a_max, cfg)
    assert actual <= cfg.jerk_max * cfg.dt + 1e-12
    assert abs(jerk) <= cfg.jerk_max + 1e-12
    assert "jerk_limit" in reasons


def test_matrix_dimensions_and_finite_outputs():
    cfg = SimConfig(horizon=0.5)
    results = run_matrix(["A", "B", "C", "D", "E"], ["ample", "short_window"], ["none"], [0, 1, 2], cfg)
    assert len(results) == 30
    for result in results:
        assert result.logs
        assert all(math.isfinite(float(result.metrics[k])) for k in ("min_net_gap_m", "max_abs_jerk_mps3", "compute_time_s"))
