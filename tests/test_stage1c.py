import math

from src.swarmalator_cav.run_stage1c import independent_trajectory_check
from src.swarmalator_cav.stage1c_simulation import (
    PairController,
    SimConfig,
    TaskState,
    VehicleState,
    _task_event,
    build_scenario,
    geometry,
    make_controller,
    make_vehicles,
    process_gap_gradient,
    run_episode,
)


def test_completion_step_crossing_boundary_is_rejected():
    cfg = SimConfig()
    sc = build_scenario("ample")
    vehicles = make_vehicles(sc, cfg)
    vehicles["M"].s = 170.0001
    vehicles["M"].merge_progress = 1.0
    task = TaskState(phase="EXECUTE")
    controller = make_controller("A", sc, cfg)
    event = _task_event(task, 4.0, vehicles, controller, sc, cfg)
    assert event == "merge_exit_before_completion"
    assert not task.success


def test_old_false_witness_endpoint_is_rejected_by_independent_checker():
    cfg = SimConfig()
    sc = build_scenario("ample")
    v = make_vehicles(sc, cfg)
    rows = []
    for t in (0.0, 0.05):
        for vid, state in v.items():
            rows.append({"t_s": t, "vehicle": vid, "scenario": "ample", "disturbance": "none", "init_variant": 0,
                         "s_m": 172.5 if vid == "M" and t > 0 else state.s, "y_m": state.y,
                         "v_mps": state.v, "a_mps2": state.a, "a_actual_mps2": 0.0,
                         "a_nom_mps2": 0.0, "merge_progress": 1.0 if vid == "M" else 0.0,
                         "task_phase": "EXECUTE"})
    result = independent_trajectory_check(rows, cfg, "ample", "none", 0)
    assert result["mission_ok"] == 0
    assert "endpoint_crossed_completion" in result["error"] or "endpoint_outside_slot" in result["error"]


def test_process_gradient_uses_new_target_span_and_is_zero_when_independent():
    cfg = SimConfig()
    ample = build_scenario("ample")
    short = build_scenario("short_window")
    assert abs(process_gap_gradient(5.0, ample, cfg)) < 1e-12
    expected = -5.0 * max(0.0, 21.4 - short.initial_gap) / (cfg.ell_g ** 2)
    assert math.isclose(process_gap_gradient(5.0, short, cfg), expected)
    ctrl = PairController("C", ample, cfg)
    out = ctrl.compute_pair(make_vehicles(ample, cfg), TaskState(), cfg.dt)
    assert abs(float(out["R"]["grad_phi_space"])) < 1e-12


def test_process_gradient_matches_finite_difference_of_potential():
    cfg = SimConfig()
    sc = build_scenario("short_window")
    gap = 17.0
    q = 0.37
    delta = max(0.0, 21.4 - sc.initial_gap)
    def phi(qr):
        err = gap - (sc.initial_gap + delta * qr)
        return 0.5 * err * err / (cfg.ell_g * cfg.ell_g)
    eps = 1e-6
    finite = (phi(q + eps) - phi(q - eps)) / (2 * eps)
    analytic = process_gap_gradient(gap - (sc.initial_gap + delta * q), sc, cfg)
    assert math.isclose(finite, analytic, rel_tol=1e-6, abs_tol=1e-9)


def test_e_zero_error_current_reference_is_zero_command():
    cfg = SimConfig()
    sc = build_scenario("ample")
    vehicles = make_vehicles(sc, cfg)
    vehicles["F"].v = vehicles["R"].v = vehicles["M"].v = 18.0
    g = geometry(vehicles, sc, cfg)
    vehicles["M"].s = g["c"]
    ctrl = PairController("E", sc, cfg)
    out = ctrl.compute_pair(vehicles, TaskState(), cfg.dt)
    assert abs(float(out["M"]["a_nom"])) < 1e-9
    assert math.isclose(float(out["M"]["s_ref"]), vehicles["M"].s)
    assert abs(float(out["M"]["s_ref_ddot"])) < 1e-9


def test_same_state_full_replay_matches_saved_command():
    cfg = SimConfig(horizon=5.0)
    result = run_episode("C", "ample", "none", 0, cfg)
    rows = [r for r in result.logs if r["vehicle"] == "M" and r.get("task_phase") == "EXECUTE"]
    assert rows
    row = rows[0]
    sc = build_scenario("ample")
    snap_rows = {r["vehicle"]: r for r in result.logs if r["t_s"] == row["t_s"]}
    snapshot = {vid: VehicleState(vid, float(snap_rows[vid]["s_m"]), float(snap_rows[vid]["y_m"]), float(snap_rows[vid]["v_mps"]), float(snap_rows[vid]["a_mps2"]), merge_progress=float(snap_rows[vid]["merge_progress"])) for vid in ("M", "R", "F", "B")}
    ctrl = PairController("C", sc, cfg)
    ctrl.load_state({"q_M": row["ctrl_pre_q_M"], "q_R": row["ctrl_pre_q_R"], "e_M0": row["ctrl_pre_e_M0"], "c0": row["ctrl_pre_c0"], "prev_k_mr": row["ctrl_pre_prev_k_mr"], "prev_target_time": row["ctrl_pre_prev_target_time"]})
    out = ctrl.compute_pair(snapshot, TaskState(phase="EXECUTE"), cfg.dt)
    assert abs(float(out["M"]["a_nom"]) - float(row["a_nom_mps2"])) < 1e-9
    assert abs(float(out["R"]["qdot_clipped"]) - float(next(r for r in snap_rows.values() if r["vehicle"] == "R")["qdot_clipped_s-1"])) < 1e-9


def test_fixed_w_and_partner_delete_are_distinct_on_same_state():
    cfg = SimConfig()
    sc = build_scenario("short_window")
    snap = make_vehicles(sc, cfg)
    task = TaskState(phase="EXECUTE")
    full = make_controller("C", sc, cfg); fixed = make_controller("C", sc, cfg, ablation="fixed_W"); deleted = make_controller("C", sc, cfg, ablation="no_partner")
    state = {"q_M": 0.2, "q_R": 0.8, "e_M0": None, "c0": None, "prev_k_mr": cfg.k_total / 2.0, "prev_target_time": 3.5}
    for c in (full, fixed, deleted): c.load_state(state)
    a = full.compute_pair(snap, task, cfg.dt); b = fixed.compute_pair(snap, task, cfg.dt); d = deleted.compute_pair(snap, task, cfg.dt)
    assert abs(float(b["M"]["W_to_partner"]) - cfg.fixed_partner_weight) < 1e-12
    assert abs(float(d["M"]["qdot_partner"])) < 1e-12
    assert abs(float(b["M"]["qdot_partner"]) - float(d["M"]["qdot_partner"])) > 1e-8


def test_post_success_r_selects_actual_m_front():
    cfg = SimConfig()
    sc = build_scenario("ample")
    v = make_vehicles(sc, cfg)
    v["M"].s = 110.0; v["R"].s = 100.0; v["M"].y = 0.0; v["R"].y = 0.0; v["M"].merge_progress = 1.0
    c = PairController("A", sc, cfg)
    out = c.compute_pair(v, TaskState(phase="SUCCESS_RELEASE"), cfg.dt)
    assert out["R"]["post_front_id"] == "M"


def test_jerk_and_checker_postconditions_are_recorded():
    cfg = SimConfig(horizon=0.5)
    r = run_episode("D", "ample", "none", 0, cfg)
    assert max(abs(float(x["jerk_mps3"])) for x in r.logs if x.get("final_state") != 1) <= cfg.jerk_max + 1e-9
    check = independent_trajectory_check(r, cfg)
    assert "error" in check and "min_h_full_m" in check
