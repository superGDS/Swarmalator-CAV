import math
import numpy as np
from src.swarmalator_cav.stage1c_check import independent_trajectory_check
from src.swarmalator_cav.stage1c_simulation import build_scenario, make_vehicles
from src.swarmalator_cav.stage2b_check import reference_continuity_check, validate_stage2b
from src.swarmalator_cav.stage2b_common import Config, array_state, control_command, physical_step
from src.swarmalator_cav.stage2b_planning import Controller, Predictor
from src.swarmalator_cav.run_stage2b import run_episode


def test_control_arm_switches_are_explicit():
    cfg=Config(); sc=build_scenario("collaborative",0); x=array_state(make_vehicles(sc,cfg)); ref=x[:,:3].copy()
    infos={}
    for method in ("P","S1","S2","S3"):
        c=Controller(method,cfg); _,infos[method]=c.decide(x,ref,0.0,"PREPARE",force=True)
    assert infos["S1"]["partner_enabled"]==0 and infos["S1"]["feedback_enabled"]==0
    assert abs(float(infos["S1"]["partner_M"]))<1e-12 and abs(float(infos["S1"]["feedback_M"]))<1e-12
    assert infos["S2"]["partner_enabled"]==1 and infos["S2"]["feedback_enabled"]==1


def test_symmetric_k_s3_clones_s2_on_same_state():
    cfg=Config(); sc=build_scenario("collaborative",0); x=array_state(make_vehicles(sc,cfg)); ref=x[:,:3].copy()
    s2=Controller("S2",cfg); s3=Controller("S3",cfg,ablation="symmetric_K")
    a,ia=s2.decide(x,ref,0.0,"PREPARE",force=True); b,ib=s3.decide(x,ref,0.0,"PREPARE",force=True)
    assert np.allclose(a,b,atol=1e-10); assert abs(float(ia["K_MR"])-float(ib["K_MR"]))<1e-12; assert abs(float(ia["K_RM"])-float(ib["K_RM"]))<1e-12


def test_stop_boundary_preserves_jerk_and_marks_empty_interval():
    cfg=Config(); sc=build_scenario("ample",0); x=array_state(make_vehicles(sc,cfg)); x[:,1]=0.01; x[:,2]=-4.0
    nominal=np.zeros((1,4)); out,actual,diag=physical_step(x[None],nominal,np.array([False]),np.array([False]),cfg)
    assert bool(diag["actuator_interval_empty"][0]); assert not bool(diag["one_step_feasible"][0]); assert bool(diag["jerk_valid"][0]); assert np.all(out[:,:,1]>=cfg.v_min-1e-9)

def test_frozen_candidate_prediction_and_execution_share_first_action():
    cfg=Config(); sc=build_scenario("collaborative",0); x=array_state(make_vehicles(sc,cfg)); ref=x[:,:3].copy(); z=np.array([[2/3,1/3]],dtype=float)
    pred=Predictor(cfg); ev=pred.evaluate(x,ref,0.0,[1.5],z,keep_trace=True); trace=ev["trace"][0]
    pair,nref,effective=control_command(x[None],ref[None],(-3*z),"PREPARE",cfg)
    nominal=np.zeros((1,4)); nominal[0,:2]=pair[0]
    from src.swarmalator_cav.stage2b_common import idm
    nominal[0,3]=idm(x[None],3,1,cfg)[0]
    assert np.allclose(trace["nominal"],nominal[0],atol=1e-10)


def test_common_reference_interface_is_used_after_handoff():
    cfg=Config(horizon=6.0); result=run_episode("P","collaborative","none",0,cfg)
    stages={str(r.get("control_stage")) for r in result.logs}
    assert "handoff" in stages and "normal_follow" in stages
    active=[r for r in result.logs if r["vehicle"]=="M" and r["task_phase"] in ("PREPARE","EXECUTE")]
    assert active and all(math.isfinite(float(r["s_ref_dot_mps"])) for r in active)
    assert all("a_actuator_mps2" in r and "safety_correction_applied" in r for r in result.logs)
    handoff=[r for r in result.logs if r["vehicle"]=="M" and r.get("control_stage")=="handoff"]
    assert handoff and max(abs(float(r["jerk_mps3"])) for r in handoff)<cfg.jerk_max+1e-8


def test_reference_and_physical_errors_are_separate():
    cfg=Config(horizon=6.0); result=run_episode("S2","collaborative","none",0,cfg); c=validate_stage2b(result,cfg)
    assert "error" in c and "reference_error" in c and "combined_error" in c
    assert c["reference_reference_ok"]==1


def test_checker_rejects_corrupted_action():
    cfg=Config(horizon=6.0); result=run_episode("P","collaborative","none",0,cfg); rows=[dict(r) for r in result.logs]
    for row in rows:
        if row["vehicle"]=="M" and float(row["t_s"])<1.0: row["a_actual_mps2"]=100.0; break
    check=independent_trajectory_check(rows,cfg,"collaborative","none",0)
    assert check["mission_ok"]==0 and ("accel_bound_M" in check["error"] or "dynamics_s_M" in check["error"])
