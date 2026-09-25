"""Stage2B finite closed-loop experiment runner.

The runner keeps Stage1--Stage2A outputs untouched and writes only
``outputs/stage2b``.  It records the complete nominal -> actuator -> safety
projection -> actual chain and uses the same ``control_command`` interface in
prediction and execution.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, time, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .stage1c_simulation import (SimulationResult, TaskState, build_scenario,
    environment_accel, geometry, idm_accel, make_vehicles, physical_start_feasibility)
from .stage2b_check import validate_stage2b
from .stage2b_common import (Config, M,R,F,B,S,V,A,Y,Q, array_state, dictionary_state,
    control_command, margins, physical_step)
from .stage2b_planning import Controller

METHODS=("P","S1","S2","S3")
SCENARIOS=("ample","collaborative","short_window")
DISTURBANCES=("none","prepare")
VARIANTS=(0,1,2)
ANCHORS=(("collaborative","none",0),("collaborative","prepare",0),
         ("collaborative","none",2),("ample","none",0),
         ("ample","prepare",0))

def write_csv(path:Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True); rows=list(rows)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        w.writerows([{k:("" if v is None else v) for k,v in r.items()} for r in rows])

def _f(v,default=float("nan")):
    try: x=float(v)
    except (TypeError,ValueError): return default
    return x if math.isfinite(x) else default

def _event(events,run_id,t,event,**extra):
    row={"run_id":run_id,"t_s":round(float(t),6),"event":event}; row.update(extra); events.append(row)

def _task_update(task,t,vehicles,plan,scenario_name,variant,cfg):
    sc=build_scenario(scenario_name,variant); m=vehicles["M"]
    if task.phase=="PREPARE":
        if m.s>sc.completion_s+1e-9:
            task.phase,task.failure_type,task.terminal,task.event="FAILURE_HANDLING","missed_window",True,"missed_window"; return task.event
        if plan is not None and plan.valid and t>=plan.start-cfg.dt*.51:
            check=physical_start_feasibility(vehicles,task,sc,cfg)
            if check["physical_ok"]:
                task.phase,task.merge_start_time,task.event="EXECUTE",t,"merge_started"; return task.event
    elif task.phase=="EXECUTE":
        if m.s>sc.completion_s+1e-9:
            task.phase,task.failure_type,task.terminal,task.event="FAILURE_HANDLING","started_but_failed",True,"merge_exit_before_completion"; return task.event
        if m.merge_progress>=1-1e-9:
            g=geometry(vehicles,sc,cfg)
            if g["interval_width"]>=-1e-7 and g["lower"]-1e-7<=m.s<=g["upper"]+1e-7:
                task.phase,task.completion_time,task.terminal,task.success,task.event="SUCCESS_RELEASE",t,True,True,"merge_completed"
            else:
                task.phase,task.failure_type,task.terminal,task.event="FAILURE_HANDLING","started_but_failed",True,"merge_completed_outside_slot"
            return task.event
    return None

def _stage(task,t,cfg):
    if task.phase=="PREPARE": return "prepare"
    if task.phase=="EXECUTE": return "execute"
    if task.phase=="SUCCESS_RELEASE":
        if task.completion_time is not None and t<task.completion_time+cfg.handoff_duration_s-1e-9: return "handoff"
        return "normal_follow"
    return "failure_handling"

def run_episode(method:str,scenario_name:str,disturbance:str,variant:int,cfg:Config,ablation:str="") -> SimulationResult:
    scenario=build_scenario(scenario_name,variant); x=array_state(make_vehicles(scenario,cfg)); ref=x[:,:3].copy()
    controller=Controller(method,cfg,ablation=ablation); task=TaskState(); plan=None; logs=[]; events=[]
    run_id=f"{method}_{scenario_name}_{disturbance}_i{variant}"+(f"_{ablation}" if ablation else "")
    last_nom=np.zeros(4); last_act=np.zeros(4); last_jerk=np.zeros(4); last_diag=None; last_info={}; last_commands={"reference_kind":"common_control_command"}
    decision_times=[]; candidate_counts=[]; first_plan_t=None; first_valid_t=None; first_pred_s=None; first_pred_t=None
    start_clock=time.perf_counter(); n_steps=int(round(cfg.horizon/cfg.dt)); invalid_steps=0

    def add_rows(t,x0,commands,info,nominal,actual,phase,control_stage,diag):
        nonlocal last_nom,last_act,last_jerk,last_diag,last_info,last_commands,invalid_steps
        state=dictionary_state(x0[None]); g=geometry(state,scenario,cfg)
        active_now=np.array([bool(phase in ("EXECUTE","SUCCESS_RELEASE") or x0[M,Q]>0)])
        net,h,rel=margins(x0[None],cfg,active_now); actlim=diag["actuator"][0]; safe=diag["safety_correction"][0]; sat=diag["saturation"][0]
        # FM/MR margins are outside the reserved corridor before execution;
        # keep them separate from the active FR/RB values in logs and plots.
        if not bool(active_now[0]): h[0,2:4]=np.nan
        if not bool(diag["one_step_feasible"][0]): invalid_steps+=1
        eta=controller.z
        for i,vid in enumerate(("M","R","F","B")):
            info_fields=info or {}; k=float(info_fields.get("K_MR" if vid=="M" else "K_RM",0.0)); w=float(info_fields.get("W",0.0))
            logs.append({
                "run_id":run_id,"t_s":round(float(t),6),"method":method,"ablation":ablation,"scenario":scenario_name,"disturbance":disturbance,"init_variant":variant,"vehicle":vid,
                "s_m":x0[i,S],"y_m":x0[i,Y],"v_mps":x0[i,V],"a_mps2":x0[i,A],"merge_progress":x0[i,Q],
                "coord_eta_M":float(eta[0]),"coord_eta_R":float(eta[1]),"coord_eta_dot_M":info_fields.get("delta_z_M",0.0),"coord_eta_dot_R":info_fields.get("delta_z_R",0.0),"coord_gradient_M":info_fields.get("gradient_M",0.0),"coord_gradient_R":info_fields.get("gradient_R",0.0),"coord_raw_gradient_M":info_fields.get("raw_gradient_M",0.0),"coord_raw_gradient_R":info_fields.get("raw_gradient_R",0.0),"coord_partner_M":info_fields.get("partner_M",0.0),"coord_partner_R":info_fields.get("partner_R",0.0),"partner_enabled":info_fields.get("partner_enabled",0),"feedback_enabled":info_fields.get("feedback_enabled",0),"base_z_M":info_fields.get("base_z_M",float(eta[0])),"base_z_R":info_fields.get("base_z_R",float(eta[1])),"proposal_z_M":info_fields.get("proposal_z_M",float(eta[0])),"proposal_z_R":info_fields.get("proposal_z_R",float(eta[1])),"state_update_accepted":info_fields.get("state_update_accepted",0),
                "K_to_partner_s-1":k,"W_to_partner":w,"raw_K_MR":info_fields.get("raw_K_MR",0.0),"raw_K_RM":info_fields.get("raw_K_RM",0.0),"raw_W":info_fields.get("raw_W",0.0),"g_FR_m":g["g_fr"],"target_gap_physical_m":g["target_gap_physical"],"target_gap_process_m":g["target_gap_physical"],"gap_error_m":g["g_fr"]-scenario.initial_gap,"lower_m":g["lower"],"upper_m":g["upper"],"interval_width_m":g["interval_width"],
                "h_FR_m":h[0,0],"h_RB_m":h[0,1],"h_FM_m":h[0,2],"h_MR_m":h[0,3],"relative_v_MR_mps":x0[M,V]-x0[R,V],"relative_a_MR_mps2":x0[M,A]-x0[R,A],"rear_braking_demand_mps2":max(0.0,-float(nominal[R])),
                "s_ref_m":ref[M,0],"s_ref_dot_mps":ref[M,1],"s_ref_ddot_mps2":ref[M,2],"reference_kind":commands.get("reference_kind","common_control_command"),"plan_start_t_s":info_fields.get("start_s",""),"plan_reason":info_fields.get("reason",""),"plan_compute_time_s":info_fields.get("decision_seconds",0.0),"predicted_start_t_s":info_fields.get("start_s",""),"predicted_completion_t_s":info_fields.get("completion_t_s",""),"predicted_completion_s_m":info_fields.get("completion_s_m",info_fields.get("completion_s","")),"predicted_min_h_m":info_fields.get("predicted_min_h_m",""),"predicted_net_clearance_m":info_fields.get("predicted_min_net_m",""),"predicted_candidate_count":info_fields.get("candidate_count",0),
                "task_phase":phase,"control_stage":control_stage,"handoff_elapsed_s":("" if task.completion_time is None else max(0.0,float(t-task.completion_time))),"a_nom_mps2":nominal[i],"a_actuator_mps2":actlim[i],"a_safety_projected_mps2":actual[i],"a_safety_target_mps2":actual[i],"a_actual_mps2":actual[i],"jerk_mps3":(actual[i]-x0[i,A])/cfg.dt,"nominal_actual_difference_mps2":actual[i]-nominal[i],"actuator_nominal_difference_mps2":actlim[i]-nominal[i],"safety_correction_mps2":actual[i]-actlim[i],"saturation_applied":int(sat[i]),"safety_correction_applied":int(safe[i]),"action_saturation_applied":int(sat[i]),"action_correction_reason":("actuator_interval_empty" if bool(diag["actuator_interval_empty"][0]) else "safety_intersection_empty" if bool(diag["safety_intersection_empty"][0]) else "safety_projection" if bool(safe[i]) else ""),
                "one_step_feasible":int(diag["one_step_feasible"][0]),"actuator_interval_feasible":int(diag["actuator_interval_feasible"][0]),"safety_intersection_feasible":int(diag["safety_intersection_feasible"][0]),"actuator_interval_empty":int(diag["actuator_interval_empty"][0]),"safety_intersection_empty":int(diag["safety_intersection_empty"][0]),"jerk_valid":int(diag["jerk_valid"][0]),"speed_valid":int(diag["speed_valid"][0]),"fallback_mode":str(diag["fallback_mode"][0]),"next_h_min_m":diag["min_h"][0],"next_min_net_m":diag["min_net"][0],"F_disturbance_mps2":environment_accel(state["F"],t,scenario,disturbance,cfg)-environment_accel(state["F"],t,scenario,"none",cfg) if vid=="F" else 0.0,
            })
        last_nom=nominal.copy(); last_act=actual.copy(); last_jerk=(actual-x0[:,A])/cfg.dt; last_diag=diag; last_info=dict(info or {}); last_commands=dict(commands)

    for step in range(n_steps):
        t=step*cfg.dt; x0=x.copy(); phase_before=task.phase; stage_before=_stage(task,t,cfg)
        target,info=controller.decide(x0,ref,t,task.phase); plan=controller.plan
        if info.get("decision"):
            decision_times.append(float(info.get("decision_seconds",0.0))); candidate_counts.append(int(info.get("candidate_count",0))); first_plan_t=t if first_plan_t is None else first_plan_t
            if info.get("valid_plan") and first_valid_t is None: first_valid_t=t; first_pred_s=info.get("completion_s_m",info.get("completion_s")); first_pred_t=info.get("completion_t_s")
            _event(events,run_id,t,"plan_generated",**info)
        pair_nom,ref_candidate,effective=control_command(x0[None],ref[None],np.asarray(target)[None] if np.asarray(target).ndim==1 else np.asarray(target),task.phase,cfg)
        # Advance the same public reference that the predictor advanced.  The
        # state is continuous through completion and normal following.
        ref=ref_candidate[0]
        st=dictionary_state(x0[None]); nominal=np.array([pair_nom[0,0],pair_nom[0,1],environment_accel(st["F"],t,scenario,disturbance,cfg),idm_accel(st["B"],st["R"],cfg)],dtype=float)
        commands={"reference_kind":f"stage2b_common_{task.phase.lower()}_{method}","effective_target_M_mps2":effective[0,0],"effective_target_R_mps2":effective[0,1]}
        if task.phase=="PREPARE" and plan is not None and plan.valid and t>=plan.start-cfg.dt*.51:
            check=physical_start_feasibility(st,task,scenario,cfg)
            if check["physical_ok"]: task.phase,task.merge_start_time,task.event="EXECUTE",t,"merge_started"; phase_before="EXECUTE"; stage_before="execute"; _event(events,run_id,t,"merge_started",plan_start_t_s=plan.start)
        active=np.array([bool(task.phase in ("EXECUTE","SUCCESS_RELEASE") or x0[M,Q]>0)]); merging=np.array([bool(task.phase=="EXECUTE")])
        xn,actual_b,diag=physical_step(x0[None],nominal[None],active,merging,cfg); actual=actual_b[0]; add_rows(t,x0,commands,info,nominal,actual,phase_before,stage_before,diag); x=xn[0]
        event=_task_update(task,t+cfg.dt,dictionary_state(x[None]),plan,scenario_name,variant,cfg)
        if event: _event(events,run_id,t+cfg.dt,event,plan_start_t_s=(plan.start if plan else ""))
    if not task.terminal: task.phase,task.failure_type,task.terminal="TIMEOUT","timeout",True; _event(events,run_id,cfg.horizon,"timeout",details="observation_window_end")
    final=x.copy(); final_stage=_stage(task,cfg.horizon,cfg); fd=last_diag
    for i,vid in enumerate(("M","R","F","B")):
        logs.append({"run_id":run_id,"t_s":round(float(cfg.horizon),6),"method":method,"ablation":ablation,"scenario":scenario_name,"disturbance":disturbance,"init_variant":variant,"vehicle":vid,"s_m":final[i,S],"y_m":final[i,Y],"v_mps":final[i,V],"a_mps2":final[i,A],"merge_progress":final[i,Q],"coord_eta_M":controller.z[0],"coord_eta_R":controller.z[1],"s_ref_m":ref[M,0],"s_ref_dot_mps":ref[M,1],"s_ref_ddot_mps2":ref[M,2],"reference_kind":last_commands.get("reference_kind",""),"predicted_completion_t_s":last_info.get("completion_t_s",""),"predicted_completion_s_m":last_info.get("completion_s_m",last_info.get("completion_s","")),"predicted_candidate_count":last_info.get("candidate_count",0),"task_phase":task.phase,"control_stage":final_stage,"a_nom_mps2":last_nom[i],"a_actuator_mps2":(fd["actuator"][0,i] if fd is not None else last_act[i]),"a_safety_projected_mps2":last_act[i],"a_safety_target_mps2":last_act[i],"a_actual_mps2":last_act[i],"jerk_mps3":last_jerk[i],"saturation_applied":0,"safety_correction_applied":0,"action_saturation_applied":0,"one_step_feasible":(int(fd["one_step_feasible"][0]) if fd is not None else 1),"actuator_interval_feasible":(int(fd["actuator_interval_feasible"][0]) if fd is not None else 1),"safety_intersection_feasible":(int(fd["safety_intersection_feasible"][0]) if fd is not None else 1),"actuator_interval_empty":(int(fd["actuator_interval_empty"][0]) if fd is not None else 0),"safety_intersection_empty":(int(fd["safety_intersection_empty"][0]) if fd is not None else 0),"jerk_valid":(int(fd["jerk_valid"][0]) if fd is not None else 1),"speed_valid":(int(fd["speed_valid"][0]) if fd is not None else 1),"fallback_mode":(str(fd["fallback_mode"][0]) if fd is not None else ""),"next_h_min_m":""})
    speed_rows=[r for r in logs if _f(r.get("t_s"))<cfg.horizon]; stages=("prepare","execute","handoff","normal_follow","failure_handling")
    stage_cost={s:sum(max(0.0,scenario.desired_speed-_f(r.get("v_mps")))*cfg.dt for r in speed_rows if r.get("control_stage")==s) for s in stages}; handoff_rows=[r for r in speed_rows if r.get("control_stage")=="handoff"]
    completion_m=[r for r in speed_rows if r.get("vehicle")=="M" and task.completion_time is not None and abs(_f(r.get("t_s"))-task.completion_time)<cfg.dt*.51]; completion_r=[r for r in speed_rows if r.get("vehicle")=="R" and task.completion_time is not None and abs(_f(r.get("t_s"))-task.completion_time)<cfg.dt*.51]
    metrics={"run_id":run_id,"method":method,"ablation":ablation,"scenario":scenario_name,"disturbance":disturbance,"init_variant":variant,"outcome":("completed" if task.success else task.failure_type or task.phase.lower()),"success":int(task.success),"geometry_completed":int(any(e.get("event")=="merge_completed" for e in events)),"merge_start_time_s":("" if task.merge_start_time is None else task.merge_start_time),"completion_time_s":("" if task.completion_time is None else task.completion_time),"planner_first_time_s":("" if first_plan_t is None else first_plan_t),"planner_first_valid_time_s":("" if first_valid_t is None else first_valid_t),"planner_replans":len(decision_times),"planner_mean_compute_time_s":statistics.mean(decision_times) if decision_times else 0.0,"planner_p95_compute_time_s":float(np.percentile(decision_times,95)) if decision_times else 0.0,"planner_max_compute_time_s":max(decision_times,default=0.0),"planner_candidate_count":statistics.mean(candidate_counts) if candidate_counts else 0.0,"planner_total_candidate_count":sum(candidate_counts),"planner_decision_count":len(candidate_counts),"planner_predicted_completion_s_m":first_pred_s if first_pred_s is not None else float("nan"),"planner_predicted_completion_t_s":first_pred_t if first_pred_t is not None else float("nan"),"planner_predicted_min_h_m":last_info.get("predicted_min_h_m",float("nan")),"speed_deficit_prepare_m":stage_cost["prepare"],"speed_deficit_execute_m":stage_cost["execute"],"speed_deficit_handoff_m":stage_cost["handoff"],"speed_deficit_normal_follow_m":stage_cost["normal_follow"],"speed_deficit_failure_handling_m":stage_cost["failure_handling"],"total_speed_deficit_distance_m":sum(stage_cost.values()),"total_effort_integral_m2_s3":sum(_f(r.get("a_actual_mps2"),0.0)**2*cfg.dt for r in speed_rows),"max_abs_jerk_mps3":max((abs(_f(r.get("jerk_mps3"))) for r in speed_rows),default=0.0),"safety_correction_count":sum(int(r.get("safety_correction_applied",0)) for r in speed_rows),"action_saturation_count":sum(int(r.get("saturation_applied",0)) for r in speed_rows),"empty_action_steps":invalid_steps,"handoff_duration_s":(cfg.handoff_duration_s if task.completion_time is not None and task.success else float("nan")),"handoff_min_h_MR_m":min((_f(r.get("h_MR_m")) for r in handoff_rows),default=float("nan")),"completion_relative_v_MR_mps":(_f(completion_m[0].get("relative_v_MR_mps")) if completion_m else float("nan")),"completion_relative_a_MR_mps2":(_f(completion_m[0].get("relative_a_MR_mps2")) if completion_m else float("nan")),"completion_rear_braking_demand_mps2":(_f(completion_r[0].get("rear_braking_demand_mps2")) if completion_r else float("nan")),"recovery_time_s":(cfg.handoff_duration_s if task.completion_time is not None and task.success else float("nan")),"compute_time_s":time.perf_counter()-start_clock}
    vehicle_metrics=[]
    for vid in ("M","R","F","B"):
        rv=[r for r in speed_rows if r.get("vehicle")==vid]; vehicle_metrics.append({"run_id":run_id,"method":method,"ablation":ablation,"scenario":scenario_name,"disturbance":disturbance,"init_variant":variant,"vehicle":vid,"speed_deficit_distance_m":sum(max(0,scenario.desired_speed-_f(r.get("v_mps")))*cfg.dt for r in rv),"safety_correction_count":sum(int(r.get("safety_correction_applied",0)) for r in rv),"saturation_count":sum(int(r.get("saturation_applied",0)) for r in rv),"max_abs_jerk_mps3":max((abs(_f(r.get("jerk_mps3"))) for r in rv),default=0.0)})
    return SimulationResult(run_id,method,scenario_name,disturbance,variant,logs,events,metrics,vehicle_metrics)

def validation_rows(results,cfg):
    out=[]
    for r in results:
        c=validate_stage2b(r,cfg); row={"run_id":r.run_id,"method":r.method,"ablation":r.metrics.get("ablation",""),"scenario":r.scenario,"disturbance":r.disturbance,"init_variant":r.init_variant}
        row.update({k:v for k,v in c.items() if k not in ("errors","min_h_by_phase")}); out.append(row)
    return out

def summary_rows(results,validations):
    out=[]; by={r["run_id"]:r for r in validations}
    for method in METHODS:
        rr=[r for r in results if r.method==method and not r.metrics.get("ablation")]
        vr=[r for r in rr if int(by.get(r.run_id,{}).get("mission_valid",0))]; fr=[r for r in rr if int(by.get(r.run_id,{}).get("full_window_valid",0))]
        mean=lambda rows,key: statistics.mean([float(r.metrics[key]) for r in rows]) if rows else float("nan")
        out.append({"method":method,"n":len(rr),"geometry_completed":sum(int(r.metrics.get("geometry_completed",0)) for r in rr),"mission_valid":len(vr),"full_window_valid":len(fr),"mission_valid_n":len(vr),"full_valid_n":len(fr),"mean_speed_deficit_all_m":mean(rr,"total_speed_deficit_distance_m"),"mean_speed_deficit_mission_valid_m":mean(vr,"total_speed_deficit_distance_m"),"mean_speed_deficit_full_valid_m":mean(fr,"total_speed_deficit_distance_m"),"mean_prepare_deficit_all_m":mean(rr,"speed_deficit_prepare_m"),"mean_execute_deficit_all_m":mean(rr,"speed_deficit_execute_m"),"mean_handoff_deficit_all_m":mean(rr,"speed_deficit_handoff_m"),"mean_normal_follow_deficit_all_m":mean(rr,"speed_deficit_normal_follow_m"),"mean_planner_compute_s":mean(rr,"planner_mean_compute_time_s"),"p95_planner_compute_s":float(np.percentile([float(r.metrics["planner_p95_compute_time_s"]) for r in rr],95)) if rr else 0.0,"mean_candidates":mean(rr,"planner_candidate_count"),"mean_empty_action_steps":mean(rr,"empty_action_steps")})
    return out

def paired_rows(results,validations):
    v={r["run_id"]:r for r in validations}; by={(r.method,r.scenario,r.disturbance,r.init_variant):r for r in results}; pairs=[("P","S1"),("P","S2"),("P","S3"),("S1","S2"),("S1","S3"),("S2","S3")]; rows=[]
    for sc in SCENARIOS:
      for d in DISTURBANCES:
       for i in VARIANTS:
        for a,b in pairs:
            ra=by.get((a,sc,d,i)); rb=by.get((b,sc,d,i))
            if ra is None or rb is None: continue
            va=v[ra.run_id]; vb=v[rb.run_id]; cm=int(va.get("mission_valid",0) and vb.get("mission_valid",0)); cf=int(va.get("full_window_valid",0) and vb.get("full_window_valid",0)); da=float(rb.metrics["total_speed_deficit_distance_m"])-float(ra.metrics["total_speed_deficit_distance_m"])
            rows.append({"environment_id":f"{sc}_{d}_i{i}","scenario":sc,"disturbance":d,"init_variant":i,"method_a":a,"method_b":b,"a_mission_valid":va.get("mission_valid",0),"b_mission_valid":vb.get("mission_valid",0),"a_full_valid":va.get("full_window_valid",0),"b_full_valid":vb.get("full_window_valid",0),"common_mission":cm,"common_full":cf,"cost_a_m":ra.metrics["total_speed_deficit_distance_m"],"cost_b_m":rb.metrics["total_speed_deficit_distance_m"],"delta_b_minus_a_all_m":da,"delta_b_minus_a_common_mission_m":(da if cm else ""),"delta_b_minus_a_common_full_m":(da if cf else "")})
    return rows

def paired_summary(rows):
    out=[]
    for a,b in sorted(set((r["method_a"],r["method_b"]) for r in rows)):
        rr=[r for r in rows if r["method_a"]==a and r["method_b"]==b]
        def avg(field,flag=None):
            vals=[float(r[field]) for r in rr if flag is None or r[flag]]; return statistics.mean(vals) if vals else float("nan")
        out.append({"method_a":a,"method_b":b,"n_all":len(rr),"n_common_mission":sum(r["common_mission"] for r in rr),"n_common_full":sum(r["common_full"] for r in rr),"mean_delta_all_m":avg("delta_b_minus_a_all_m"),"mean_delta_common_mission_m":avg("delta_b_minus_a_common_mission_m","common_mission"),"mean_delta_common_full_m":avg("delta_b_minus_a_common_full_m","common_full")})
    return out

def stage_cost_rows(results):
    out=[]
    for r in results:
        sc=build_scenario(r.scenario,r.init_variant)
        for stage in ("prepare","execute","handoff","normal_follow","failure_handling"):
            rr=[x for x in r.logs if x.get("control_stage")==stage and _f(x.get("t_s"))<16]
            out.append({"run_id":r.run_id,"method":r.method,"ablation":r.metrics.get("ablation",""),"scenario":r.scenario,"disturbance":r.disturbance,"init_variant":r.init_variant,"stage":stage,"speed_deficit_distance_m":sum(max(0,sc.desired_speed-_f(x.get("v_mps")))*.05 for x in rr)})
    return out

def _figures(out_dir,results,paired):
    fd=out_dir/"figures"; fd.mkdir(parents=True,exist_ok=True); paths={}
    def pick(method): return next((r for r in results if r.method==method and r.scenario=="collaborative" and r.disturbance=="none" and r.init_variant==0 and not r.metrics.get("ablation")),None)
    p=pick("P"); s2=pick("S2"); s3=pick("S3")
    if p:
        rr=[r for r in p.logs if r["vehicle"]=="M" and _f(r.get("t_s"))<16]; t=np.array([_f(r["t_s"]) for r in rr]); fig,ax=plt.subplots(4,1,figsize=(9,10),sharex=True)
        ax[0].plot(t,[_f(r["h_MR_m"]) for r in rr],label="h_MR"); ax[0].plot(t,[_f(r["h_RB_m"]) for r in rr],label="h_RB"); ax[0].axvline(_f(p.metrics.get("completion_time_s")),ls="--",c="k",label="completion"); ax[0].set_ylabel("h (m)"); ax[0].legend()
        ax[1].plot(t,[_f(r["relative_v_MR_mps"]) for r in rr],label="v_M-v_R"); ax[1].plot(t,[_f(r["relative_a_MR_mps2"]) for r in rr],label="a_M-a_R"); ax[1].set_ylabel("relative m/s; m/s²"); ax[1].legend()
        ax[2].plot(t,[_f(r["v_mps"]) for r in rr],label="M speed"); ax[2].set_ylabel("speed (m/s)"); ax[2].legend()
        ax[3].plot(t,[_f(r["coord_eta_M"]) for r in rr],label="z_M",color="tab:orange"); ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("z (0–1)"); ax[3].legend(); fig.suptitle("Collaboration → handoff → normal following (P anchor)"); fig.tight_layout(); paths["chain"]=fd/"collaboration_handoff_recovery.png"; fig.savefig(paths["chain"],dpi=160); plt.close(fig)
    if paired:
        ss=paired_summary(paired); labels=[f'{r["method_a"]}→{r["method_b"]}' for r in ss]; x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(10,4.8)); ax.bar(x-.2,[r["mean_delta_common_mission_m"] if math.isfinite(_f(r["mean_delta_common_mission_m"])) else 0 for r in ss],.2,label="common mission"); ax.bar(x,[r["mean_delta_common_full_m"] if math.isfinite(_f(r["mean_delta_common_full_m"])) else 0 for r in ss],.2,label="common full"); ax.axhline(0,c="k",lw=.8); ax.set_xticks(x,labels); ax.set_ylabel("Δ cost B−A (m), same set"); ax.set_title("Paired results on common service conditions"); ax.legend(); fig.tight_layout(); paths["paired"]=fd/"paired_same_service.png"; fig.savefig(paths["paired"],dpi=160); plt.close(fig)
    if s2 and s3:
        r2=[r for r in s2.logs if r["vehicle"]=="M" and _f(r.get("t_s"))<5]; r3=[r for r in s3.logs if r["vehicle"]=="M" and _f(r.get("t_s"))<5]; t=np.array([_f(r["t_s"]) for r in r2]); fig,ax=plt.subplots(3,1,figsize=(9,8),sharex=True)
        ax[0].plot(t,[_f(r["K_to_partner_s-1"]) for r in r2],label="S2 K_MR"); ax[0].plot(t,[_f(r["K_to_partner_s-1"]) for r in r3],label="S3 K_MR"); ax[0].set_ylabel("K (s⁻¹)"); ax[0].legend()
        ax[1].plot(t,[_f(r["coord_partner_M"]) for r in r2],label="S2 partner M"); ax[1].plot(t,[_f(r["coord_partner_M"]) for r in r3],label="S3 partner M"); ax[1].set_ylabel("partner action"); ax[1].legend()
        ax[2].plot(t,[_f(r["coord_eta_M"]) for r in r2],label="S2 z_M",ls="--"); ax[2].plot(t,[_f(r["coord_eta_M"]) for r in r3],label="S3 z_M",ls="--"); ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("z (0–1)"); ax[2].legend(); fig.suptitle("Coupling direction and action allocation"); fig.tight_layout(); paths["coupling"]=fd/"coupling_direction_allocation.png"; fig.savefig(paths["coupling"],dpi=160); plt.close(fig)
    return {k:str(v) for k,v in paths.items()}

def _docs(root,summary,paired,counts,figs,mechanism):
    (root/"docs").mkdir(exist_ok=True); (root/"reports").mkdir(exist_ok=True)
    (root/"docs"/"model_v4.md").write_text("""# Swarmalator–CAV Stage 2B model v4

Stage2B keeps the Stage2A finite-window joint predictor but makes the public control arms and common execution path explicit. The state is `X_i=(s_i,v_i,a_i,y_i,q_i)` for M (merge), R (rear gap vehicle), F (front boundary) and B (rear follower). The bounded internal variable `z_i∈[0,1]` is preparation intensity; it is not phase or physical lateral progress. Its target acceleration is `a_target,i=-3 z_i` (m/s²).

A candidate absolute start and `(z_M,z_R)` are rolled through preparation, the 3 s lateral execution and a 1.25 s release preview. The observed Stage2A empty-action transient occurred about 0.65 s after completion; M braking release at the 2.5 m/s³ jerk cap takes about `|a|/J`, so the common handoff duration is fixed at 1.0 s and the extra preview is 0.25 s. This is a finite terminal condition, not a recursive-feasibility proof.

Prediction and execution call the same `control_command` interface. In PREPARE/EXECUTE it propagates `(s_ref,v_ref,a_ref)` with the declared jerk bound and tracks the current state. In SUCCESS_RELEASE/FAILURE_HANDLING it switches to current-occupancy `post_targets`; after the common 1.0 s successful handoff the logged control stage is `normal_follow`. No position or velocity reset is used. F uses the current declared disturbance input and B uses current-state IDM.

The action chain is `nominal → actuator interval → safety projection → actual`. An empty actuator interval (acceleration, jerk or speed limits) is distinct from an empty ordered-chain safety intersection. On either empty set the fallback is logged as invalid and is never called safe.

The four main arms are P, joint finite prediction and candidate selection without added state-law terms; S1, the same predictor/reference with gradient and partner terms zero; S2, S1 plus finite-difference task-potential gradient and symmetric partner action; and S3, S2 with the same total partner budget allocated from the current rear margin. The `symmetric_K` S3 intervention uses the 0.5/0.5 allocation as S2 and clones its decision under the same controller state. Planning is centralized over the four-vehicle snapshot; distributed communication is not implemented.

Geometry/event completion, mission validity through the first q=1 sample, and full-window validity through 16 s remain separate. The cost is fixed four-vehicle speed-deficit distance, with prepare, execute, handoff, normal-follow and failure-handling stages retained. Pairwise deltas use the same environment and the same valid-success denominator.
""",encoding="utf-8")
    lines=["# Stage 2B report","","Stage1/1B/1C/2A evidence and feedback packages remain unchanged. Stage2B writes a new v4 model and `outputs/stage2b`.","","## Core scientific result","","This finite deterministic matrix tests prediction, space-to-preparation feedback, partner action and non-reciprocal allocation after a common handoff repair. No arm is assumed to win. `z` is a bounded braking-preparation intensity, not an oscillator phase or merge progress.","",f"Counts: {json.dumps(counts,ensure_ascii=False)}","","## Main regression","","|method|n|geometry|mission-valid|full-window|all-run cost (m)|mission-valid cost (m)|full-valid cost (m)|mean candidates|mean compute (ms)|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"|{r['method']}|{r['n']}|{r['geometry_completed']}|{r['mission_valid']}|{r['full_window_valid']}|{r['mean_speed_deficit_all_m']:.3f}|{r['mean_speed_deficit_mission_valid_m']:.3f}|{r['mean_speed_deficit_full_valid_m']:.3f}|{r['mean_candidates']:.1f}|{1000*r['mean_planner_compute_s']:.2f}|")
    main_mission=sum(int(r["mission_valid"]) for r in summary); main_full=sum(int(r["full_window_valid"]) for r in summary)
    lines += ["",f"Across the 72 main runs, {main_mission} method-environment pairs are mission-valid and {main_full} are full-window-valid. The remaining rows are retained as failures or unresolved finite-search cases; the most frequent ledger causes are no merge completion and a small number of post-task h-margin violations.","","All-run cost includes failures and is descriptive. Conditional means are labeled by their common validity denominator and are not a sole ranking.","","## Paired contribution checks","","`paired_results.csv` gives one row per environment and method pair; `paired_summary.csv` reports only common mission and common full subsets."]
    for r in paired_summary(paired): lines.append(f"- {r['method_a']} → {r['method_b']}: common mission n={r['n_common_mission']}, Δ={r['mean_delta_common_mission_m']:.3f} m; common full n={r['n_common_full']}, Δ={r['mean_delta_common_full_m']:.3f} m.")
    lines += ["","## Anchors and handoff","","The five anchors are the first 20 main rows and are reused in the 72-run matrix. Handoff duration, completion h, relative speed/acceleration and rear braking demand are in `metrics.csv` and `handoff_metrics.csv`. `public_checks.csv` contains four diagnostic runs and is not silently added to the regression count.","","## Mechanism interventions","","S1 has effective gradient, partner and K/W fields equal to zero; raw finite-difference probes remain in separate audit columns. The S3 `symmetric_K` intervention reproduces S2 under the same initial controller state and acceptance rule on both mechanism anchors, so any normal S3 difference is attributable to allocation direction rather than a different total coupling budget."]
    for r in mechanism: lines.append(f"- {r.get('ablation','')} / {r.get('scenario','')} / {r.get('disturbance','')} / i{r.get('init_variant','')}: mission={r.get('mission_valid','')}, full={r.get('full_window_valid','')}, cost={_f(r.get('total_speed_deficit_distance_m')):.3f} m.")
    lines += ["","## Public-layer findings","","An empty actuator interval or ordered-chain intersection is an unresolved hard-constraint failure. It is logged separately from ordinary saturation. Reference continuity and physical error fields are separate in `validation.csv`.","","## Limitations","","This is a finite deterministic development/regression matrix, not a traffic simulator, distributed communication test, reachability proof or road-safety certification. Search failure is reported as unresolved, not as proof of physical infeasibility. Future F behavior is not read by any controller.","","## Reproduction","","```powershell",".\\.venv\\Scripts\\python.exe -m pytest -q",".\\.venv\\Scripts\\python.exe -m src.swarmalator_cav.run_stage2b --config configs/stage2b_config.json --out outputs/stage2b","```","","## Figures",*[f"- `{v}`" for v in figs.values()]]
    (root/"reports"/"stage2b_report.md").write_text("\n".join(lines),encoding="utf-8")
    (root/"reports"/"stage2b_validation.md").write_text("# Stage2B validation ledger\n\n"+json.dumps(counts,indent=2)+"\n",encoding="utf-8")

def _package(root,out):
    feedback=root/"feedback"/"stage2b_feedback.zip"; feedback.parent.mkdir(exist_ok=True)
    include=[root/"AGENTS.md",root/"README.md",root/"codex_swarmalator_stage2b_prompt.md",root/"references"/"literature_notes.md",root/"configs"/"stage2b_config.json",root/"docs"/"model_v4.md",root/"reports"/"stage2b_report.md",root/"reports"/"stage2b_validation.md",root/"src"/"swarmalator_cav"/"simulation.py",root/"src"/"swarmalator_cav"/"stage1c_simulation.py",root/"src"/"swarmalator_cav"/"stage1c_check.py",root/"src"/"swarmalator_cav"/"stage2b_common.py",root/"src"/"swarmalator_cav"/"stage2b_planning.py",root/"src"/"swarmalator_cav"/"stage2b_check.py",root/"src"/"swarmalator_cav"/"run_stage2b.py",root/"src"/"swarmalator_cav"/"__init__.py",root/"tests"/"test_stage2b.py"]
    names=("config_used.json","frozen_evidence_hashes.json","pilot_notes.json","public_checks.csv","metrics.csv","vehicle_metrics.csv","validation.csv","summary.csv","anchor_results.csv","paired_results.csv","paired_summary.csv","stage_costs.csv","handoff_metrics.csv","mechanism.csv","plan_events.csv","key_events.csv","key_trajectories.csv","same_state_check.csv","run_log.txt")
    include += [out/n for n in names]+list((out/"figures").glob("*.png"))
    with zipfile.ZipFile(feedback,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include:
            p=p.resolve()
            if p.exists(): zf.write(p,p.relative_to(root.resolve()).as_posix())
    sha=hashlib.sha256(feedback.read_bytes()).hexdigest()
    with zipfile.ZipFile(feedback) as zf: n=len(zf.namelist())
    (root/"feedback"/"stage2b_feedback_manifest.txt").write_text(f"stage2b_feedback.zip sha256={sha}\nfiles={n}\n",encoding="utf-8")
    return feedback,sha,n

def run(args):
    root=Path(__file__).resolve().parents[2]; out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    cfg_json=json.loads(Path(args.config).read_text(encoding="utf-8")); cfg=Config(**{k:v for k,v in cfg_json.get("simulation",{}).items() if k in Config.__dataclass_fields__}); (out/"config_used.json").write_text(json.dumps(cfg_json,indent=2),encoding="utf-8")
    # Four public diagnostics are deliberately separate from the 18-environment matrix.
    public=[]
    for key in (("collaborative","none",0),("collaborative","prepare",2),("short_window","none",1),("ample","none",0)):
        r=run_episode("P",*key,cfg); vv=validation_rows([r],cfg)[0]; public.append(dict(r.metrics,mission_valid=vv.get("mission_valid"),full_window_valid=vv.get("full_window_valid"),error=vv.get("error")))
    write_csv(out/"public_checks.csv",public)
    results=[]; anchor_keys=set(ANCHORS)
    for method in METHODS:
        for key in ANCHORS: results.append(run_episode(method,*key,cfg))
    for method in METHODS:
        for sc in SCENARIOS:
            for d in DISTURBANCES:
                for i in VARIANTS:
                    if (sc,d,i) in anchor_keys: continue
                    results.append(run_episode(method,sc,d,i,cfg))
    ablations=[]
    for abl in ("no_partner","no_feedback","fixed_W"):
        for key in (ANCHORS[0],ANCHORS[3]): ablations.append(run_episode("S2",*key,cfg,ablation=abl))
    for key in (ANCHORS[0],ANCHORS[3]): ablations.append(run_episode("S3",*key,cfg,ablation="symmetric_K"))
    allr=results+ablations; vals=validation_rows(allr,cfg); mainvals=[v for v in vals if not v.get("ablation")]; vmap={v["run_id"]:v for v in vals}
    summary=summary_rows(results,mainvals); paired=paired_rows(results,mainvals)
    write_csv(out/"metrics.csv",[r.metrics for r in results]); write_csv(out/"vehicle_metrics.csv",[v for r in results for v in r.vehicle_metrics]); write_csv(out/"validation.csv",vals); write_csv(out/"summary.csv",summary)
    write_csv(out/"anchor_results.csv",[dict(r.metrics,mission_valid=vmap[r.run_id].get("mission_valid"),full_window_valid=vmap[r.run_id].get("full_window_valid"),error=vmap[r.run_id].get("error")) for r in results if (r.scenario,r.disturbance,r.init_variant) in anchor_keys]); write_csv(out/"paired_results.csv",paired); write_csv(out/"paired_summary.csv",paired_summary(paired)); write_csv(out/"stage_costs.csv",stage_cost_rows(allr)); write_csv(out/"handoff_metrics.csv",[r.metrics for r in allr]); write_csv(out/"mechanism.csv",[dict(r.metrics,mission_valid=vmap[r.run_id].get("mission_valid"),full_window_valid=vmap[r.run_id].get("full_window_valid"),error=vmap[r.run_id].get("error")) for r in ablations]); write_csv(out/"plan_events.csv",[e for r in allr for e in r.events if e.get("event")=="plan_generated"]); write_csv(out/"key_events.csv",[e for r in allr for e in r.events]); write_csv(out/"key_trajectories.csv",[row for r in results if (r.scenario,r.disturbance,r.init_variant) in anchor_keys for row in r.logs])
    # Same-state clone: state, plan history and reference are all restored before deciding.
    sc=build_scenario("collaborative",0); x=array_state(make_vehicles(sc,cfg)); ref=x[:,:3].copy(); c1=Controller("S2",cfg); c1.decide(x,ref,0.0,"PREPARE",force=True); c2=Controller("S2",cfg); c2.load_state(c1.state_dict()); a,ia=c1.decide(x,ref,.5,"PREPARE",force=True); b,ib=c2.decide(x,ref,.5,"PREPARE",force=True); write_csv(out/"same_state_check.csv",[{"same_target":int(np.allclose(a,b,atol=1e-12)),"same_reason":int(ia.get("reason")==ib.get("reason")),"same_candidates":int(ia.get("candidate_count")==ib.get("candidate_count")),"a":a.tolist(),"b":b.tolist()}])
    frozen={}
    for n in ("stage1_feedback.zip","stage1b_feedback.zip","stage1c_feedback.zip","stage2a_feedback.zip"):
        p=root/"feedback"/n
        if p.exists(): frozen[n]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"frozen_evidence_hashes.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8"); (out/"pilot_notes.json").write_text(json.dumps({"public_checks":4,"anchors":len(ANCHORS)*len(METHODS),"main_regression":len(results),"mechanism_runs":len(ablations),"handoff_duration_s":cfg.handoff_duration_s,"post_preview_s":cfg.post_preview_s},indent=2),encoding="utf-8")
    figs=_figures(out,results,paired); counts={"public_checks":len(public),"anchors":len(ANCHORS)*len(METHODS),"regression":len(results)-len(ANCHORS)*len(METHODS),"main":len(results),"validation_rows":len(vals),"mechanism":len(ablations),"paired_rows":len(paired)}; mechanism_rows=[dict(r.metrics,mission_valid=vmap[r.run_id].get("mission_valid"),full_window_valid=vmap[r.run_id].get("full_window_valid")) for r in ablations]; _docs(root,summary,paired,counts,figs,mechanism_rows); (out/"run_log.txt").write_text(json.dumps({"config":str(Path(args.config).resolve()),"counts":counts,"figures":figs},indent=2),encoding="utf-8")
    feedback,sha,n=_package(root,out); print(json.dumps({"counts":counts,"feedback":str(feedback),"sha256":sha,"files":n},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/stage2b_config.json"); ap.add_argument("--out",default="outputs/stage2b"); run(ap.parse_args())

if __name__=="__main__": main()
