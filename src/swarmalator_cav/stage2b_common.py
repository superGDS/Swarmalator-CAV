"""Stage2B shared physical layer.

Stage2B is a clean copy of the Stage2A public model.  It keeps the Stage1C
state convention (M, R, F, B; s, v, a, y, q) while making the preparation,
release and fallback paths explicit.  The predictor and the runner call the
same ``control_command`` and ``physical_step`` functions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .stage1c_simulation import SimConfig, VehicleState

IDS = ('M','R','F','B')
M,R,F,B = range(4)
S,V,A,Y,Q = range(5)
POST_CONTROL_PHASES = ('SUCCESS_RELEASE','FAILURE_HANDLING','NORMAL_FOLLOW')

@dataclass
class Config(SimConfig):
    refresh_s: float = 0.5
    # 0.65 s was the observed post-completion empty-action time in Stage2A;
    # 1.0 s covers that transient and the M braking-release time |a|/J.
    handoff_duration_s: float = 1.0
    handoff_preview_s: float = 0.25
    post_preview_s: float = 1.25
    max_prepare_s: float = 2.5
    reference_kp: float = 0.10
    reference_kv: float = 0.40
    z_step: float = 0.05
    gradient_gain: float = 0.018
    partner_total: float = 0.32
    fixed_W: float = 0.40
    intrinsic_M: float = 0.010
    intrinsic_R: float = 0.016
    planner_wait_values: tuple = (0.0,0.5,1.0,1.5,2.0,2.5)
    planner_action_values: tuple = (1.0,2.0/3.0,1.0/3.0,0.0)
    planner_max_candidates: int = 160
    planner_action_effort_weight: float = 0.04
    planner_wait_weight: float = 0.20


def array_state(snapshot):
    return np.array([[snapshot[k].s,snapshot[k].v,snapshot[k].a,snapshot[k].y,snapshot[k].merge_progress] for k in IDS],dtype=float)


def dictionary_state(x):
    if x.ndim == 3: x=x[0]
    return {k:VehicleState(k,float(x[i,S]),float(x[i,Y]),float(x[i,V]),float(x[i,A]),merge_progress=float(x[i,Q])) for i,k in enumerate(IDS)}


def idm(x,follower,leader,cfg):
    v=x[:,follower,V]
    gap=np.maximum(x[:,leader,S]-x[:,follower,S]-4.8,0.1)
    dv=v-x[:,leader,V]
    star=cfg.min_base_gap+cfg.time_headway*v+v*dv/(2*np.sqrt(1.3*2.2))
    return np.clip(1.3*(1-(v/20)**4-(np.maximum(star,0.1)/gap)**2),cfg.a_min,cfg.a_max)


def margins(x,cfg,active):
    active=np.asarray(active,dtype=bool)
    if active.ndim==0: active=np.full(len(x),bool(active),dtype=bool)
    pairs=((F,R),(R,B),(F,M),(M,R))
    net=np.stack([x[:,l,S]-x[:,f,S]-4.8 for l,f in pairs],axis=1)
    h=net-np.stack([cfg.min_base_gap+cfg.time_headway*x[:,f,V] for _,f in pairs],axis=1)
    relevant=np.stack([np.ones(len(x),bool),np.ones(len(x),bool),active,active],axis=1)
    return net,h,relevant


def slot(x,cfg):
    lower=x[:,R,S]+4.8+cfg.min_base_gap+cfg.time_headway*x[:,R,V]
    upper=x[:,F,S]-4.8-cfg.min_base_gap-cfg.time_headway*x[:,M,V]
    return lower,upper


def post_targets(x,cfg):
    """Common hand-off/normal-follow target based on current occupancy."""
    overlap=(np.abs(x[:,M,Y]-x[:,R,Y])<=1.9)&(x[:,M,S]>x[:,R,S])
    r=np.where(overlap,idm(x,R,M,cfg),idm(x,R,F,cfg))
    m=np.where((x[:,F,S]>x[:,M,S])&(np.abs(x[:,F,Y]-x[:,M,Y])<=1.9),
               idm(x,M,F,cfg),0.85*(18-x[:,M,V]))
    return np.stack([m,r],axis=1)


def reference_command(x,ref,target,cfg):
    """Propagate the public (s_ref,v_ref,a_ref) state without endpoint reset."""
    target=np.asarray(target,dtype=float)
    if target.ndim==1: target=target[None,:]
    ar=ref[:,:2,2]+np.clip(target-ref[:,:2,2],-cfg.jerk_max*cfg.dt,cfg.jerk_max*cfg.dt)
    ar=np.clip(ar,cfg.a_min,cfg.a_max)
    nominal=ar+cfg.reference_kp*(ref[:,:2,0]-x[:,:2,S])+cfg.reference_kv*(ref[:,:2,1]-x[:,:2,V])
    nr=ref.copy()
    nr[:,:2,0]+=ref[:,:2,1]*cfg.dt+0.5*ar*cfg.dt**2
    nr[:,:2,1]=np.clip(ref[:,:2,1]+ar*cfg.dt,cfg.v_min,cfg.v_max)
    nr[:,:2,2]=ar
    return nominal,nr


def control_command(x,ref,target,phase,cfg):
    """Single target/reference interface used by prediction and execution.

    ``phase`` may be one string for the scalar runner or a boolean/string mask
    for a vectorized predictor; both paths use exactly the same reference law.
    """
    target=np.asarray(target,dtype=float)
    if target.ndim==1: target=target[None,:]
    if isinstance(phase,str):
        is_post=phase in POST_CONTROL_PHASES
        effective=post_targets(x,cfg) if is_post else target
        pair,nref=reference_command(x,ref,effective,cfg)
        return pair,nref,effective
    phase_arr=np.asarray(phase)
    is_post=np.array([str(v) in POST_CONTROL_PHASES for v in phase_arr],dtype=bool)
    pre_pair,pre_ref=reference_command(x,ref,target,cfg)
    post_pair,post_ref=reference_command(x,ref,post_targets(x,cfg),cfg)
    pair=np.where(is_post[:,None],post_pair,pre_pair)
    nref=np.where(is_post[:,None,None],post_ref,pre_ref)
    effective=np.where(is_post[:,None],post_targets(x,cfg),target)
    return pair,nref,effective


def physical_step(x,nominal,active,merging,cfg):
    """Jerk/acceleration/velocity action plus ordered-chain projection.

    ``actuator_interval_empty`` and ``safety_intersection_empty`` are separate
    diagnostics.  On either empty set the fallback is the actuator-compatible
    action (or a speed-bound emergency action if the actuator interval itself
    is empty); it is explicitly marked invalid and never called a safety proof.
    """
    dt=cfg.dt; n=len(x); eps=1e-9
    raw_lo=np.maximum(np.maximum(cfg.a_min,x[:,:,A]-cfg.jerk_max*dt),(cfg.v_min-x[:,:,V])/dt)
    raw_hi=np.minimum(np.minimum(cfg.a_max,x[:,:,A]+cfg.jerk_max*dt),(cfg.v_max-x[:,:,V])/dt)
    actuator_ok=np.all(raw_lo<=raw_hi+eps,axis=1)
    # When speed and jerk constraints conflict, preserve the declared jerk
    # bound and mark the speed-side conflict invalid.  A speed-priority
    # fallback would create the old 80 m/s^3 acceleration jump.
    jerk_lo=np.maximum(cfg.a_min,x[:,:,A]-cfg.jerk_max*dt)
    jerk_hi=np.minimum(cfg.a_max,x[:,:,A]+cfg.jerk_max*dt)
    emergency=np.minimum(np.maximum(nominal,jerk_lo),jerk_hi)
    bounded=np.minimum(np.maximum(nominal,raw_lo),raw_hi)
    actuator=np.where(actuator_ok[:,None],bounded,emergency)
    actual=actuator.copy()
    safety_ok=actuator_ok.copy()
    c=.5*dt**2; d=c+cfg.time_headway*dt
    for mask,chain in ((~active,(F,R,B)),(active,(F,M,R,B))):
        idx=np.flatnonzero(mask)
        if not len(idx): continue
        xx=x[idx]; lows=raw_lo[idx].copy(); highs=raw_hi[idx].copy()
        chain_ok=actuator_ok[idx].copy()
        base={}
        for leader,follower in zip(chain,chain[1:]):
            base[leader,follower]=xx[:,leader,S]-xx[:,follower,S]-4.8-cfg.min_base_gap-cfg.time_headway*xx[:,follower,V]+dt*(xx[:,leader,V]-xx[:,follower,V])
        for leader,follower in reversed(list(zip(chain,chain[1:]))):
            lows[:,leader]=np.maximum(lows[:,leader],(d*lows[:,follower]-base[leader,follower])/c)
            chain_ok &= lows[:,leader]<=highs[:,leader]+eps
        chain_ok &= lows[:,F]<=actuator[idx,F]+eps
        # A safety projection is only accepted when every interval in the
        # chain is non-empty.  Otherwise retain the same actuator action.
        for leader,follower in zip(chain,chain[1:]):
            upper=np.minimum(highs[:,follower],(base[leader,follower]+c*actuator[idx,leader])/d)
            selected=np.minimum(np.maximum(nominal[idx,follower],lows[:,follower]),upper)
            actual[idx,follower]=np.where(chain_ok,selected,actuator[idx,follower])
        if M not in chain:
            actual[idx,M]=actuator[idx,M]
        safety_ok[idx]=chain_ok
    # Keep state speeds in the declared physical range.  The checker uses the
    # same clipped update; an empty action is still invalid through the diag.
    out=x.copy()
    out[:,:,S]+=x[:,:,V]*dt+0.5*actual*dt**2
    out[:,:,V]=np.clip(x[:,:,V]+actual*dt,cfg.v_min,cfg.v_max)
    out[:,:,A]=actual
    out[:,M,Q]=np.minimum(1.,x[:,M,Q]+merging*dt/cfg.merge_duration)
    out[:,M,Y]=cfg.lane_ramp+(cfg.lane_main-cfg.lane_ramp)*out[:,M,Q]
    net,h,relevant=margins(out,cfg,active)
    jerk=(actual-x[:,:,A])/dt
    jerk_ok=np.all(np.abs(jerk)<=cfg.jerk_max+1e-8,axis=1)
    speed_ok=np.all((out[:,:,V]>=cfg.v_min-1e-8)&(out[:,:,V]<=cfg.v_max+1e-8),axis=1)
    one_step=actuator_ok & safety_ok & jerk_ok & speed_ok
    return out,actual,{
        'one_step_feasible':one_step,
        'actuator_interval_feasible':actuator_ok,
        'safety_intersection_feasible':safety_ok,
        'actuator_interval_empty':~actuator_ok,
        'safety_intersection_empty':actuator_ok & ~safety_ok,
        'actuator':actuator,
        'saturation':np.abs(actuator-nominal)>1e-9,
        'safety_correction':np.abs(actual-actuator)>1e-9,
        'jerk_valid':jerk_ok,
        'speed_valid':speed_ok,
        'jerk_mps3':jerk,
        'fallback_mode':np.where(~actuator_ok,'actuator_interval_empty',np.where(actuator_ok & ~safety_ok,'safety_intersection_empty','')),
        'min_h':np.min(np.where(relevant,h,np.inf),axis=1),
        'min_net':np.min(np.where(relevant,net,np.inf),axis=1),
    }
