"""Shared Stage2A physical execution and smooth acceleration references.

Only Stage1C's immutable state/config/environment definitions are reused.
Vectorized prediction and scalar simulation both call ``physical_step``.
Column order is M,R,F,B; state order is s,v,a,y,physical lateral progress.
No environment or scenario identifier enters these numerical functions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .stage1c_simulation import SimConfig, VehicleState

IDS = ('M', 'R', 'F', 'B')
M, R, F, B = range(4)
S, V, A, Y, Q = range(5)

@dataclass
class Config(SimConfig):
    refresh_s: float = 0.5
    post_preview_s: float = 0.5
    max_prepare_s: float = 2.5
    reference_kp: float = 0.10
    reference_kv: float = 0.40
    z_step: float = 0.05
    gradient_gain: float = 0.018
    partner_total: float = 0.32
    fixed_W: float = 0.40
    intrinsic_M: float = 0.010
    intrinsic_R: float = 0.016
    # The finite candidate budget is part of the reproducible model contract.
    # Values are bounded role intensities z, not a hidden scenario table.
    planner_wait_values: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    planner_action_values: tuple = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)
    planner_max_candidates: int = 160
    planner_action_effort_weight: float = 0.04
    planner_wait_weight: float = 0.20

def array_state(snapshot):
    return np.array([[snapshot[k].s,snapshot[k].v,snapshot[k].a,snapshot[k].y,snapshot[k].merge_progress] for k in IDS],dtype=float)

def dictionary_state(x):
    if x.ndim == 3:
        x = x[0]
    return {k:VehicleState(k,float(x[i,S]),float(x[i,Y]),float(x[i,V]),float(x[i,A]),merge_progress=float(x[i,Q])) for i,k in enumerate(IDS)}

def idm(x, follower, leader, cfg):
    v=x[:,follower,V]; gap=np.maximum(x[:,leader,S]-x[:,follower,S]-4.8,0.1)
    dv=v-x[:,leader,V]
    star=cfg.min_base_gap+cfg.time_headway*v+v*dv/(2*np.sqrt(1.3*2.2))
    return np.clip(1.3*(1-(v/20)**4-(np.maximum(star,0.1)/gap)**2),cfg.a_min,cfg.a_max)

def margins(x,cfg,active):
    """Return physical clearance and h in FR,RB,FM,MR order.

    FM/MR h is required throughout the declared EXECUTE task, before body
    overlap, and in the occupied mainline after it. This matches Stage1C.
    """
    # Accept a scalar for one-step probes as well as the vector used by the
    # batched predictor.  The latter is the public execution path, but making
    # the diagnostic total here avoids a silent broadcasting change between
    # prediction and the independent checker.
    active=np.asarray(active,dtype=bool)
    if active.ndim == 0:
        active=np.full(len(x),bool(active),dtype=bool)
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
    # Following changes at the physical lateral overlap, not at a phase flag.
    overlap=(np.abs(x[:,M,Y]-x[:,R,Y])<=1.9)&(x[:,M,S]>x[:,R,S])
    r=np.where(overlap,idm(x,R,M,cfg),idm(x,R,F,cfg))
    m=np.where((x[:,F,S]>x[:,M,S])&(np.abs(x[:,F,Y]-x[:,M,Y])<=1.9),
               idm(x,M,F,cfg),0.85*(18-x[:,M,V]))
    return np.stack([m,r],axis=1)

def reference_command(x,ref,target,cfg):
    """C2 integrated s reference with bounded piecewise-linear acceleration.

    ref columns s,v,a; acceleration target changes are governed by jerk.
    The same integrator continues through preparation, execution and release.
    There is no q endpoint/reset and no dropped derivative term.
    """
    ar=ref[:,:2,2]+np.clip(target-ref[:,:2,2],-cfg.jerk_max*cfg.dt,cfg.jerk_max*cfg.dt)
    ar=np.clip(ar,cfg.a_min,cfg.a_max)
    nominal=ar+cfg.reference_kp*(ref[:,:2,0]-x[:,:2,S])+cfg.reference_kv*(ref[:,:2,1]-x[:,:2,V])
    nr=ref.copy()
    nr[:,:2,0]+=ref[:,:2,1]*cfg.dt+0.5*ar*cfg.dt**2
    nr[:,:2,1]+=ar*cfg.dt
    nr[:,:2,2]=ar
    return nominal,nr

def physical_step(x,nominal,active,merging,cfg):
    """One step of common chain projection and kinematic execution.

    F is fixed by its jerk-executable external nominal input. On an empty
    action intersection all controlled vehicles take their feasible maximum
    braking action; this is explicitly a fallback, never a safety guarantee.
    Returns states, actual actions and diagnostics for every batch member.
    """
    dt=cfg.dt; n=len(x)
    lo=np.maximum(np.maximum(cfg.a_min,x[:,:,A]-cfg.jerk_max*dt),(cfg.v_min-x[:,:,V])/dt)
    hi=np.minimum(np.minimum(cfg.a_max,x[:,:,A]+cfg.jerk_max*dt),(cfg.v_max-x[:,:,V])/dt)
    actuator=np.minimum(np.maximum(nominal,lo),hi)
    actual=actuator.copy(); feasible=np.all(lo<=hi+1e-9,axis=1)
    c=.5*dt**2; d=c+cfg.time_headway*dt
    for mask,chain in ((~active,(F,R,B)),(active,(F,M,R,B))):
        idx=np.flatnonzero(mask)
        if not len(idx):continue
        xx=x[idx]; lows=lo[idx].copy(); highs=hi[idx]; base={}
        ok=feasible[idx].copy()
        for leader,follower in zip(chain,chain[1:]):
            base[leader,follower]=xx[:,leader,S]-xx[:,follower,S]-4.8-cfg.min_base_gap-cfg.time_headway*xx[:,follower,V]+dt*(xx[:,leader,V]-xx[:,follower,V])
        for leader,follower in reversed(list(zip(chain,chain[1:]))):
            lows[:,leader]=np.maximum(lows[:,leader],(d*lows[:,follower]-base[leader,follower])/c)
            ok &= lows[:,leader]<=highs[:,leader]+1e-9
        ok &= lows[:,F]<=actual[idx,F]+1e-9
        for leader,follower in zip(chain,chain[1:]):
            upper=np.minimum(highs[:,follower],(base[leader,follower]+c*actual[idx,leader])/d)
            selected=np.minimum(np.maximum(nominal[idx,follower],lows[:,follower]),upper)
            actual[idx,follower]=np.where(ok,selected,lo[idx,follower])
        if M not in chain:actual[idx,M]=np.where(ok,actuator[idx,M],lo[idx,M])
        feasible[idx]=ok
    out=x.copy()
    out[:,:,S]+=x[:,:,V]*dt+.5*actual*dt**2
    out[:,:,V]+=actual*dt
    out[:,:,A]=actual
    out[:,M,Q]=np.minimum(1.,x[:,M,Q]+merging*dt/cfg.merge_duration)
    out[:,M,Y]=cfg.lane_ramp+(cfg.lane_main-cfg.lane_ramp)*out[:,M,Q]
    net,h,relevant=margins(out,cfg,active)
    return out,actual,{'one_step_feasible':feasible,'actuator':actuator,
        'saturation':np.abs(actuator-nominal)>1e-9,
        'safety_correction':np.abs(actual-actuator)>1e-9,
        'min_h':np.min(np.where(relevant,h,np.inf),axis=1),
        'min_net':np.min(np.where(relevant,net,np.inf),axis=1)}
