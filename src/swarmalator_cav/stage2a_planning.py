"""Fixed-budget causal joint preparation shooting and internal coordination.
No scenario name, initial-variant id, disturbance label or witness table enters
this module. Plans use absolute times. All rollouts call the actual executor.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,asdict
import time
import numpy as np
from .stage2a_common import Config, S,V,A,Y,Q,M,R,F,B,slot,margins,idm,post_targets,reference_command,physical_step

@dataclass
class Plan:
    start: float
    z: list
    created: float
    valid: bool=True

class Predictor:
    def __init__(self,cfg):self.cfg=cfg

    def evaluate(self,x,ref,t,starts,z,executing=False,keep_trace=False):
        c=self.cfg; starts=np.asarray(starts,float); z=np.asarray(z,float)
        n=len(starts); xx=np.repeat(x[None],n,axis=0); rr=np.repeat(ref[None],n,axis=0)
        completes=starts+c.merge_duration
        if executing:completes=np.full(n,t+(1-x[M,Q])*c.merge_duration)
        ends=completes+c.post_preview_s
        ok=np.ones(n,bool); reasons=np.full(n,'',object)
        min_h=np.full(n,np.inf); min_net=np.full(n,np.inf); cost=np.zeros(n); effort=np.zeros(n)
        completion_s=np.full(n,np.nan); start_low=np.full(n,np.nan); start_up=np.full(n,np.nan)
        min_action=np.full(n,np.inf); trace=[]
        def reject(mask,reason):
            mask=np.asarray(mask,dtype=bool)
            chosen=mask&ok; reasons[chosen]=reason; ok[mask]=False
        nsteps=max(1,int(round((float(np.max(ends))-t)/c.dt)))
        for k in range(nsteps+1):
            now=t+k*c.dt; alive=now<=ends+1e-8
            active=(now>=starts-1e-8)|(xx[:,M,Q]>0)
            merging=(now>=starts-1e-8)&(now<completes-1e-8)
            post=now>=completes-1e-8
            net,h,rel=margins(xx,c,active)
            min_h=np.minimum(min_h,np.where(alive,np.min(np.where(rel,h,np.inf),axis=1),np.inf))
            min_net=np.minimum(min_net,np.where(alive,np.min(np.where(rel,net,np.inf),axis=1),np.inf))
            reject(alive&(np.min(np.where(rel,h,np.inf),axis=1)<-1e-7),'dynamic_h_negative')
            at_start=(np.abs(now-starts)<c.dt*.4)&(not executing)
            lo,hi=slot(xx,c)
            start_low=np.where(at_start,xx[:,M,S]-lo,start_low)
            start_up=np.where(at_start,hi-xx[:,M,S],start_up)
            reject(at_start&((xx[:,M,S]<55-1e-8)|(xx[:,M,S]>170+1e-8)|(xx[:,M,S]<lo-1e-7)|(xx[:,M,S]>hi+1e-7)),'start_outside_slot')
            at_end=np.abs(now-completes)<c.dt*.4
            completion_s=np.where(at_end,xx[:,M,S],completion_s)
            reject(at_end&((xx[:,M,S]>170+1e-7)|(xx[:,M,S]<lo-1e-7)|(xx[:,M,S]>hi+1e-7)),'endpoint_boundary_or_slot')
            reject(alive&np.any((xx[:,:,S]<-1-1e-7)|(xx[:,:,S]>450+1e-7),axis=1),'road')
            if k==nsteps:break
            target=-3*z
            target=np.where(post[:,None],post_targets(xx,c),target)
            mr,nr=reference_command(xx,rr,target,c)
            nominal=np.zeros((n,4)); nominal[:,:2]=mr
            # Public desired speed and present state only: no future script.
            nominal[:,F]=np.clip(.85*(18-xx[:,F,V]),c.a_min,c.a_max)
            nominal[:,B]=idm(xx,B,R,c)
            nxt,actual,diag=physical_step(xx,nominal,active,merging,c)
            reject(alive&(~diag['one_step_feasible']),'empty_action_set')
            interval_margin=np.min(np.minimum(actual-c.a_min,c.a_max-actual),axis=1)
            min_action=np.minimum(min_action,np.where(alive,interval_margin,np.inf))
            cost+=np.where(alive,np.sum(np.maximum(0,18-xx[:,:,V]),axis=1)*c.dt,0)
            effort+=np.where(alive,np.sum(actual**2,axis=1)*c.dt,0)
            if keep_trace:
                trace.append(dict(t_s=round(now,6),x=xx[0].tolist(),ref=rr[0].tolist(),nominal=nominal[0].tolist(),actual=actual[0].tolist(),active=bool(active[0]),h=h[0].tolist()))
            xx=nxt; rr=nr
        completion_bad=~np.isfinite(completion_s); reject(completion_bad,'completion_not_in_horizon')
        # Cost over preparation, maneuver and the common 0.5 s release tail.
        # Infeasibility penalties are only used by the finite-difference state
        # update. Selection ALWAYS puts feasible candidates ahead of failures.
        score=cost+c.planner_action_effort_weight*effort+c.planner_wait_weight*(completes-t)
        penalty=20*np.maximum(0,-min_h)**2+2*np.maximum(0,np.nan_to_num(completion_s,nan=200)-170)**2
        penalty+=10*(np.maximum(0,-np.nan_to_num(start_low))**2+np.maximum(0,-np.nan_to_num(start_up))**2)
        potential=score/20+penalty
        return dict(valid=ok,reason=reasons,min_h=min_h,min_net=min_net,cost=cost,effort=effort,
                    score=score,potential=potential,completion_s=completion_s,
                    completion_t_s=completes,min_action=min_action,trace=trace)

class Controller:
    def __init__(self,method,cfg,ablation=''):
        assert method in ('P','S1','S2','S3')
        self.method=method;self.cfg=cfg;self.ablation=ablation;self.predictor=Predictor(cfg)
        self.plan=None;self.last_decision=-1e9;self.z=np.array([2/3,2/3],float)
        self.first_valid=None;self.history=[];self.last_info={};self.last_gradient=np.zeros(2)

    def state_dict(self):
        return deepcopy(dict(plan=None if self.plan is None else asdict(self.plan),last_decision=self.last_decision,
            z=self.z.tolist(),first_valid=self.first_valid,history=self.history,last_info=self.last_info,
            last_gradient=self.last_gradient.tolist()))

    def load_state(self,state):
        s=deepcopy(state); self.plan=None if s['plan'] is None else Plan(**s['plan'])
        self.last_decision=s['last_decision']; self.z=np.array(s['z']); self.first_valid=s['first_valid']
        self.history=s['history'];self.last_info=s['last_info'];self.last_gradient=np.array(s['last_gradient'])

    def decide(self,x,ref,t,phase,force=False):
        c=self.cfg
        if phase not in ('PREPARE','EXECUTE'):return post_targets(x[None],c)[0],dict(decision=0)
        if not force and t<self.last_decision+c.refresh_s-1e-8:
            return -3*self.z,{**self.last_info,'decision':0}
        tick=time.perf_counter(); executing=phase=='EXECUTE'; candidate_count=0
        # Incumbent is evaluated first, with its absolute start unchanged.
        base_diag=None
        if self.plan is not None:
            starts=[self.plan.start]; zz=[self.z.copy()]
            inc=self.predictor.evaluate(x,ref,t,starts,zz,executing)
            candidate_count+=1; incumbent_ok=bool(inc['valid'][0])
            if incumbent_ok:
                base_diag={key:inc[key][0] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}
        else:inc=None;incumbent_ok=False
        # If a plan is still feasible, retain its absolute timing and seek only
        # amplitude changes. An invalid preparation plan opens the fixed grid.
        if incumbent_ok or executing:
            start=self.plan.start if self.plan is not None else t-x[M,Q]*c.merge_duration
            base_z=self.z.copy(); action='hold_and_refine' if incumbent_ok else 'executing_replan'
        else:
            waits=np.asarray(c.planner_wait_values,dtype=float)
            waits=waits[(waits >= -1e-9) & (waits <= c.max_prepare_s + 1e-9)]
            intensities=np.asarray(c.planner_action_values,dtype=float)
            intensities=intensities[(intensities >= 0.0) & (intensities <= 1.0)]
            candidates=[(t+float(w),a,b) for w in waits for a in intensities for b in intensities]
            if len(candidates) > c.planner_max_candidates:
                candidates=candidates[:c.planner_max_candidates]
            starts=np.array([v[0] for v in candidates]);zz=np.array([v[1:] for v in candidates])
            ev=self.predictor.evaluate(x,ref,t,starts,zz,executing)
            candidate_count+=len(candidates)
            good=np.flatnonzero(ev['valid'])
            if not len(good):
                # Explicit bounded-search failure, never physical infeasibility.
                self.last_decision=t
                self.last_info=dict(decision=1,valid_plan=0,reason='finite_search_no_valid_plan',
                    candidate_count=candidate_count,decision_seconds=time.perf_counter()-tick,
                    action='continue_bounded_preparation',start_s='',completion_s='',predicted_min_h_m=float(np.max(ev['min_h'])),
                    gradient_M=0.,gradient_R=0.,partner_M=0.,partner_R=0.,K_MR=0.,K_RM=0.,W=0.)
                # Common bounded brake while unresolved. No known answer read.
                self.z=np.array([2/3,2/3]);self.plan=None
                self.history.append(dict(t_s=t,**self.last_info));return -3*self.z,self.last_info
            best=good[np.argmin(ev['score'][good])]
            start=float(starts[best]);base_z=zz[best].copy();action='new_joint_plan' if self.plan is None else 'invalid_plan_replaced'
            base_diag={key:ev[key][best] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}
        # Five probes are the same reference family and the same finite budget
        # for P and all S variants. P chooses the best probe without a latent
        # state law; S evolves intensity and validates that sixth candidate.
        probes=[base_z.copy()]
        for axis in range(2):
            for direction in (-1,1):
                z=base_z.copy(); z[axis]=np.clip(z[axis]+direction*c.z_step,0,1);probes.append(z)
        ev=self.predictor.evaluate(x,ref,t,[start]*5,probes,executing);candidate_count+=5
        gradients=[]
        for axis in range(2):
            lm,lp=1+axis*2,2+axis*2
            denominator=probes[lp][axis]-probes[lm][axis]
            gradients.append(float((ev['potential'][lp]-ev['potential'][lm])/max(denominator,1e-9)))
        gradients=np.array(gradients); self.last_gradient=gradients
        lo,hi=slot(x[None],c); W=float(.2+.6*np.exp(-abs(float(hi[0]-lo[0]))/10))
        if self.ablation=='fixed_W':W=c.fixed_W
        if self.method=='S3' and self.ablation!='symmetric_k':
            # R's upstream pressure makes M more responsive to R's intensity.
            rear_h=float(x[R,S]-x[B,S]-4.8-c.min_base_gap-c.time_headway*x[B,V])
            fraction=.5+.3/(1+np.exp(np.clip((rear_h-10)/4,-30,30)))
        else:fraction=.5
        km=c.partner_total*fraction;kr=c.partner_total-km
        delta=np.sin(np.pi*(base_z[1]-base_z[0]))
        partner=np.array([km*W*delta,-kr*W*delta])
        if self.ablation=='no_partner' or self.method=='P':partner[:]=0
        drift=-np.array([c.intrinsic_M,c.intrinsic_R])*base_z
        feedback=-c.gradient_gain*np.clip(gradients,-4,4)
        if self.method=='S1' or self.ablation=='no_feedback':feedback[:]=0
        proposal=np.clip(base_z+c.refresh_s*(drift+feedback+partner),0,1)
        # P receives the same sixth evaluation. Its candidate follows the
        # local gradient, but it has no intrinsic drift or partner state.
        if self.method=='P':proposal=np.clip(base_z-c.refresh_s*c.gradient_gain*np.clip(gradients,-4,4),0,1)
        proposed=self.predictor.evaluate(x,ref,t,[start],[proposal],executing);candidate_count+=1
        if self.method=='P':
            choices=[(float(ev['score'][i]),probes[i],{key:ev[key][i] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}) for i in range(5) if ev['valid'][i]]
            if proposed['valid'][0]:choices.append((float(proposed['score'][0]),proposal,{key:proposed[key][0] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}))
            if choices:_,chosen,diag=min(choices,key=lambda v:v[0])
            else:chosen=base_z;diag=base_diag or {key:ev[key][0] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}
        else:
            if proposed['valid'][0]:chosen=proposal;diag={key:proposed[key][0] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}
            else:
                chosen=base_z;diag=base_diag or {key:ev[key][0] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')};action+=';state_update_rejected'
        self.z=np.array(chosen);self.plan=Plan(start,self.z.tolist(),t,bool(diag['valid']))
        if diag['valid'] and self.first_valid is None:self.first_valid=t
        self.last_decision=t
        self.last_info=dict(decision=1,valid_plan=int(diag['valid']),reason=str(diag['reason']),candidate_count=candidate_count,
            decision_seconds=time.perf_counter()-tick,action=action,start_s=start,
            completion_s=float(diag['completion_s']),completion_s_m=float(diag['completion_s']),
            completion_t_s=float(diag['completion_t_s']),
            predicted_min_h_m=float(diag['min_h']),predicted_min_net_m=float(diag['min_net']),execution_margin_mps2=float(diag['min_action']),
            gradient_M=float(gradients[0]),gradient_R=float(gradients[1]),partner_M=float(partner[0]),partner_R=float(partner[1]),
            feedback_M=float(feedback[0]),feedback_R=float(feedback[1]),K_MR=km,K_RM=kr,W=W,
            z_M=float(self.z[0]),z_R=float(self.z[1]),delta_z_M=float(self.z[0]-base_z[0]),delta_z_R=float(self.z[1]-base_z[1]))
        self.history.append(dict(t_s=t,**self.last_info));return -3*self.z,self.last_info
