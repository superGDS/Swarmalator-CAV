"""Stage2B finite-window predictor and correctly separated control arms."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,asdict
import time
import numpy as np
from .stage2b_common import (Config,S,V,A,Y,Q,M,R,F,B,slot,margins,idm,
                             post_targets,reference_command,control_command,
                             physical_step)

@dataclass
class Plan:
    start: float
    z: list
    created: float
    valid: bool=True

class Predictor:
    def __init__(self,cfg): self.cfg=cfg

    def evaluate(self,x,ref,t,starts,z,executing=False,keep_trace=False):
        c=self.cfg; starts=np.asarray(starts,dtype=float); z=np.asarray(z,dtype=float)
        if z.ndim==1: z=z[None,:]
        n=len(starts); xx=np.repeat(x[None],n,axis=0); rr=np.repeat(ref[None],n,axis=0)
        completes=starts+c.merge_duration
        if executing: completes=np.full(n,t+(1-x[M,Q])*c.merge_duration)
        ends=completes+c.post_preview_s
        ok=np.ones(n,bool); reasons=np.full(n,'',object)
        min_h=np.full(n,np.inf); min_net=np.full(n,np.inf); cost=np.zeros(n); effort=np.zeros(n)
        completion_s=np.full(n,np.nan); start_low=np.full(n,np.nan); start_up=np.full(n,np.nan)
        min_action=np.full(n,np.inf); trace=[]; empty_steps=np.zeros(n,int)
        def reject(mask,reason):
            mask=np.asarray(mask,dtype=bool); chosen=mask&ok; reasons[chosen]=reason; ok[mask]=False
        nsteps=max(1,int(round((float(np.max(ends))-t)/c.dt)))
        for k in range(nsteps+1):
            now=t+k*c.dt; alive=now<=ends+1e-8
            active=(now>=starts-1e-8)|(xx[:,M,Q]>0)
            merging=(now>=starts-1e-8)&(now<completes-1e-8)
            post=now>=completes-1e-8
            phase=np.where(post,'SUCCESS_RELEASE',np.where(merging,'EXECUTE','PREPARE'))
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
            if k==nsteps: break
            target=np.repeat((-3*z)[:,None,:],1,axis=1).reshape(n,2)
            mr,nr,_=control_command(xx,rr,target,phase,c)
            nominal=np.zeros((n,4)); nominal[:,:2]=mr
            nominal[:,F]=np.clip(.85*(18-xx[:,F,V]),c.a_min,c.a_max)
            nominal[:,B]=idm(xx,B,R,c)
            nxt,actual,diag=physical_step(xx,nominal,active,merging,c)
            bad=alive&(~diag['one_step_feasible']); empty_steps+=bad.astype(int)
            reject(bad,'empty_action_set')
            interval_margin=np.min(np.minimum(actual-c.a_min,c.a_max-actual),axis=1)
            min_action=np.minimum(min_action,np.where(alive,interval_margin,np.inf))
            cost+=np.where(alive,np.sum(np.maximum(0,18-xx[:,:,V]),axis=1)*c.dt,0)
            effort+=np.where(alive,np.sum(actual**2,axis=1)*c.dt,0)
            if keep_trace:
                trace.append(dict(t_s=round(now,6),x=xx[0].tolist(),ref=rr[0].tolist(),nominal=nominal[0].tolist(),actual=actual[0].tolist(),active=bool(active[0]),phase=str(phase[0]),h=h[0].tolist(),one_step_feasible=bool(diag['one_step_feasible'][0]),fallback_mode=str(diag['fallback_mode'][0])))
            xx=nxt; rr=nr
        reject(~np.isfinite(completion_s),'completion_not_in_horizon')
        score=cost+c.planner_action_effort_weight*effort+c.planner_wait_weight*(completes-t)
        penalty=20*np.maximum(0,-min_h)**2+2*np.maximum(0,np.nan_to_num(completion_s,nan=200)-170)**2
        penalty+=10*(np.maximum(0,-np.nan_to_num(start_low))**2+np.maximum(0,-np.nan_to_num(start_up))**2)
        penalty+=2*empty_steps
        potential=score/20+penalty
        return dict(valid=ok,reason=reasons,min_h=min_h,min_net=min_net,cost=cost,effort=effort,
                    score=score,potential=potential,completion_s=completion_s,
                    completion_t_s=completes,min_action=min_action,empty_action_steps=empty_steps,trace=trace)

class Controller:
    def __init__(self,method,cfg,ablation=''):
        assert method in ('P','S1','S2','S3')
        self.method=method; self.cfg=cfg; self.ablation=ablation; self.predictor=Predictor(cfg)
        self.plan=None; self.last_decision=-1e9; self.z=np.array([2/3,2/3],float)
        self.first_valid=None; self.history=[]; self.last_info={}; self.last_gradient=np.zeros(2)

    def state_dict(self):
        return deepcopy(dict(plan=None if self.plan is None else asdict(self.plan),last_decision=self.last_decision,
            z=self.z.tolist(),first_valid=self.first_valid,history=self.history,last_info=self.last_info,
            last_gradient=self.last_gradient.tolist()))

    def load_state(self,state):
        s=deepcopy(state); self.plan=None if s['plan'] is None else Plan(**s['plan'])
        self.last_decision=s['last_decision']; self.z=np.array(s['z']); self.first_valid=s['first_valid']
        self.history=s['history']; self.last_info=s['last_info']; self.last_gradient=np.array(s['last_gradient'])

    @staticmethod
    def _diag(ev,i):
        return {key:ev[key][i] for key in ('valid','reason','min_h','min_net','cost','completion_s','completion_t_s','min_action')}

    def decide(self,x,ref,t,phase,force=False):
        c=self.cfg
        if phase not in ('PREPARE','EXECUTE'):
            return post_targets(x[None],c)[0],{**self.last_info,'decision':0,'phase':phase}
        if not force and t<self.last_decision+c.refresh_s-1e-8:
            return -3*self.z,{**self.last_info,'decision':0,'phase':phase}
        tick=time.perf_counter(); executing=phase=='EXECUTE'; candidate_count=0
        base_diag=None; incumbent_ok=False
        if self.plan is not None:
            inc=self.predictor.evaluate(x,ref,t,[self.plan.start],[self.z.copy()],executing)
            candidate_count+=1; incumbent_ok=bool(inc['valid'][0])
            if incumbent_ok: base_diag=self._diag(inc,0)
        if incumbent_ok or executing:
            start=self.plan.start if self.plan is not None else t-x[M,Q]*c.merge_duration
            base_z=self.z.copy(); action='hold_and_refine' if incumbent_ok else 'executing_replan'
        else:
            waits=np.asarray(c.planner_wait_values,dtype=float)
            waits=waits[(waits>=-1e-9)&(waits<=c.max_prepare_s+1e-9)]
            levels=np.asarray(c.planner_action_values,dtype=float)
            levels=levels[(levels>=0)&(levels<=1)]
            candidates=[(t+float(w),a,b) for w in waits for a in levels for b in levels]
            candidates=candidates[:int(c.planner_max_candidates)]
            starts=np.array([v[0] for v in candidates]); zz=np.array([v[1:] for v in candidates])
            ev=self.predictor.evaluate(x,ref,t,starts,zz,executing); candidate_count+=len(candidates)
            good=np.flatnonzero(ev['valid'])
            if not len(good):
                self.last_decision=t; self.z=np.array([2/3,2/3]); self.plan=None
                self.last_info=dict(decision=1,valid_plan=0,reason='finite_search_no_valid_plan',candidate_count=candidate_count,
                    decision_seconds=time.perf_counter()-tick,action='continue_bounded_preparation',start_s='',completion_s='',completion_t_s='',
                    predicted_min_h_m=float(np.max(ev['min_h'])),predicted_min_net_m=float(np.max(ev['min_net'])),gradient_M=0.,gradient_R=0.,
                    partner_M=0.,partner_R=0.,feedback_M=0.,feedback_R=0.,K_MR=0.,K_RM=0.,W=0.,base_z_M=2/3,base_z_R=2/3,
                    proposal_z_M=2/3,proposal_z_R=2/3,accepted_z_M=2/3,accepted_z_R=2/3,state_update_accepted=0,
                    partner_enabled=0,feedback_enabled=0,phase=phase)
                self.history.append(dict(t_s=t,**self.last_info)); return -3*self.z,self.last_info
            best=good[np.argmin(ev['score'][good])]; start=float(starts[best]); base_z=zz[best].copy(); action='new_joint_plan'
            base_diag=self._diag(ev,best)
        probes=[base_z.copy()]
        for axis in range(2):
            for direction in (-1,1):
                zz=base_z.copy(); zz[axis]=np.clip(zz[axis]+direction*c.z_step,0,1); probes.append(zz)
        ev=self.predictor.evaluate(x,ref,t,[start]*5,np.asarray(probes),executing); candidate_count+=5
        gradients=[]
        for axis in range(2):
            lm,lp=1+axis*2,2+axis*2
            gradients.append(float((ev['potential'][lp]-ev['potential'][lm])/max(probes[lp][axis]-probes[lm][axis],1e-9)))
        gradients=np.asarray(gradients); self.last_gradient=gradients
        lo,hi=slot(x[None],c); W=float(.2+.6*np.exp(-abs(float(hi[0]-lo[0]))/10))
        if self.ablation.lower()=='fixed_w': W=c.fixed_W
        symmetric=(self.ablation.lower()=='symmetric_k')
        if self.method=='S3' and not symmetric:
            rear_h=float(x[R,S]-x[B,S]-4.8-c.min_base_gap-c.time_headway*x[B,V])
            fraction=.5+.3/(1+np.exp(np.clip((rear_h-10)/4,-30,30)))
        else: fraction=.5
        km=c.partner_total*fraction; kr=c.partner_total-km
        delta=np.sin(np.pi*(base_z[1]-base_z[0]))
        partner=np.array([km*W*delta,-kr*W*delta])
        # Correct arms: P and S1 have no added partner; S1 also has no gradient.
        partner_enabled=int(self.method in ('S2','S3') and self.ablation.lower()!='no_partner')
        if not partner_enabled: partner[:]=0.0
        feedback=-c.gradient_gain*np.clip(gradients,-4,4)
        feedback_enabled=int(self.method in ('S2','S3') and self.ablation.lower()!='no_feedback')
        if not feedback_enabled: feedback[:]=0.0
        drift=-np.array([c.intrinsic_M,c.intrinsic_R])*base_z if self.method in ('S2','S3') else np.zeros(2)
        proposal=np.clip(base_z+c.refresh_s*(drift+feedback+partner),0,1)
        if self.method=='P': proposal=base_z.copy()
        proposed=self.predictor.evaluate(x,ref,t,[start],[proposal],executing); candidate_count+=1
        if self.method=='P':
            choices=[(float(ev['score'][i]),probes[i],self._diag(ev,i)) for i in range(5) if ev['valid'][i]]
            if proposed['valid'][0]: choices.append((float(proposed['score'][0]),proposal,self._diag(proposed,0)))
            if choices: _,chosen,diag=min(choices,key=lambda v:v[0]); accepted=True
            else: chosen=base_z; diag=base_diag or self._diag(ev,0); accepted=False
        elif proposed['valid'][0]: chosen=proposal; diag=self._diag(proposed,0); accepted=True
        else: chosen=base_z; diag=base_diag or self._diag(ev,0); accepted=False; action+=';state_update_rejected'
        self.z=np.asarray(chosen,dtype=float); self.plan=Plan(float(start),self.z.tolist(),float(t),bool(diag['valid']))
        if diag['valid'] and self.first_valid is None: self.first_valid=t
        self.last_decision=t
        self.last_info=dict(decision=1,valid_plan=int(diag['valid']),reason=str(diag['reason']),candidate_count=candidate_count,
            decision_seconds=time.perf_counter()-tick,action=action,start_s=float(start),completion_s=float(diag['completion_s']),completion_s_m=float(diag['completion_s']),completion_t_s=float(diag['completion_t_s']),
            predicted_min_h_m=float(diag['min_h']),predicted_min_net_m=float(diag['min_net']),execution_margin_mps2=float(diag['min_action']),
            # raw terms are retained for audit; public gradient/K/W fields are
            # the terms actually applied by this control arm.
            raw_gradient_M=float(gradients[0]),raw_gradient_R=float(gradients[1]),gradient_M=float(np.clip(gradients[0],-4,4) if feedback_enabled else 0.0),gradient_R=float(np.clip(gradients[1],-4,4) if feedback_enabled else 0.0),partner_M=float(partner[0]),partner_R=float(partner[1]),feedback_M=float(feedback[0]),feedback_R=float(feedback[1]),
            raw_K_MR=float(km),raw_K_RM=float(kr),raw_W=float(W),K_MR=float(km if partner_enabled else 0.0),K_RM=float(kr if partner_enabled else 0.0),W=float(W if partner_enabled else 0.0),z_M=float(self.z[0]),z_R=float(self.z[1]),delta_z_M=float(self.z[0]-base_z[0]),delta_z_R=float(self.z[1]-base_z[1]),
            base_z_M=float(base_z[0]),base_z_R=float(base_z[1]),proposal_z_M=float(proposal[0]),proposal_z_R=float(proposal[1]),accepted_z_M=float(self.z[0]),accepted_z_R=float(self.z[1]),
            state_update_accepted=int(accepted),proposal_valid=int(proposed['valid'][0]),partner_enabled=partner_enabled,feedback_enabled=feedback_enabled,
            coupling_ablation=self.ablation,phase=phase)
        self.history.append(dict(t_s=t,**self.last_info)); return -3*self.z,self.last_info
