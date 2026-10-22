"""Exact finite-law joint capacity probability using an active-interval frontier.

This is a specialized implementation of probabilistic variable elimination,
not a claim that variable elimination or joint chance constraints are new.
Floating-point probability, not an aviation safety certificate.
"""
import math
from fractions import Fraction
from numbers import Integral
import numpy as np


def rounded_down(value):
    return np.maximum(0.,np.nextafter(value,-np.inf))


def sum_lower(values,axis=None):
    """Lower bound for a sum of nonnegative lower bounds (IEEE double).

    Standard positive-summation error bound with at most n-1 additions; an
    outward step covers arithmetic used to form the error bound and division.
    """
    n=values.size if axis is None else values.shape[axis]
    # For n*u <= 1/2, gamma_(n-1) <= 2*n*u = n*eps. This loose
    # bound avoids relying on roundoff in evaluating gamma itself.
    if n*np.finfo(float).eps>1:return np.zeros_like(np.sum(values,axis=axis))
    denominator=np.nextafter(1+n*np.finfo(float).eps,np.inf)
    return rounded_down(np.sum(values,axis=axis)/denominator)


def joint_capacity_risk(usage,capacities,q=(.7,.3),delta=.01,max_states=100000):
    """usage[job, outcome, resource, time]; one binary outcome per entire job.

    Conditional on common Z=z, outcomes are independent with probabilities
    (1-delta)*1[y=z]+delta*q[y]. The unconditional one-job marginal stays q.
    Resource occupancy is checked jointly over every resource and time slot.
    A state-limit return is unresolved and its upper risk bound is one.
    """
    u=np.asarray(usage);cap=np.asarray(capacities);prior=np.asarray(q,dtype=float)
    if u.ndim!=4 or u.shape[1]!=2 or cap.shape!=u.shape[2:] or cap.size==0:
        raise ValueError('Expected job x 2 x resource x time occupancy')
    if not np.isfinite(u).all() or not np.isfinite(cap).all() or np.any(u<0) or np.any(cap<0):
        raise ValueError('Invalid capacity/usage values')
    if np.any(u!=np.floor(u)) or np.any(cap!=np.floor(cap)):
        raise ValueError('Discrete resource occupancy must be integer-valued')
    if np.any(u>2**50/max(1,len(u))) or np.any(cap>2**50):
        raise ValueError('Occupancies too large for exact integer load arithmetic')
    if prior.shape!=(2,) or not np.isfinite(prior).all() or np.any(prior<=0) or not math.isclose(float(prior.sum()),1.,abs_tol=1e-12):
        raise ValueError('Positive two-state probability distribution required')
    if not math.isfinite(delta) or not 0<=delta<=1:raise ValueError('Invalid resampling probability')
    if not isinstance(max_states,Integral) or isinstance(max_states,bool) or max_states<1:raise ValueError('Invalid state limit')
    u=u.astype(np.int64);cap=cap.astype(np.int64)
    starts={};ends={};nonempty=[]
    for j in range(len(u)):
        times=np.flatnonzero(u[j].sum(axis=(0,1)))
        if len(times):
            starts.setdefault(int(times[0]),[]).append(j);ends[j]=int(times[-1]);nonempty.append(j)
    # An envelope proof gives an exact zero without roundoff or enumeration.
    if not nonempty or np.all(u.max(axis=1).sum(axis=0)<=cap):
        return {'status':'exact','risk':0.,'risk_upper':0.,'success_probability':1.,
                'peak_states':1,'peak_active_jobs':0,'state_visits':1,'certificate':'envelope'}
    # Normalize the input floating priors as exact rational numbers, then
    # retain lower bounds throughout positive products and marginalization.
    rational=[Fraction(float(v)) for v in prior];normalizer=sum(rational)
    rational=[v/normalizer for v in rational];d=Fraction(float(delta))
    exact_transition=[[(1-d)*int(y==z)+d*rational[y] for y in (0,1)] for z in (0,1)]
    transition=np.array([[float(v) for v in row] for row in exact_transition])
    transition_lower=np.array([[0. if v==0 else np.nextafter(float(v),-np.inf) for v in row] for row in exact_transition])
    prior=np.array([float(v) for v in rational])
    active=[];masks=np.array([0],dtype=np.uint64);mass=prior[None,:].copy()
    mass_lower=rounded_down(prior)[None,:]
    peak=1;width=0;visits=0

    def unresolved(reason):
        return {'status':'state_limit','risk':None,'risk_upper':1.,'success_probability':None,
                'peak_states':peak,'peak_active_jobs':width,'state_visits':visits,'reason':reason}

    for t in range(u.shape[-1]):
        for j in starts.get(t,[]):
            if len(active)>=63:return unresolved('mask_width')
            bit=np.uint64(1)<<np.uint64(len(active));active.append(j)
            masks=np.r_[masks,masks|bit]
            mass=np.vstack([mass*transition[:,0][None,:],mass*transition[:,1][None,:]])
            mass_lower=np.vstack([rounded_down(mass_lower*transition_lower[:,0][None,:]),
                                  rounded_down(mass_lower*transition_lower[:,1][None,:])])
            supported=mass.sum(axis=1)>0;masks=masks[supported];mass=mass[supported];mass_lower=mass_lower[supported]
            width=max(width,len(active));peak=max(peak,len(masks))
            if len(masks)>max_states:return unresolved('frontier_states')
        if active and len(masks):
            labels=((masks[:,None]>>np.arange(len(active),dtype=np.uint64))&np.uint64(1)).astype(np.int64)
            current=u[active,:,:,t]
            loads=current[:,0,:].sum(axis=0)[None,:]+labels@(current[:,1,:]-current[:,0,:])
            feasible=(loads<=cap[:,t]).all(axis=1)
            masks=masks[feasible];mass=mass[feasible];mass_lower=mass_lower[feasible];visits+=len(feasible)
        if not len(masks):
            return {'status':'exact','risk':1.,'risk_upper':1.,'success_probability':0.,
                    'peak_states':peak,'peak_active_jobs':width,'state_visits':visits,'certificate':'all_states_fail'}
        keep=[i for i,j in enumerate(active) if ends[j]>t]
        if len(keep)!=len(active):
            reduced=np.zeros(len(masks),dtype=np.uint64)
            for new,old in enumerate(keep):
                reduced|=((masks>>np.uint64(old))&np.uint64(1))<<np.uint64(new)
            unique,inverse=np.unique(reduced,return_inverse=True)
            collapsed=np.zeros((len(unique),2));np.add.at(collapsed,inverse,mass)
            collapsed_lower=np.zeros_like(collapsed);np.add.at(collapsed_lower,inverse,mass_lower)
            counts=np.bincount(inverse,minlength=len(unique))[:,None]
            denominator=np.nextafter(1+counts*np.finfo(float).eps,np.inf)
            mass_lower=rounded_down(collapsed_lower/denominator)
            masks=unique;mass=collapsed;active=[active[i] for i in keep]
    success=float(np.clip(mass.sum(),0.,1.));risk=1-success
    success_lower=float(np.clip(sum_lower(mass_lower),0.,1.))
    risk_upper=min(1.,float(np.nextafter(1-success_lower,np.inf)))
    return {'status':'exact','risk':risk,'risk_upper':risk_upper,
            'success_probability':success,'peak_states':peak,'peak_active_jobs':width,
            'state_visits':visits,'success_lower':success_lower,'roundoff_margin':risk_upper-risk,
            'certificate':'frontier_variable_elimination_with_lower_probability_bound'}
