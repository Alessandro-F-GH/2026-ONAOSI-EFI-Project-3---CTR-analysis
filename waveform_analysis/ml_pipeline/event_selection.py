from __future__ import annotations
from typing import Any
import numpy as np
from utils.photopeak import fit_photopeak,photopeak_mask

def apply_energy_preselection(amplitudes_mV,noise_rms_mV,trigger_index,*,energy_channels,selection,photopeak,logger):
    a=np.asarray(amplitudes_mV,float); n=np.asarray(noise_rms_mV,float); t=np.asarray(trigger_index,np.int64)
    if a.ndim!=2 or a.shape[1]!=2 or n.shape!=a.shape or t.shape!=a.shape: raise ValueError('Energy preselection arrays must have shape [event,2]')
    valid=np.all(np.isfinite(a),axis=1)&np.all(np.isfinite(n),axis=1)&np.all(t>=0,axis=1)
    summary={'stage':'raw_energy_first_pass_before_timing_preprocessing','source_signal_variant':'raw_energy','scanned_events':int(a.shape[0]),'valid_basic_energy_features':int(valid.sum())}
    tr=selection.get('energy_trigger_index_range')
    if tr is not None:
        lo,hi=map(int,tr); valid&=np.all((t>lo)&(t<hi),axis=1); summary['energy_trigger_index_range']=[lo,hi]
    limit=selection.get('energy_noise_max_mV')
    if limit is not None:
        limits=np.asarray(limit if isinstance(limit,(list,tuple)) else [limit,limit],float).reshape(-1)
        if limits.size!=2: raise ValueError('energy_noise_max_mV must be scalar or length 2')
        valid&=(n[:,0]<limits[0])&(n[:,1]<limits[1]); summary['energy_noise_max_mV']=limits.tolist()
    summary['eligible_before_photopeak']=int(valid.sum()); fits=[]
    if bool(photopeak.get('enabled',False)):
        fi=np.flatnonzero(valid)
        if fi.size<20: raise RuntimeError(f'Too few selected events ({fi.size}) for photopeak fitting')
        for pos,ch in enumerate(energy_channels):
            result=fit_photopeak(a[fi,pos],channel=int(ch),config=photopeak)
            if not result.success: raise RuntimeError(f'Photopeak fit failed for energy channel {ch}: {result.message}')
            valid&=photopeak_mask(a[:,pos],result); fits.append(result.as_dict())
    summary['photopeak']=fits; summary['photopeak_enabled']=bool(photopeak.get('enabled',False)); summary['selected_events']=int(valid.sum())
    minimum=int(selection.get('minimum_events',selection.get('minimum_events_per_split',100)))
    if valid.sum()<minimum: raise RuntimeError(f'Only {int(valid.sum())} events remain after photopeak selection; need at least {minimum}')
    logger.info('Energy/photopeak preselection | retained=%d/%d',int(valid.sum()),valid.size)
    return valid,summary
