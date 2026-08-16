from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from .metrics import residual_metrics, ctr_bootstrap_uncertainty


def short_model_label(name: str) -> str:
    mapping={
        'led':'LED','cfd':'CFD','linear_svr':'SVR','constructive_mlp_encoder':'MLP',
        'constructive_mlp':'MLP','cnn_regressor':'CNN','cnn':'CNN','multithreshold_svr':'MT-SVR'
    }
    key=str(name).strip().lower()
    return mapping.get(key, str(name).replace('_regressor','').replace('_',' ').title())


def short_mode_label(mode: str) -> str:
    return str(mode).replace('energy_to_energy','energy → energy').replace('energy_to_timing','energy → timing').replace('timing_to_timing','timing → timing')


def format_ctr(ctr_ps: float, uncertainty_ps: float | None = None) -> str:
    if not np.isfinite(float(ctr_ps)): return 'n/a'
    center=int(round(float(ctr_ps)))
    if uncertainty_ps is None or not np.isfinite(float(uncertainty_ps)):
        return f'{center} ps'
    return f'{center} ± {max(0,int(round(float(uncertainty_ps))))} ps'


def _save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(path,dpi=int(dpi),bbox_inches='tight'); plt.close(fig)


def _robust_bounds(groups: dict[str,np.ndarray]) -> tuple[float,float]:
    allv=np.concatenate([np.asarray(v,float)[np.isfinite(v)] for v in groups.values() if np.asarray(v).size])
    if allv.size<2:return (-100,100)
    med=float(np.median(allv)); mad=float(np.median(np.abs(allv-med))); scale=1.4826*mad
    if not np.isfinite(scale) or scale<=0: scale=float(np.std(allv,ddof=1)) or 1.0
    return med-7*scale, med+7*scale


def eligible(methods: dict[str,np.ndarray], ratio_limit: float) -> set[str]:
    if 'led' not in methods:return set(methods)
    led=residual_metrics(methods['led'])['ctr_ps']
    if not np.isfinite(led) or led<=0:return set(methods)
    out={'led'}
    for name,values in methods.items():
        if name=='led':continue
        ctr=residual_metrics(values)['ctr_ps']
        if np.isfinite(ctr) and ctr<=float(ratio_limit)*led:out.add(name)
    return out


def plot_blind_distribution(path:Path, *, mode:str, methods:dict[str,np.ndarray], dpi:int, ratio_limit:float, bootstrap_samples:int, seed:int) -> dict[str,float]:
    keep=eligible(methods,ratio_limit); visible={k:np.asarray(v,float) for k,v in methods.items() if k in keep and np.sum(np.isfinite(v))>=2}
    if not visible:return {}
    lo,hi=_robust_bounds(visible); bins=np.linspace(lo,hi,81); fig,ax=plt.subplots(figsize=(8.8,5.0)); unc={}
    for j,(name,values) in enumerate(visible.items()):
        values=values[np.isfinite(values)]; m=residual_metrics(values); u=ctr_bootstrap_uncertainty(values,bootstrap_samples,seed+137*j); unc[name]=u
        ax.hist(values[(values>=lo)&(values<=hi)],bins=bins,histtype='step',density=True,linewidth=1.4,label=f'{short_model_label(name)} — CTR {format_ctr(m["ctr_ps"],u)}')
    ax.set_title(short_mode_label(mode)); ax.set_xlabel('Residual timing error [ps]'); ax.set_ylabel('Density'); ax.grid(alpha=.22); ax.legend(frameon=False,fontsize=9,loc='best')
    _save(fig,path,dpi); return unc


def _series_rows(rows: list[dict[str,Any]], stage: str, selected_only=True):
    out=[r for r in rows if r.get('stage_name')==stage and (not selected_only or int(r.get('selected',0))==1)]
    return out


def plot_ctr_vs_voltage(path:Path, *, rows:list[dict[str,Any]], stage:str, dpi:int, ratio_limit:float, title:str|None=None):
    selected=_series_rows(rows,stage,True); selected=[r for r in selected if r.get('model') not in ('led','cfd') or True]
    if not selected:return
    modes=sorted(set(str(r['mode']) for r in selected)); fig,axes=plt.subplots(len(modes),1,figsize=(9.0,4.2*len(modes)),squeeze=False)
    for ax,mode in zip(axes[:,0],modes):
        subset=[r for r in selected if r['mode']==mode]
        for model in sorted(set(r['model'] for r in subset),key=lambda x:(x not in ('led','cfd'),x)):
            pts=sorted([r for r in subset if r['model']==model and int(r.get('plot_included',1))==1],key=lambda r:float(r['voltage_V']))
            if not pts:continue
            x=[float(r['voltage_V']) for r in pts]; y=[float(r['ctr_ps']) for r in pts]
            err=[float(r.get('ctr_uncertainty_ps',r.get('ctr_fold_std_ps',np.nan))) for r in pts]
            if np.all(np.isfinite(err)):ax.errorbar(x,y,yerr=err,marker='o',linewidth=1.2,capsize=2,label=short_model_label(model))
            else:ax.plot(x,y,marker='o',linewidth=1.2,label=short_model_label(model))
        ax.set_title(short_mode_label(mode)); ax.set_xlabel('Bias voltage [V]'); ax.set_ylabel('CTR [ps]'); ax.grid(alpha=.22); ax.legend(frameon=False,ncol=min(4,max(1,len(set(r['model'] for r in subset)))),fontsize=8)
    if title:fig.suptitle(title,y=1.01)
    _save(fig,path,dpi)



def plot_window_scan_bars(
    output_dir: Path,
    *,
    candidate_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    codebooks: dict[str, dict[str, int]],
    windows: list[dict[str, Any]],
    dpi: int,
    ratio_limit: float,
) -> None:
    """One per-file validation bar plot for a physical window-ablation study.

    For each model/window pair, only the best development-selection candidate
    (hyperparameters, fixed input variant and subsampling) is shown.  LED/CFD
    are window-independent validation baselines.  Blind results are deliberately
    not used here, so this figure can be used to choose/localize an informative
    time region without opening the blind set multiple times.
    """

    if not candidate_rows or not windows:
        return

    reverse_file = {int(v): str(k) for k, v in codebooks["file"].items()}
    reverse_mode = {int(v): str(k) for k, v in codebooks["mode"].items()}
    reverse_model = {int(v): str(k) for k, v in codebooks["model"].items()}
    window_id_to_index = {str(w["id"]): int(codebooks["window"][str(w["id"])]) for w in windows}

    def window_label(window: dict[str, Any]) -> str:
        start = float(window.get("start_ns", -float(window["before_ns"])))
        end = float(window.get("end_ns", float(window["after_ns"])))
        return f"[{start:g},{end:+g}]"

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_id, file_name in sorted(reverse_file.items()):
        file_candidates = [
            row for row in candidate_rows
            if int(row.get("stage", -1)) == 0
            and int(row.get("file_id", -1)) == file_id
        ]
        if not file_candidates:
            continue

        mode_ids = sorted({
            int(row["mode_id"]) for row in file_candidates
            if int(row.get("mode_id", -1)) in reverse_mode
        })
        if not mode_ids:
            continue

        fig, axes = plt.subplots(
            len(mode_ids), 1,
            figsize=(10.5, 4.4 * len(mode_ids)),
            squeeze=False,
            constrained_layout=True,
        )

        for ax, mode_id in zip(axes[:, 0], mode_ids):
            mode = reverse_mode[mode_id]
            validation_rows = [
                row for row in report_rows
                if int(row.get("file_id", -1)) == file_id
                and str(row.get("mode", "")) == mode
                and str(row.get("stage_name", "")) == "validation"
            ]
            led_row = next((r for r in validation_rows if str(r.get("model")) == "led"), None)
            cfd_row = next((r for r in validation_rows if str(r.get("model")) == "cfd"), None)
            led_ctr = float(led_row["ctr_ps"]) if led_row is not None else float("nan")
            cfd_ctr = float(cfd_row["ctr_ps"]) if cfd_row is not None else float("nan")

            model_ids = sorted({
                int(row["model_id"]) for row in file_candidates
                if int(row.get("mode_id", -1)) == mode_id
                and reverse_model.get(int(row["model_id"]), "") not in {"led", "cfd"}
            })
            if not model_ids:
                continue

            x = np.arange(len(windows), dtype=np.float64)
            width = 0.78 / max(1, len(model_ids))

            for model_pos, model_id in enumerate(model_ids):
                model = reverse_model[model_id]
                values: list[float] = []
                excluded: list[bool] = []

                for window in windows:
                    window_index = window_id_to_index[str(window["id"])]
                    matches = [
                        row for row in file_candidates
                        if int(row.get("mode_id", -1)) == mode_id
                        and int(row.get("model_id", -1)) == model_id
                        and int(row.get("window_id", -999)) == window_index
                        and np.isfinite(float(row.get("ctr_ps", np.nan)))
                    ]
                    if not matches:
                        values.append(float("nan"))
                        excluded.append(False)
                        continue

                    best = min(matches, key=lambda r: float(r["ctr_ps"]))
                    ctr = float(best["ctr_ps"])
                    bad = (
                        np.isfinite(led_ctr)
                        and led_ctr > 0.0
                        and ctr > float(ratio_limit) * led_ctr
                    )
                    values.append(float("nan") if bad else ctr)
                    excluded.append(bad)

                offsets = (
                    x - 0.39 + width / 2.0 + model_pos * width
                    if len(model_ids) > 1 else x
                )
                bars = ax.bar(
                    offsets,
                    values,
                    width=(width if len(model_ids) > 1 else 0.62),
                    label=short_model_label(model),
                )
                for bar, ctr in zip(bars, values):
                    if np.isfinite(ctr):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            ctr,
                            f"{ctr:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                for xpos, is_excluded in zip(offsets, excluded):
                    if is_excluded:
                        ax.text(
                            xpos, 0.02,
                            ">2× LED",
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom",
                            rotation=90, fontsize=7,
                        )

            if np.isfinite(led_ctr):
                ax.axhline(
                    led_ctr, linestyle="--", linewidth=1.2,
                    label=f"LED {led_ctr:.0f} ps",
                )
            if np.isfinite(cfd_ctr):
                ax.axhline(
                    cfd_ctr, linestyle=":", linewidth=1.2,
                    label=f"CFD {cfd_ctr:.0f} ps",
                )

            ax.set_xticks(x, [window_label(w) for w in windows])
            ax.set_xlabel("Disjoint LED-relative window [ns]")
            ax.set_ylabel("Validation s-CTR [ps]")
            ax.set_title(short_mode_label(mode))
            ax.grid(axis="y", alpha=0.22)
            ax.legend(frameon=False, fontsize=8, ncol=min(4, len(model_ids) + 2))

        fig.suptitle(
            f"{Path(file_name).stem} · disjoint-window validation scan",
            fontsize=13,
        )
        _save(fig, output_dir / f"{Path(file_name).stem}.png", dpi)


def plot_final_bars(path:Path, *, rows:list[dict[str,Any]], dpi:int):
    blind=[r for r in _series_rows(rows,'blind',True) if int(r.get('plot_included',1))==1]
    if not blind:return
    modes=sorted(set(r['mode'] for r in blind)); fig,axes=plt.subplots(len(modes),1,figsize=(10.5,4.5*len(modes)),squeeze=False)
    for ax,mode in zip(axes[:,0],modes):
        sub=[r for r in blind if r['mode']==mode]; voltages=sorted(set(float(r['voltage_V']) for r in sub)); models=sorted(set(r['model'] for r in sub),key=lambda x:(x not in ('led','cfd'),x))
        width=.8/max(1,len(models)); center=np.arange(len(voltages))
        for j,model in enumerate(models):
            vals=[]; labels=[]
            for v in voltages:
                match=next((r for r in sub if float(r['voltage_V'])==v and r['model']==model),None); vals.append(float(match['ctr_ps']) if match else np.nan)
                if match and model not in ('led','cfd') and match.get('window_before_ns','')!='': labels.append((len(vals)-1,f"-{float(match['window_before_ns']):g}/+{float(match['window_after_ns']):g}"))
            xs=center-.4+width/2+j*width; bars=ax.bar(xs,vals,width=width,label=short_model_label(model))
            for idx,text in labels:
                val=vals[idx]
                if np.isfinite(val): ax.text(xs[idx],val,text,ha='center',va='bottom',rotation=90,fontsize=6)
        ax.set_xticks(center, [f'{v:g}' for v in voltages]); ax.set_xlabel('Bias voltage [V]'); ax.set_ylabel('Blind CTR [ps]'); ax.set_title(short_mode_label(mode)); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,ncol=min(5,len(models)),fontsize=8)
    _save(fig,path,dpi)


def plot_selection_vs_blind(path:Path, *, rows:list[dict[str,Any]], selection_stage:str, dpi:int):
    sel=[r for r in _series_rows(rows,selection_stage,True) if r['model'] not in ('led','cfd') and int(r.get('plot_included',1))==1]; blind=_series_rows(rows,'blind',True)
    pairs=[]
    for r in sel:
        b=next((x for x in blind if x['file']==r['file'] and x['mode']==r['mode'] and x['model']==r['model'] and int(x.get('plot_included',1))==1),None)
        if b:pairs.append((r,b))
    if not pairs:return
    fig,ax=plt.subplots(figsize=(6.7,6.0))
    for model in sorted(set(a['model'] for a,b in pairs)):
        pts=[(a,b) for a,b in pairs if a['model']==model]; ax.scatter([float(a['ctr_ps']) for a,b in pts],[float(b['ctr_ps']) for a,b in pts],label=short_model_label(model))
    vals=np.array([float(x['ctr_ps']) for p in pairs for x in p]); lo=float(np.nanmin(vals)); hi=float(np.nanmax(vals)); pad=.05*(hi-lo if hi>lo else 1); ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',linewidth=1)
    x=np.asarray([float(a['ctr_ps']) for a,b in pairs]); y=np.asarray([float(b['ctr_ps']) for a,b in pairs]); r=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else np.nan
    ax.set_xlabel(f'{selection_stage.replace("_"," ").title()} CTR [ps]'); ax.set_ylabel('Blind CTR [ps]'); ax.set_title(f'Selection vs blind' + (f' · r={r:.2f}' if np.isfinite(r) else '')); ax.grid(alpha=.22); ax.legend(frameon=False,fontsize=8)
    _save(fig,path,dpi)


def plot_correction_matrix(path:Path, *, corrections:dict[str,Any], dpi:int, title:str):
    # Values can be bare correction arrays (identical population) or
    # (event_indices, corrections) pairs. The latter is used whenever coverage
    # can differ (notably multithreshold), so correlations are always computed
    # on the exact same events rather than by accidental positional truncation.
    normalized={}
    for name,value in corrections.items():
        if isinstance(value,tuple) and len(value)==2:
            idx=np.asarray(value[0],dtype=np.int64); val=np.asarray(value[1],dtype=float);
            if len(idx)==len(val) and len(idx)>=3: normalized[name]=(idx,val)
        else:
            val=np.asarray(value,dtype=float);
            if val.size>=3: normalized[name]=(np.arange(val.size,dtype=np.int64),val)
    names=list(normalized)
    if len(names)<2:return
    common=set(normalized[names[0]][0].tolist())
    for name in names[1:]: common.intersection_update(normalized[name][0].tolist())
    common=np.asarray(sorted(common),dtype=np.int64)
    if common.size<3:return
    columns=[]
    for name in names:
        idx,val=normalized[name]; lookup={int(i):float(v) for i,v in zip(idx,val)}; columns.append(np.asarray([lookup[int(i)] for i in common],dtype=float))
    mat=np.column_stack(columns); finite=np.all(np.isfinite(mat),axis=1); mat=mat[finite]
    if len(mat)<3:return
    corr=np.corrcoef(mat,rowvar=False)
    fig,ax=plt.subplots(figsize=(max(5,1+0.75*len(names)),max(4.5,1+0.7*len(names)))); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(names)),[short_model_label(x) for x in names],rotation=35,ha='right'); ax.set_yticks(range(len(names)),[short_model_label(x) for x in names]); ax.set_title(title+f' · n={len(mat)}')
    for i in range(len(names)):
        for j in range(len(names)):ax.text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Pearson r'); _save(fig,path,dpi)


def plot_top_corrections(
    path: Path, *, time_ps: np.ndarray, waveforms: np.ndarray,
    led_residual: np.ndarray, corrected_residual: np.ndarray,
    calibration_offset_ps: float, model: str, mode: str, k: int, dpi: int,
    window_before_ns: float | None = None, window_after_ns: float | None = None,
) -> None:
    before = np.abs(np.asarray(led_residual, float) - float(calibration_offset_ps))
    after = np.abs(np.asarray(corrected_residual, float) - float(calibration_offset_ps))
    gain = before - after
    good = np.flatnonzero(np.isfinite(gain))
    order = good[np.argsort(gain[good])[::-1]][: max(1, int(k))]
    if order.size == 0:
        return
    fig, axes = plt.subplots(order.size, 1, figsize=(9.2, 2.85 * order.size), squeeze=False)
    x = np.asarray(time_ps, dtype=np.float64) / 1000.0
    for rank, (ax, idx) in enumerate(zip(axes[:, 0], order), 1):
        if window_before_ns is not None and window_after_ns is not None:
            ax.axvspan(-float(window_before_ns), float(window_after_ns), alpha=0.08, label='selected window')
        ax.plot(x, waveforms[idx, 0], linewidth=1.05, label='ch1')
        ax.plot(x, waveforms[idx, 1], linewidth=1.05, label='ch2')
        ax.axvline(0.0, linewidth=0.8, linestyle='--')
        ax.set_ylabel('mV')
        ax.grid(alpha=.18)
        ax.text(
            .99, .96,
            f'#{rank} · LED {before[idx]:.0f} ps → corrected {after[idx]:.0f} ps · improvement {gain[idx]:+.0f} ps',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox={'facecolor':'white','alpha':.80,'edgecolor':'none'},
        )
        if rank == 1:
            ax.legend(frameon=False, ncol=3, fontsize=8, loc='best')
    axes[-1, 0].set_xlabel('Time relative to LED-aligned native anchor [ns]')
    fig.suptitle(f'{short_model_label(model)} · {short_mode_label(mode)} · top corrections', fontsize=11)
    _save(fig, path, dpi)


def write_csv(path:Path, rows:list[dict[str,Any]]):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def write_summary_results(path:Path, rows:list[dict[str,Any]]):
    blind=[dict(r) for r in rows if r.get('stage_name')=='blind' and int(r.get('selected',0))==1]
    columns=['file','voltage_V','mode','model','window_id','window_before_ns','window_after_ns','subsampling','hyperparameters_json','validation_strategy','validation_ctr_ps','validation_ctr_uncertainty_ps','n','mean_ps','std_ps','ctr_ps','ctr_uncertainty_ps','rmse_ps','led_ctr_ps','ctr_over_led','plot_included']
    out=[]
    for r in blind: out.append({k:r.get(k,'') for k in columns})
    write_csv(path,out)
