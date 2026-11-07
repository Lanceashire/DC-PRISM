"""Fresh random-split confirmation of the frozen repaired online P4 pipeline.

Real cohorts are reused; this is conditional split/stream confirmation and does
not introduce new institutions, raw credit outcomes, or verified calendar splits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import run_p3_p4 as runner
import prism_engine as eng
from validate_p4 import validate

SOURCE_PATHS = ['experiments/p4_confirmation_v1/run.py','src/prism_engine.py',
                'src/run_p3_p4.py','scripts/validate_p4.py']


def summarize(out, protocol):
    traj = pd.read_csv(out/'p4_real_labels/p4_trajectories_10rounds.csv')
    per_seed = traj.sort_values('round').groupby(['dataset','seed','policy'],as_index=False).agg(
        alc_brier=('brier_t','mean'), round10_brier=('brier_t','last'))
    per_seed.to_csv(out/'seed_metrics.csv',index=False)
    records = []; m = protocol['primary_family_size']
    for dataset in protocol['datasets']:
        g = per_seed[per_seed.dataset == dataset].pivot(index='seed',columns='policy',values='alc_brier')
        assert set(g.index) == set(protocol['seeds'])
        for comp in protocol['primary_competitors']:
            x = (g[comp]-g['DC-PRISM']).to_numpy(); n = len(x)
            mu, sd = float(x.mean()), float(x.std(ddof=1)); se = sd/np.sqrt(n)
            c = float(t.ppf(1-protocol['alpha']/(2*m),n-1)); lo,hi = mu-c*se,mu+c*se
            p = float(ttest_1samp(x,0).pvalue) if sd>1e-15 else (1. if abs(mu)<1e-15 else 0.)
            delta = protocol['material_delta']
            decision = ('material_positive' if lo>delta else 'material_negative' if hi < -delta else
                        'within_material_band' if lo>=-delta and hi<=delta else
                        'positive' if lo>0 else 'negative' if hi<0 else 'indeterminate')
            records.append(dict(dataset=dataset,competitor=comp,n=n,mean_gain=mu,
                mean_dc=float(g['DC-PRISM'].mean()),mean_competitor=float(g[comp].mean()),
                simultaneous_low=lo,simultaneous_high=hi,p_raw=p,decision=decision))
    stats = pd.DataFrame(records); assert len(stats) == m
    adjusted = np.empty(m); running = 0.
    for rank,i in enumerate(np.argsort(stats.p_raw)):
        running = max(running,min(1.,stats.p_raw.iloc[i]*(m-rank))); adjusted[i] = running
    stats['p_holm_global12'] = adjusted
    stats.to_csv(out/'primary_statistics.csv',index=False)
    return traj, per_seed, stats


def main():
    parser = argparse.ArgumentParser()
    for key in ['protocol','freeze','output','data-dir']:
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    assert sources == freeze['source_sha256']
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    for sub in ['p3_realx','p4_real_labels','results/audit']:
        (out/sub).mkdir(parents=True)
    runner.PKG_ROOT = str(out); runner.AUDIT_DIR = str(out/'results/audit'); runner.DATA_DIR = str(args.data_dir.resolve())
    (out/'pre_run_freeze.json').write_text(json.dumps(freeze,indent=2))
    manifest = dict(status='RUNNING',protocol=protocol,protocol_sha256=freeze['protocol_sha256'],
                    source_sha256=sources,python=platform.python_version(),numpy=np.__version__,compute_backend='CPU / frozen SciPy BFGS')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    started = time.time(); fits = []; original = eng.minimize
    def tracked(*a,**kw):
        result = original(*a,**kw)
        fits.append(dict(success=bool(result.success),status=int(result.status),grad_inf=float(np.max(np.abs(result.jac)))))
        return result
    eng.minimize = tracked
    try:
        datasets = runner.load_four_real_datasets()
        assert set(datasets) == set(protocol['datasets'])
        runner.run_p4_real_labels_benchmark(datasets,seeds=protocol['seeds'])
        audit = validate(out/'p4_real_labels/p4_trajectories_10rounds.csv')
        traj, metrics, stats = summarize(out,protocol)
        assert len(traj) == 6400 and len(metrics) == 640
        fit_audit = dict(total_fits=len(fits),non_success=sum(not x['success'] for x in fits),
                         max_gradient_inf=max(x['grad_inf'] for x in fits))
        (out/'optimizer_audit.json').write_text(json.dumps(dict(summary=fit_audit,fits=fits),indent=2))
        # Preserve the frozen optimizer's outcomes. Any failure blocks a clean acceptance claim.
        manifest.update(status='COMPLETE' if fit_audit['non_success']==0 else 'COMPUTED_OPTIMIZER_REVIEW_REQUIRED',
            validation=audit,optimizer_audit=fit_audit,trajectory_rows=len(traj),method_trajectories=len(metrics),
            primary_comparisons=len(stats),elapsed_seconds=time.time()-started)
        print(json.dumps({k:v for k,v in manifest.items() if k!='protocol'},indent=2),flush=True)
    except BaseException as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        eng.minimize = original
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
