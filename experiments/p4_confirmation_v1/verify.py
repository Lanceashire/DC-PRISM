"""Validate saved online trajectories, fresh seeds, statistics, and replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirmation_runner', HERE/'run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def verify(out, data_dir, replay_output):
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE', manifest['status']
    protocol = manifest['protocol']; m = protocol['primary_family_size']
    assert hashlib.sha256((run.ROOT/'configs/p4_confirmation_v1.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
    for path, h in manifest['source_sha256'].items():
        assert hashlib.sha256((run.ROOT/path).read_bytes()).hexdigest() == h
    assert hashlib.sha256((data_dir/'real_credit_datasets_authentic.npz').read_bytes()).hexdigest() == protocol['data_sha256']
    validation = run.validate(out/'p4_real_labels/p4_trajectories_10rounds.csv')
    traj = pd.read_csv(out/'p4_real_labels/p4_trajectories_10rounds.csv')
    assert len(traj) == 6400 and set(traj.dataset) == set(protocol['datasets'])
    for _, g in traj.groupby('dataset'):
        assert set(g.seed) == set(protocol['seeds'])
        assert g.groupby(['seed','policy']).size().eq(10).all()
    values = traj.groupby(['dataset','seed','policy']).brier_t.mean().unstack('policy')
    stats = pd.read_csv(out/'primary_statistics.csv'); assert len(stats) == m
    pvals = []
    for i, row in stats.iterrows():
        x = (values.loc[row.dataset,row.competitor]-values.loc[row.dataset,'DC-PRISM']).to_numpy()
        se = x.std(ddof=1)/np.sqrt(len(x)); c = t.ppf(1-protocol['alpha']/(2*m),len(x)-1)
        assert len(x) == row.n == 16
        assert abs(row.mean_gain-x.mean()) < 1e-12
        assert abs(row.simultaneous_low-(x.mean()-c*se)) < 1e-12
        assert abs(row.simultaneous_high-(x.mean()+c*se)) < 1e-12
        p = float(ttest_1samp(x,0).pvalue) if x.std(ddof=1)>1e-15 else (1. if abs(x.mean())<1e-15 else 0.)
        assert abs(row.p_raw-p) < 1e-12; pvals.append((i,p))
    running = 0.
    for rank,(i,p) in enumerate(sorted(pvals,key=lambda x:x[1])):
        running = max(running,min(1.,p*(m-rank)))
        assert abs(stats.loc[i,'p_holm_global12']-running) < 1e-12
    fit = json.loads((out/'optimizer_audit.json').read_text())
    assert len(fit['fits']) == fit['summary']['total_fits']
    assert not any(not a['success'] for a in fit['fits'])
    replay_output.mkdir(parents=True,exist_ok=False)
    for sub in ['p3_realx','p4_real_labels','results/audit']:
        (replay_output/sub).mkdir(parents=True)
    run.runner.PKG_ROOT = str(replay_output.resolve())
    run.runner.AUDIT_DIR = str((replay_output/'results/audit').resolve())
    run.runner.DATA_DIR = str(data_dir.resolve())
    data = run.runner.load_four_real_datasets()
    seeds = [protocol['seeds'][0],protocol['seeds'][-1]]
    run.runner.run_p4_real_labels_benchmark(data,seeds=seeds)
    replay = pd.read_csv(replay_output/'p4_real_labels/p4_trajectories_10rounds.csv')
    keys = ['dataset','seed','round','policy']
    saved = traj[traj.seed.isin(seeds)].sort_values(keys).reset_index(drop=True)
    replay = replay.sort_values(keys).reset_index(drop=True)
    assert len(replay) == len(saved) == 800
    hashed = ['D_hash','O_hash','C_hash','Q_hash','budget_hash','selected_ids','released_ids']
    # All identity columns that actually exist are audited, plus every non-timing column.
    excluded = ['selection_time_sec']
    max_error = 0.; numeric = 0; strings = 0
    assert list(saved.columns) == list(replay.columns)
    for col in saved.columns:
        if col in excluded: continue
        if pd.api.types.is_numeric_dtype(saved[col]):
            a,b = saved[col].to_numpy(),replay[col].to_numpy()
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
            if len(a) and np.isfinite(a).any():
                max_error = max(max_error,float(np.nanmax(np.abs(a.astype(float)-b.astype(float)))))
            numeric += 1
        else:
            assert saved[col].fillna('').equals(replay[col].fillna('')),col
            strings += 1
    return dict(status='PASS',**validation,primary_global_statistics_recomputed=m,
                new_seed_range=[min(protocol['seeds']),max(protocol['seeds'])],
                deterministic_trajectories_replayed=80,deterministic_rows_replayed=len(replay),
                replay_numeric_columns=numeric,replay_identity_columns=strings,max_replay_error=max_error,
                all_optimizer_fits=fit['summary'],source_and_data_hashes='PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ['output','data-dir','replay-output']: p.add_argument('--'+arg,type=Path,required=True)
    args = p.parse_args(); result = verify(args.output,args.data_dir,args.replay_output)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
