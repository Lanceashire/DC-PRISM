"""Recompute statistics from saved rows and audit all saved family choices."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('coverage_runner', HERE/'run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def verify(out, data_dir):
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE'
    protocol = manifest['protocol']; m = protocol['primary_family_size']
    freeze = json.loads((out/'pre_run_freeze.json').read_text())
    assert freeze['protocol_sha256'] == manifest['protocol_sha256']
    assert hashlib.sha256((run.ROOT/'configs/supplement_v2.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
    for path, h in manifest['source_sha256'].items():
        assert hashlib.sha256((run.ROOT/path).read_bytes()).hexdigest() == h
    df = pd.read_csv(out/'world_results.csv.gz'); fam = pd.read_csv(out/'family_results.csv.gz')
    assert not df.duplicated(['cohort','seed','arm','policy']).any()
    assert not fam.duplicated(['cohort','seed','arm','family_index']).any()
    assert set(df.cohort) == set(protocol['cohorts'])
    pairs = 0
    for cohort, config in protocol['cohorts'].items():
        g = df[df.cohort == cohort]
        assert set(g.seed) == set(range(config['seed_start'],config['seed_start']+config['n_pairs']))
        assert g.groupby('seed').size().eq(12).all()
        pairs += config['n_pairs']
    assert len(df) == pairs*12
    assert np.isfinite(df[['brier','expected_brier','B','envelope','expected_regret','family_headroom']]).all().all()
    assert (df.envelope <= df.B+1e-10).all()
    assert (df.expected_regret >= -1e-12).all() and (df.family_headroom >= -1e-12).all()
    assert (df.gap_history_axis_mean < df.broad_history_axis_mean).all()
    for (cohort,seed), g in df.groupby(['cohort','seed']):
        for col in ['q_hash','B','K','budget_hash','family_hash','shared_hidden_hash','shared_covariate_hash','history_pool_hash']:
            assert g[col].nunique() == 1
        f = fam[(fam.cohort == cohort) & (fam.seed == seed)]
        for arm, a in g.groupby('arm'):
            assert set(a.policy) == set(run.POLICIES)
            fa = f[f.arm == arm].set_index('family_index')
            other = f[f.arm != arm].set_index('family_index')
            assert fa.selected_indices.equals(other.selected_indices)
            assert (fa.envelope <= float(a.B.iloc[0])+1e-10).all()
            anchor = a[a.policy == 'Safe-MinRisk'].iloc[0]
            for row in a.itertuples():
                s = json.loads(row.selected_indices)
                assert len(s) == row.k and len(set(s)) == row.k and all(0<=i<24 for i in s)
                assert row.selected_indices == fa.loc[row.family_index,'selected_indices']
                assert abs(row.expected_brier-fa.loc[row.family_index,'expected_brier']) < 1e-12
                assert abs(row.family_oracle_expected_brier-fa.expected_brier.min()) < 1e-12
                assert abs(row.expected_gain-(anchor.expected_brier-row.expected_brier)) < 1e-12
                if row.policy == 'Fixed-ceil': assert row.k == int(np.ceil(.75*row.K))
    stats = pd.read_csv(out/'primary_statistics.csv'); checks = []
    # Deliberately reconstruct endpoints without using the runner's statistics helpers.
    for cohort, g in df.groupby('cohort'):
        full = g[g.policy == 'Full'].pivot(index='seed',columns='arm',values=['family_headroom','expected_gain'])
        point = g[(g.policy == 'PointValue') & (g.arm == 'gap')].sort_values('seed')
        vectors = [full.family_headroom.gap-full.family_headroom.broad,
                   full.expected_gain.gap-full.expected_gain.broad,
                   full.expected_gain.gap, point.expected_gain]
        for name, values in zip(protocol['primary_endpoints'], vectors):
            row = stats[(stats.cohort == cohort) & (stats.endpoint == name)].iloc[0]
            x = np.asarray(values); se = x.std(ddof=1)/np.sqrt(len(x)); critical = t.ppf(1-protocol['alpha']/(2*m),len(x)-1)
            assert row.n == len(x) and abs(row['mean']-x.mean()) < 1e-12
            assert abs(row.simultaneous_low-(x.mean()-critical*se)) < 1e-12
            assert abs(row.simultaneous_high-(x.mean()+critical*se)) < 1e-12
            p = float(ttest_1samp(x,0).pvalue) if x.std(ddof=1)>1e-15 else (1. if abs(x.mean())<1e-15 else 0.)
            assert abs(row.p_raw-p) < 1e-12
            checks.append((row.name,p))
    assert len(checks) == len(stats) == m
    running = 0.
    for rank,(i,p) in enumerate(sorted(checks,key=lambda x:x[1])):
        running = max(running,min(1.,p*(m-rank)))
        assert abs(stats.loc[i,'p_holm']-running) < 1e-12
    audit = json.loads((out/'optimizer_audit.json').read_text())
    assert audit['summary']['total_fits'] == len(audit['fits'])
    assert sum(not f['success'] for f in audit['fits']) == audit['summary']['non_success'] == 0
    datafile = data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    archive = np.load(datafile,allow_pickle=False); replayed = 0; max_error = 0.
    for cohort, settings in protocol['cohorts'].items():
        data = {k:archive[settings['dataset']+'_'+k] for k in ['X','ead']}
        for seed in [settings['seed_start'],settings['seed_start']+settings['n_pairs']-1]:
            fresh, family = run.one_pair(data,seed,settings['mechanism'],cohort)
            saved = df[(df.cohort == cohort) & (df.seed == seed)].set_index(['arm','policy'])
            for row in fresh:
                old = saved.loc[(row['arm'],row['policy'])]
                for col in ['selected_hash','family_hash','q_hash','budget_hash','visible_history_hash','shared_hidden_hash']:
                    assert row[col] == old[col]
                for col in ['brier','expected_brier','envelope','family_headroom','expected_gain','expected_regret']:
                    err = abs(row[col]-old[col]); max_error = max(max_error,err); assert err < 1e-12
                replayed += 1
    return dict(status='PASS',independent_pairs=pairs,arm_states=2*pairs,method_rows=len(df),
                family_rows=len(fam),all_family_choices_and_risk='PASS',
                shared_pair_state_q_budget_family='PASS',primary_statistics_recomputed=m,
                deterministic_pairs_replayed=8,deterministic_rows_replayed=replayed,max_replay_error=max_error,
                all_optimizer_fits=audit['summary'],source_hashes='PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--data-dir',type=Path,required=True)
    args = p.parse_args(); result = verify(args.output,args.data_dir)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
