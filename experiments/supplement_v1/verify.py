"""Independent structural/statistical validation and deterministic state replay."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('supplement_runner',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def verify(output,data_dir):
    manifest=json.loads((output/'run_manifest.json').read_text());assert manifest['status']=='COMPLETE'
    protocol=manifest['protocol'];df=pd.read_csv(output/('world_results.csv' if (output/'world_results.csv').exists() else 'world_results.csv.gz'))
    assert len(df)==8064 and not df.duplicated(['cohort','seed','policy']).any()
    assert set(df.cohort)==set(protocol['cohorts'])
    for cohort,settings in protocol['cohorts'].items():
        sub=df[df.cohort==cohort]
        assert sub.seed.nunique()==settings['n_worlds'] and sub.groupby('seed').size().eq(21).all()
        assert set(sub.seed)==set(range(settings['seed_start'],settings['seed_start']+settings['n_worlds']))
    assert (df.envelope<=df.B+1e-10).all()
    assert np.isfinite(df[['brier','expected_brier','B','envelope']].to_numpy()).all()
    for _,g in df.groupby(['cohort','seed']):
        for column in ['state_visible_hash','family_hash','budget_hash','B','K']:
            assert g[column].nunique()==1
        for row in g.itertuples():
            S=json.loads(row.selected_indices)
            assert len(S)==row.k and len(S)==len(set(S)) and all(0<=i<24 for i in S)
            if row.policy=='Fixed-ceil':assert row.k==np.ceil(.75*row.K)
    stats=pd.read_csv(output/'primary_statistics.csv')
    assert len(stats)==35
    for row in stats.itertuples():
        p=df[df.cohort==row.cohort].pivot(index='seed',columns='policy',values='brier')
        values=p[row.competitor]-p['P1Q1L1G1']
        se=values.std(ddof=1)/np.sqrt(len(values));critical=t.ppf(1-.05/(2*35),len(values)-1)
        assert abs(values.mean()-row.mean_gain)<1e-12
        assert abs(values.mean()-critical*se-row.simultaneous_bonferroni_low)<1e-12
        assert abs(values.mean()+critical*se-row.simultaneous_bonferroni_high)<1e-12
    # Replay the first and last world of every cohort and audit optimizer status.
    archive=np.load(data_dir/'real_credit_datasets_authentic.npz',allow_pickle=False)
    _,generate=run.original_modules();replayed=0;fits=[];original_minimize=run.eng.minimize
    def tracked(*args,**kwargs):
        result=original_minimize(*args,**kwargs)
        fits.append({'success':bool(result.success),'status':int(result.status),'grad_inf':float(np.max(np.abs(result.jac)))})
        return result
    run.eng.minimize=tracked
    try:
        for cohort,settings in protocol['cohorts'].items():
            for seed in [settings['seed_start'],settings['seed_start']+settings['n_worlds']-1]:
                if cohort=='Controlled':st=generate(seed)
                else:
                    d={key:archive[settings['dataset']+'_'+key] for key in ['X','y','ead']}
                    st=run.realx_state(d,seed,settings['mechanism'])
                records=run.one_world(st,cohort,seed)
                old=df[(df.cohort==cohort)&(df.seed==seed)].set_index('policy')
                for row in records:
                    saved=old.loc[row['policy']]
                    for col in ['selected_hash','family_hash','budget_hash','state_visible_hash']:
                        assert row[col]==saved[col]
                    for col in ['brier','expected_brier','envelope','realized_loss']:
                        assert abs(row[col]-saved[col])<1e-12
                    replayed+=1
    finally:
        run.eng.minimize=original_minimize
    repair=pd.read_csv(output/('p2_repair.csv' if (output/'p2_repair.csv').exists() else 'p2_repair.csv.gz'));fixed=repair[repair.policy=='Fixed-Round-Repaired']
    assert len(repair)==640 and len(fixed)==128 and fixed.k.eq(fixed.target_k).all()
    assert (fixed.envelope<=fixed.B+1e-10).all()
    result={'world_rows':len(df),'fresh_worlds':384,'primary_statistics_recomputed':35,'deterministic_worlds_replayed':10,'deterministic_rows_replayed':replayed,
            'fixed_round_repair_worlds':128,'shared_state_budget_and_family':'PASS','risk_feasibility':'PASS','cardinality':'PASS',
            'sampled_optimizer_fits':len(fits),'sampled_optimizer_non_success':sum(not f['success'] for f in fits),
            'sampled_optimizer_max_gradient_inf':max(f['grad_inf'] for f in fits)}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args();result=verify(args.output,args.data_dir)
    (args.output/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
