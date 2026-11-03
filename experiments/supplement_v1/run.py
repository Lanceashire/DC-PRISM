"""Versioned P2 repair, orthogonal rescoring and unmodified Real-X diagnostics.

Selection APIs take visible labels only. Hidden outcomes are consumed only after
all choices are locked, for common-state post-settlement evaluation.
"""
import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t, ttest_1samp, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import prism_engine as eng


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def array_hash(*values):
    h=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(value);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def original_modules():
    path=ROOT/'archive/submitted_v2_2/src'
    spec=importlib.util.spec_from_file_location('submitted_engine',path/'prism_engine.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    # Load precisely the frozen DGP functions, without importing or running its master runner.
    tree=ast.parse((path/'run_p0_p1_p2.py').read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'true_dgp_prob','generate_synthetic_credit_state'}]
    scope={'np':np,'expit':expit}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path/'run_p0_p1_p2.py'),'exec'),scope)
    return original, scope['generate_synthetic_credit_state']


def fixed_repair(original, st, learner, q, B, rounding='round'):
    M,_=original.solve_safe_minrisk_exact_or_bounded(q,st['ead_C'],B,exact_max_n=16)
    K=len(M)
    target=int(np.round(.75*K)) if rounding=='round' else int(np.ceil(.75*K))
    target=max(target,1) if K else 0
    S,_,_=original.select_acquisition_policy('Fixed-Eps-Safe-PA',st['X_D'],st['y_D'],st['X_O'],st['X_C'],st['ead_C'],st['X_Q'],B,
                                             learner=learner,q_C_precomputed=q)
    fallback=len(S)!=target
    if fallback:
        S=np.sort(M[:target])
    assert len(S)==target and original.bernstein_envelope(q[S],st['ead_C'][S])<=B+1e-10
    return S,target,fallback


def evaluate(st, selected, learner, module=eng):
    ids=np.asarray(selected,int)
    X=np.vstack([st['X_D'],st['X_O'],st['X_C'][ids]])
    y=np.concatenate([st['y_D'],st['y_O'],st['y_C'][ids]])
    fitted=module.UnifiedLogisticLearner(l2=.45).fit(X,y,w_init=learner.w)
    p=fitted.predict_proba(st['X_T'])
    expected=float(np.mean((p-st['p_T_true'])**2+st['p_T_true']*(1-st['p_T_true'])))
    return module.brier_score(st['y_T'],p),expected


def repair_old_worlds(output):
    original,generate=original_modules()
    saved=pd.read_csv(ROOT/'archive/submitted_v2_2/p2_confirmation/p2_confirmation_worlds.csv')
    records=[]
    for seed in range(2001,2129):
        st=generate(seed);learner=original.UnifiedLogisticLearner(l2=.45).fit(st['X_D'],st['y_D'])
        q,_=original.compute_calibrated_upper_risk(learner,st['X_D'],st['y_D'],st['X_C'],rng=np.random.default_rng(seed+100))
        B,_=original.compute_exogenous_budget(q,st['ead_C'],target_k=6)
        w=original.sample_common_parameter_worlds(learner,n_worlds=20,rng=np.random.default_rng(seed+700))
        for policy in ['DC-PRISM','Fixed-Eps-Safe-PA','Safe-MinRisk','BAIT-Cost-Safe']:
            S,M,_=original.select_acquisition_policy(policy,st['X_D'],st['y_D'],st['X_O'],st['X_C'],st['ead_C'],st['X_Q'],B,
                            learner=learner,q_C_precomputed=q,w_worlds_precomputed=w,rng=np.random.default_rng(seed+400))
            b,expected=evaluate(st,S,learner,original)
            old=saved[(saved.world_seed==seed)&(saved.policy==policy)].iloc[0]
            assert abs(b-old.brier)<1e-12, f'Frozen P2 reproduction failed: {seed}/{policy}'
            records.append(dict(cohort='P2_repair',seed=seed,policy=policy,brier=b,expected_brier=expected,k=len(S),K=len(M),B=B,
                                envelope=original.bernstein_envelope(q[S],st['ead_C'][S]),fallback=False,target_k=-1))
        S,target,fallback=fixed_repair(original,st,learner,q,B)
        b,expected=evaluate(st,S,learner,original)
        records.append(dict(cohort='P2_repair',seed=seed,policy='Fixed-Round-Repaired',brier=b,expected_brier=expected,k=len(S),K=len(M),B=B,
                            envelope=original.bernstein_envelope(q[S],st['ead_C'][S]),fallback=fallback,target_k=target))
    df=pd.DataFrame(records);df.to_csv(output/'p2_repair.csv',index=False)
    return df


def realx_state(data, seed, mechanism):
    rng=np.random.default_rng(seed)
    ids=rng.permutation(len(data['y']))[:546]
    Xraw=data['X'][ids];eraw=data['ead'][ids]
    prep=eng.SplitBeforePreprocessor(tail_scale=1).fit(Xraw[:80],eraw[:80])
    prep.tau_aux=np.zeros(2)  # Retain standardized coordinates; no artificial tail deadzone or group distortion.
    X=prep.transform_X(Xraw);e=prep.transform_ead(eraw)
    logit=-.8+X@np.array([.45,.65,-.5,.35,.2,-.15])
    if mechanism=='nonlinear':
        logit+=.5*np.sin(X[:,1])+.35*X[:,2]*X[:,3]
    prob=expit(logit);y=rng.binomial(1,prob)
    return {'X_D':X[:80],'y_D':y[:80],'X_O':X[80:102],'y_O':y[80:102],
            'X_C':X[102:126],'y_C':y[102:126],'ead_C':e[102:126],'p_C_true':prob[102:126],
            'X_Q':X[126:306],'X_T':X[306:546],'y_T':y[306:546],'p_T_true':prob[306:546]}


def common_selection(X_D,y_D,X_O,X_C,ead,X_Q,seed):
    """No hidden candidate, ordinary, test labels or true probabilities accepted."""
    learner=eng.UnifiedLogisticLearner(l2=.45).fit(X_D,y_D)
    q,_=eng.compute_calibrated_upper_risk(learner,X_D,y_D,X_C,rng=np.random.default_rng(seed+100))
    B,bhash=eng.compute_exogenous_budget(q,ead,target_k=6)
    M,certificate=eng.solve_safe_minrisk_exact_or_bounded(q,ead,B)
    worlds=eng.sample_common_parameter_worlds(learner,n_worlds=20,rng=np.random.default_rng(seed+700))
    # Family construction deliberately ignores O. Pending on/off therefore changes
    # only the rescoring geometry, with neither hidden O access nor a changing search space.
    family,*_=eng.generate_candidate_family(X_D,None,X_C,X_Q,q,ead,B,M,learner)
    candidates={tuple(np.sort(x)):np.sort(x) for x in family if len(x)<=len(M)}
    for k in range(len(M)+1):
        x=np.sort(M[:k]);candidates[tuple(x)]=x
    family=[candidates[k] for k in sorted(candidates)]
    fam_hash=digest([x.tolist() for x in family]);anchor=next(i for i,x in enumerate(family) if np.array_equal(x,M))
    Xb=np.column_stack([np.ones(len(X_C)),X_C]);deltas={};point_values={}
    for pending in [0,1]:
        O=X_O if pending else None
        H,Hi,G=eng.build_pending_fisher_and_target(X_D,O,X_Q,learner.w)
        wc=np.clip(learner.predict_proba(X_C)*(1-learner.predict_proba(X_C)),1e-6,None)
        point_values[pending]=np.array([eng.pending_aware_value(S,Xb,wc,H,Hi,G) for S in family])
        all_values=[]
        for w in worlds[1:]:
            H,Hi,G=eng.build_pending_fisher_and_target(X_D,O,X_Q,w)
            p=expit(Xb@w)
            weights=np.clip(p*(1-p)*np.exp(-.45*(Xb@(w-learner.w))**2),1e-6,None)
            values=np.array([eng.pending_aware_value(S,Xb,weights,H,Hi,G) for S in family])
            all_values.append(np.tanh((values-max(values[anchor],1e-6))/(2.5*max(values[anchor],1e-6))))
        deltas[pending]=np.array(all_values)
    choices={};scores={};details={}
    for pending,quantile,penalty,gate in itertools.product([0,1],repeat=4):
        name=f'P{pending}Q{quantile}L{penalty}G{gate}'
        raw=np.quantile(deltas[pending],.2 if quantile else .8,axis=0)
        score=raw-.22*penalty*np.array([max(len(M)-len(S),0)/max(len(M),1) for S in family])
        best=int(np.argmax(score))
        use=best if not gate or score[best]>.08 else anchor
        choices[name]=family[use];scores[name]=score
        details[name]={'surplus':float(score[best]),'gate_triggered':bool(use!=anchor),'chosen_score':float(score[use])}
    for rounding in ['round','ceil']:
        k=int(np.round(.75*len(M))) if rounding=='round' else int(np.ceil(.75*len(M)))
        k=max(k,1) if len(M) else 0
        eligible=[i for i,S in enumerate(family) if len(S)==k]
        best=max(eligible,key=lambda i:point_values[1][i]);choices['Fixed-'+rounding]=family[best]
    choices['Safe-MinRisk']=M
    for policy in ['DC-PRISM','BAIT-Cost-Safe']:
        choices['Package-'+policy],_,_=eng.select_acquisition_policy(policy,X_D,y_D,X_O,X_C,ead,X_Q,B,
                        learner=learner,q_C_precomputed=q,w_worlds_precomputed=worlds,rng=np.random.default_rng(seed+400))
    for S in choices.values():
        assert len(set(S))==len(S) and eng.bernstein_envelope(q[S],ead[S])<=B+1e-10
    return choices,learner,q,B,bhash,family,fam_hash,scores,details,certificate


def one_world(st, cohort, seed):
    start=time.perf_counter()
    args=[st[k] for k in ['X_D','y_D','X_O','X_C','ead_C','X_Q']]
    choices,learner,q,B,bhash,family,fhash,scores,details,cert=common_selection(*args,seed)
    # Every method's selection is locked before the evaluation service touches hidden labels.
    locked={k:v.copy() for k,v in choices.items()}
    cache={}
    def evaluate_cached(S):
        key=tuple(S)
        if key not in cache:cache[key]=evaluate(st,S,learner)
        return cache[key]
    family_values=[evaluate_cached(S) for S in family]
    expected=np.array([x[1] for x in family_values]);realized=np.array([x[0] for x in family_values])
    ranks=spearmanr(scores['P1Q1L1G1'],-expected).statistic if len(family)>2 and np.ptp(expected)>1e-12 and np.ptp(scores['P1Q1L1G1'])>1e-12 else np.nan
    rows=[]
    for policy,S in locked.items():
        b,eb=evaluate_cached(S);risk=eng.bernstein_envelope(q[S],st['ead_C'][S])
        rows.append({'cohort':cohort,'seed':seed,'policy':policy,'brier':b,'expected_brier':eb,'k':len(S),'K':cert['k_safe_lower'],
          'B':B,'budget_hash':bhash,'envelope':risk,'realized_loss':float(np.sum(st['y_C'][S]*st['ead_C'][S])),
          'q_selected_coverage':bool(np.all(q[S]>=st['p_C_true'][S])),
          'family_hash':fhash,'family_size':len(family),'selected_indices':json.dumps(S.tolist()),
          'selected_hash':array_hash(S),'state_visible_hash':array_hash(*args),
          'expected_regret_to_family_oracle':eb-float(expected.min()),
          'full_score_expected_gain_spearman':ranks,'anchor_count_upper':cert['k_safe_upper'],
          **details.get(policy,{})})
    elapsed=time.perf_counter()-start
    for row in rows:row['world_total_seconds']=elapsed
    return rows


CONTRASTS=[('gate','P1Q1L1G0'),('quantity_penalty','P1Q1L0G1'),('quantile_only','P1Q0L1G1'),
           ('pending_rescoring','P0Q1L1G1'),('fixed_cardinality','Fixed-ceil'),('risk_anchor','Safe-MinRisk'),('fisher_adaptation','Package-BAIT-Cost-Safe')]


def summarize(df, output):
    summary=df.groupby(['cohort','policy'],as_index=False).agg(n_worlds=('seed','nunique'),mean_brier=('brier','mean'),mean_expected_brier=('expected_brier','mean'),mean_k=('k','mean'),mean_envelope=('envelope','mean'),mean_realized_loss=('realized_loss','mean'),q_selected_coverage=('q_selected_coverage','mean'),mean_expected_family_regret=('expected_regret_to_family_oracle','mean'),mean_proxy_expected_rank=('full_score_expected_gain_spearman','mean'))
    summary.to_csv(output/'method_summary.csv',index=False)
    rows=[];m=35  # Five predeclared cohorts times seven primary contrasts.
    for cohort,g in df.groupby('cohort'):
        pivot=g.pivot(index='seed',columns='policy',values='brier')
        for name,comp in CONTRASTS:
            diffs=(pivot[comp]-pivot['P1Q1L1G1']).to_numpy();n=len(diffs);mean=float(diffs.mean());sd=float(diffs.std(ddof=1));se=sd/np.sqrt(n)
            critical=float(t.ppf(1-.05/(2*m),n-1));lo=mean-critical*se;hi=mean+critical*se
            p=float(ttest_1samp(diffs,0).pvalue) if sd>1e-15 else (1. if abs(mean)<1e-15 else 0.)
            decision='material_positive' if lo>.0001 else ('positive_below_material_threshold' if lo>0 else ('negative' if hi<0 else ('within_material_band' if lo>-.0001 and hi<.0001 else 'indeterminate')))
            rows.append(dict(cohort=cohort,contrast=name,competitor=comp,n=n,mean_gain=mean,ci95_low=mean-float(t.ppf(.975,n-1))*se,ci95_high=mean+float(t.ppf(.975,n-1))*se,
                             simultaneous_bonferroni_low=lo,simultaneous_bonferroni_high=hi,p_raw=p,decision=decision))
    stats=pd.DataFrame(rows);order=np.argsort(stats.p_raw.to_numpy());adjusted=np.empty(len(stats));running=0
    for rank,i in enumerate(order):running=max(running,min(1,stats.p_raw.iloc[i]*(len(stats)-rank)));adjusted[i]=running
    stats['p_holm_global35']=adjusted;stats.to_csv(output/'primary_statistics.csv',index=False)
    return summary,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original,generate=original_modules()
    datafile=args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest()==protocol['data_sha256']
    started=time.time();manifest={'status':'RUNNING','protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'protocol':protocol,'python':platform.python_version(),'numpy':np.__version__,
          'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/prism_engine.py',ROOT/'archive/submitted_v2_2/src/prism_engine.py',ROOT/'archive/submitted_v2_2/src/run_p0_p1_p2.py']}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        repair=repair_old_worlds(out)
        print('P2 frozen-world repair completed',flush=True)
        archive=np.load(datafile,allow_pickle=False);rows=[]
        for cohort,settings in protocol['cohorts'].items():
            for seed in range(settings['seed_start'],settings['seed_start']+settings['n_worlds']):
                if cohort=='Controlled':st=generate(seed)
                else:
                    data={k:archive[settings['dataset']+'_'+k] for k in ['X','y','ead']}
                    st=realx_state(data,seed,settings['mechanism'])
                rows.extend(one_world(st,cohort,seed))
            print(f'{cohort}: {settings["n_worlds"]} worlds complete',flush=True)
        df=pd.DataFrame(rows);df.to_csv(out/'world_results.csv',index=False)
        summary,stats=summarize(df,out)
        assert len(df)==sum(x['n_worlds'] for x in protocol['cohorts'].values())*21
        assert df.groupby(['cohort','seed']).family_hash.nunique().eq(1).all()
        assert (df.envelope<=df.B+1e-9).all()
        old=repair.pivot(index='seed',columns='policy',values='brier');gain=old['Fixed-Round-Repaired']-old['DC-PRISM']
        se=gain.std(ddof=1)/np.sqrt(len(gain));crit=t.ppf(.975,len(gain)-1)
        repair_summary={'worlds':128,'archived_rows_verified':512,'fallbacks':int(repair.fallback.sum()),'original_fixed_mean':float(old['Fixed-Eps-Safe-PA'].mean()),'repaired_fixed_mean':float(old['Fixed-Round-Repaired'].mean()),'DC_mean':float(old['DC-PRISM'].mean()),
                         'gain_repaired_fixed_minus_DC':float(gain.mean()),'ci95':[float(gain.mean()-crit*se),float(gain.mean()+crit*se)],'study_role':'repair reanalysis of already-used worlds, not new confirmation'}
        (out/'p2_repair_summary.json').write_text(json.dumps(repair_summary,indent=2))
        manifest.update(status='COMPLETE',elapsed_seconds=time.time()-started,world_rows=len(df),fresh_simulation_worlds=df.groupby(['cohort','seed']).ngroups,primary_comparisons=len(stats),p2_repair=repair_summary)
        print(json.dumps(repair_summary,indent=2),flush=True)
    except BaseException as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
