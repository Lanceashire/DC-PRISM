"""Prespecified paired-seed inference; absence of significance is not equivalence."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ORACLES=['oracle-test','oracle-split','oracle-exante']
STRATEGIES=['Random','Uncertainty','CoreSet']

def family(items,alpha,delta):
    m=len(items);out=[]
    for key,values in items:
        v=np.asarray(values,float); n=len(v); mean=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
        width=student_t.ppf(1-alpha/(2*m),n-1)*se
        p=float(student_t.sf(mean/se,n-1)) if se>0 else float(mean<=0)
        pm=float(student_t.sf((mean-delta)/se,n-1)) if se>0 else float(mean<=delta)
        out.append(dict(**key,n=n,mean=mean,se=se,ci_low=mean-width,ci_high=mean+width,p_positive=p,p_material=pm,
                        positive=bool(mean-width>0),material_positive=bool(mean-width>delta),
                        certified_not_material_positive=bool(mean+width<=delta)))
    df=pd.DataFrame(out)
    for col in ['p_positive','p_material']:
        order=np.argsort(df[col].to_numpy()); adjusted=np.empty(m); running=0.
        for k,j in enumerate(order):
            running=max(running,min(1.,(m-k)*df.iloc[j][col]));adjusted[j]=running
        df[col+'_holm']=adjusted
    return df

def summarize(out,protocol):
    P=json.loads(protocol.read_text());M=json.loads((out/'run_manifest.json').read_text());assert M['status']=='COMPLETE' and not M['smoke']
    files=[out/f'{ds}_{s}.csv' for ds in P['datasets'] for s in P['seeds']]
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    assert len(df)==len(P['datasets'])*len(P['seeds'])*6*31
    assert df.groupby(['dataset','seed','method']).step.apply(lambda x:set(x)==set(range(31))).all()
    assert (df.n_labels==68+df.step*10).all() and np.isfinite(df.select_dtypes('number')).all().all()
    df.to_csv(out/'rows.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    tabs={};summary=[]
    for window,(lo,hi) in {'all':(1,30),'early':(1,10),'late':(21,30)}.items():
        x=df[df.step.between(lo,hi)].pivot_table(index=['dataset','seed'],columns='method',values='brier_T',aggfunc='mean')
        x['reuse']=x['oracle-split']-x['oracle-test'];x['luck']=x['oracle-exante']-x['oracle-split'];tabs[window]=x
        x.to_csv(out/f'per_trajectory_{window}.csv')
        for ds,g in x.groupby(level=0):
            means=g.mean();best=means[STRATEGIES].idxmin()
            summary.append({'window':window,'dataset':ds,'best_strategy_descriptive':best,
                            **{m:means[m] for m in ORACLES+STRATEGIES},'reuse':means['reuse'],'luck':means['luck'],
                            'best_minus_test':means[best]-means['oracle-test'],'best_minus_exante':means[best]-means['oracle-exante']})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    alpha=P['inference']['alpha'];delta=P['inference']['material_delta'];items=[]
    for ds in P['datasets']:
        for endpoint in ['reuse','luck']:items.append(({'dataset':ds,'endpoint':endpoint},tabs['all'].loc[ds,endpoint]))
    whole=family(items,alpha,delta);whole.to_csv(out/'primary_components.csv',index=False)
    items=[]
    for ds in P['datasets']:
        e=tabs['early'].loc[ds,'luck'];l=tabs['late'].loc[ds,'luck']
        items.extend([({'dataset':ds,'endpoint':'early_luck'},e),({'dataset':ds,'endpoint':'early_minus_late_luck'},e-l)])
    early=family(items,alpha,delta);early.to_csv(out/'primary_early.csv',index=False)
    items=[]
    for ds in P['datasets']:
        for o in ORACLES:
            for s in STRATEGIES:items.append(({'dataset':ds,'oracle':o,'strategy':s},tabs['all'].loc[ds,s]-tabs['all'].loc[ds,o]))
    short=family(items,alpha,delta);short.to_csv(out/'primary_shortfalls.csv',index=False)
    checks={}
    checks['H1']=bool(whole[whole.endpoint=='reuse'].positive.all())
    checks['H2']=bool(early.positive.all())
    h3=[]
    for ds in P['datasets']:
        z=short[short.dataset==ds]
        alltest=bool(z[z.oracle=='oracle-test'].material_positive.all())
        anyex=bool(z[z.oracle=='oracle-exante'].certified_not_material_positive.any())
        h3.append(alltest and anyex)
        checks[ds]={'all_materially_short':{o:bool(z[z.oracle==o].material_positive.all()) for o in ORACLES},
                    'any_certified_not_materially_short':{o:bool(z[z.oracle==o].certified_not_material_positive.any()) for o in ORACLES}}
    checks['H3']=all(h3)
    (out/'prediction_check.json').write_text(json.dumps(checks,indent=2))
    (out/'integrity_check.json').write_text(json.dumps({'status':'PASS','trajectories':len(files),'rows':len(df),'all_curves_complete':True,'uniform_label_budget':True,'finite_metrics':True},indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args();summarize(a.output,a.protocol)
