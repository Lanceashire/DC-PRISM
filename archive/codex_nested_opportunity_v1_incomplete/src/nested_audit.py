"""Fixed-family, independent-label-bank opportunity audit. No hidden outcomes enter operational selectors."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,importlib.util,json,time,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t
from concurrent.futures import ProcessPoolExecutor,as_completed
BASE=Path(__file__).resolve().parents[1]
REPO=BASE/'repo'
spec=importlib.util.spec_from_file_location('coverage',REPO/'experiments/supplement_v2/run.py')
cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
eng=cov.eng
DATA=Path('<LOCAL_WORKSPACE>/2026-11-03/xian/work/sources/2/DC_PRISM_DASFAA2027_Revision_Complete_Package/data/real_credit_datasets_authentic.npz')
ORIGINAL_MINIMIZE=eng.minimize
AUDIT={}
def checked_minimize(*args,**kwargs):
 r=ORIGINAL_MINIMIZE(*args,**kwargs)
 AUDIT['calls']=AUDIT.get('calls',0)+1
 if not r.success:
  AUDIT['initial_non_success']=AUDIT.get('initial_non_success',0)+1
  retry=dict(kwargs);retry['options']={'maxiter':300,'gtol':1e-5}
  r2=ORIGINAL_MINIMIZE(args[0],r.x,*args[2:],**retry)
  AUDIT['retries']=AUDIT.get('retries',0)+1
  if r2.fun<=r.fun+1e-10:r=r2
 g=float(np.max(np.abs(r.jac)))
 AUDIT['max_gradient_inf']=max(AUDIT.get('max_gradient_inf',0),g)
 if not np.isfinite(r.fun) or not np.isfinite(r.x).all() or g>1e-4:
  raise RuntimeError(f'Unacceptable logistic fit: success={r.success}, gradient={g}')
 return r
eng.minimize=checked_minimize

def probability(X,mechanism):
 logit=-.8+X@np.array([.45,.65,-.5,.35,.2,-.15])
 if mechanism=='nonlinear':logit+=.5*np.sin(X[:,1])+.35*X[:,2]*X[:,3]
 return expit(logit)

def one(job):
 global AUDIT
 AUDIT={};cell,seed,R,output=job;start=time.time()
 dataset,mechanism=cell.rsplit('_',1)
 with np.load(DATA) as a:data={k:a[f'{dataset}_{k}'] for k in ['X','y','ead']}
 arms,_=cov.make_pair(data,seed,mechanism)
 st=arms['broad'];ctx=cov.reference_context(*[st[k] for k in ['X_D','y_D','X_C','ead_C','X_Q']],seed)
 # Operational selections are all locked before independent hidden-label banks exist.
 locks={arm:cov.select(*[st[k] for k in ['X_D','y_D','X_O','X_C','X_Q']],ctx,seed) for arm,st in arms.items()}
 family=ctx['family'];mapping={tuple(s):i for i,s in enumerate(family)};anchor=mapping[tuple(ctx['M'])]
 pO=probability(st['X_O'],mechanism);pC=probability(st['X_C'],mechanism)
 rng=np.random.default_rng(seed+700000)
 outcomes=rng.binomial(1,np.r_[pO,pC],size=(2,R,len(pO)+len(pC)))
 assert np.max(np.abs(pC-st['p_C_true']))<1e-12
 assert np.max(np.abs(probability(st['X_T'],mechanism)-st['p_T_true']))<1e-12
 records=[];save={};symmetry=[]
 for arm,st in arms.items():
  choices,learner,scores,points=locks[arm]
  loss=np.empty((2,R,len(family)))
  for bank in range(2):
   for draw in range(R):
    evalstate=dict(st,y_O=outcomes[bank,draw,:len(pO)],y_C=outcomes[bank,draw,len(pO):])
    for i,S in enumerate(family):loss[bank,draw,i]=cov.sup.evaluate(evalstate,S,learner)[1]
  means=loss.mean(axis=1)
  selected=[int(np.argmin(means[bank])) for bank in range(2)]
  cross_gain=.5*sum(means[1-bank,anchor]-means[1-bank,selected[bank]] for bank in range(2))
  plugin_gain=float(loss[:,:,anchor].mean()-loss.mean(axis=(0,1)).min())
  post=float((loss[:,:,anchor]-loss.min(axis=2)).mean())
  row={'cell':cell,'seed':seed,'arm':arm,'R_per_bank':R,'family_size':len(family),'anchor_k':len(ctx['M']),
       'post_headroom':post,'crossfit_known_dgp_gain':float(cross_gain),'plugin_pre_gain':plugin_gain,
       'crossfit_choices_agree':bool(selected[0]==selected[1]),'bankA_choice':selected[0],'bankB_choice':selected[1],
       'full_chooses_anchor':bool(np.array_equal(choices['Full'],ctx['M']))}
  for name,S in choices.items():row['gain_'+name]=float((loss[:,:,anchor]-loss[:,:,mapping[tuple(S)]]).mean())
  row['crossfit_minus_full']=row['crossfit_known_dgp_gain']-row['gain_Full']
  row['post_minus_crossfit']=post-cross_gain
  records.append(row);save[arm+'_loss']=loss;save[arm+'_full_index']=mapping[tuple(choices['Full'])];save[arm+'_point_index']=mapping[tuple(choices['PointValue'])]
 save['anchor_index']=anchor;save['family_sizes']=np.array([len(s) for s in family]);save['family_hash']=ctx['family_hash']
 dest=Path(output)/'states';dest.mkdir(exist_ok=True,parents=True)
 np.savez_compressed(dest/f'{cell}_{seed}.npz',**save)
 metadata={'cell':cell,'seed':seed,'records':records,'optimizer':dict(AUDIT),'elapsed_seconds':time.time()-start,
           'family_hash':ctx['family_hash'],'budget':ctx['B'],'choices_locked_before_outcomes':True}
 (dest/f'{cell}_{seed}.json').write_text(json.dumps(metadata,indent=2))
 return metadata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');ap.add_argument('--protocol');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
 if a.engineering:
  r=one(('HMEQ_logistic',99021,4,str(BASE/'engineering')))
  print(json.dumps({k:r[k] for k in ['elapsed_seconds','optimizer','family_hash']},indent=2));print('family_size',r['records'][0]['family_size']);return
 p=json.loads(Path(a.protocol).read_text());out=BASE/'results'/p['id']
 out.mkdir(exist_ok=False,parents=True)
 frozen={'protocol':p,'protocol_sha256':hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),
         'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
 (out/'freeze.json').write_text(json.dumps(frozen,indent=2))
 jobs=[(cell,seed,p['draws_per_bank'],str(out)) for cell,ss in p['seeds'].items() for seed in ss]
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(one,j):j for j in jobs}
  for f in as_completed(fs):
   r=f.result();results.append(r)
   print(f"completed {len(results)}/{len(jobs)} {r['cell']} {r['seed']} fits={r['optimizer']['calls']} seconds={r['elapsed_seconds']:.1f}",flush=True)
 rows=[x for r in results for x in r['records']];pd.DataFrame(rows).sort_values(['cell','seed','arm']).to_csv(out/'state_metrics.csv',index=False)
 (out/'run_manifest.json').write_text(json.dumps({'complete':True,'states':len(jobs),'arms':len(rows),'optimizer_calls':sum(r['optimizer']['calls'] for r in results),'optimizer_initial_non_success':sum(r['optimizer'].get('initial_non_success',0) for r in results),'max_gradient_inf':max(r['optimizer']['max_gradient_inf'] for r in results)},indent=2))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
