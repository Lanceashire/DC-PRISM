"""Full acquisition curves for frozen deep encoders + a refitted nonlinear neural head.

This is an external adaptation of v13, not an end-to-end CDALBench reproduction.
Candidate search and training budgets are the same for all three oracles.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from gpu_head import predict_many

ROOT=Path(__file__).resolve().parent
METHODS=['oracle-test','oracle-split','oracle-exante','Random','Uncertainty','CoreSet']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(name,seed,P):
    a=np.load(ROOT/(name+'.npz')); X=a['X']; y=a['y']; ids=a['ids']; sp=a['official_split']
    rng=np.random.default_rng(seed)
    tr=rng.permutation(np.flatnonzero(sp=='train')); te=rng.permutation(np.flatnonzero(sp=='test'))[:P['test_size']]
    naux=P['aux_size']; nv=P['validation_size']; nq=P['reference_size']; nh=P['initial_labels']; ns=P['steps']*P['pool']
    aux=tr[:naux]; val=tr[naux:naux+nv]; q=tr[naux+nv:naux+nv+nq]
    h=tr[naux+nv+nq:naux+nv+nq+nh]; stream=tr[naux+nv+nq+nh:naux+nv+nq+nh+ns]
    parts=[aux,val,q,h,stream,te]
    assert len(stream)==ns
    assert sum(map(len,parts))==len(set(np.concatenate(parts).tolist()))
    # No target/test/validation labels enter the feature transform or outcome model.
    pca=PCA(n_components=P['pca_dim'],svd_solver='full').fit(X[aux])
    scale=np.sqrt(pca.explained_variance_).clip(.01)
    def tf(z): return np.clip(pca.transform(z)/scale,-5,5).astype('float32')
    Z=tf(X)
    if P['phat']=='logistic': f=LogisticRegression(C=1.,max_iter=500,random_state=seed).fit(Z[aux],y[aux])
    else: f=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1.,random_state=seed).fit(Z[aux],y[aux])
    ph=f.predict_proba(Z[stream])[:,1].clip(.01,.99)
    # Calibration diagnostics use a held-out subset of the auxiliary set only.
    ac=aux[:int(.8*naux)]; ae=aux[int(.8*naux):]
    if P['phat']=='logistic': fc=LogisticRegression(C=1.,max_iter=500,random_state=seed).fit(Z[ac],y[ac])
    else: fc=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1.,random_state=seed).fit(Z[ac],y[ac])
    pp=fc.predict_proba(Z[ae])[:,1]
    audit={'dataset':name,'seed':seed,'counts':dict(zip(['aux','validation','reference','history','stream','test'],map(len,parts))),
           'pairwise_disjoint':True,'split_hashes':{k:hashlib.sha256('\n'.join(ids[z]).encode()).hexdigest() for k,z in zip(['aux','validation','reference','history','stream','test'],parts)},
           'phat_aux_holdout_brier':float(np.mean((pp-y[ae])**2)), 'phat_aux_holdout_accuracy':float(np.mean((pp>.5)==y[ae])),
           'noise_index':float(np.mean(ph*(1-ph)))}
    return dict(XH=Z[h],yH=y[h],XT=Z[te],yT=y[te],XV=Z[val],yV=y[val],XQ=Z[q],XS=Z[stream],yS=y[stream],pS=ph,audit=audit)

def greedy(XD,yD,XC,labels,XE,yE,b,seed,P):
    # 'labels' is either one recorded-label world or R sampled worlds. The function
    # cannot access any other candidate labels. All candidate fits use the same seed.
    selected=[]; R=len(labels)
    for _ in range(b):
        remaining=[i for i in range(len(XC)) if i not in selected]
        jobs=[]
        for i in remaining:
            idx=selected+[i]; x=np.vstack([XD,XC[idx]])
            jobs.extend((x,np.concatenate([yD,l[idx]])) for l in labels)
        pred=predict_many(jobs,XE,seed,epochs=P['head_epochs'],hidden=P['head_hidden'],chunk=P['fit_chunk'])
        score=((pred-yE)**2).mean(1).reshape(len(remaining),R).mean(1)
        selected.append(remaining[int(np.argmin(score))])
    return np.asarray(selected)

def coreset(XD,XC,b):
    dist=((XC[:,None,:]-XD[None,:,:])**2).sum(-1).min(1); pick=[]
    for _ in range(b):
        i=int(np.argmax(dist)); pick.append(i)
        dist=np.minimum(dist,((XC-XC[i])**2).sum(1)); dist[pick]=-np.inf
    return np.asarray(pick)

def trajectory(name,seed,P,out):
    started=time.time(); S=build(name,seed,P); b=P['batch']; pool=P['pool']; R=P['replicates']
    lab={m:(S['XH'].copy(),S['yH'].copy()) for m in METHODS}
    E=np.vstack([S['XT'],S['XV']]); nt=len(S['XT']); nv=len(S['XV'])
    def record(step,method,p):
        return dict(dataset=name,seed=seed,method=method,step=step,n_labels=P['initial_labels']+step*b,
                    brier_T=float(np.mean((p[:nt]-S['yT'])**2)),brier_V=float(np.mean((p[nt:]-S['yV'])**2)),
                    accuracy_T=float(np.mean((p[:nt]>.5)==S['yT'])),noise_index=S['audit']['noise_index'])
    p0=predict_many([(S['XH'],S['yH'])],E,seed,epochs=P['head_epochs'],hidden=P['head_hidden'],chunk=P['fit_chunk'])[0]
    rows=[record(0,m,p0) for m in METHODS]
    for t in range(P['steps']):
        sl=slice(t*pool,(t+1)*pool); XC,yC,pC=S['XS'][sl],S['yS'][sl],S['pS'][sl]
        reps=(np.random.default_rng([seed,t,180003]).random((R,pool))<pC).astype('int64')
        for m in METHODS:
            XD,yD=lab[m]
            if m=='oracle-test': pick=greedy(XD,yD,XC,yC[None,:],S['XT'],S['yT'],b,seed,P)
            elif m=='oracle-split': pick=greedy(XD,yD,XC,yC[None,:],S['XV'],S['yV'],b,seed,P)
            elif m=='oracle-exante': pick=greedy(XD,yD,XC,reps,S['XV'],S['yV'],b,seed,P)
            elif m=='Random': pick=np.random.default_rng([seed,t,180004]).choice(pool,size=b,replace=False)
            elif m=='CoreSet': pick=coreset(XD,XC,b)
            else:
                p=predict_many([(XD,yD)],XC,seed,epochs=P['head_epochs'],hidden=P['head_hidden'],chunk=P['fit_chunk'])[0]
                pick=np.argsort(abs(p-.5),kind='stable')[:b]
            assert len(pick)==b and len(set(pick.tolist()))==b
            XD=np.vstack([XD,XC[pick]]); yD=np.concatenate([yD,yC[pick]]); lab[m]=(XD,yD)
            pred=predict_many([(XD,yD)],E,seed,epochs=P['head_epochs'],hidden=P['head_hidden'],chunk=P['fit_chunk'])[0]
            rows.append(record(t+1,m,pred))
        (out/'progress.json').write_text(json.dumps({'dataset':name,'seed':seed,'step':t+1,'elapsed_trajectory_seconds':time.time()-started}))
    pd.DataFrame(rows).to_csv(out/f'{name}_{seed}.csv',index=False)
    (out/f'{name}_{seed}_audit.json').write_text(json.dumps(S['audit'],indent=2))
    return time.time()-started

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    P=json.loads(args.protocol.read_text()); formal=not args.smoke
    if formal:
        F=json.loads((args.protocol.parent/'freeze.json').read_text())
        assert F['protocol_sha256']==sha(args.protocol)
        for name,h in F['sources'].items(): assert sha(ROOT/name)==h,name
        for name,h in F['data_sha256'].items(): assert sha(ROOT/name)==h,name
    else:
        P.update(steps=2,pool=6,batch=2,replicates=2,seeds=[99001])
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    started=time.time();M={'status':'RUNNING','smoke':args.smoke,'protocol_sha256':sha(args.protocol),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'completed':[]}
    (out/'run_manifest.json').write_text(json.dumps(M,indent=2))
    try:
        with threadpool_limits(limits=4):
            for ds in P['datasets']:
                for s in P['seeds']:
                    seconds=trajectory(ds,s,P,out);M['completed'].append({'dataset':ds,'seed':s,'seconds':seconds})
                    (out/'run_manifest.json').write_text(json.dumps(M,indent=2))
                    print(ds,s,round(seconds,1),flush=True)
        M.update(status='COMPLETE',elapsed_seconds=time.time()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        M.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally: (out/'run_manifest.json').write_text(json.dumps(M,indent=2))

if __name__=='__main__': main()
