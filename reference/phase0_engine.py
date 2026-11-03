import json, hashlib, random, inspect, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT=Path(os.environ.get("ECOMIC_ROOT", Path(__file__).resolve().parents[1])); DATA=ROOT/"data/raw/hmeq.csv"; OUT=ROOT/"results_phase0"; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[0,1]; RHOS=[0.025,0.05,0.10,0.20]; ROUNDS=10; TAU=0.5; DEVICE="cuda:0"

FAVE_GAMMA=1.0
FAVE_DELTA=0.5
POLICIES=["NoExplore","CountOnly","CountOnly-MinRisk","LeastRisky","MostRisky","Uncertainty","RiskTilted-Uncertainty","FAVE-v2.1"]
def policy_score(name,p,cost):
    if name=="CountOnly": return -1e-9*np.arange(len(p))
    if name=="CountOnly-MinRisk": return -cost
    if name=="LeastRisky": return -p
    if name=="MostRisky": return p
    if name=="Uncertainty": return p*(1-p)
    if name=="RiskTilted-Uncertainty": return p*(1-p)*np.sqrt(np.clip(p,1e-9,1))
    raise ValueError(name)

def optimize(cost,score,budget):
    n=len(cost)
    if n==0 or budget<=0: return np.array([],int),0
    integ=np.ones(n,int); bnd=Bounds(np.zeros(n),np.ones(n)); con=LinearConstraint(cost.reshape(1,-1),-np.inf,budget)
    r1=milp(c=-np.ones(n),integrality=integ,bounds=bnd,constraints=con)
    if not r1.success: raise RuntimeError("stage1 MILP failed")
    k=int(np.rint(r1.x).sum())
    r2=milp(c=-np.asarray(score,float),integrality=integ,bounds=bnd,constraints=[con,LinearConstraint(np.ones((1,n)),k,k)])
    if not r2.success: raise RuntimeError("stage2 MILP failed")
    chosen=np.flatnonzero(np.rint(r2.x)>0.5)
    return chosen,k

def make_pre(raw):
    num=[c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]; cat=[c for c in raw.columns if c not in num]
    return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="Missing")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])

def fit_model(raw_X,y,visible,seed):
    pre=make_pre(raw_X); Xt=pre.fit_transform(raw_X.iloc[visible]); Xa=pre.transform(raw_X)
    model=XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,objective="binary:logistic",eval_metric="logloss",random_state=seed,n_jobs=1,tree_method="hist",device=DEVICE)
    model.fit(Xt,y[visible]); return pre,model,model.predict_proba(Xa)[:,1]

def predict(raw_X,y,visible,seed):
    return fit_model(raw_X,y,visible,seed)[2]

def select_round(policy, p, cost, budget, fave_score=None):
    """Select from current-round predictions only; no applicant labels are accepted."""
    p=np.asarray(p,float); cost=np.asarray(cost,float)
    if policy=="NoExplore" or len(p)==0:
        return np.array([],dtype=int), 0
    score=fave_score if policy=="FAVE-v2.1" else policy_score(policy,p,cost)
    local,k=optimize(cost,score,budget)
    return local,k

def _ridge_inverse(a):
    """Small dense inverse using scalar operations; avoids unstable BLAS calls."""
    a=np.asarray(a,dtype=float); n=a.shape[0]
    aug=[[float(a[i,j]) for j in range(n)] + [1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot=max(range(col,n),key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            aug[col][col] += 1e-8
            pivot=max(range(col,n),key=lambda r: abs(aug[r][col]))
        if pivot != col: aug[col],aug[pivot]=aug[pivot],aug[col]
        scale=aug[col][col]
        aug[col]=[v/scale for v in aug[col]]
        for r in range(n):
            if r==col: continue
            factor=aug[r][col]
            if factor: aug[r]=[x-factor*y for x,y in zip(aug[r],aug[col])]
    return np.asarray([row[n:] for row in aug],dtype=float)

def fave_value(raw_X, visible, rejected, reference, pre, model, gamma=FAVE_GAMMA, delta=FAVE_DELTA):
    """Frozen FAVE-v2.1 local Fisher/influence EVI, with no label access."""
    visible=np.asarray(visible,dtype=int); rejected=np.asarray(rejected,dtype=int); reference=np.asarray(reference,dtype=int)
    if len(rejected)==0: return np.empty(0,dtype=float)
    Zv=pre.transform(raw_X.iloc[visible]); Zr=pre.transform(raw_X.iloc[rejected]); Zf=pre.transform(raw_X.iloc[reference])
    pv=np.clip(model.predict_proba(Zv)[:,1],1e-9,1-1e-9)
    pr=np.clip(model.predict_proba(Zr)[:,1],1e-9,1-1e-9)
    pf=np.clip(model.predict_proba(Zf)[:,1],1e-9,1-1e-9)
    V=np.column_stack([np.ones(len(Zv)),Zv]); R=np.column_stack([np.ones(len(Zr)),Zr]); F=np.column_stack([np.ones(len(Zf)),Zf])
    wv=np.clip(pv*(1-pv),1e-6,None)
    wi=np.clip(pr*(1-pr)*(pr**float(delta)),1e-10,None)
    Vw=V*np.sqrt(wv)[:,None]
    H=np.einsum("ni,nj->ij",Vw,Vw)
    ridge=max(float(np.trace(H)/H.shape[0])*1e-6,1e-10)
    Hinv=_ridge_inverse(H+ridge*np.eye(H.shape[0]))
    G=np.einsum("n,ni,nj->ij",pf**float(gamma),F,F)/max(len(F),1)
    HG=np.einsum("ij,jk,kl->il",Hinv,G,Hinv)
    lev=np.einsum("ij,jk,ik->i",R,Hinv,R)
    num=np.einsum("ij,jk,ik->i",R,HG,R)
    value=wi*num/(1+wi*lev)
    return np.maximum(np.nan_to_num(value,nan=0.0,posinf=0.0,neginf=0.0),0.0)

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); val=0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
        if m.any(): val += m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
    return float(val)

def eval_metrics(y,p,visible,pop_rate):
    pred=(p>=TAU).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); spec=tn/(tn+fp) if tn+fp else np.nan; rec=recall_score(y,pred,zero_division=0); vis_tdr=float(np.mean(visible)) if len(visible) else np.nan
    return {"Accuracy":accuracy_score(y,pred),"Precision":precision_score(y,pred,zero_division=0),"Recall":rec,"Specificity":spec,"BalancedAccuracy":balanced_accuracy_score(y,pred),"Gmean":float(np.sqrt(max(rec*spec,0))),"ROCAUC":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,"PRAUC":average_precision_score(y,p) if len(np.unique(y))>1 else np.nan,"Brier":brier_score_loss(y,p),"ECE":ece(y,p),"CalibrationSlope":np.nan,"CalibrationIntercept":np.nan,"VisibleTDR":vis_tdr,"PopulationDefaultRate":pop_rate,"FDRatio":vis_tdr/pop_rate if pop_rate>0 else np.nan}

def run_episode(df,seed,rho,policy):
    y=df.BAD.astype(int).to_numpy(); exposure=df.LOAN.astype(float).to_numpy(); raw=df.drop(columns=["BAD"])
    idx=np.arange(len(df)); rng=np.random.RandomState(seed); ntest=int(.25*len(df)); perm=rng.permutation(idx); test=perm[:ntest]; pool=perm[ntest:]
    init_n=max(100,int(.2*len(pool))); visible,remaining=np.array(pool[:init_n]),np.array(pool[init_n:])
    batches=np.array_split(remaining,ROUNDS)
    batch_sets=[set(b.tolist()) for b in batches]
    partition_union=set().union(*batch_sets) if batch_sets else set()
    partition_disjoint=all(batch_sets[i].isdisjoint(batch_sets[j]) for i in range(len(batch_sets)) for j in range(i))
    partition_complete=partition_disjoint and all(len(b)>0 for b in batches) and partition_union==set(remaining.tolist())
    rows=[]; cumulative=0; cum_er=0.; cum_rb=0.; pop_rate=float(y.mean())
    hidden_api_ok="y" not in inspect.signature(select_round).parameters
    for t,batch in enumerate(batches):
        batch_set=batch_sets[t]; prior_union=set().union(*batch_sets[:t]) if t else set()
        pre,model,p_before=fit_model(raw,y,visible,seed+t)
        accepted=batch[p_before[batch]<TAU]; rejected=batch[p_before[batch]>=TAU]; candidate=rejected.copy()
        p=p_before[candidate]; cost=p*exposure[candidate]; budget=float(rho*cost.sum())
        selected=np.array([],int); k=0; score=None
        if policy!="NoExplore" and len(candidate):
            if policy=="FAVE-v2.1":
                score=fave_value(raw,visible,candidate,batch,pre,model)
            selected_local,k=select_round(policy,p,cost,budget,score); selected=candidate[selected_local]
        expected_visible=np.concatenate([visible,accepted,selected]).astype(int)
        visible_next=expected_visible.copy()
        selected_set=set(selected.tolist()); candidate_set=set(candidate.tolist())
        used=float(cost[np.isin(candidate,selected)].sum()) if len(selected) else 0.
        bad_exp=float(exposure[selected][y[selected]==1].sum()) if len(selected) else 0.
        current_only=(candidate_set==set(rejected.tolist()) and candidate_set.issubset(batch_set) and candidate_set.isdisjoint(prior_union) and selected_set.issubset(candidate_set))
        no_duplicate=(len(np.unique(visible_next))==len(visible_next) and selected_set.isdisjoint(set(visible.tolist())))
        visible_update=np.array_equal(visible_next,expected_visible) and set(visible_next.tolist()).issubset(set(pool.tolist()))
        test_leakage=set(test.tolist()).isdisjoint(set(visible_next.tolist()))
        risk_ok=bool(used<=budget+1e-6)
        kstar_ok=(len(selected)==k and k>=0 and (used<=budget+1e-6))
        batch_ok=bool(len(batch)>0 and batch_set.isdisjoint(prior_union))
        visible=visible_next; cumulative+=len(selected); cum_er+=used; cum_rb+=bad_exp
        p_after=predict(raw,y,visible,seed+t+10000); mm=eval_metrics(y[test],p_after[test],y[visible],pop_rate)
        mm.update({"seed":seed,"rho":rho,"round":t,"policy":policy,"n_batch":len(batch),"n_accepted":len(accepted),"n_rejected":len(rejected),"n_explored":len(selected),"k_star":k,"risk_budget":budget,"risk_used":used,"risk_constraint_ok":risk_ok,"candidate_current_round_only":current_only,"no_duplicate_applicant":no_duplicate,"hidden_label_access_ok":hidden_api_ok,"visible_update_ok":visible_update,"test_leakage_ok":test_leakage,"kstar_consistency_ok":kstar_ok,"cumulative_explored_labels":cumulative,"cumulative_expected_risk":cum_er,"cumulative_realized_bad_exposure":cum_rb,"batch_partition_ok":batch_ok,"all_applicants_partitioned":partition_complete})
        rows.append(mm)
    assert partition_complete
    assert len(set(visible.tolist()).intersection(set(test.tolist())))==0
    return rows

def main():
    df=pd.read_csv(DATA); rows=[]
    for seed in SEEDS:
        for rho in RHOS:
            for policy in POLICIES: rows.extend(run_episode(df,seed,rho,policy))
        print("seed=%d done"%seed,flush=True)
    out=pd.DataFrame(rows); out.to_csv(OUT/"hmeq_phase0_sanity_raw.csv",index=False)
    inv=[c for c in ["risk_constraint_ok","candidate_current_round_only","no_duplicate_applicant","hidden_label_access_ok","visible_update_ok","test_leakage_ok","kstar_consistency_ok","batch_partition_ok","all_applicants_partitioned"]]
    out[["seed","rho","round","policy","n_batch","n_accepted","n_rejected","n_explored","k_star","risk_budget","risk_used"]+inv].to_csv(OUT/"protocol_invariant_audit.csv",index=False)
    print("rows",len(out),"invariant_failures",int((~out[inv].all(axis=1)).sum()))
if __name__=="__main__": main()



