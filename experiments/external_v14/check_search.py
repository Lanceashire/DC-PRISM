"""Search/accounting checks using a tiny deterministic reference fitter, without formal seeds."""
import numpy as np
import run

def fake(sets,Xe,seed,**kwargs):
    return np.asarray([np.clip(y.mean()+.02*Xe[:,0]+.01*x[:,0].mean(),.01,.99) for x,y in sets])

run.predict_many=fake
XD=np.asarray([[0.],[1.],[2.]],dtype='float32'); yD=np.asarray([0,0,1])
XC=np.asarray([[3.],[4.],[5.],[6.]],dtype='float32')
worlds=np.asarray([[0,1,1,0],[1,0,1,1]])
E=np.asarray([[1.],[2.],[3.]],dtype='float32'); yE=np.asarray([0,1,1])
P={'head_epochs':60,'head_hidden':32,'fit_chunk':1536}
actual=run.greedy(XD,yD,XC,worlds,E,yE,2,99001,P)
chosen=[]
for _ in range(2):
    scores={}
    for i in range(4):
        if i in chosen:continue
        idx=chosen+[i]; scores[i]=np.mean([np.mean((fake([(np.vstack([XD,XC[idx]]),np.r_[yD,w[idx]])],E,99001)[0]-yE)**2) for w in worlds])
    chosen.append(min(scores,key=scores.get))
assert np.array_equal(actual,chosen)
assert len(set(actual.tolist()))==2
p=run.coreset(XD,XC,3)
assert len(set(p.tolist()))==3
print('PASS: greedy picks match independent sequential enumeration; exactly-b; CoreSet uniqueness.')
