"""Deterministic batched nonlinear heads; each model gets its own independent parameters."""
import numpy as np
import torch

def predict_many(sets, Xe, seed, epochs=60, hidden=32, chunk=1536):
    ps=[]
    for start in range(0,len(sets),chunk):
        part=sets[start:start+chunk]; N=len(part); d=part[0][0].shape[1]; n=max(len(y) for _,y in part)
        X=np.zeros((N,n,d),np.float32); Y=np.zeros((N,n),np.float32); mask=np.zeros_like(Y)
        for j,(x,y) in enumerate(part):
            X[j,:len(y)]=x; Y[j,:len(y)]=y; mask[j,:len(y)]=1
        X,Y,mask=[torch.as_tensor(a,device='cuda') for a in (X,Y,mask)]
        g=torch.Generator().manual_seed(seed)
        W=(torch.randn(1,d,hidden,generator=g)*np.sqrt(2/d)).repeat(N,1,1).cuda().requires_grad_()
        b=torch.zeros(N,1,hidden,device='cuda',requires_grad=True)
        v=(torch.randn(1,hidden,1,generator=g)*np.sqrt(1/hidden)).repeat(N,1,1).cuda().requires_grad_()
        c=torch.zeros(N,1,1,device='cuda',requires_grad=True)
        # One common initialization, including output bias, irrespective of hypothetical labels.
        opt=torch.optim.Adam([W,b,v,c],lr=.02,foreach=True)
        den=mask.sum(1)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            z=(torch.relu(X@W+b)@v+c).squeeze(-1)
            losses=(torch.nn.functional.binary_cross_entropy_with_logits(z,Y,reduction='none')*mask).sum(1)/den
            loss=losses.sum()+.001*(W.square().sum()+v.square().sum())
            if not torch.isfinite(loss): raise RuntimeError('nonfinite head loss')
            loss.backward(); opt.step()
        with torch.inference_mode():
            E=torch.as_tensor(np.asarray(Xe,np.float32),device='cuda').unsqueeze(0).expand(N,-1,-1)
            p=torch.sigmoid((torch.relu(E@W+b)@v+c).squeeze(-1))
            ps.append(p.cpu().numpy().astype(float))
    return np.concatenate(ps)

if __name__=='__main__':
    import time,json
    torch.set_num_threads(4)
    rng=np.random.default_rng(99003)
    results=[]
    for N in [1,40,320]:
        sets=[(rng.normal(size=(168,32)).astype('float32'),rng.integers(2,size=168)) for _ in range(N)]
        E=rng.normal(size=(1024,32)).astype('float32')
        t=time.time(); p=predict_many(sets,E,99003); elapsed=time.time()-t
        results.append({'models':N,'seconds':elapsed,'shape':list(p.shape),'peak_MB':torch.cuda.max_memory_allocated()/2**20})
    # Model packing/chunking must not affect a fit. Labels for one fit must not affect another.
    X=rng.normal(size=(80,32)).astype('float32'); y=rng.integers(2,size=80)
    a=predict_many([(X,y)],E,99001)
    b=predict_many([(X,y),(X,1-y)],E,99001,chunk=2)
    c=predict_many([(X,y),(X,1-y)],E,99001,chunk=1)
    assert np.max(abs(a[0]-b[0]))<2e-5
    assert np.max(abs(b-c))<2e-5
    print(json.dumps({'benchmark':results,'packing_check_max_abs':float(np.max(abs(b-c))),'status':'PASS'},indent=2))
