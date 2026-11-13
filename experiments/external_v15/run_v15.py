"""external-v15: three oracles along full learning curves on eight standard multiclass benchmarks.

Design (supplement-v13 / external-v14, scaled up): frozen pretrained encoders, PCA-64 (fitted on label-free auxiliary
features), a retrained softmax head (MLP 64 hidden units, or linear). Start from 68 labels; 30 steps; each step 40 fresh
candidates, exactly 10 acquired. Every method follows its own trajectory; all are scored on the same test sample with
recorded labels (multiclass Brier = mean over records of sum_k (p_k - 1[y=k])^2).

  oracle-test   : greedy with the candidates' recorded labels, selecting on the test sample (CDALBench / BoSS style)
  oracle-split  : the same, selecting on a separate validation sample (Zhou et al. 2021 style)
  oracle-exante : greedy on the validation sample by the mean over R replicates of the candidates' labels drawn from a
                  multiclass GBDT outcome model fitted on disjoint auxiliary records; never sees the candidates' labels
  strategies    : Random, Margin, Entropy, CoreSet (PCA space), BADGE (gradient embeddings, k-means++)

Grids: A = primary encoder x MLP head; B = secondary encoder x MLP head; C = primary encoder x linear head.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ['oracle-test', 'oracle-split', 'oracle-exante', 'Random', 'Margin', 'Entropy', 'CoreSet', 'BADGE']
PRIMARY = {'image': 'dinov2s', 'text': 'bge'}; SECONDARY = {'image': 'mnv3s', 'text': 'minilm'}
KIND = {'cifar10': 'image', 'cifar100c': 'image', 'fmnist': 'image', 'svhn': 'image',
        'agnews': 'text', 'dbpedia': 'text', 'imdb': 'text', 'yelp': 'text'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def onehot(y, K): return np.eye(K)[y]


def brier(p, y, K): return float(np.mean(((p-onehot(y, K))**2).sum(-1)))


def build(dataset, encoder, seed, P):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    a = np.load(ROOT/'features'/f'{dataset}__{encoder}.npz'); X, y, sp = a['X'], a['y'], a['official_split']
    K = int(y.max())+1; rng = np.random.default_rng(seed)
    tr = rng.permutation(np.flatnonzero(sp == 'train')); te = rng.permutation(np.flatnonzero(sp == 'test'))[:P['test_size']]
    sizes = [P['aux_size'], P['validation_size'], P['reference_size'], P['initial_labels'], P['steps']*P['pool']]
    cut = np.cumsum([0]+sizes); aux, val, ref, hist, stream = (tr[cut[i]:cut[i+1]] for i in range(5))
    assert len(stream) == sizes[-1]
    pca = PCA(n_components=P['pca_dim'], svd_solver='full', random_state=0).fit(X[aux])
    sc = np.sqrt(pca.explained_variance_).clip(.01)
    Z = np.clip(pca.transform(X)/sc, -5, 5).astype('float32')
    gb = lambda idx: HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1., random_state=seed % (2**31)).fit(Z[idx], y[idx])
    f = gb(aux); pS = np.clip(f.predict_proba(Z[stream]), 1e-4, 1)
    full = np.zeros((len(stream), K)); full[:, f.classes_] = pS; pS = full/full.sum(1, keepdims=True)
    n8 = int(.8*len(aux)); fc = gb(aux[:n8]); pe = np.zeros((len(aux)-n8, K)); pe[:, fc.classes_] = fc.predict_proba(Z[aux[n8:]])
    audit = dict(dataset=dataset, encoder=encoder, seed=seed, classes=K, noise_index=float(np.mean(1-(pS**2).sum(1))),
                 phat_holdout_brier=brier(pe, y[aux[n8:]], K), phat_holdout_acc=float(np.mean(pe.argmax(1) == y[aux[n8:]])),
                 disjoint=len(set(np.concatenate([aux, val, ref, hist, stream, te]).tolist())) == sum(sizes)+len(te))
    assert audit['disjoint']
    return dict(K=K, XH=Z[hist], yH=y[hist], XT=Z[te], yT=y[te], XV=Z[val], yV=y[val], XS=Z[stream], yS=y[stream], pS=pS, audit=audit)


def run_trajectory(task):
    dataset, encoder, head, seed, P, outdir = task
    import torch, heads
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(outdir); f = out/f'{dataset}__{encoder}__{head}__{seed}.csv'
    if f.exists(): return str(f), 0.
    t0 = time.time(); S = build(dataset, encoder, seed, P); K = S['K']
    hid = P['hidden'] if head == 'mlp' else 0; ep = P['epochs']; lr = P['lr_mlp'] if head == 'mlp' else P['lr_linear']
    fit_pred = lambda sets, Xe: heads.predict_many(sets, Xe, K, seed, hid, ep, lr, P['chunk'])
    b, pool, R = P['batch'], P['pool'], P['replicates']; nt = len(S['yT'])
    E = np.vstack([S['XT'], S['XV']])
    lab = {m: (S['XH'].copy(), S['yH'].copy()) for m in METHODS}

    def greedy(XD, yD, XC, worlds, XE, yE):
        return heads.greedy_gpu(XD, yD, XC, worlds, XE, yE, K, b, seed, hid, ep, lr, P['chunk'])

    rows = []
    p0 = fit_pred([(S['XH'], S['yH'])], E)[0]
    for m in METHODS:
        rows.append(dict(step=0, method=m, brier_T=brier(p0[:nt], S['yT'], K), brier_V=brier(p0[nt:], S['yV'], K), acc_T=float(np.mean(p0[:nt].argmax(1) == S['yT']))))
    for t in range(P['steps']):
        sl = slice(t*pool, (t+1)*pool); XC, yC, pC = S['XS'][sl], S['yS'][sl], S['pS'][sl]
        u = np.random.default_rng([seed, t, 195003]).random((R, pool, 1)); reps = (u > pC.cumsum(1)[None]).sum(-1).clip(0, K-1)
        for m in METHODS:
            XD, yD = lab[m]
            if m == 'oracle-test': pick = greedy(XD, yD, XC, yC[None], S['XT'], S['yT'])
            elif m == 'oracle-split': pick = greedy(XD, yD, XC, yC[None], S['XV'], S['yV'])
            elif m == 'oracle-exante': pick = greedy(XD, yD, XC, reps, S['XV'], S['yV'])
            elif m == 'Random': pick = np.random.default_rng([seed, t, 195004]).choice(pool, b, replace=False)
            elif m == 'CoreSet':
                dist = ((XC[:, None]-XD[None])**2).sum(-1).min(1); pick = []
                for _ in range(b):
                    i = int(np.argmax(dist)); pick.append(i); dist = np.minimum(dist, ((XC-XC[i])**2).sum(1)); dist[pick] = -np.inf
                pick = np.array(pick)
            else:
                Pm = heads.train([(XD, yD)], K, seed, hid, ep, lr); pc, hc = heads.predict(Pm, XC, hid, return_hidden=True); pc, hc = pc[0], hc[0]
                if m == 'Margin':
                    s2 = np.sort(pc, 1); pick = np.argsort(s2[:, -1]-s2[:, -2], kind='stable')[:b]
                elif m == 'Entropy':
                    pick = np.argsort(-(-(pc*np.log(np.clip(pc, 1e-12, 1))).sum(1)), kind='stable')[:b]
                else:   # BADGE: gradient embedding (p - onehot(argmax p)) (x) hidden, k-means++ seeding
                    g = ((pc-onehot(pc.argmax(1), K))[:, :, None]*hc[:, None, :]).reshape(len(pc), -1)
                    rb = np.random.default_rng([seed, t, 195005]); pick = [int(np.argmax((g**2).sum(1)))]
                    d2 = ((g-g[pick[0]])**2).sum(1)
                    for _ in range(b-1):
                        pr_ = d2/d2.sum() if d2.sum() > 0 else None
                        i = int(rb.choice(len(g), p=pr_)) if pr_ is not None else int(rb.choice([j for j in range(len(g)) if j not in pick]))
                        while i in pick: i = int(rb.choice(len(g)))
                        pick.append(i); d2 = np.minimum(d2, ((g-g[i])**2).sum(1))
                    pick = np.array(pick)
            assert len(pick) == b and len(set(pick.tolist())) == b, m
            XD = np.vstack([XD, XC[pick]]); yD = np.concatenate([yD, yC[pick]]); lab[m] = (XD, yD)
            pr = fit_pred([(XD, yD)], E)[0]
            rows.append(dict(step=t+1, method=m, brier_T=brier(pr[:nt], S['yT'], K), brier_V=brier(pr[nt:], S['yV'], K),
                             acc_T=float(np.mean(pr[:nt].argmax(1) == S['yT']))))
    df = pd.DataFrame(rows).assign(dataset=dataset, encoder=encoder, head=head, seed=seed, n_labels=lambda d: P['initial_labels']+d.step*b,
                                   noise_index=S['audit']['noise_index'], classes=K)
    df.to_csv(f, index=False); (out/f'{f.stem}_audit.json').write_text(json.dumps(S['audit']))
    return str(f), time.time()-t0


def tasks_for(P, grid, smoke):
    G = P['grids'][grid]; seeds = [99001] if smoke else list(range(G['seed_start'], G['seed_start']+G['n_seeds']))
    ds = G['datasets'][:2] if smoke else G['datasets']
    enc = lambda d: (PRIMARY if G['encoder'] == 'primary' else SECONDARY)[KIND[d]]
    return [(d, enc(d), G['head'], s) for s in seeds for d in ds]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--grids', default='A,B,C'); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); P = json.loads(args.protocol.read_text())
    if args.smoke:
        P.update(P['smoke_overrides'])
    else:
        F = json.loads((args.protocol.parent/'freeze.json').read_text()); assert F['protocol_sha256'] == sha(args.protocol)
        for k, h in F['sources'].items(): assert sha(ROOT/k) == h, k
        for k, h in F['features'].items(): assert sha(ROOT/'features'/k) == h, k
    args.output.mkdir(parents=True, exist_ok=True)
    import torch.multiprocessing as mp
    allt = []
    for g in args.grids.split(','):
        allt += [t+(P, str(args.output)) for t in tasks_for(P, g, args.smoke)]
    M = dict(status='RUNNING', smoke=args.smoke, grids=args.grids, tasks=len(allt), started=time.time(), done=0)
    (args.output/'run_manifest.json').write_text(json.dumps(M, indent=2))
    ctx = mp.get_context('spawn'); t0 = time.time()
    try:
        with ctx.Pool(args.workers) as pool:
            for n, (f, s) in enumerate(pool.imap_unordered(run_trajectory, allt, chunksize=1)):
                M['done'] = n+1; M['elapsed'] = time.time()-t0
                (args.output/'run_manifest.json').write_text(json.dumps(M, indent=2))
                (args.output/'progress.txt').write_text(f'{n+1}/{len(allt)} trajectories, {time.time()-t0:.0f}s, last {Path(f).name} {s:.0f}s\n')
        M.update(status='COMPLETE', elapsed=time.time()-t0)
    except BaseException as e:
        M.update(status='FAILED', error=f'{type(e).__name__}: {e}'); raise
    finally:
        (args.output/'run_manifest.json').write_text(json.dumps(M, indent=2))


if __name__ == '__main__':
    main()
