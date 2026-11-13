"""Batched multiclass heads on one GPU (external-v15).

Every model in a batch has its own parameters, trained from one shared initialization (common random numbers
across hypothetical label sets) with full-batch Adam and masked cross-entropy. hidden=0 gives a linear softmax
head (multinomial logistic regression); hidden>0 a one-hidden-layer ReLU head.
"""
import numpy as np
import torch


def _init(d, K, hidden, seed, N, dev):
    g = torch.Generator().manual_seed(int(seed) % (2**63))
    if hidden:
        W = (torch.randn(1, d, hidden, generator=g)*np.sqrt(2/d)).repeat(N, 1, 1)
        V = (torch.randn(1, hidden, K, generator=g)*np.sqrt(1/hidden)).repeat(N, 1, 1)
        P = [W, torch.zeros(N, 1, hidden), V, torch.zeros(N, 1, K)]
    else:
        P = [(torch.randn(1, d, K, generator=g)*0.01).repeat(N, 1, 1), torch.zeros(N, 1, K)]
    return [p.to(dev).requires_grad_() for p in P]


def _forward(P, X, hidden, return_hidden=False):
    if hidden:
        H = torch.relu(X @ P[0] + P[1]); Z = H @ P[2] + P[3]
        return (Z, H) if return_hidden else Z
    Z = X @ P[0] + P[1]
    return (Z, X) if return_hidden else Z


def train(sets, K, seed, hidden, epochs, lr, wd=1e-3, dev='cuda'):
    N = len(sets); d = sets[0][0].shape[1]; n = max(len(y) for _, y in sets)
    X = np.zeros((N, n, d), np.float32); Y = np.zeros((N, n), np.int64); M = np.zeros((N, n), np.float32)
    for j, (x, y) in enumerate(sets):
        X[j, :len(y)] = x; Y[j, :len(y)] = y; M[j, :len(y)] = 1
    X, Y, M = (torch.as_tensor(a, device=dev) for a in (X, Y, M))
    P = _init(d, K, hidden, seed, N, dev); opt = torch.optim.Adam(P, lr=lr, foreach=True); den = M.sum(1)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        Z = _forward(P, X, hidden)
        ce = torch.nn.functional.cross_entropy(Z.reshape(-1, K), Y.reshape(-1), reduction='none').reshape(N, n)
        loss = ((ce*M).sum(1)/den).sum() + wd*sum((p.square().sum() for p in P[0::2]))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite head loss')
        loss.backward(); opt.step()
    return [p.detach() for p in P]


def predict(P, Xe, hidden, return_hidden=False):
    with torch.inference_mode():
        E = torch.as_tensor(np.asarray(Xe, np.float32), device=P[0].device).unsqueeze(0).expand(P[0].shape[0], -1, -1)
        out = _forward(P, E, hidden, return_hidden)
        if return_hidden:
            return torch.softmax(out[0], -1).cpu().numpy().astype(float), out[1].cpu().numpy().astype(float)
        return torch.softmax(out, -1).cpu().numpy().astype(float)


def predict_many(sets, Xe, K, seed, hidden, epochs, lr, chunk):
    out = []
    for s in range(0, len(sets), chunk):
        out.append(predict(train(sets[s:s+chunk], K, seed, hidden, epochs, lr), Xe, hidden))
    return np.concatenate(out)


def greedy_gpu(XD, yD, XC, worlds, XE, yE, K, b, seed, hidden, epochs, lr, chunk, dev='cuda'):
    """Greedy oracle with all hypothetical training sets assembled on the GPU (equal lengths, no padding).
    worlds: (W, c) candidate labels (one recorded world or W sampled worlds). Picks exactly b candidates
    minimizing the mean over worlds of the multiclass Brier score on (XE, yE)."""
    Xa = torch.as_tensor(np.vstack([XD, XC]).astype(np.float32), device=dev); n = len(XD)
    yDt = torch.as_tensor(np.asarray(yD), device=dev); Wl = torch.as_tensor(np.asarray(worlds), device=dev)
    Et = torch.as_tensor(np.asarray(XE, np.float32), device=dev); OH = torch.nn.functional.one_hot(torch.as_tensor(yE, device=dev), K).float()
    sel = []; W = Wl.shape[0]
    for _ in range(b):
        rem = [i for i in range(XC.shape[0]) if i not in sel]
        cand = torch.tensor([sel+[i] for i in rem], device=dev)                    # (C, k)
        idx = torch.cat([torch.arange(n, device=dev).expand(len(rem), n), n+cand], 1)  # (C, n+k)
        idx = idx.repeat_interleave(W, 0)                                           # (C*W, n+k)
        lab = torch.cat([yDt.expand(len(rem)*W, n), Wl[:, cand].permute(1, 0, 2).reshape(len(rem)*W, -1)], 1)
        scores = []
        for s in range(0, idx.shape[0], chunk):
            X = Xa[idx[s:s+chunk]]; Y = lab[s:s+chunk]; N = X.shape[0]
            P = _init(X.shape[2], K, hidden, seed, N, dev); opt = torch.optim.Adam(P, lr=lr, foreach=True)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                Z = _forward(P, X, hidden)
                loss = torch.nn.functional.cross_entropy(Z.reshape(-1, K), Y.reshape(-1), reduction='sum')/X.shape[1] \
                    + 1e-3*sum((p.square().sum() for p in P[0::2]))
                loss.backward(); opt.step()
            with torch.inference_mode():
                pr = torch.softmax(_forward([p.detach() for p in P], Et.unsqueeze(0).expand(N, -1, -1), hidden), -1)
                scores.append(((pr-OH)**2).sum(-1).mean(-1))
        sc = torch.cat(scores).reshape(len(rem), W).mean(1)
        sel.append(rem[int(torch.argmin(sc))])
    return np.array(sel)
