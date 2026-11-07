"""Batched training of many small MLP classifiers on one GPU.

Each of N models has its own training set (rows padded to a common length and
masked) and all share the evaluation inputs. One hidden ReLU layer, sigmoid
output, weight decay, full-batch Adam for a fixed number of steps; deterministic
given the seed. Used as the downstream learner in supplement-v10.
"""
import numpy as np
import torch

HIDDEN, STEPS, LR, WD = 32, 300, 0.01, 1e-3


def predict_many(train_sets, X_eval, seed, device='cuda', chunk=1024):
    """train_sets: list of (X [n_i, d] float array, y [n_i] 0/1 array). Returns [N, n_eval] probabilities."""
    out = []
    for s in range(0, len(train_sets), chunk):
        out.append(_batch(train_sets[s:s+chunk], X_eval, seed, device))
    return np.concatenate(out, axis=0)


def _batch(train_sets, X_eval, seed, device):
    N = len(train_sets); d = train_sets[0][0].shape[1]; n = max(len(y) for _, y in train_sets)
    X = np.zeros((N, n, d), np.float32); Y = np.zeros((N, n), np.float32); Mk = np.zeros((N, n), np.float32)
    for i, (x, y) in enumerate(train_sets):
        X[i, :len(y)] = x; Y[i, :len(y)] = y; Mk[i, :len(y)] = 1.
    X, Y, Mk = (torch.from_numpy(a).to(device) for a in (X, Y, Mk))
    g = torch.Generator(device='cpu').manual_seed(int(seed) % (2**63))
    # one shared initialization for all N models (common random numbers across training sets)
    W1 = (torch.randn(1, d, HIDDEN, generator=g)*np.sqrt(2./d)).repeat(N, 1, 1).to(device).requires_grad_()
    b1 = torch.zeros(N, 1, HIDDEN, device=device, requires_grad=True)
    w2 = (torch.randn(1, HIDDEN, 1, generator=g)*np.sqrt(1./HIDDEN)).repeat(N, 1, 1).to(device).requires_grad_()
    b2 = torch.zeros(N, 1, 1, device=device, requires_grad=True)
    with torch.no_grad():                         # start the output at the training base rate
        b2 += torch.logit((Y*Mk).sum(1)/Mk.sum(1), eps=1e-3).view(N, 1, 1)
    opt = torch.optim.Adam([W1, b1, w2, b2], lr=LR)
    cnt = Mk.sum(1)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        z = (torch.relu(X @ W1 + b1) @ w2 + b2).squeeze(-1)
        nll = (torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='none')*Mk).sum(1)/cnt
        loss = nll.sum() + WD*((W1**2).sum()+(w2**2).sum())
        loss.backward(); opt.step()
    with torch.no_grad():
        Xe = torch.from_numpy(np.asarray(X_eval, np.float32)).to(device).unsqueeze(0).expand(N, -1, -1)
        p = torch.sigmoid((torch.relu(Xe @ W1 + b1) @ w2 + b2).squeeze(-1))
    return p.cpu().numpy().astype(np.float64)
