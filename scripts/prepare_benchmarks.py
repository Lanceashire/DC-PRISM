"""Prepare three non-credit binary cohorts for supplement-v9 (array format of the credit NPZ).

Spambase (UCI 94): label = spam; 57 features standardized, unsupervised PCA to 10 components.
USPS (LIBSVM train+test, used by CDALBench): label = digit >= 5; 256 pixels, PCA to 10 components.
Bank Marketing (UCI 222, bank-full.csv): label = term-deposit subscription; six numeric covariates
  (age, balance, day, campaign, pdays, previous); call duration excluded because it leaks the outcome.
Exposure is 1 for every record. PCA uses covariates only (no labels), like pre-encoded benchmark data.
Usage: python -I scripts/prepare_benchmarks.py <spambase.data> <usps.bz2> <usps.t.bz2> <bank-full.csv> <out.npz>
"""
import bz2, csv, hashlib, sys
import numpy as np

spam_p, usps_p, uspst_p, bank_p, out = sys.argv[1:6]


def pca(X, k):
    Z = (X-X.mean(axis=0))/np.where(X.std(axis=0) < 1e-9, 1., X.std(axis=0))
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:k].T*np.sign(Vt[:k].T[np.abs(Vt[:k].T).argmax(axis=0), range(k)])   # deterministic signs
    return Z @ V, float((S[:k]**2).sum()/(S**2).sum())


def libsvm(path, d=256):
    X, y = [], []
    for line in bz2.open(path, 'rt'):
        parts = line.split(); y.append(int(float(parts[0]))); row = np.zeros(d)
        for t in parts[1:]:
            i, v = t.split(':'); row[int(i)-1] = float(v)
        X.append(row)
    return np.array(X), np.array(y)


res, info = {}, {}
S = np.loadtxt(spam_p, delimiter=',')
res['Spambase_X'], ev = pca(S[:, :57], 10); res['Spambase_y'] = S[:, 57].astype(np.int8); info['Spambase'] = (len(S), ev)
Xa, ya = libsvm(usps_p); Xb, yb = libsvm(uspst_p); Xu = np.vstack([Xa, Xb]); digit = np.concatenate([ya, yb])-1
res['USPS_X'], ev = pca(Xu, 10); res['USPS_y'] = (digit >= 5).astype(np.int8); info['USPS'] = (len(Xu), ev)
rows = list(csv.DictReader(open(bank_p, newline=''), delimiter=';'))
res['BankMarketing_X'] = np.array([[float(r[k]) for k in ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']] for r in rows])
res['BankMarketing_y'] = np.array([r['y'] == 'yes' for r in rows], np.int8); info['BankMarketing'] = (len(rows), 1.0)
for k in ['Spambase', 'USPS', 'BankMarketing']:
    res[f'{k}_ead'] = np.ones(len(res[f'{k}_y']))
np.savez(out, **res)
for k, (n, ev) in info.items():
    print(k, dict(rows=n, d=res[f'{k}_X'].shape[1], positive_rate=round(float(res[f'{k}_y'].mean()), 4), pca_variance=round(ev, 3)))
for p in [spam_p, usps_p, uspst_p, bank_p]:
    print(p.split('/')[-1], hashlib.sha256(open(p, 'rb').read()).hexdigest())
