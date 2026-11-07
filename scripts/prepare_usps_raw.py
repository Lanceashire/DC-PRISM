"""Raw-pixel USPS (256 features in [-1, 1], LIBSVM train+test) for the MLP study of supplement-v10.
Label: digit >= 5. Exposure 1. Usage: python -I scripts/prepare_usps_raw.py <usps.bz2> <usps.t.bz2> <out.npz>"""
import bz2, hashlib, sys
import numpy as np


def libsvm(path, d=256):
    X, y = [], []
    for line in bz2.open(path, 'rt'):
        parts = line.split(); y.append(int(float(parts[0]))); row = np.zeros(d)
        for t in parts[1:]:
            i, v = t.split(':'); row[int(i)-1] = float(v)
        X.append(row)
    return np.array(X), np.array(y)


a, b, out = sys.argv[1:4]
Xa, ya = libsvm(a); Xb, yb = libsvm(b); X = np.vstack([Xa, Xb]); digit = np.concatenate([ya, yb])-1
np.savez(out, USPSraw_X=X, USPSraw_y=(digit >= 5).astype(np.int8), USPSraw_ead=np.ones(len(X)))
print(dict(rows=len(X), d=X.shape[1], positive_rate=round(float((digit >= 5).mean()), 4)))
