"""supplement-v6: the opportunity protocol without known outcome probabilities.

The protocol of the paper needs p to resample hidden training labels. Here p is
replaced by an estimate p_hat fitted on records that are disjoint from the state
(auxiliary rows of the same cohort), and the protocol is otherwise unchanged.

V (validation). On the 512 supplement-v3 Real-X states the true p is known.
  Auxiliary cohort rows get labels drawn from the true p; p_hat is fitted on them
  (logistic main effects, and gradient boosting) and the protocol is run with
  p (truth) and with each p_hat on common random numbers. Bias of H_post, H_pre,
  foresight and policy gains measures whether p_hat is good enough.
R (real labels). Round-1 states of the delayed replay (engineered and natural
  streams, four cohorts, fresh seeds). p_hat is fitted on cohort rows outside the
  state. Test loss uses the recorded test labels. Besides the p_hat protocol, the
  *realized* gains under the recorded training labels are computed for every set,
  which gives a model-free hindsight headroom H_post_real per state; its mean is
  compared with the p_hat-based H_post as a calibration check.

Selectors receive visible inputs only; all choices are locked before p_hat,
recorded labels or resamples are used.
"""
import argparse
import hashlib
import importlib.util as ilu
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import run_p3_p4 as runner


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v3 = _load('v3_run_offline', 'experiments/supplement_v3/run_offline.py')
v3on = _load('v3_run_online', 'experiments/supplement_v3/run_online.py')
v5 = _load('v5_run_offline', 'experiments/supplement_v5/run_offline.py')
SOURCE_PATHS = ['experiments/supplement_v6/run.py', 'experiments/supplement_v6/stats_v6.py',
                'experiments/supplement_v5/run_offline.py'] + v3.SOURCE_PATHS + ['experiments/supplement_v3/run_online.py', 'src/run_p3_p4.py']
_S = {}
AUX_CAP = 20000


# ------------------------------------------------------------------ p_hat models
def fit_phat(kind, X, y, seed):
    if kind == 'logit':
        med = np.nanmedian(X, axis=0)                      # imputation fitted on auxiliary rows only
        imp = lambda Z: np.where(np.isnan(Z), med, Z)
        Xi = imp(X); mu, sd = Xi.mean(axis=0), Xi.std(axis=0)+1e-9
        m = eng.UnifiedLogisticLearner(l2=1.0).fit((Xi-mu)/sd, y)
        return lambda Z: np.clip(m.predict_proba((imp(Z)-mu)/sd), 1e-4, 1-1e-4)
    if kind == 'gbdt':
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=False, random_state=seed % (2**31))
        m.fit(X, y)
        return lambda Z: np.clip(m.predict_proba(Z)[:, 1], 1e-4, 1-1e-4)
    raise ValueError(kind)


def calib(p, y):
    return dict(brier=float(np.mean((p-y)**2)), mean_p=float(p.mean()), mean_y=float(y.mean()))


# ------------------------------------------------------------------ evaluator
def loss_matrix(st, sets, w_init, U_O, U_C, pO, pC, test):
    """L[r, j]: test loss of set j when hidden training labels are 1{U < p}."""
    R = U_O.shape[0]; L = np.empty((R, len(sets)))
    YO = (U_O < pO).astype(int); YC = (U_C < pC).astype(int)
    for j, s in enumerate(sets):
        X = np.vstack([st['X_D'], st['X_O'], st['X_C'][s]])
        for r in range(R):
            y = np.concatenate([st['y_D'], YO[r], YC[r, s]])
            f = eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=w_init)
            L[r, j] = test(f.predict_proba(st['X_T']))
    return L


def gains(L, anchor, idx):
    return {k: float(np.mean(L[:, anchor]-L[:, i])) for k, i in idx.items()}


def decomposition(L, anchor):
    return v3.decomposition(L, anchor)


def family_and_choices(st, ctx, seed):
    ch, learner, _ = v5.choose_all(st, ctx, seed, ablation=False)
    fam = {tuple(s.tolist()): s for s in ctx['family']}
    for s in ch.values():
        fam.setdefault(tuple(s.tolist()), s)
    keys = sorted(fam); sets = [np.array(k, int) for k in keys]
    pv = v5.point_values(learner, st['X_D'], st['X_O'], st['X_C'], st['X_Q'], sets)
    ch['ProxyArgmax-F'] = sets[int(np.argmax(pv))]
    idx = {p: keys.index(tuple(np.sort(s).tolist())) for p, s in ch.items()}
    return sets, idx, keys.index(tuple(ctx['M'].tolist())), learner


# ------------------------------------------------------------------ study V
def study_v(task):
    group, cohort, seed, arm, R, code = task
    cfg = v3._STATE['cohorts'][cohort]; data = v3._STATE['data'][cfg['dataset']]
    arms = v3.realx_pair(data, seed, cfg['mechanism'])
    st, ref = arms[arm], arms['broad']
    ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    sets, idx, anchor, learner = family_and_choices(st, ctx, seed)
    # ---- choices locked; auxiliary truth and p_hat only from here on ----
    rng = np.random.default_rng(seed); perm = rng.permutation(len(data['X'])); ids = perm[:706]
    prep = eng.SplitBeforePreprocessor(tail_scale=1).fit(data['X'][ids][:80], data['ead'][ids][:80]); prep.tau_aux = np.zeros(2)
    aux = perm[706:][:AUX_CAP]
    Xa = prep.transform_X(data['X'][aux])
    def truth(X):
        z = -.8 + X @ np.array([.45, .65, -.5, .35, .2, -.15])
        if cfg['mechanism'] == 'nonlinear':
            z = z + .5*np.sin(X[:, 1]) + .35*X[:, 2]*X[:, 3]
        return expit(z)
    assert np.allclose(truth(st['X_C']), st['p_C_true'])
    ya = (np.random.default_rng([seed, code, 6001]).random(len(aux)) < truth(Xa)).astype(int)
    rr = np.random.default_rng([seed, code, 6002])
    U_O, U_C = rr.random((R, len(st['X_O']))), rr.random((R, len(st['X_C'])))
    pT = st['p_T_true']; noise = float(np.mean(pT*(1-pT)))
    test = lambda p: float(np.mean((p-pT)**2))+noise
    rows = []
    probs = {'truth': (st['p_O_true'], st['p_C_true'])}
    for kind in _S['models']:
        f = fit_phat(kind, Xa, ya, seed)
        probs[kind] = (f(st['X_O']), f(st['X_C']))
    for model, (pO, pC) in probs.items():
        L = loss_matrix(st, sets, learner.w, U_O, U_C, pO, pC, test)
        post, plug, cf = decomposition(L, anchor)
        g = gains(L, anchor, idx)
        rows.append(dict(group=group, seed=seed, model=model, family_size=len(sets), H_post=post, H_pre_plugin=plug,
                         H_pre_crossfit=cf, foresight=post-plug,
                         p_mae_C=float(np.mean(np.abs(pC-st['p_C_true']))),
                         **{f'gain_{k}': v for k, v in g.items()}))
    return rows


# ------------------------------------------------------------------ study R
def study_r(task):
    stream, dataset, seed, R = task
    X_raw, y_raw, ead_raw = v3on._DATA[dataset]
    part = (runner.build_strictly_hidden_y_free_real_state(X_raw, y_raw, ead_raw, seed=seed, n_rounds=10)
            if stream == 'engineered' else v3on.natural_state(X_raw, y_raw, ead_raw, seed))
    r1 = part['rounds'][0]
    st = dict(X_D=part['X_D0'], y_D=part['y_D0'], X_O=r1['X_O'], X_C=r1['X_C'], ead_C=r1['ead_C'], X_Q=part['X_Q'],
              X_T=part['X_T'])
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets, idx, anchor, learner = family_and_choices(st, ctx, seed)
    # ---- choices locked; recorded labels and p_hat only from here on ----
    used = np.concatenate([part['D0_ids'], part['Q_ids'], part['T_ids'], r1['O_ids'], r1['C_ids']]).astype(int)
    mask = np.ones(len(y_raw), bool); mask[used] = False
    aux = np.flatnonzero(mask)
    aux = np.random.default_rng([seed, 6003]).permutation(aux)[:AUX_CAP]
    n_hold = len(aux)//5; hold, fit_ids = aux[:n_hold], aux[n_hold:]
    yT = part['y_T'].astype(float)
    test = lambda p: float(np.mean((p-yT)**2))
    rr = np.random.default_rng([seed, 6004])
    U_O, U_C = rr.random((R, len(r1['O_ids']))), rr.random((R, len(r1['C_ids'])))
    base = dict(stream=stream, dataset=dataset, seed=seed, K=len(ctx['M']), family_size=len(sets),
                mk_gap=ctx['certificate']['optimality_gap'])
    rows = []
    # realized (recorded labels) -- one draw per state, model-free
    yO, yC = r1['y_O'].astype(float), r1['y_C'].astype(float)
    Lr = loss_matrix(st, sets, learner.w, np.zeros((1, len(yO))), np.zeros((1, len(yC))), yO, yC, test)
    rows.append(dict(base, model='realized', H_post=float(Lr[0, anchor]-Lr[0].min()), H_pre_plugin=np.nan,
                     H_pre_crossfit=np.nan, foresight=np.nan, **{f'gain_{k}': v for k, v in gains(Lr, anchor, idx).items()}))
    for kind in _S['models']:
        f = fit_phat(kind, X_raw[fit_ids], y_raw[fit_ids].astype(int), seed)
        cal = calib(f(X_raw[hold]), y_raw[hold].astype(float))
        pO, pC = f(X_raw[r1['O_ids']]), f(X_raw[r1['C_ids']])
        L = loss_matrix(st, sets, learner.w, U_O, U_C, pO, pC, test)
        post, plug, cf = decomposition(L, anchor)
        rows.append(dict(base, model=kind, H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf, foresight=post-plug,
                         holdout_brier=cal['brier'], holdout_mean_p=cal['mean_p'], holdout_mean_y=cal['mean_y'],
                         **{f'gain_{k}': v for k, v in gains(L, anchor, idx).items()}))
    return rows


def _init(datafile, cohorts, models):
    v3._init(datafile, cohorts); v3on._init(datafile); _S['models'] = models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--study', choices=['V', 'R'], required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'experiments/supplement_v6')); import stats_v6
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    import sklearn
    manifest = dict(status='RUNNING', study=args.study, smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'],
                    source_sha256=sources, python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        cfg = protocol[f'study_{args.study}']
        cohorts = cfg.get('cohorts', {})
        if args.study == 'V':
            tasks = v3.tasks_from({'offline': {'cohorts': cohorts, 'label_resamples': cfg['label_resamples']}})
            if args.smoke:
                tasks = [(g, c, 99001, a, 16, code) for (g, c, s, a, R, code) in tasks if s == cohorts[c]['seed_start']]
            fn = study_v
        else:
            seeds = [99001] if args.smoke else range(cfg['seed_start'], cfg['seed_start']+cfg['n_seeds'])
            tasks = [(st, d, s, 16 if args.smoke else cfg['label_resamples']) for st in cfg['streams'] for d in cfg['datasets'] for s in seeds]
            fn = study_r
        with Pool(args.workers, initializer=_init, initargs=(str(datafile), cohorts, protocol['phat_models'])) as pool:
            res = pool.map(fn, tasks, chunksize=1)
        df = pd.DataFrame([r for x in res for r in x])
        df.to_csv(out/f'study_{args.study}_rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v6.run(args.study, df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
