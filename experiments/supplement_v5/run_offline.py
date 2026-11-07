"""supplement-v5 offline studies E, F, G (response to the strict review).

E. Family-size dependence of the foresight share (review M2). Diagnostic on the
   640 supplement-v3 states with their exact label resamples. The v3 family is
   enlarged with 64 random feasible sets (visible information only); H_post and
   H_pre are recomputed on random sub-families of fixed size that contain M_K.
F. Conservative-layer ablation on fresh states (review M3). The decision rule
   (draw quantile, curvature damping, reference normalisation, quantity penalty,
   gate) is applied to a family that also contains the TargetVar-greedy set, and
   compared with the point proxy's own maximiser on that same family, and with
   the greedy set itself. Also records selection wall time (M7) and the exact
   envelope-exceedance probability under the true p (M6).
G. Generator perturbation of the controlled worlds (review M4): pending overlap
   x risk-information coupling grid, tail regime removed, doubled candidate
   pool with K=6 and K=12.

Selectors receive visible inputs only; every choice is locked before the
evaluator draws hidden labels.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v5'))
import prism_engine as eng
import baselines as bl
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('v3_run_offline', ROOT/'experiments/supplement_v3/run_offline.py')
v3 = _ilu.module_from_spec(_spec); sys.modules['v3_run_offline'] = v3; _spec.loader.exec_module(v3)

SOURCE_PATHS = ['experiments/supplement_v5/run_offline.py', 'experiments/supplement_v5/stats_v5.py'] + v3.SOURCE_PATHS
ALPHA = 0.10
_STATE = {}


# ---------------------------------------------------------------- generators
def _generator(tail=True):
    path = ROOT/'archive/submitted_v2_2/src/run_p0_p1_p2.py'
    tree = ast.parse(path.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'true_dgp_prob', 'generate_synthetic_credit_state'}]
    scope = {'np': np, 'expit': expit}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), scope)
    if not tail:
        def logistic_only(X_mat, w_true, is_shift=False):
            Xb = np.column_stack([np.ones(len(X_mat)), X_mat])
            return np.clip(expit(Xb @ w_true), 1e-4, 0.999)
        scope['true_dgp_prob'] = logistic_only
    return scope['generate_synthetic_credit_state'], scope['true_dgp_prob']


def perturbed_state(seed, overlap='High', coupling='Positive', tail=True, double=False):
    gen, truth = _generator(tail)
    st = gen(seed, pending_overlap=overlap, risk_info_coupling=coupling)
    if double:
        st2 = gen(seed+1_000_000, pending_overlap=overlap, risk_info_coupling=coupling)
        for k in ['X_C', 'ead_C', 'p_C_true', 'y_C']:
            st[k] = np.concatenate([st[k], st2[k]])
    assert np.allclose(truth(st['X_C'], v3.W_TRUE), st['p_C_true'])
    st['p_O_true'] = truth(st['X_O'], v3.W_TRUE)
    return st


def reference_context(X_D, y_D, X_C, ead_C, X_Q, seed, target_k=6):
    """supplement-v2 reference_context with target_k exposed (identical for target_k=6)."""
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(X_D, y_D)
    q, _ = eng.compute_calibrated_upper_risk(learner, X_D, y_D, X_C, rng=np.random.default_rng(seed+100))
    B, bhash = eng.compute_exogenous_budget(q, ead_C, target_k=target_k)
    t0 = time.perf_counter()
    M, certificate = eng.solve_safe_minrisk_exact_or_bounded(q, ead_C, B)
    t_mk = time.perf_counter()-t0
    family, *_ = eng.generate_candidate_family(X_D, None, X_C, X_Q, q, ead_C, B, M, learner)
    candidates = {tuple(np.sort(s)): np.sort(s) for s in family if len(s) <= len(M)}
    for k in range(len(M)+1):
        s = np.sort(M[:k]); candidates[tuple(s)] = s
    family = [candidates[k] for k in sorted(candidates)]
    return dict(q=q, B=B, M=M, family=family, certificate=certificate, t_mk=t_mk)


# ---------------------------------------------------------------- decision-rule pieces
def layer_deltas(learner, X_D, X_O, X_C, X_Q, family, M, worlds, damping=.45, scale=2.5, point=False):
    """Per-draw normalised deltas of the package rule on an arbitrary family."""
    Xb = np.column_stack([np.ones(len(X_C)), X_C])
    draws = [learner.w] if point else (worlds[1:] if len(worlds) > 1 else worlds)
    out = np.empty((len(draws), len(family)))
    for m, w in enumerate(draws):
        H, Hi, G = eng.build_pending_fisher_and_target(X_D, X_O, X_Q, w, l2=learner.l2)
        p = expit(Xb @ w)
        wc = np.clip(p*(1-p)*np.exp(-damping*(Xb @ (w-learner.w))**2), 1e-6, None)
        a = max(eng.pending_aware_value(M, Xb, wc, H, Hi, G), 1e-6)
        for j, s in enumerate(family):
            out[m, j] = np.tanh((eng.pending_aware_value(s, Xb, wc, H, Hi, G)-a)/(scale*a))
    return out


def point_values(learner, X_D, X_O, X_C, X_Q, family):
    H, Hi, G = eng.build_pending_fisher_and_target(X_D, X_O, X_Q, learner.w, l2=learner.l2)
    Xb = np.column_stack([np.ones(len(X_C)), X_C])
    p = learner.predict_proba(X_C); wc = np.clip(p*(1-p), 1e-6, None)
    return np.array([eng.pending_aware_value(s, Xb, wc, H, Hi, G) for s in family])


def random_feasible_sets(q, ead, B, K, n, rng):
    """Visible-information random feasible sets of random size 1..K (deduplicated)."""
    out = {}
    tries = 0
    while len(out) < n and tries < 50*n:
        tries += 1
        k = int(rng.integers(1, max(K, 1)+1))
        chosen = []
        for i in rng.permutation(len(q)):
            trial = np.array(chosen+[int(i)])
            if eng.bernstein_envelope(q[trial], ead[trial], alpha=ALPHA) <= B+1e-12:
                chosen.append(int(i))
            if len(chosen) == k:
                break
        s = np.sort(np.array(chosen, int))
        if len(s):
            out.setdefault(tuple(s.tolist()), s)
    return list(out.values())


def exceedance_probability(st, s, q, n_draw, rng):
    """Pr{sum_{i in S} e_i Y_i > R_alpha(S)} under the true p (Monte Carlo)."""
    if len(s) == 0:
        return 0.
    R = eng.bernstein_envelope(q[s], st['ead_C'][s], alpha=ALPHA)
    Y = rng.random((n_draw, len(s))) < st['p_C_true'][s]
    return float(np.mean(Y.astype(float) @ st['ead_C'][s] > R+1e-12))


# ---------------------------------------------------------------- study E
def study_e(task):
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        st = v3.controlled_state(seed)['only']; ref = st
    else:
        data = v3._STATE['data'][v3._STATE['cohorts'][cohort]['dataset']]
        arms = v3.realx_pair(data, seed, v3._STATE['cohorts'][cohort]['mechanism'])
        st, ref = arms[arm], arms['broad']
    ctx = v3.sup2.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    choices, _, learner = v3.select_all(st, ctx, seed)
    fam = {tuple(s.tolist()): s for s in ctx['family']}
    for s in choices.values():
        fam.setdefault(tuple(s.tolist()), s)
    base_keys = sorted(fam)
    extra = random_feasible_sets(ctx['q'], st['ead_C'], ctx['B'], len(ctx['M']), _STATE['n_random'],
                                 np.random.default_rng([seed, code, 8888]))
    rand_keys = sorted({tuple(s.tolist()) for s in extra} - set(base_keys))
    keys = base_keys + rand_keys
    # ---- family locked; evaluator draws exactly the supplement-v3 labels ----
    L, _ = v3.evaluate_sets(st, [np.array(k, int) for k in keys], learner, R, np.random.default_rng([seed, code, 7777]))
    anchor = keys.index(tuple(ctx['M'].tolist()))
    nb = len(base_keys)
    post_b, plug_b, cf_b = v3.decomposition(L[:, :nb], anchor)
    rows = [dict(group=group, seed=seed, family='v3_base', size=nb, draws=1, H_post=post_b, H_pre_plugin=plug_b, H_pre_crossfit=cf_b)]
    rng = np.random.default_rng([seed, code, 9999])
    others = np.array([i for i in range(len(keys)) if i != anchor])
    for m in _STATE['sizes']:
        if m > len(keys):
            continue
        acc = np.zeros(3); n_sub = 1 if m == len(keys) else _STATE['n_sub']
        for _ in range(n_sub):
            sub = np.r_[anchor, rng.choice(others, m-1, replace=False)] if m < len(keys) else np.arange(len(keys))
            a_pos = 0 if m < len(keys) else anchor
            acc += np.array(v3.decomposition(L[:, sub], a_pos))
        acc /= n_sub
        rows.append(dict(group=group, seed=seed, family='random_subfamily', size=m, draws=n_sub,
                         H_post=acc[0], H_pre_plugin=acc[1], H_pre_crossfit=acc[2]))
    full = v3.decomposition(L, anchor)
    rows.append(dict(group=group, seed=seed, family='enlarged_full', size=len(keys), draws=1,
                     H_post=full[0], H_pre_plugin=full[1], H_pre_crossfit=full[2]))
    # anchor + random-only sets, size matched to the v3 base family
    r_idx = [keys.index(k) for k in rand_keys]
    if len(r_idx) >= nb-1:
        acc = np.zeros(3)
        for _ in range(_STATE['n_sub']):
            sub = np.r_[anchor, rng.choice(r_idx, nb-1, replace=False)]
            acc += np.array(v3.decomposition(L[:, sub], 0))
        acc /= _STATE['n_sub']
        rows.append(dict(group=group, seed=seed, family='random_only_matched', size=nb, draws=_STATE['n_sub'],
                         H_post=acc[0], H_pre_plugin=acc[1], H_pre_crossfit=acc[2]))
    return rows


# ---------------------------------------------------------------- study F / G shared state evaluation
def choose_all(st, ctx, seed, ablation):
    X_D, y_D, X_O, X_C, ead, X_Q = (st[k] for k in ['X_D', 'y_D', 'X_O', 'X_C', 'ead_C', 'X_Q'])
    q, B, M = ctx['q'], ctx['B'], ctx['M']
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(X_D, y_D)
    worlds = eng.sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed+700))
    timing, ch = {'Safe-MinRisk': ctx['t_mk']}, {'Safe-MinRisk': M.copy()}
    for name, pol in [('DC-PRISM', 'DC-PRISM'), ('Fisher-cost', 'BAIT-Cost-Safe')]:
        t0 = time.perf_counter()
        s, m_k, _ = eng.select_acquisition_policy(pol, X_D, y_D, X_O, X_C, ead, X_Q, B, learner=learner,
                        q_C_precomputed=q, w_worlds_precomputed=worlds, rng=np.random.default_rng(seed+400))
        timing[name] = time.perf_counter()-t0
        assert np.array_equal(np.sort(m_k), np.sort(M))
        ch[name] = np.sort(np.asarray(s, int))
    for name, fn in [('TargetVar-Greedy', lambda: bl.targetvar_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)),
                     ('BAIT-Greedy', lambda: bl.bait_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)),
                     ('Uncertainty', lambda: bl.uncertainty_greedy(learner, X_C, q, ead, B)),
                     ('Random-Feasible', lambda: bl.random_feasible(X_C, q, ead, B, np.random.default_rng(seed+900)))]:
        t0 = time.perf_counter(); ch[name] = fn(); timing[name] = time.perf_counter()-t0
    pv_common = point_values(learner, X_D, X_O, X_C, X_Q, ctx['family'])
    ch['PointValue'] = ctx['family'][int(np.argmax(pv_common))].copy()
    if ablation:
        pkg, pkg_d = bl.package_family_scores(learner, X_D, X_O, X_C, X_Q, q, ead, B, M, worlds)
        s_pkg, _ = bl.choose_from_scores(pkg, pkg_d, M)
        assert np.array_equal(np.sort(s_pkg), ch['DC-PRISM']), 'package rule re-implementation mismatch'
        g = ch['TargetVar-Greedy']
        fplus = {tuple(np.sort(s).tolist()): np.sort(np.asarray(s, int)) for s in pkg}
        fplus.setdefault(tuple(g.tolist()), g)
        fplus = [fplus[k] for k in sorted(fplus)]
        pv_pkg = point_values(learner, X_D, X_O, X_C, X_Q, pkg)
        pv_plus = point_values(learner, X_D, X_O, X_C, X_Q, fplus)
        ch['ProxyArgmax-Fpkg'] = np.sort(np.asarray(pkg[int(np.argmax(pv_pkg))], int))
        ch['ProxyArgmax-F+'] = fplus[int(np.argmax(pv_plus))].copy()
        d_plus = layer_deltas(learner, X_D, X_O, X_C, X_Q, fplus, M, worlds)
        ch['Layer-F+'] = np.sort(bl.choose_from_scores(fplus, d_plus, M)[0])
        ch['Layer-F+-median'] = np.sort(bl.choose_from_scores(fplus, d_plus, M, quantile=.5)[0])
        ch['Layer-F+-nopenalty-nogate'] = np.sort(bl.choose_from_scores(fplus, d_plus, M, penalty=0., gate=-1e9)[0])
        d_pt = layer_deltas(learner, X_D, X_O, X_C, X_Q, fplus, M, worlds, point=True)
        ch['PointLayer-F+'] = np.sort(bl.choose_from_scores(fplus, d_pt, M)[0])
        gi = next(j for j, s in enumerate(fplus) if np.array_equal(s, g))
        ch['Gated-Greedy'] = np.sort(bl.choose_from_scores([fplus[gi]], d_plus[:, [gi]], M)[0])
    for s in ch.values():
        assert len(set(s.tolist())) == len(s) and eng.bernstein_envelope(q[s], ead[s]) <= B+1e-10
    return ch, learner, timing


def evaluate_state(group, st, ctx, seed, code, R, ablation):
    ch, learner, timing = choose_all(st, ctx, seed, ablation)
    # ---- selections locked ----
    fam = {tuple(s.tolist()): s for s in ctx['family']}
    nF0 = len(fam)
    for s in ch.values():
        fam.setdefault(tuple(s.tolist()), s)
    keys = sorted(fam); sets = [np.array(k, int) for k in keys]
    L, realized = v3.evaluate_sets(st, sets, learner, R, np.random.default_rng([seed, code, 5555]))
    idx = {k: i for i, k in enumerate(keys)}; anchor = idx[tuple(ctx['M'].tolist())]
    post, plug, cf = v3.decomposition(L, anchor)
    means = L.mean(axis=0)
    ex_rng = np.random.default_rng([seed, code, 4444])
    base = dict(group=group, seed=seed, K=len(ctx['M']), n_C=len(st['X_C']), B=ctx['B'], family_size=len(keys),
                common_family_size=nF0, mk_status=ctx['certificate']['solver_status'],
                mk_gap=ctx['certificate']['optimality_gap'], H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf)
    rows = []
    for pol, s in ch.items():
        i = idx[tuple(s.tolist())]
        rows.append(dict(base, policy=pol, k=len(s), equals_anchor=bool(i == anchor),
                         exante_loss=float(means[i]), exante_gain_vs_anchor=float(np.mean(L[:, anchor]-L[:, i])),
                         realized_brier=realized[i][0], seconds=timing.get(pol, np.nan),
                         q_covers_p=bool(np.all(ctx['q'][s] >= st['p_C_true'][s])) if len(s) else True,
                         exceed_prob=exceedance_probability(st, s, ctx['q'], 4000, ex_rng),
                         selected=json.dumps(s.tolist())))
    return rows


def study_f(task):
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        st = v3.controlled_state(seed)['only']; ref = st
    else:
        data = v3._STATE['data'][v3._STATE['cohorts'][cohort]['dataset']]
        arms = v3.realx_pair(data, seed, v3._STATE['cohorts'][cohort]['mechanism'])
        st, ref = arms[arm], arms['broad']
    ctx = reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    ctx_v2 = v3.sup2.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    assert np.array_equal(ctx['M'], ctx_v2['M']) and len(ctx['family']) == len(ctx_v2['family'])
    return evaluate_state(group, st, ctx, seed, code, R, ablation=True)


def study_g(task):
    cell, cfg, seed, R = task
    st = perturbed_state(seed, cfg['overlap'], cfg['coupling'], cfg['tail'], cfg['double'])
    ctx = reference_context(*[st[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed, target_k=cfg['target_k'])
    rows = evaluate_state(cell, st, ctx, seed, 0, R, ablation=False)
    for r in rows:
        r.update(cell=cell, **{f'cfg_{k}': v for k, v in cfg.items()})
    return rows


# ---------------------------------------------------------------- driver
def _init(datafile, cohorts, extra):
    v3._init(datafile, cohorts)
    _STATE.update(extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--study', choices=['E', 'F', 'G'], required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import stats_v5
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', study=args.study, smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'],
                    source_sha256=sources, python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        cfgS = protocol[f'study_{args.study}']
        if args.study == 'G':
            cohorts = {}
            tasks = [(cell, c, s, cfgS['label_resamples']) for cell, c in cfgS['cells'].items()
                     for s in ([99001] if args.smoke else range(c['seed_start'], c['seed_start']+cfgS['worlds_per_cell']))]
            fn, extra = study_g, {}
        else:
            src = protocol['study_E']['source_cohorts'] if args.study == 'E' else cfgS['cohorts']
            cohorts = src
            tasks = v3.tasks_from({'offline': {'cohorts': src, 'label_resamples': cfgS['label_resamples']}})
            if args.smoke:
                tasks = [(g, c, 99001, a, R, code) for (g, c, s, a, R, code) in tasks if s == src[c]['seed_start']]
            fn = study_e if args.study == 'E' else study_f
            extra = dict(n_random=cfgS.get('random_sets', 0), sizes=cfgS.get('subfamily_sizes', []), n_sub=cfgS.get('subfamily_draws', 0))
        with Pool(args.workers, initializer=_init, initargs=(str(datafile), cohorts, extra)) as pool:
            results = pool.map(fn, tasks, chunksize=1)
        df = pd.DataFrame([r for res in results for r in res])
        df.to_csv(out/f'study_{args.study}_rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v5.run(args.study, df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
