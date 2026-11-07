"""Paired intervention on history coverage with a frozen decision problem.

All methods use only visible history labels. Hidden labels and known truth are
available only to the post-lock evaluator. Reference preprocessing, q, B, anchor,
and the candidate family are held fixed across the two history arms.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, t, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('supplement_v1', ROOT/'experiments/supplement_v1/run.py')
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)
eng = sup.eng
POLICIES = ['Full', 'NoPending', 'NoGate', 'Fixed-ceil', 'Safe-MinRisk', 'PointValue']
SOURCE_PATHS = ['experiments/supplement_v2/run.py', 'experiments/supplement_v1/run.py', 'src/prism_engine.py']


def make_pair(data, seed, mechanism):
    """Two histories from one pool; no true labels or probabilities select D.

    Preprocessing is fitted once on the uniform reference history. This makes
    the estimand conditional on a shared, frozen preprocessing/calibration.
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(data['X']))[:706]
    raw, eraw = data['X'][ids], data['ead'][ids]
    prep = eng.SplitBeforePreprocessor(tail_scale=1).fit(raw[:80], eraw[:80])
    prep.tau_aux = np.zeros(2)
    X, e = prep.transform_X(raw), prep.transform_ead(eraw)
    logits = -.8 + X @ np.array([.45, .65, -.5, .35, .2, -.15])
    if mechanism == 'nonlinear':
        logits += .5*np.sin(X[:, 1]) + .35*X[:, 2]*X[:, 3]
    elif mechanism != 'logistic':
        raise ValueError(mechanism)
    p = expit(logits)
    y = rng.binomial(1, p)
    broad = np.arange(80)
    gap = np.sort(np.lexsort((np.arange(240), X[:240, 1]))[:80])
    shared = dict(X_O=X[240:262], y_O=y[240:262], X_C=X[262:286],
                  y_C=y[262:286], ead_C=e[262:286], p_C_true=p[262:286],
                  X_Q=X[286:466], X_T=X[466:706], y_T=y[466:706], p_T_true=p[466:706])
    arms = {}
    for name, idx in [('broad', broad), ('gap', gap)]:
        arms[name] = dict(shared, X_D=X[idx], y_D=y[idx])
    metadata = {'shared_hidden_hash': sup.array_hash(y[240:], p[240:]),
                'shared_covariate_hash': sup.array_hash(X[240:], e[262:286]),
                'history_pool_hash': sup.array_hash(X[:240], y[:240]),
                'broad_history_hash': sup.array_hash(broad), 'gap_history_hash': sup.array_hash(gap),
                'broad_history_axis_mean': float(X[broad, 1].mean()),
                'gap_history_axis_mean': float(X[gap, 1].mean()),
                'history_overlap': int(len(np.intersect1d(broad, gap))),
                'query_axis_mean': float(X[286:466, 1].mean())}
    return arms, metadata


def reference_context(X_D, y_D, X_C, ead_C, X_Q, seed):
    """Freeze a reference problem using visible broad-history inputs only."""
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(X_D, y_D)
    q, _ = eng.compute_calibrated_upper_risk(learner, X_D, y_D, X_C, rng=np.random.default_rng(seed+100))
    B, bhash = eng.compute_exogenous_budget(q, ead_C, target_k=6)
    M, certificate = eng.solve_safe_minrisk_exact_or_bounded(q, ead_C, B)
    family, *_ = eng.generate_candidate_family(X_D, None, X_C, X_Q, q, ead_C, B, M, learner)
    candidates = {tuple(np.sort(s)): np.sort(s) for s in family if len(s) <= len(M)}
    for k in range(len(M)+1):
        s = np.sort(M[:k]); candidates[tuple(s)] = s
    family = [candidates[k] for k in sorted(candidates)]
    return dict(q=q, B=B, budget_hash=bhash, M=M, family=family,
                family_hash=sup.digest([s.tolist() for s in family]),
                q_hash=sup.array_hash(q), certificate=certificate)


def select(X_D, y_D, X_O, X_C, X_Q, ctx, seed):
    """No hidden labels, true probabilities, or test inputs accepted here."""
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(X_D, y_D)
    worlds = eng.sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed+700))
    family, M = ctx['family'], ctx['M']
    anchor = next(i for i, s in enumerate(family) if np.array_equal(s, M))
    Xb = np.column_stack([np.ones(len(X_C)), X_C])
    penalty = .22*np.array([(len(M)-len(s))/max(len(M), 1) for s in family])
    scores, points = {}, {}
    for pending in [0, 1]:
        O = X_O if pending else None
        H, Hi, G = eng.build_pending_fisher_and_target(X_D, O, X_Q, learner.w)
        p = learner.predict_proba(X_C)
        points[pending] = np.array([eng.pending_aware_value(s, Xb, np.clip(p*(1-p), 1e-6, None), H, Hi, G) for s in family])
        draws = []
        for w in worlds[1:]:
            H, Hi, G = eng.build_pending_fisher_and_target(X_D, O, X_Q, w)
            p = expit(Xb@w)
            weights = np.clip(p*(1-p)*np.exp(-.45*(Xb@(w-learner.w))**2), 1e-6, None)
            vals = np.array([eng.pending_aware_value(s, Xb, weights, H, Hi, G) for s in family])
            a = max(vals[anchor], 1e-6)
            draws.append(np.tanh((vals-a)/(2.5*a)))
        scores[pending] = np.quantile(draws, .2, axis=0)-penalty
    full_best, no_pending_best = int(np.argmax(scores[1])), int(np.argmax(scores[0]))
    k = max(int(np.ceil(.75*len(M))), 1) if len(M) else 0
    eligible = [i for i, s in enumerate(family) if len(s) == k]
    indices = {'Full': full_best if scores[1][full_best] > .08 else anchor,
               'NoPending': no_pending_best if scores[0][no_pending_best] > .08 else anchor,
               'NoGate': full_best, 'Fixed-ceil': max(eligible, key=lambda i: points[1][i]),
               'Safe-MinRisk': anchor, 'PointValue': int(np.argmax(points[1]))}
    choices = {name: family[i].copy() for name, i in indices.items()}
    return choices, learner, scores[1], points[1]


def one_pair(data, seed, mechanism, cohort):
    arms, metadata = make_pair(data, seed, mechanism)
    broad = arms['broad']
    ctx = reference_context(*[broad[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    # Lock both arms before either arm's evaluator reads hidden outcomes.
    locked = {arm: select(*[st[k] for k in ['X_D', 'y_D', 'X_O', 'X_C', 'X_Q']], ctx, seed)
              for arm, st in arms.items()}
    rows, family_rows = [], []
    for arm, st in arms.items():
        choices, learner, scores, points = locked[arm]
        values = [sup.evaluate(st, s, learner) for s in ctx['family']]
        expected = np.array([x[1] for x in values])
        oracle = float(expected.min())
        mapping = {tuple(s): i for i, s in enumerate(ctx['family'])}
        ai = mapping[tuple(ctx['M'])]
        rank = float(spearmanr(scores, -expected).statistic) if np.ptp(scores) > 1e-12 and np.ptp(expected) > 1e-12 else np.nan
        for i, s in enumerate(ctx['family']):
            family_rows.append(dict(cohort=cohort, seed=seed, arm=arm, family_index=i,
                selected_indices=json.dumps(s.tolist()), k=len(s), brier=values[i][0], expected_brier=values[i][1],
                full_score=float(scores[i]), point_value=float(points[i]),
                envelope=eng.bernstein_envelope(ctx['q'][s], st['ead_C'][s]),
                family_hash=ctx['family_hash']))
        for policy, s in choices.items():
            i = mapping[tuple(s)]
            rows.append(dict(cohort=cohort, seed=seed, arm=arm, policy=policy,
                brier=values[i][0], expected_brier=values[i][1], k=len(s), K=len(ctx['M']), B=ctx['B'],
                budget_hash=ctx['budget_hash'], q_hash=ctx['q_hash'], family_hash=ctx['family_hash'],
                family_size=len(ctx['family']), family_index=i, selected_indices=json.dumps(s.tolist()),
                selected_hash=sup.array_hash(s), visible_history_hash=sup.array_hash(st['X_D'], st['y_D']),
                envelope=eng.bernstein_envelope(ctx['q'][s], st['ead_C'][s]),
                realized_loss=float(np.sum(st['y_C'][s]*st['ead_C'][s])),
                selected_q_coverage=bool(np.all(ctx['q'][s] >= st['p_C_true'][s])),
                anchor_expected_brier=float(expected[ai]), family_oracle_expected_brier=oracle,
                family_headroom=float(expected[ai]-oracle), expected_gain=float(expected[ai]-expected[i]),
                expected_regret=float(expected[i]-oracle), proxy_expected_spearman=rank,
                full_gate_explores=bool(not np.array_equal(choices['Full'], ctx['M'])), **metadata))
    return rows, family_rows


def primary_vectors(df):
    """Four endpoints per cohort, independent simulation pair is the unit."""
    result = []
    for cohort, g in df.groupby('cohort'):
        full = g[g.policy == 'Full'].pivot(index='seed', columns='arm', values=['family_headroom', 'expected_gain'])
        point = g[(g.policy == 'PointValue') & (g.arm == 'gap')].set_index('seed').expected_gain.sort_index()
        endpoints = {
            'headroom_gap_minus_broad': full['family_headroom']['gap']-full['family_headroom']['broad'],
            'full_gain_gap_minus_broad': full['expected_gain']['gap']-full['expected_gain']['broad'],
            'full_gain_in_gap': full['expected_gain']['gap'],
            'point_value_gain_in_gap': point,
        }
        result.extend((cohort, name, series.to_numpy()) for name, series in endpoints.items())
    return result


def summarize(df, out, protocol):
    summary = df.groupby(['cohort', 'arm', 'policy'], as_index=False).agg(
        worlds=('seed', 'nunique'), expected_brier=('expected_brier', 'mean'), brier=('brier', 'mean'),
        family_headroom=('family_headroom', 'mean'), expected_gain=('expected_gain', 'mean'),
        expected_regret=('expected_regret', 'mean'), k=('k', 'mean'),
        proxy_rank=('proxy_expected_spearman', 'mean'), gate_explores=('full_gate_explores', 'mean'),
        selected_q_coverage=('selected_q_coverage', 'mean'))
    summary.to_csv(out/'method_summary.csv', index=False)
    stats = []
    m, alpha, delta = protocol['primary_family_size'], protocol['alpha'], protocol['material_delta']
    for cohort, endpoint, x in primary_vectors(df):
        n, mu, sd = len(x), float(np.mean(x)), float(np.std(x, ddof=1))
        se = sd/np.sqrt(n); c = float(t.ppf(1-alpha/(2*m), n-1))
        lo, hi = mu-c*se, mu+c*se
        p = float(ttest_1samp(x, 0).pvalue) if sd > 1e-15 else (1. if abs(mu) < 1e-15 else 0.)
        decision = ('material_positive' if lo > delta else 'material_negative' if hi < -delta else
                    'within_material_band' if lo >= -delta and hi <= delta else
                    'positive' if lo > 0 else 'negative' if hi < 0 else 'indeterminate')
        stats.append(dict(cohort=cohort, endpoint=endpoint, n=n, mean=mu,
                          simultaneous_low=lo, simultaneous_high=hi, p_raw=p, decision=decision))
    assert len(stats) == m
    stats = pd.DataFrame(stats)
    order = np.argsort(stats.p_raw); adj = np.empty(len(stats)); running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., stats.p_raw.iloc[i]*(len(stats)-rank))); adj[i] = running
    stats['p_holm'] = adj
    stats.to_csv(out/'primary_statistics.csv', index=False)
    return summary, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    source_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    assert source_hashes == freeze['source_sha256'], 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    archive = np.load(datafile, allow_pickle=False)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = dict(status='RUNNING', protocol=protocol, protocol_sha256=freeze['protocol_sha256'],
                    source_sha256=source_hashes, python=platform.python_version(), numpy=np.__version__)
    (out/'pre_run_freeze.json').write_text(json.dumps(freeze, indent=2))
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    fits = []; original = eng.minimize
    def tracked(*a, **kw):
        res = original(*a, **kw)
        fits.append(dict(success=bool(res.success), status=int(res.status), grad_inf=float(np.max(np.abs(res.jac)))))
        return res
    eng.minimize = tracked
    try:
        rows, families = [], []
        for cohort, settings in protocol['cohorts'].items():
            data = {k: archive[settings['dataset']+'_'+k] for k in ['X', 'ead']}
            for seed in range(settings['seed_start'], settings['seed_start']+settings['n_pairs']):
                a, b = one_pair(data, seed, settings['mechanism'], cohort); rows.extend(a); families.extend(b)
            # Persist each complete cohort for recovery, without inspecting outcomes.
            pd.DataFrame(rows).to_csv(out/'world_results.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})
            pd.DataFrame(families).to_csv(out/'family_results.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})
            print(f'{cohort}: {settings["n_pairs"]} paired worlds complete', flush=True)
        df = pd.DataFrame(rows); _, stats = summarize(df, out, protocol)
        expected_pairs = sum(x['n_pairs'] for x in protocol['cohorts'].values())
        assert len(df) == expected_pairs*2*len(POLICIES)
        assert (df.envelope <= df.B+1e-10).all()
        assert np.min(df.expected_regret) >= -1e-12
        fit_audit = dict(total_fits=len(fits), non_success=sum(not f['success'] for f in fits),
                         max_gradient_inf=max(f['grad_inf'] for f in fits))
        (out/'optimizer_audit.json').write_text(json.dumps(dict(summary=fit_audit, fits=fits), indent=2))
        assert fit_audit['non_success'] == 0, 'Optimizer failures require a separate audit before interpretation'
        manifest.update(status='COMPLETE', independent_simulation_pairs=expected_pairs,
                        arm_states=expected_pairs*2, method_rows=len(df), family_rows=len(families),
                        primary_comparisons=len(stats), elapsed_seconds=time.time()-started, optimizer_audit=fit_audit)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        eng.minimize = original
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
