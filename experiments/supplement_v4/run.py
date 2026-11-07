"""supplement-v4: where does the pre-outcome selection gap come from?

Diagnostic, not a confirmation of DC-PRISM. It reuses the 640 supplement-v3
states and their exact label resamples (identical RNG streams), so every loss
here is paired with the archived v3 H_pre of the same state.

On the same finite family F (common family + v3 main-policy choices), the
target-weighted information proxy V_theta(S) (Eq. value in the paper) ranks
sets with two parameter values:
  theta_hat : the visible-history logistic fit (deployable);
  theta_pop : the population KL projection of the true p onto the logistic
              class, fitted on all state covariates with soft labels p
              (equals the true parameter in logistic cells; not deployable).
`argmax_F V_theta` is the proxy's own pre-outcome choice. Comparing the two
choices separates parameter error from proxy-approximation error; the cross-
fitted MC choice (H_pre) is the reference ceiling within F.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import run_offline as v3
import stats as st_mod

SOURCE_PATHS = ['experiments/supplement_v4/run.py'] + v3.SOURCE_PATHS


def theta_pop(st):
    """Soft-label logistic fit on every covariate in the state (tiny ridge)."""
    X = np.vstack([st['X_D'], st['X_O'], st['X_C'], st['X_Q'], st['X_T']])
    p = np.concatenate([st['p_D_true'], st['p_O_true'], st['p_C_true'], st['p_Q_true'], st['p_T_true']])
    Xb = np.column_stack([np.ones(len(X)), X])

    def f(w):
        z = Xb @ w; q = expit(z)
        nll = -np.sum(p*np.log(np.clip(q, 1e-12, 1))+(1-p)*np.log(np.clip(1-q, 1e-12, 1))) + 5e-4*np.sum(w[1:]**2)
        return nll, Xb.T @ (q-p) + 1e-3*np.r_[0, w[1:]]
    res = minimize(f, np.zeros(Xb.shape[1]), jac=True, method='L-BFGS-B', options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 2000})
    assert np.max(np.abs(res.jac)) < 1e-4, f'theta_pop not converged: {np.max(np.abs(res.jac))}'
    return res.x


def proxy_values(st, family, w):
    H, Hi, G = eng.build_pending_fisher_and_target(st['X_D'], st['X_O'], st['X_Q'], w, l2=.45)
    Xb = np.column_stack([np.ones(len(st['X_C'])), st['X_C']])
    p = expit(Xb @ w); wc = np.clip(p*(1-p), 1e-6, None)
    return np.array([eng.pending_aware_value(s, Xb, wc, H, Hi, G) for s in family])


def with_truth(task):
    """Rebuild a v3 state and attach the true p for D and Q (needed for theta_pop only)."""
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        gen, truth = v3._controlled_generators()
        st = gen(seed); st['p_O_true'] = truth(st['X_O'], v3.W_TRUE)
        st['p_D_true'] = truth(st['X_D'], v3.W_TRUE); st['p_Q_true'] = truth(st['X_Q'], v3.W_TRUE)
        return st, st
    cfg = v3._STATE['cohorts'][cohort]; data = v3._STATE['data'][cfg['dataset']]
    arms = v3.realx_pair(data, seed, cfg['mechanism'])
    rng = np.random.default_rng(seed); ids = rng.permutation(len(data['X']))[:706]
    prep = eng.SplitBeforePreprocessor(tail_scale=1).fit(data['X'][ids][:80], data['ead'][ids][:80]); prep.tau_aux = np.zeros(2)
    X = prep.transform_X(data['X'][ids]); logit = -.8 + X @ np.array([.45, .65, -.5, .35, .2, -.15])
    if cfg['mechanism'] == 'nonlinear':
        logit += .5*np.sin(X[:, 1]) + .35*X[:, 2]*X[:, 3]
    p = expit(logit)
    broad = np.arange(80); gap = np.sort(np.lexsort((np.arange(240), X[:240, 1]))[:80])
    for name, idx in [('broad', broad), ('gap', gap)]:
        assert np.array_equal(arms[name]['X_D'], X[idx])
        arms[name]['p_D_true'] = p[idx]; arms[name]['p_Q_true'] = p[286:466]
    return arms[arm], arms['broad']


def one_state(task):
    group, cohort, seed, arm, R, code = task
    st, ref = with_truth(task)
    ctx = v3.sup2.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    choices, _, learner = v3.select_all(st, ctx, seed)
    fam = {tuple(s.tolist()): s for s in ctx['family']}
    for s in choices.values():
        fam.setdefault(tuple(s.tolist()), s)
    keys = sorted(fam); family = [np.array(k, int) for k in keys]
    anchor = keys.index(tuple(ctx['M'].tolist()))
    w_pop = theta_pop(st)
    v_hat, v_pop = proxy_values(st, family, learner.w), proxy_values(st, family, w_pop)
    i_hat, i_pop = int(np.argmax(v_hat)), int(np.argmax(v_pop))
    # ---- choices locked; evaluator draws the same labels as supplement-v3 ----
    L, _ = v3.evaluate_sets(st, family, learner, R, np.random.default_rng([seed, code, 7777]))
    post, plug, cf = v3.decomposition(L, anchor)
    gain = L[:, anchor].mean() - L.mean(axis=0)
    rho = lambda v: float(spearmanr(v, gain).statistic) if np.ptp(v) > 1e-12 and np.ptp(gain) > 1e-12 else np.nan
    return dict(group=group, cohort=cohort, seed=seed, arm=arm, family_size=len(family),
                anchor_exante_loss=float(L[:, anchor].mean()), H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf,
                gain_proxy_hat=float(gain[i_hat]), gain_proxy_pop=float(gain[i_pop]),
                k_proxy_hat=len(family[i_hat]), k_proxy_pop=len(family[i_pop]),
                rank_corr_hat=rho(v_hat), rank_corr_pop=rho(v_pop),
                theta_distance=float(np.linalg.norm(learner.w-w_pop)))


def statistics(df, protocol, out):
    cfg = protocol['inference']; recs = []
    groups = sorted(df.group.unique(), key=lambda g: (g != 'Controlled', g))
    for g in groups:
        x = df[df.group == g].sort_values('seed')
        recs.append(({'group': g, 'endpoint': 'parameter_effect_gain_pop_minus_hat'}, (x.gain_proxy_pop-x.gain_proxy_hat).to_numpy()))
        recs.append(({'group': g, 'endpoint': 'approximation_gap_Hpre_cf_minus_gain_pop'}, (x.H_pre_crossfit-x.gain_proxy_pop).to_numpy()))
    t = st_mod.family_table(recs, cfg['alpha'], cfg['material_delta'], cfg['legacy_delta'])
    assert len(t) == protocol['primary_family_size']
    t.to_csv(out/'primary_statistics.csv', index=False)
    df.groupby('group', as_index=False)[['H_post', 'H_pre_crossfit', 'gain_proxy_hat', 'gain_proxy_pop', 'k_proxy_hat',
        'k_proxy_pop', 'rank_corr_hat', 'rank_corr_pop', 'theta_distance']].mean().to_csv(out/'summary.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    for k in ['protocol', 'freeze', 'data-dir', 'output']:
        ap.add_argument('--'+k, type=Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text()); freeze = json.loads(a.freeze.read_text())
    assert hashlib.sha256(a.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    src = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    assert src == freeze['source_sha256'], 'Frozen source changed'
    v3p = json.loads((ROOT/'configs/supplement_v3.json').read_text())
    datafile = a.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == v3p['data_sha256']
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', protocol_sha256=freeze['protocol_sha256'], source_sha256=src,
                    python=platform.python_version(), numpy=np.__version__); t0 = time.time()
    try:
        tasks = v3.tasks_from(v3p)
        with Pool(a.workers, initializer=v3._init, initargs=(str(datafile), v3p['offline']['cohorts'])) as pool:
            df = pd.DataFrame(pool.map(one_state, tasks, chunksize=1))
        archived = pd.read_csv(ROOT/'results/supplement_v3/offline_policy_results.csv.gz')
        arch = archived[archived.policy == 'Safe-MinRisk'].set_index(['group', 'seed'])
        cur = df.set_index(['group', 'seed'])
        err = float(max((cur.H_pre_crossfit-arch.H_pre_crossfit.loc[cur.index]).abs().max(),
                        (cur.anchor_exante_loss-arch.exante_loss.loc[cur.index]).abs().max()))
        assert err < 1e-12, f'v3 pairing broken: {err}'
        df.to_csv(out/'state_results.csv', index=False)
        statistics(df, protocol, out)
        manifest.update(status='COMPLETE', states=len(df), v3_pairing_max_abs_error=err, elapsed_seconds=time.time()-t0)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
