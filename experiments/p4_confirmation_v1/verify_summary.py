"""Recheck public seed-level primary statistics without real credit data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


def verify(output):
    manifest = json.loads((output/'run_manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE'
    protocol = manifest['protocol']; m = protocol['primary_family_size']
    seeds = pd.read_csv(output/'seed_metrics.csv')
    assert len(seeds) == 640 and not seeds.duplicated(['dataset','seed','policy']).any()
    for dataset,g in seeds.groupby('dataset'):
        assert dataset in protocol['datasets']
        assert set(g.seed) == set(protocol['seeds']) and g.groupby('seed').size().eq(10).all()
    table = seeds.pivot(index=['dataset','seed'],columns='policy',values='alc_brier')
    stats = pd.read_csv(output/'primary_statistics.csv'); assert len(stats) == m
    pvals = []
    for i,r in stats.iterrows():
        x = (table.loc[r.dataset,r.competitor]-table.loc[r.dataset,'DC-PRISM']).to_numpy()
        se = x.std(ddof=1)/np.sqrt(len(x)); c = t.ppf(1-protocol['alpha']/(2*m),len(x)-1)
        assert r.n == len(x) == 16 and abs(r.mean_gain-x.mean()) < 1e-12
        assert abs(r.simultaneous_low-(x.mean()-c*se)) < 1e-12
        assert abs(r.simultaneous_high-(x.mean()+c*se)) < 1e-12
        p = float(ttest_1samp(x,0).pvalue) if x.std(ddof=1)>1e-15 else (1. if abs(x.mean())<1e-15 else 0.)
        assert abs(r.p_raw-p) < 1e-12; pvals.append((i,p))
    running = 0.
    for rank,(i,p) in enumerate(sorted(pvals,key=lambda x:x[1])):
        running = max(running,min(1.,p*(m-rank)))
        assert abs(stats.loc[i,'p_holm_global12']-running) < 1e-12
    return {'status':'PASS','seed_metrics':len(seeds),'primary_statistics_recomputed':m,
            'scope':'public statistical recomputation; full trajectory/release audit and replay require local data'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output',type=Path)
    print(json.dumps(verify(p.parse_args().output),indent=2))
