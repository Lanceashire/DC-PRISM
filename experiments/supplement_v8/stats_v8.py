"""Registered summaries for supplement-v8 (greedy-oracle gap decomposition)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v8', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def run(df, protocol, out):
    inf = protocol['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    df = df.assign(test_reuse=df.reported-df.ex_post, floor=df.ex_post-df.deployable,
                   gap_vs_best_reported=df.reported-df.best_practical, gap_vs_best_deployable=df.deployable-df.best_practical)
    cols = ['reported', 'test_reuse', 'ex_post', 'floor', 'deployable', 'best_practical', 'gap_vs_best_reported',
            'gap_vs_best_deployable', 'oracle_k_post', 'oracle_k_dep', 'K']
    g = df.groupby(['setting', 'group'])[cols].mean()
    for c in ['test_reuse', 'floor', 'deployable']:
        g[f'share_{c}'] = g[c]/g['reported']
    g.to_csv(out/'summary_by_group.csv')
    s = df.groupby('setting')[cols].mean()
    for c in ['test_reuse', 'floor', 'deployable']:
        s[f'share_{c}'] = s[c]/s['reported']
    s.to_csv(out/'summary_by_setting.csv')
    recs = []
    for (setting, grp), x in df.groupby(['setting', 'group']):
        x = x.sort_values('seed')
        for e in protocol['study']['primary_endpoints']:
            recs.append(({'setting': setting, 'group': grp, 'endpoint': e}, x[e].to_numpy()))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study']['primary_family_size']
    p.to_csv(out/'primary_statistics.csv', index=False)
