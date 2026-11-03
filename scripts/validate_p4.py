"""Validate release schedules, disjointness, budgets and first-round causality."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SAFE = {'Safe-MinRisk','Safe-PA-NoGate','Fixed-Eps-Safe-PA','DC-PRISM',
        'D-Only-Safe-PA','PRISM-ValueUCB','BAIT-Cost-Safe'}


def validate(path):
    df = pd.read_csv(path)
    assert not df.duplicated(['dataset','seed','round','policy']).any(), 'Duplicate state-policy rows'
    assert not df.isna()[['brier_t','B_t','risk_envelope']].any().any(), 'Missing metrics'
    assert df.loc[df.policy.isin(SAFE),'envelope_violation'].eq(0).all(), 'Unsafe selected set'
    fixed=df[df.policy=='Fixed-Eps-Safe-PA']
    assert np.array_equal(fixed.k_selected.to_numpy(),np.ceil(.75*fixed.k_safe.to_numpy())), 'Fixed-epsilon cardinality target missed'
    for _, g in df.groupby(['dataset','seed','round']):
        assert g.B_t.nunique() == 1 and g.budget_hash.nunique() == 1, 'Non-shared budget'
    for _, first in df[df['round']==1].groupby(['dataset','seed']):
        assert all(json.loads(x)==[] for x in first.released_ids), 'Unexpected first-round release'
        assert np.ptp(first.brier_t.to_numpy()) < 1e-12, 'Unreleased labels affected first-round Brier'
    for _, group in df.groupby(['dataset','seed','policy']):
        due = {}; explored=set(); observed=set()
        for row in group.sort_values('round').itertuples():
            released=set(json.loads(row.released_ids))
            assert released == due.pop(row.round,set()), 'Incorrect label release schedule'
            observed.update(released)
            selected=set(json.loads(row.selected_ids))
            candidate=set(json.loads(row.candidate_ids))
            ordinary=set(json.loads(row.current_O_ids))
            assert len(selected)==row.k_selected and selected <= candidate, 'Invalid selection IDs'
            assert not selected & (explored|observed), 'Repeated exploration'
            assert not candidate & ordinary, 'C/O overlap'
            explored.update(selected)
            due.setdefault(row.round+1,set()).update(selected)
            due.setdefault(row.round+2,set()).update(ordinary)
    return {'rows':len(df),'trajectories':df.groupby(['dataset','seed','policy']).ngroups,
            'first_round_causality':'PASS','release_schedule':'PASS','shared_budget':'PASS',
            'safe_envelope_feasibility':'PASS','repeat_exploration':'PASS'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('trajectory', type=Path)
    print(json.dumps(validate(parser.parse_args().trajectory),indent=2))
