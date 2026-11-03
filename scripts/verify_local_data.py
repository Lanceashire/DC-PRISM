"""Check the submitted NPZ bytes and structure, without claiming raw-data provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED_SHA = '8a4e800f5d64188fd718dc732be01007409a4d0953343b7c5febf0d713a238b5'
EXPECTED_ROWS = {'LendingClub':10624,'HMEQ':5960,'UCI_Credit':30000,'PPDai':55596}


def verify(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError('Input differs from submitted v2.2; register a new cohort instead of mixing results')
    details={}
    with np.load(path,allow_pickle=False) as data:
        for name,n in EXPECTED_ROWS.items():
            X,y,e=(data[name+'_'+key] for key in ['X','y','ead'])
            assert X.shape==(n,6) and y.shape==(n,) and e.shape==(n,)
            assert np.isin(y,[0,1]).all()
            details[name]={'rows':n,'features':6,'default_rate':float(y.mean()),
                           'missing_covariates':int(np.isnan(X).sum())}
    return {'sha256':digest,'byte_identity':'PASS','datasets':details,
            'raw_source_provenance':'NOT_INDEPENDENTLY_VERIFIED',
            'redistribution_rights':'NOT_ESTABLISHED_BY_HASH_CHECK'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path)
    print(json.dumps(verify(p.parse_args().archive),indent=2))
