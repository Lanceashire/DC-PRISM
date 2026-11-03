"""Regenerate the unmodified submitted P4, then verify semantic column hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    from verify_local_data import verify
    verify(args.data_dir/'real_credit_datasets_authentic.npz')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    for sub in ['p4_real_labels','results/audit']:(output/sub).mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'archive/submitted_v2_2/src'))
    import run_p3_p4 as original
    original.DATA_DIR=str(args.data_dir.resolve())
    original.PKG_ROOT=str(output);original.AUDIT_DIR=str(output/'results/audit')
    original.run_p4_real_labels_benchmark(original.load_four_real_datasets())
    expected=json.loads((ROOT/'provenance/submitted_p4_semantic_hashes.json').read_text())
    df=pd.read_csv(output/'p4_real_labels/p4_trajectories_10rounds.csv')
    df=df.sort_values(expected['sort_keys']).reset_index(drop=True)
    actual={c:hashlib.sha256(json.dumps(df[c].tolist(),separators=(',',':')).encode()).hexdigest()
            for c in expected['columns']}
    status={'rows':len(df),'column_matches':{c:actual[c]==v for c,v in expected['columns'].items()}}
    (output/'semantic_verification.json').write_text(json.dumps(status,indent=2))
    if len(df)!=expected['rows'] or not all(status['column_matches'].values()):
        raise RuntimeError('Submitted P4 semantic regeneration differs; inspect the environment and source')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
