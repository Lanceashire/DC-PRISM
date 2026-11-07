"""Verify the saved publication with Python's standard library; no original data needed."""
from pathlib import Path
import csv, gzip, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    manifest=ROOT/'provenance/publication_manifest_20261107.json'
    mismatches=[]
    if manifest.exists():
        for x in json.loads(manifest.read_text())['files']:
            p=ROOT/x['path']
            if not p.is_file() or p.stat().st_size!=x['bytes'] or digest(p)!=x['sha256']:
                mismatches.append(x['path'])
    v11=ROOT/'results/supplement_v11'
    run=json.loads((v11/'run_manifest.json').read_text())
    source_mismatches=[p for p,h in run['source_sha256'].items() if digest(ROOT/p)!=h]
    with gzip.open(v11/'rows.csv.gz','rt',newline='') as f:
        rows=list(csv.DictReader(f))
    a=[r for r in rows if r['part']=='A']
    bad_bounds=[i for i,r in enumerate(a) if not float(r['F_lo'])-1e-12<=float(r['F'])<=float(r['F_up'])+1e-12]
    hpath=ROOT/'results/supplement_v5/study_H_trajectories.csv.gz'
    with gzip.open(hpath,'rt',newline='') as f:
        count=sum(1 for _ in csv.DictReader(f))
    with (v11/'E_exact_mk.csv').open(newline='') as f:
        exact_rows=list(csv.DictReader(f))
    out={'file_mismatches':mismatches,'v11_status':run['status'],'v11_saved_rows':len(rows),
         'v11_source_hash_mismatches':source_mismatches,'family_states':len(a),'bad_empirical_bounds':len(bad_bounds),
         'exact_reference_rows':len(exact_rows),'v5_H_trajectory_round_rows':count,
         'scope':'Saved-file integrity and stored-result algebra; no complete raw-data experiment rerun.',
         'manuscript_status':'Working drafts with unresolved issues; see docs/REVIEW_STATUS_20261107.md.'}
    assert not mismatches and not source_mismatches and len(rows)==4728 and len(a)==624 and not bad_bounds
    assert count==368640 and len(exact_rows)==88
    return out
if __name__=='__main__':
    result=verify()
    print(json.dumps(result,ensure_ascii=False,indent=2))
