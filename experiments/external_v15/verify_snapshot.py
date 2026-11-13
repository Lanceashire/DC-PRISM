"""Verify captured evidence and complete-file structure without computing endpoints."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'provenance/snapshot_sha256.json').read_text())
for name,expected in manifest.items():
 p=root/name
 if not p.is_file() and name.startswith('runs/main/'):
  p=root/'runs/l40_snapshot_99'/Path(name).name
 assert p.is_file(), f'Missing snapshot file: {name}'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, f'Snapshot modified: {name}'
p=json.loads((root/'protocol.json').read_text())
methods={'oracle-test','oracle-split','oracle-exante',*p['strategies']}
required={(s,m) for s in range(p['steps']+1) for m in methods}
allowed=set()
for g in p['grids'].values():
 for seed in range(g['seed_start'],g['seed_start']+g['n_seeds']):
  for ds in g['datasets']:
   image=ds in {'cifar10','cifar100c','fmnist','svhn'}
   enc=('dinov2s' if image else 'bge') if g['encoder']=='primary' else ('mnv3s' if image else 'minilm')
   allowed.add((ds,enc,g['head'],str(seed)))
files=list((root/'runs/main').glob('*.csv'))
for path in files:
 with path.open(newline='') as f: rows=list(csv.DictReader(f))
 assert len(rows)==len(required) and {(int(r['step']),r['method']) for r in rows}==required, f'Incomplete: {path}'
 key=tuple(rows[0][k] for k in ('dataset','encoder','head','seed'))
 assert key in allowed and path.stem=='__'.join(key), f'Unregistered trajectory: {path}'
 assert all(tuple(r[k] for k in ('dataset','encoder','head','seed'))==key for r in rows),path
 assert all(math.isfinite(float(r[k])) for r in rows for k in ('brier_T','brier_V','acc_T')),path
 assert path.with_name(path.stem+'_audit.json').is_file(),path
print(f'PASS: {len(manifest)} snapshot files retain SHA-256; {len(files)}/384 complete formal trajectories.')
