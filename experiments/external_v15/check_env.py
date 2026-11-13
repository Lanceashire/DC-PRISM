"""Pre-flight check: GPU visible, frozen sources/features unchanged, a tiny head trains."""
import hashlib, json, sys
from pathlib import Path
import torch
R = Path(__file__).resolve().parent; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
F = json.loads((R/'freeze.json').read_text()); ok = True
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f'  GPU{i}: {p.name}, {p.total_memory/2**30:.1f} GiB')
if not torch.cuda.is_available(): print('ERROR: no CUDA GPU'); ok = False
if F['protocol_sha256'] != sha(R/'protocol.json'): print('ERROR protocol changed'); ok = False
for k, h in F['sources'].items():
    if sha(R/k) != h: print('ERROR source changed:', k); ok = False
for k, h in F['features'].items():
    if not (R/'features'/k).exists() or sha(R/'features'/k) != h: print('ERROR feature missing/changed:', k); ok = False
if ok:
    sys.path.insert(0, str(R)); import heads, numpy as np
    rng = np.random.default_rng(0); X = rng.normal(size=(80, 64)).astype('float32'); y = rng.integers(10, size=80)
    p = heads.predict_many([(X, y)], X, 10, 1, 64, 60, .02, 16); print('head smoke test OK, prob sum', round(float(p.sum(-1).mean()), 4))
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
