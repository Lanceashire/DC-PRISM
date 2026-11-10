"""Create a separately identified replication freeze after regenerating features."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args()
original=json.loads((ROOT/'freeze.json').read_text())
for name,h in original['sources'].items():
    if name!='assets_manifest.json':assert sha(ROOT/name)==h, f'Source changed: {name}'
a.directory.mkdir(parents=True,exist_ok=False)
shutil.copyfile(ROOT/'protocol.json',a.directory/'protocol.json')
f={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'protocol_sha256':sha(ROOT/'protocol.json'),'original_freeze_sha256':sha(ROOT/'freeze.json'),
   'sources':{n:sha(ROOT/n) for n in original['sources']},
   'data_sha256':{n:sha(ROOT/n) for n in original['data_sha256']},
   'scope':'Independent replication freeze; original archived freeze retained unchanged.'}
(a.directory/'freeze.json').write_text(json.dumps(f,indent=2))
print(a.directory/'freeze.json')
