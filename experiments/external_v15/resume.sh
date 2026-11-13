#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
workers=${1:-2}
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Usage: bash resume.sh [positive worker count]'; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
venv/bin/python verify_snapshot.py
venv/bin/python check_env.py
venv/bin/python - "$workers" <<'PY'
import datetime, importlib.metadata, json, subprocess, sys
from pathlib import Path
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
p = Path('provenance/continuations'); p.mkdir(parents=True, exist_ok=True)
record = dict(started=datetime.datetime.now().astimezone().isoformat(), workers=int(sys.argv[1]),
              completed_at_launch=sorted(x.name for x in Path('runs/main').glob('*.csv')),
              origin_snapshot='provenance/snapshot.json',
              dependencies={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','scikit-learn']},
              gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],text=True))
with open('runs/supervisor_console.log','a') as log:
    child=subprocess.Popen(['venv/bin/python','-u','server_supervisor.py',sys.argv[1]],
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['supervisor_pid']=child.pid
(p / f'{stamp}.json').write_text(json.dumps(record,indent=2))
print('Started background supervisor:',child.pid)
print('Check actual state with: bash status.sh')
PY
