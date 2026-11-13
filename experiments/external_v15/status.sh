#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
venv/bin/python - <<'PY'
import datetime,json,os
from pathlib import Path
print(datetime.datetime.now().astimezone().isoformat())
print('Saved formal trajectories:',len(list(Path('runs/main').glob('*.csv'))),'/ 384')
for name in ['runs/supervisor_state.json','runs/main/run_manifest.json','runs/main/progress.txt']:
 p=Path(name)
 if p.exists():
  print(name);print(p.read_text())
  if p.suffix=='.json':
   data=json.loads(p.read_text())
   if 'supervisor_pid' in data:
    try:
     os.kill(data['supervisor_pid'],0)
     cmd=Path(f'/proc/{data["supervisor_pid"]}/cmdline').read_bytes()
     print('Supervisor process:', 'alive' if b'server_supervisor.py' in cmd else 'PID reused; supervisor absent')
    except (ProcessLookupError,FileNotFoundError):print('Supervisor process: absent; JSON status may be stale')
PY
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
tail -n 10 runs/supervisor_console.log 2>/dev/null || true
