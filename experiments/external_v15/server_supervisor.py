"""Run the unchanged frozen v15 protocol and summarize only complete results."""
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
STATE = ROOT / 'runs' / 'supervisor_state.json'
ROOT.joinpath('runs').mkdir(exist_ok=True)
LOCK = open(ROOT / 'runs' / 'supervisor.lock', 'w')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
env = os.environ.copy()
env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHONWARNINGS='ignore')
state = dict(status='STARTING', started=time.time(), supervisor_pid=os.getpid(),
             workers=workers, total=384, log=f'runs/main_{STAMP}.log',
             note='Frozen protocol and research sources unchanged; local18 excluded.')


def save(**changes):
    state.update(changes)
    state['updated'] = time.time()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def validate_all():
    p = json.loads(Path('protocol.json').read_text())
    primary = dict(image='dinov2s', text='bge')
    secondary = dict(image='mnv3s', text='minilm')
    methods = {'oracle-test', 'oracle-split', 'oracle-exante', *p['strategies']}
    required = {(step, method) for step in range(p['steps'] + 1) for method in methods}
    expected = []
    for grid in p['grids'].values():
        for seed in range(grid['seed_start'], grid['seed_start'] + grid['n_seeds']):
            for dataset in grid['datasets']:
                kind = 'image' if dataset in {'cifar10', 'cifar100c', 'fmnist', 'svhn'} else 'text'
                encoder = (primary if grid['encoder'] == 'primary' else secondary)[kind]
                expected.append((dataset, encoder, grid['head'], seed))
    assert len(expected) == 384
    for dataset, encoder, head, seed in expected:
        path = Path('runs/main') / f'{dataset}__{encoder}__{head}__{seed}.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(required), f'incomplete trajectory: {path}'
        assert {(int(row['step']), row['method']) for row in rows} == required, path
        for row in rows:
            assert (row['dataset'], row['encoder'], row['head'], int(row['seed'])) == (dataset, encoder, head, seed), path
            assert all(math.isfinite(float(row[k])) for k in ('brier_T', 'brier_V', 'acc_T')), path
        assert path.with_name(path.stem + '_audit.json').is_file(), path
    assert len(list(Path('runs/main').glob('*.csv'))) == len(expected)
    return len(expected)


save()
try:
    out = Path('runs/main')
    out.mkdir(exist_ok=True)
    old = out / 'run_manifest.json'
    if old.exists():
        old.rename(out / f'run_manifest_before_codex_{STAMP}.json')
    command = ['venv/bin/python', '-u', 'run_v15.py', '--protocol', 'protocol.json',
               '--output', 'runs/main', '--grids', 'A,B,C', '--workers', str(workers)]
    with open(state['log'], 'a') as log:
        log.write(json.dumps(dict(command=command, started=state['started'])) + '\n')
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        save(status='RUNNING', runner_pid=child.pid)
        rc = child.wait()
    save(runner_exit_code=rc, training_finished=time.time())
    if rc != 0:
        raise RuntimeError(f'runner failed, exit {rc}; see {state["log"]}')
    count = validate_all()
    save(status='SUMMARIZING', validated_trajectories=count)
    with open(f'runs/stats_{STAMP}.log', 'a') as log:
        subprocess.run(['venv/bin/python', 'stats_v15.py', '--protocol', 'protocol.json',
                        '--runs', 'runs/main', '--output', 'results'],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    save(status='COMPLETE', finished=time.time(), results='results')
except BaseException as exc:
    save(status='FAILED', error=f'{type(exc).__name__}: {exc}', failed=time.time())
    raise
