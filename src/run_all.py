"""Run audited experiments into a new directory; never restore archived tables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['checks', 'p4', 'p3'], default='checks')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--smoke', action='store_true', help='One HMEQ seed for P4; never a confirmation result')
    args = p.parse_args()
    if args.data_dir:
        os.environ['DC_PRISM_DATA_DIR'] = str(args.data_dir.resolve())
    if args.suite == 'checks':
        subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT/'tests')], check=True, cwd=ROOT)
        return
    if not args.data_dir or not args.output_dir:
        p.error('Experiments require --data-dir and a new --output-dir')
    data = args.data_dir.resolve()/'real_credit_datasets_authentic.npz'
    if not data.is_file():
        p.error(f'Missing local input: {data}')
    output = args.output_dir.resolve()
    if output == ROOT or ROOT/'archive' in output.parents or output == ROOT/'archive':
        p.error('Output must not replace repository source or archive')
    if args.suite == 'p3' and args.smoke:
        p.error('P3 smoke is not implemented; use P4 smoke or checks')
    output.mkdir(parents=True, exist_ok=False)
    for sub in ['p3_realx', 'p4_real_labels', 'results/audit']:
        (output/sub).mkdir(parents=True)
    import numpy as np
    import scipy
    import pandas as pd
    import run_p3_p4 as runner
    runner.PKG_ROOT = str(output)
    runner.AUDIT_DIR = str(output/'results/audit')
    runner.DATA_DIR = str(args.data_dir.resolve())
    manifest = {
        'protocol_version': 'dc-prism-integration-v1',
        'suite': args.suite, 'mode': 'smoke' if args.smoke else 'full',
        'status': 'RUNNING', 'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
        'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'source_sha256': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'src').glob('*.py')},
        'script_sha256': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'scripts').glob('*.py')},
        'metric_timing': 'start_of_round_after_due_releases',
        'confirmation_claim': False,
    }
    manifest_path = output/'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        datasets = runner.load_four_real_datasets()
        if args.smoke:
            datasets = {'HMEQ': datasets['HMEQ']}
        if args.suite == 'p4':
            audit = runner.run_hidden_label_mutation_audit(datasets)
            if not audit['all_passed']:
                raise RuntimeError('Hidden-label mutation audit failed')
            runner.run_p4_real_labels_benchmark(datasets, seeds=[4001] if args.smoke else range(4001,4017))
            sys.path.insert(0, str(ROOT/'scripts'))
            from validate_p4 import validate
            manifest['validation'] = validate(output/'p4_real_labels/p4_trajectories_10rounds.csv')
        else:
            runner.run_p3_realx_benchmark(datasets)
        manifest['status'] = 'COMPLETE'
    except BaseException as exc:
        manifest['status'] = 'FAILED'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
