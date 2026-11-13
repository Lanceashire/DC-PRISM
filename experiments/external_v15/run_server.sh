#!/bin/bash
# Usage:
#   bash run_server.sh single [WORKERS]        # one GPU: grids A,B,C in runs/main (default WORKERS=3)
#   bash run_server.sh multi  [WORKERS]        # >=3 GPUs: grid A on GPU0, B on GPU1, C on GPU2 (separate dirs)
#   bash run_server.sh status                  # progress
#   bash run_server.sh stats                   # after completion: merge runs and compute registered statistics
# Resumable: completed trajectory CSVs are skipped. Optional: cp runs/main_local18/*.csv runs/main/ to reuse the
# 18 trajectories computed locally (RTX 5060) before starting; by default everything is recomputed on the server.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore
MODE=${1:-single}; W=${2:-3}
run() { # $1 grids $2 outdir $3 gpu
  mkdir -p "$2"; CUDA_VISIBLE_DEVICES=$3 nohup venv/bin/python run_v15.py --protocol protocol.json --output "$2" --grids "$1" --workers $W > "$2.log" 2>&1 &
  echo "started grids $1 on GPU $3 -> $2 (pid $!)"; }
case $MODE in
  single) [ -f runs/main/run_manifest.json ] && mv runs/main/run_manifest.json runs/main/run_manifest_prev_$(date +%s).json; run A,B,C runs/main 0 ;;
  multi)  for g in A B C; do d=runs/main_$g; [ -f $d/run_manifest.json ] && mv $d/run_manifest.json $d/run_manifest_prev_$(date +%s).json; done
          run A runs/main_A 0; run B runs/main_B 1; run C runs/main_C 2 ;;
  status) for d in runs/main runs/main_A runs/main_B runs/main_C; do [ -f $d/progress.txt ] && echo "$d: $(cat $d/progress.txt)"; [ -f $d/run_manifest.json ] && grep -h '"status"' $d/run_manifest.json; done
          echo "completed trajectories: $(ls runs/main*/*__*__*__*.csv 2>/dev/null | grep -v local18 | wc -l)/384"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader ;;
  stats)  mkdir -p runs/merged; for f in runs/main*/*__*__*__*.csv; do case $f in *local18*) ;; *) ln -sf "$(realpath $f)" runs/merged/;; esac; done
          echo "merged $(ls runs/merged | wc -l) trajectories"; venv/bin/python stats_v15.py --protocol protocol.json --runs runs/merged --output results ;;
esac
