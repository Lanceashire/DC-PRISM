#!/bin/bash
# Usage:
#   bash setup.sh system   # recommended on cloud images: reuse the image's preinstalled PyTorch (no big download)
#   bash setup.sh cu128    # fresh venv with PyTorch from the official index (cu128 / cu126 / cu124)
set -e; cd "$(dirname "$0")"; MODE=${1:-system}
if [ "$MODE" = system ]; then
  python3 -c "import torch; assert torch.cuda.is_available(), 'image torch has no CUDA'; print('using image torch', torch.__version__)"
  python3 -m venv --system-site-packages venv
  venv/bin/pip install -q "numpy>=1.26" "pandas>=2.0" "scipy>=1.11" "scikit-learn>=1.4" threadpoolctl
else
  python3 -m venv venv; venv/bin/pip install --upgrade pip
  venv/bin/pip install torch --index-url https://download.pytorch.org/whl/$MODE
  venv/bin/pip install -r requirements.txt
fi
venv/bin/python check_env.py
