#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo " DC-PRISM (DASFAA 2027 Revision 2) — Fast Standalone Reproduction"
echo "====================================================================="
python3 src/run_all.py --fast
echo "====================================================================="
echo " Fast standalone reproduction and all 10 audit checks passed!"
echo "====================================================================="
