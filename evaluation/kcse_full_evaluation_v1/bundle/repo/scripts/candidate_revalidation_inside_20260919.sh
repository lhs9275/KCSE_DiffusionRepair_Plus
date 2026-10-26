#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd /bundle
sha256sum -c SHA256SUMS.txt > /output/internal-start.txt
python3 --version > /output/python-version.txt
cd /bundle/repo
python3 evaluation/evaluate.py --results_dir "/bundle/candidates/$1" --bugs_meta /bundle/assets/metadata.json --bugs "$2" --compile-timeout-seconds "$3" --workspace_root /output/workspaces --num_processes 4 --output /output/evaluation.json
cd /bundle
sha256sum -c SHA256SUMS.txt > /output/internal-end.txt
