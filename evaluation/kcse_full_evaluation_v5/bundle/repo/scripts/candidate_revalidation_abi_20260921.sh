#!/bin/bash
# Candidate evaluation for the frozen eight pandas IDs with the ABI rebuild
# enabled (protocol amendment 2026-09-21). Identical to the standard inside
# script apart from --diagnostic-abi-rebuild and the extended compile limit.
set -euo pipefail
cd /bundle && sha256sum -c SHA256SUMS.txt > /output/internal-start.txt
cd /bundle/repo
python3 evaluation/evaluate.py --results_dir "/bundle/candidates/$1" \
  --bugs_meta /bundle/assets/metadata.json --bugs "$2" \
  --diagnostic-abi-rebuild --compile-timeout-seconds "$3" \
  --workspace_root /output/workspaces --num_processes 4 --output /output/evaluation.json
cd /bundle && sha256sum -c SHA256SUMS.txt > /output/internal-end.txt
