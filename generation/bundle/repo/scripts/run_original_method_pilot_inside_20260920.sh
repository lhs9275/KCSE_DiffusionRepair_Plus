#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}" || exit 2
cd /bundle
sha256sum -c SHA256SUMS.txt > /output/internal-start.txt
cd /bundle/repo
python3 scripts/run_original_method_pilot_20260920.py
cd /bundle
sha256sum -c SHA256SUMS.txt > /output/internal-end.txt
