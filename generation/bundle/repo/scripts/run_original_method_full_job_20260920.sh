#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}" || exit 2
test -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" || exit 2
bundle=$1
expected=$2
task_index=${SLURM_ARRAY_TASK_ID:?}
readarray -t task < <(python3 - "$bundle/contract/tasks.json" "$task_index" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))[int(sys.argv[2])]
print(row['bug_id']);print(row['arm']);print(row['seed'])
PY
)
bug=${task[0]}; arm=${task[1]}; seed=${task[2]}
root=/cephfs/home/s2022810061/tmp/2026_09_20/kcse_original_method_full_v1
output=$root/results/task_${task_index}_bug_${bug}_${arm}_seed_${seed}_job_${SLURM_JOB_ID}
mkdir -m 750 "$output"
printf '%s\n' "$task_index" "$bug" "$arm" "$seed" > "$output/task.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm-start.txt"
printf '%s  %s\n' "$expected" "$bundle/SHA256SUMS.txt" | sha256sum -c - > "$output/manifest-check.txt"
(cd "$bundle" && sha256sum -c SHA256SUMS.txt) > "$output/host-start.txt"
image_id=sha256:6bc378e182e3cc95edffff6589166e63558423f07f017b89b2d1b7f32340e0ad
/usr/local/bin/cehub-docker load kcse-planner-generation:v3 > "$output/image-load.txt" 2>&1
/usr/local/bin/cehub-docker exec image inspect "$image_id" > "$output/image-inspect.json"
container=kcse-full-${SLURM_JOB_ID}-${task_index}
/usr/local/bin/cehub-docker exec create --name "$container" --device=nvidia.com/gpu=all --entrypoint /bin/bash --shm-size=1g \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" -e KCSE_ARM="$arm" -e KCSE_BUG_ID="$bug" -e KCSE_SEED="$seed" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=4 \
  --mount "type=bind,src=$bundle,dst=/bundle,readonly" --mount "type=bind,src=$output,dst=/output" \
  --mount type=bind,src=/cephfs/models/starcoder2-7b,dst=/models/starcoder2-7b,readonly \
  --mount type=bind,src=/cephfs/models/e5-small-v2,dst=/models/e5-small-v2,readonly \
  --mount type=bind,src=/cephfs/home/s2022810061/Research/KCSE2026_revision_models/planner_archived_rng_primary_v4/bd4dc70ca4206e17,dst=/planner,readonly \
  --workdir /bundle/repo "$image_id" /bundle/repo/scripts/run_original_method_pilot_inside_20260920.sh > "$output/container-id.txt"
/usr/local/bin/cehub-docker exec container inspect "$container" > "$output/container-start.json"
set +e
/usr/local/bin/cehub-docker exec start -a "$container" > "$output/container.stdout" 2> "$output/container.stderr"
rc=$?
set -e
/usr/local/bin/cehub-docker exec container inspect "$container" > "$output/container-end.json"
(cd "$bundle" && sha256sum -c SHA256SUMS.txt) > "$output/host-end.txt"
printf 'OUTPUT=%s ATTACH_EXIT=%s\n' "$output" "$rc"
exit "$rc"
