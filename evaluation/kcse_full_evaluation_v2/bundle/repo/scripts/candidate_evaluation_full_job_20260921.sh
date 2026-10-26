#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
bundle=$1
expected=$2
root="${bundle%/bundle}"
slot=${SLURM_ARRAY_TASK_ID:?}
mapfile -t args < <(python3 - "$bundle/protocol.json" "$slot" <<'PY'
import json,sys
t=json.load(open(sys.argv[1]))['tasks'][int(sys.argv[2])]
print(t['run']);print(','.join(t['ids']));print(t['compile_timeout']);print(t['task_index'])
PY
)
run=${args[0]}; ids=${args[1]}; timeout=${args[2]}; task_index=${args[3]}
output="$root/results/slot_${slot}_task_${task_index}_${run}_bug_${ids}_job_${SLURM_JOB_ID}"
mkdir -m 750 -p "$output"
printf '%s\n' "$slot" "$task_index" "$run" "$ids" > "$output/task.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/slurm.txt"
printf '%s  %s\n' "$expected" "$bundle/SHA256SUMS.txt" | sha256sum -c - > "$output/manifest-check.txt"
(cd "$bundle" && sha256sum -c SHA256SUMS.txt) > "$output/host-start.txt"
archive=/cephfs/home/s2022810061/.cehub/docker-images/kcse-bugsinpy_revision-v2-py38--2ff43c4b906e.tar.zst
printf '%s  %s\n' ae173785ec104931921c0e8acc8a739238e6d3c0c70076a74176675974ca2ac1 "$archive" | sha256sum -c - > "$output/archive-check.txt"
/usr/local/bin/cehub-docker load kcse-bugsinpy:revision-v2-py38 > "$output/image-load.txt" 2>&1
image=sha256:eb2e457d0685eb31003fa6e1ff6a6fac6ce3230765ec7bf494747fd9111b71a0
/usr/local/bin/cehub-docker exec image inspect "$image" > "$output/image-inspect.json"
container=kcse-eval-${SLURM_JOB_ID}-${slot}
/usr/local/bin/cehub-docker exec create --name "$container" --entrypoint /bin/bash \
 -e SLURM_JOB_ID="$SLURM_JOB_ID" -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
 --mount "type=bind,src=$bundle,dst=/bundle,readonly" --mount "type=bind,src=$output,dst=/output" \
 --workdir /bundle/repo "$image" /bundle/repo/scripts/candidate_revalidation_inside_20260919.sh \
 "$run" "$ids" "$timeout" > "$output/container-id.txt"
/usr/local/bin/cehub-docker exec container inspect "$container" > "$output/container-start.json"
set +e
/usr/local/bin/cehub-docker exec start -a "$container" > "$output/stdout.txt" 2> "$output/stderr.txt"
rc=$?
set -e
/usr/local/bin/cehub-docker exec container inspect "$container" > "$output/container-end.json"
(cd "$bundle" && sha256sum -c SHA256SUMS.txt) > "$output/host-end.txt"
printf '%s\n' "$rc" > "$output/rc.txt"
exit "$rc"
