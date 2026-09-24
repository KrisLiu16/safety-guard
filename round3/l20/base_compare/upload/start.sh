#!/usr/bin/env bash
set -euo pipefail
cd /work/input
sha256sum -c code_and_data.sha256
for i in $(seq 1 3600); do
  if test -f /work/output/environment_ready; then break; fi
  sleep 1
done
test -f /work/output/environment_ready
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader
if ! test -f /work/output/teacher/manifest.json; then
  /work/legacy/bin/python /work/input/export_teacher.py 2>&1 | tee /work/output/teacher_run.log
fi
for model in qwen3 qwen35; do
  /work/modern/bin/python /work/input/train_base.py --model "$model" --training-mode full --smoke-only 2>&1 | tee "/work/output/${model}_smoke.log"
  /work/modern/bin/python /work/input/train_base.py --model "$model" --training-mode full 2>&1 | tee "/work/output/${model}_train.log"
done
touch /work/output/comparison_ready
for i in $(seq 1 3600); do
  if test -f /work/output/artifacts_collected; then exit 0; fi
  sleep 1
done
echo 'Artifact collection acknowledgement timeout' >&2
exit 1
