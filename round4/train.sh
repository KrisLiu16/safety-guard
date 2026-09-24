#!/usr/bin/env bash
set -euo pipefail
cd /work/round4
sha256sum -c training_files.sha256
/work/modern/bin/python /work/round4/train_risk.py 2>&1 | tee /work/output/round4/training.log
/work/legacy/bin/python /work/round4/eval_guard_reference.py 2>&1 | tee /work/output/round4/a0_reference.log
touch /work/output/round4_ready
for i in $(seq 1 1800); do
  if test -f /work/output/round4_collected; then exit 0; fi
  sleep 1
done
exit 1
