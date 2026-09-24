#!/usr/bin/env bash
set -euo pipefail
cd /work/round4
sha256sum -c smoke_files.sha256
apt-get update > /work/output/round4/compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/round4/compiler_install.log 2>&1
/work/modern/bin/python /work/round4/test_memory_l20.py 2>&1 | tee /work/output/round4/memory_smoke.log
for i in $(seq 1 1800); do
  if test -f /work/round4/train_ready; then exec bash /work/round4/train.sh; fi
  sleep 1
done
echo 'Training package was not ready within the bounded preparation window' >&2
exit 1
