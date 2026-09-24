#!/usr/bin/env bash
set -euo pipefail
cd /work/round4/fast_runtime
sha256sum -c input_files.sha256
sha256sum -c external_files.sha256
mkdir -p /work/output/fast_runtime
apt-get update > /work/output/fast_runtime/compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/fast_runtime/compiler_install.log 2>&1
/work/modern/bin/python /work/round4/fast_runtime/run_fast_validation.py 2>&1 | tee /work/output/fast_runtime/validation.log
touch /work/output/fast_runtime_ready
for i in $(seq 1 1800); do
  if test -f /work/output/fast_runtime_collected; then exit 0; fi
  sleep 1
done
echo 'Fast text validation collection acknowledgement timed out' >&2
exit 1
