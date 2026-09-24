#!/usr/bin/env bash
set -euo pipefail
cd /work/validation
sha256sum -c input_files.sha256
mkdir -p /work/output/round4_validation
apt-get update > /work/output/round4_validation/compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/round4_validation/compiler_install.log 2>&1
/work/modern/bin/python /work/validation/run_final_validation.py 2>&1 | tee /work/output/round4_validation/validation.log
touch /work/output/round4_validation_ready
for i in $(seq 1 1800); do
  if test -f /work/output/round4_validation_collected; then exit 0; fi
  sleep 1
done
echo 'Final validation report collection acknowledgement timed out' >&2
exit 1
