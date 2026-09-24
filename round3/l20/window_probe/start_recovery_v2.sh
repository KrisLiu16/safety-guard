#!/usr/bin/env bash
set -euo pipefail
cd /work/recovery
sha256sum -c files.sha256
apt-get update > /work/output/recovery_compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/recovery_compiler_install.log 2>&1
/work/modern/bin/python /work/recovery/recover_window.py --mode structure 2>&1 | tee /work/output/window_recovery_v2_run.log
touch /work/output/window_recovery_v2_ready
for i in $(seq 1 1800); do
  if test -f /work/output/window_recovery_v2_collected; then exit 0; fi
  sleep 1
done
exit 1
