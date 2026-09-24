#!/usr/bin/env bash
set -euo pipefail
cd /work/window
sha256sum -c files.sha256
apt-get update > /work/output/window_compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/window_compiler_install.log 2>&1
/work/modern/bin/python /work/window/run_probe.py 2>&1 | tee /work/output/window_probe_run.log
touch /work/output/window_probe_ready
for i in $(seq 1 1800); do
  if test -f /work/output/window_probe_collected; then exit 0; fi
  sleep 1
done
exit 1
