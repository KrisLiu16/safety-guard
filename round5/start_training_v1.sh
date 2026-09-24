#!/usr/bin/env bash
set -euo pipefail
cd /work/round5
sha256sum -c input_files.sha256
sha256sum -c external_files.sha256
sha256sum -c launch_files.sha256
mkdir -p /work/output/round5
apt-get update > /work/output/round5/compiler_install.log 2>&1
apt-get install -y --no-install-recommends gcc libc6-dev >> /work/output/round5/compiler_install.log 2>&1
/work/modern/bin/python -B /work/round5/run_training.py 2>&1 | tee /work/output/round5/launcher.log
touch /work/output/round5_training_ready
for i in $(seq 1 3600); do
  if test -f /work/output/round5_training_collected; then exit 0; fi
  sleep 1
done
echo 'Round5 collection acknowledgement timed out' >&2
exit 1
