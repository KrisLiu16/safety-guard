#!/bin/bash
# Resume the Run B draft upload across network drops and Aster redeploys. Upload only: no submit.
# Non-network failures only continue if the CLI is behind the server (then `aster update` once per
# server version); any other failure stops without submitting.
set -u
cd "$(dirname "$0")"
LOG=batch_rest/upload.log
VERSION_ID=dsv_01M38NHHQF0DQZ1MESYZC4WBTR
updated_for=""
for attempt in $(seq 1 30); do
  ../../.venv/bin/python ../../round2/upload_dataset_chunks.py $VERSION_ID batch_rest/archives >> $LOG 2>&1
  code=$?
  if tail -1 $LOG | grep -q '"status": "uploaded"'; then break; fi
  if tail -1 $LOG | grep -q CLI_UNREACHABLE; then
    echo "attempt $attempt: network drop, resuming in 30s" >> $LOG; sleep 30; continue
  fi
  cli=$(aster version | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['version'])")
  server=$(aster health | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['version'])" 2>/dev/null)
  if [ -n "$server" ] && [ "$cli" != "$server" ] && [ "$updated_for" != "$server" ]; then
    echo "attempt $attempt: CLI $cli behind server $server -> aster update" >> $LOG
    aster update >> $LOG 2>&1; updated_for="$server"; continue
  fi
  echo "attempt $attempt: non-network failure (exit $code, cli=$cli server=$server) - stopping" >> $LOG
  break
done
tail -3 $LOG
tail -1 $LOG | grep -q '"status": "uploaded"' || { echo "UPLOAD DID NOT FINISH CLEANLY"; exit 1; }
