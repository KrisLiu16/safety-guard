#!/usr/bin/env python3
"""Download, verify, and aggregate the frozen full Run after it finishes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUN = "aster-dev-267"
OUT = ROOT / "full-output-v8"


def main() -> None:
    p = subprocess.run(["aster", "runs", "get", RUN], capture_output=True, text=True)
    response = json.loads(p.stdout or p.stderr)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", {}).get("message", "Aster status failed"))
    status = response["data"]["status"]
    if status not in ("succeeded", "failed", "canceled"):
        print(json.dumps({"run": RUN, "status": status, "ready_to_extract": False}))
        return
    extraction = subprocess.run([sys.executable, str(ROOT / "extract_luna_archives.py"), RUN,
                                 "--out", str(OUT),
                                 "--expected-prompt-version", "guard-luna-bilingual-v8"])
    if extraction.returncode:
        raise SystemExit(extraction.returncode)
    aggregation = subprocess.run([sys.executable, str(ROOT / "aggregate_category_rewards.py"),
                                  "--statuses", str(OUT / "term_status.jsonl"),
                                  "--out", str(OUT / "category_rewards.json")])
    if aggregation.returncode:
        raise SystemExit(aggregation.returncode)
    print(json.dumps({"run": RUN, "status": status, "output": str(OUT),
                      "summary": str(OUT / "summary.json"),
                      "category_rewards": str(OUT / "category_rewards.json")}))


if __name__ == "__main__":
    main()
