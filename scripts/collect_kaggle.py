"""Collect a Kaggle job's output and verify it is a completed run.

Usage: python scripts/collect_kaggle.py --kernel gjusev/mokka-decisions-pilot \
           --dest runs/kaggle-pilot

Verifies: download succeeded, run_status.json exists and names the expected
stage, training produced a checkpoint with global_step > 0 (train stage), and
prints artifact hashes. A successful push is not a completed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--expect-stage", default=None)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-m", "kaggle", "kernels", "output", args.kernel, "-p", str(dest)]
    )

    status_file = dest / "run_status.json"
    if not status_file.exists():
        print("FAIL: no run_status.json in kernel output — job did not reach its verdict")
        return 1
    status = json.loads(status_file.read_text(encoding="utf-8"))
    print("run_status:", json.dumps(status, indent=2)[:2000])

    if args.expect_stage and status.get("stage") != args.expect_stage:
        print(f"FAIL: expected stage {args.expect_stage!r}, got {status.get('stage')!r}")
        return 1

    if status.get("stage") == "train":
        if not status.get("final_global_step"):
            print("FAIL: train stage without final_global_step")
            return 1
        ckpts = list(dest.rglob("*.safetensors")) + list(dest.rglob("checkpoint_latest.pt"))
        if not ckpts:
            print("FAIL: no checkpoint artifact in output")
            return 1

    print("artifacts:")
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "run_status.json":
            print(f"  {p.relative_to(dest)}  {p.stat().st_size} B  sha256:{sha256(p)[:12]}")
    print("OK: verified completed run output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
