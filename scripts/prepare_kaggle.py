"""Assemble the private Kaggle bundle: wheel + configs + prepared data.

Two bundle shapes:

* default: the full job bundle (dataset `gjusev/mokka-decisions-bundle`);
* --checkpoint PATH: a checkpoint-transport bundle (dataset
  `gjusev/mokka-decisions-checkpoint`) carrying weights + optimizer + RNG
  states + metadata, for resume and eval jobs.

Every bundle gets BUNDLE_MANIFEST.json with sha256 of every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_wheel(out_dir: Path) -> Path:
    import os

    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIP_CONFIG_FILE="/dev/null")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out_dir), str(ROOT)],
        env=env,
    )
    wheels = sorted(out_dir.glob("mokka_decisions-*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced nothing")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--bundle", default="kaggle-bundle")
    parser.add_argument("--checkpoint", default=None, help="build a checkpoint-transport bundle instead")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if bundle.exists():
        shutil.rmtree(bundle)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        bundle.mkdir(parents=True)
        shutil.copy2(ckpt, bundle / ckpt.name)
        for sidecar in (ckpt.with_suffix(".json"),):
            if sidecar.exists():
                shutil.copy2(sidecar, bundle / sidecar.name)
        (bundle / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "mokka-decisions-checkpoint",
                    "id": "gjusev/mokka-decisions-checkpoint",
                    "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        wheel = build_wheel(bundle / "dist")
        print(f"wheel: {wheel.name}")
        for cfg in sorted((ROOT / "configs").glob("*.yaml")):
            shutil.copy2(cfg, bundle / cfg.name)
        data = Path(args.data)
        if not (data / "instances").exists():
            raise SystemExit(f"no prepared data under {data}; run scripts/prepare_data.py first")
        shutil.copytree(data, bundle / "data" / "processed" / "v1")
        (bundle / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "mokka-decisions-bundle",
                    "id": "gjusev/mokka-decisions-bundle",
                    "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "files": {
            str(p.relative_to(bundle)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(bundle.rglob("*"))
            if p.is_file() and p.name not in ("BUNDLE_MANIFEST.json",)
        },
        "cli": sys.argv,
    }
    (bundle / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    total = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"bundle: {bundle} ({total / 2**20:.1f} MiB, {len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
