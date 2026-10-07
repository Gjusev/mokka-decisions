"""Assemble the local release from collected Kaggle artifacts.

Usage:
    python scripts/export_release.py \
        --run-dir runs/kaggle-final \
        --checkpoint model_best.safetensors \
        --release-config configs/release.yaml

Verifies inputs exist (checkpoint, calibration.json, policy.json, run_report),
exports the self-contained directory and prints its manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="collected kernel output dir")
    parser.add_argument("--checkpoint", default="model_best.safetensors")
    parser.add_argument("--release-config", default="configs/release.yaml")
    args = parser.parse_args()

    import yaml

    from mokka_decisions.export import export_release

    run_dir = Path(args.run_dir)
    ckpt = run_dir / args.checkpoint
    calibration_file = run_dir / "calibration.json"
    policy_file = run_dir / "policy.json"
    missing = [p.name for p in (ckpt, calibration_file, policy_file) if not p.exists()]
    if missing:
        print(f"FAIL: missing artifacts in {run_dir}: {missing}", file=sys.stderr)
        return 1

    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    report = run_dir / "train" / "run_report.json"
    meta = {}
    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))
        meta = {
            "epochs": rep.get("epochs_completed"),
            "final_global_step": rep.get("final_global_step"),
            "best_dev_accuracy": rep.get("best_dev_accuracy"),
            "config_hash": rep.get("config_hash"),
            "data_manifest_hash": rep.get("data_manifest_hash"),
        }
    rel_cfg = yaml.safe_load(Path(args.release_config).read_text(encoding="utf-8"))["release"]

    out = export_release(
        checkpoint=ckpt,
        encoder_name="jhu-clsp/mmBERT-base",
        out_dir=rel_cfg["export_dir"],
        calibration=calibration,
        policy=policy,
        meta=meta,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print(f"release: {out}")
    for name, info in manifest["artifacts"].items():
        print(f"  {name}: {info['bytes']/2**20:.1f} MiB sha256:{info['sha256'][:12]}")

    # immediate self-check: load from the export dir and score one case
    from mokka_decisions.contracts import Option, validate_request_payload
    from mokka_decisions.export import load_exported_model

    scorer, pol = load_exported_model(out)
    case = validate_request_payload(
        {
            "state": "I was charged twice for the same order.",
            "question": "Which banking support intent does this customer request express?",
            "options": [
                {"id": "transaction_charged_twice", "description": "The same transaction was charged twice"},
                {"id": "card_arrival", "description": "Ask when a new or replacement card will arrive"},
                {"id": "none", "description": "None of the listed options applies to this request."},
            ],
        }
    )
    probs = scorer.score([case])[0]
    print("self-check probabilities:", {k: round(v, 3) for k, v in probs.items()})
    print("OK: release loads and infers offline from its own directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
