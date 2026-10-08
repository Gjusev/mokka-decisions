"""Collect the available traceability registry for the typed-decisions work.

Records hashes/data/configs/code/checkpoints/versions/hardware that EXIST in
artifacts, and explicitly marks what was never recorded (no invention, no
reconstruction of past hashes). Output: docs/internal/traceability-<date>.md
(gitignored) + runs/traceability/traceability-<date>.json. Rerunnable; rows
appear as their artifacts land.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = str(date.today())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def arm_row(name: str, ckpt: Path, report: Path | None) -> dict:
    row = {"arm": name, "checkpoint": str(ckpt.relative_to(ROOT))}
    if (ckpt / "checkpoint_best.safetensors").exists():
        row["checkpoint_best_sha256"] = sha256(ckpt / "checkpoint_best.safetensors")
        meta_p = ckpt / "checkpoint_best.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            row["best_epoch"] = meta.get("epoch")
            row["config_hash"] = meta.get("config_hash")
            row["encoder"] = meta.get("encoder")
    else:
        row["checkpoint_best_sha256"] = "NOT_COLLECTED_YET"
    if report and report.exists():
        r = json.loads(report.read_text(encoding="utf-8"))
        row["best_dev_accuracy"] = r.get("best_dev_accuracy")
    return row


def main() -> int:
    arms = [
        ("base_ord (E11)", ROOT / "runs/kaggle-typed-ord/base_ord", ROOT / "runs/kaggle-typed-ord/base_ord/run_report.json"),
        ("base_ord6 (E14)", ROOT / "runs/kaggle-typed-ord-h2/base_ord6", ROOT / "runs/kaggle-typed-ord-h2/base_ord6/run_report.json"),
        ("small_ord8 (E14)", ROOT / "runs/kaggle-typed-ord-h2/small_ord8", ROOT / "runs/kaggle-typed-ord-h2/small_ord8/run_report.json"),
        ("e15_control s42", ROOT / "runs/kaggle-typed-e15/e15_control", None),
        ("e15_probs s42", ROOT / "runs/kaggle-typed-e15/e15_probs", None),
        ("e15_control s43", ROOT / "runs/kaggle-typed-e15-seeds/e15_control_s43", None),
        ("e15_probs s43", ROOT / "runs/kaggle-typed-e15-seeds/e15_probs_s43", None),
        ("e15_control s44", ROOT / "runs/kaggle-typed-e15-seeds/e15_control_s44", None),
        ("e15_probs s44", ROOT / "runs/kaggle-typed-e15-seeds/e15_probs_s44", None),
        ("e15_probs s45 (exploratory)", ROOT / "runs/kaggle-typed-e15-seeds/e15_probs_s45", None),
    ]
    rows = [arm_row(n, c, r) for n, c, r in arms]

    typed_manifest = json.loads((ROOT / "data/typed-v1/manifest.json").read_text(encoding="utf-8"))
    indep_manifest = json.loads((ROOT / "data/independent-v1/manifest.json").read_text(encoding="utf-8"))
    bundle_manifest = ROOT / "kaggle-bundle/BUNDLE_MANIFEST.json"

    registry = {
        "date": TODAY,
        "code": {
            "git_head": git_head(),
            "generator_indep_v1_sha256": indep_manifest["generator_sha256"],
        },
        "data": {
            "typed-v1": {"revision": typed_manifest["revision"],
                         "train_parquet_sha256": typed_manifest["train_parquet_sha256"],
                         "counts": typed_manifest["counts"],
                         "probabilities": "scripts/prepare_typed_probs.py output; verified id/order alignment"},
            "independent-v1": {"cases_sha256": indep_manifest["files"]["cases.jsonl"]["sha256"],
                               "seed": indep_manifest["seed"], "cases": indep_manifest["cases"],
                               "frozen_at": indep_manifest["frozen_at"]},
            "bundle_manifest_sha256": sha256(bundle_manifest) if bundle_manifest.exists() else "NOT_FOUND",
        },
        "runs": {
            "kernels": {
                "mokka-typed-ord": "v1 (E11 base_ord), v2 (E14 base_ord6+small_ord8) - COMPLETE",
                "mokka-typed-small-ord": "v1 (E12) - COMPLETE",
                "mokka-typed-e15": "v1 (seed42 pair, COMPLETE), v2 (seeds 43/44 control+candidate, 45 candidate) - status at collection time",
                "hardware": "2x Tesla T4 per kernel hardware report (recorded in kernel logs); "
                            "GPU INSTANCE IDENTITY PER SESSION NOT RECORDED by Kaggle logs",
            },
            "julia_reference": {
                "weights_sha256": "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72",
                "engine": "official FastEngine, .venv-julia (torch 2.14.1+cpu, transformers 5.0.0)",
                "training_budget": "UNKNOWN - upstream pipeline private (declared, not invented)",
            },
        },
        "not_recorded_explicitly": [
            "GPU instance identity / interconnect per session (Kaggle logs carry model only)",
            "exact wheel sha256 installed inside past kernel sessions (bundle manifest covers content)",
            "RNG states of finished sessions beyond checkpointed ones",
        ],
        "open_hypotheses": [
            "base_ord dev 0.7123 (E11 session) vs e15_control dev 0.6754 (E15 session), same recipe+seed: "
            "cause UNKNOWN - candidates: session environment, GPU instance, CUDA nondeterminism; "
            "NOT attributed to any single cause without further controls",
        ],
        "arms": rows,
    }
    out = ROOT / "runs/traceability"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"traceability-{TODAY}.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    md = [f"# Registro de trazabilidad — {TODAY}", "",
          f"- git HEAD: `{registry['code']['git_head']}`",
          f"- typed-v1: revisión `{typed_manifest['revision'][:12]}…`, parquet sha256 `{typed_manifest['train_parquet_sha256'][:16]}…`",
          f"- independent-v1: cases sha256 `{indep_manifest['files']['cases.jsonl']['sha256'][:16]}…`, seed {indep_manifest['seed']}, congelado {indep_manifest['frozen_at']}",
          "", "| brazo | checkpoint sha256 | época mejor | config_hash | dev |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['arm']} | `{str(r.get('checkpoint_best_sha256',''))[:16]}…` | {r.get('best_epoch','—')} | "
                  f"`{str(r.get('config_hash',''))[:16]}` | {r.get('best_dev_accuracy','—')} |")
    md += ["", "**No registrado (explícito, sin invención):** " + "; ".join(registry["not_recorded_explicitly"]),
           "", "**Hipótesis abierta:** " + registry["open_hypotheses"][0]]
    dest = ROOT / f"docs/internal/traceability-{TODAY}.md"
    dest.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", dest, "and", out / f"traceability-{TODAY}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
