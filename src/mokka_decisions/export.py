"""Export a self-contained release directory.

Layout (everything needed for offline inference after first download)::

    release/mokka-decisions-<tag>/
      model.safetensors      our trained weights (encoder fine-tuned + head)
      model-meta.json        encoder id, config hash, epochs, dev metrics
      tokenizer/             tokenizer files
      calibration.json       temperature + revision
      policy.json            threshold + stats
      manifest.json          sha256 of every artifact + code revision
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .model import OptionScorer, ScorerInference, load_checkpoint


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_release(
    *,
    checkpoint: str | Path,
    encoder_name: str,
    out_dir: str | Path,
    calibration: dict,
    policy: dict,
    meta: dict,
) -> Path:
    from transformers import AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    tokenizer.save_pretrained(out / "tokenizer")

    model = OptionScorer(encoder_name)
    load_checkpoint(model, checkpoint)
    from safetensors.torch import save_file

    weights = out / "model.safetensors"
    save_file(dict(model.state_dict()), str(weights), metadata={"format": "pt"})

    (out / "model-meta.json").write_text(
        json.dumps({"encoder": encoder_name, **meta}, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "artifacts": {
            p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
        "meta": meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def load_exported_model(model_dir: str | Path):
    """Load (ScorerInference, policy) from an export directory; offline after
    the encoder tokenizer/config are cached or shipped in tokenizer/."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model-meta.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    policy = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer", local_files_only=True
    )
    model = OptionScorer(meta["encoder"])
    load_checkpoint(model, model_dir / "model.safetensors")
    scorer = ScorerInference(
        model,
        tokenizer,
        device="cpu",
        temperature=calibration.get("temperature", 1.0),
        batch_decisions=8,
    )
    scorer.model_revision = f"option-scorer@{meta.get('config_hash', '')[:12]}"
    return scorer, policy
