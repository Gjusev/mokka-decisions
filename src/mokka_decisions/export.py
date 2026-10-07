"""Export a self-contained release directory.

Layout (offline-capable: architecture instantiated from the LOCAL config, our
weights loaded directly — no Hugging Face cache or network at load time)::

    release/mokka-decisions-<tag>/
      model.safetensors      our trained weights (encoder fine-tuned + head)
      encoder-config/        architecture config (local instantiation)
      tokenizer/             tokenizer files
      model-meta.json        encoder id, config hash, epochs, dev metrics, provenance
      calibration.json       temperature + revision
      policy.json            threshold + stats
      manifest.json          sha256 of every artifact (relative-path keys) + meta
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .model import OptionScorer, load_checkpoint


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
    # architecture config saved locally so loading never touches the network
    model.encoder.config.save_pretrained(out / "encoder-config")
    from safetensors.torch import save_file

    weights = out / "model.safetensors"
    save_file(dict(model.state_dict()), str(weights), metadata={"format": "pt"})

    full_meta = {
        "encoder": encoder_name,
        "export": "mokka-decisions release",
        "contract_version": "v1 (DecisionCase/DecisionResponse)",
        **meta,
    }
    (out / "model-meta.json").write_text(
        json.dumps(full_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "artifacts": {
            str(p.relative_to(out)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
        "meta": full_meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def load_exported_model(model_dir: str | Path):
    """Load (ScorerInference, policy) fully offline from an export dir.

    The encoder architecture is instantiated from the LOCAL saved config and
    our trained weights are loaded directly — ``from_pretrained`` on a hub id
    is never called, so a fresh machine with an empty HF cache and no network
    still loads.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    import torch

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model-meta.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    policy = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    config = AutoConfig.from_pretrained(model_dir / "encoder-config", local_files_only=True)
    encoder = AutoModel.from_config(config)
    from .model import OptionScorer

    model = OptionScorer.__new__(OptionScorer)
    torch.nn.Module.__init__(model)
    model.encoder_name = meta.get("encoder", "")
    model.encoder = encoder
    hidden = encoder.config.hidden_size
    import torch.nn as nn

    model.head = nn.Sequential(
        nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(0.1), nn.Linear(hidden, 1)
    )
    model.freeze_encoder = False
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


from .model import ScorerInference  # noqa: E402  (kept last: avoids import cycle at module top)
