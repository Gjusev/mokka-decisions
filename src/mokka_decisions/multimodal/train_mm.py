"""Trainer for the multimodal scorer, with a frozen-tower image cache.

The SigLIP 2 tower is frozen, so every screenshot is encoded **once** per
build into a cache keyed by image sha256 and invalidated when the tower or
preprocessing changes (cache_version). Training then never touches images
again — only cached vectors — which keeps the loop as fast as the textual one.

Resume/checkpointing mirror the textual trainer (weights + optimizer +
scheduler + scaler + RNG + progress), in a separate output tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

from ..contracts import DecisionCase
from ..model import encode_cases, case_targets
from .model import MultimodalOptionScorer


def cache_version(tower_model: str) -> str:
    return hashlib.sha256(f"imgcache-v1|{tower_model}".encode()).hexdigest()[:10]


@torch.no_grad()
def build_image_cache(
    cases: Sequence[DecisionCase],
    media_rows: dict[str, dict],  # case_id -> {"screenshot": {"path", "sha256"}} or None
    data_root: Path,
    tower,
    processor,
    *,
    device: str = "cpu",
    batch: int = 32,
) -> dict[str, torch.Tensor]:
    """sha256 -> pooled vision vector for every distinct referenced image."""
    shas, paths = [], []
    seen = set()
    for case in cases:
        media = media_rows.get(case.id) or {}
        shot = media.get("screenshot") or {}
        sha = shot.get("sha256")
        path = shot.get("path")
        if sha and path and sha not in seen:
            seen.add(sha)
            shas.append(sha)
            paths.append(data_root / path)
    out: dict[str, torch.Tensor] = {}
    tower = tower.to(device).eval()
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        images = []
        from PIL import Image

        for p in chunk:
            images.append(Image.open(p).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        vecs = tower(inputs["pixel_values"]).float().cpu()
        for sha, vec in zip(shas[start : start + batch], vecs):
            out[sha] = vec
    return out


def load_media(case_files: Sequence[Path]) -> tuple[dict[str, dict], dict[str, str]]:
    """case_id -> media dict, and image sha -> relative path (for rebuilds)."""
    media = {}
    images = {}
    for path in case_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            m = row.get("media")
            if m:
                media[row["id"]] = m
                shot = m.get("screenshot") or {}
                if shot.get("sha256"):
                    images[shot["sha256"]] = shot.get("path")
    return media, images


def load_case_types(case_files: Sequence[Path]) -> dict[str, str]:
    """case_id -> case_type (text_suffices / vision_needed / ...)."""
    types = {}
    for path in case_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                types[row["id"]] = row.get("case_type", "")
    return types


def encode_mm_batch(
    tokenizer,
    cases: Sequence[DecisionCase],
    media: dict[str, dict],
    vec_cache: dict[str, torch.Tensor],
    *,
    max_length: int,
    device,
    vision_dim: int,
) -> dict:
    batch = encode_cases(tokenizer, cases, max_length=max_length, device=device)
    import math as _math

    vecs = torch.full((len(cases), vision_dim), float("nan"))
    for i, case in enumerate(cases):
        sha = ((media.get(case.id) or {}).get("screenshot") or {}).get("sha256")
        if sha and sha in vec_cache:
            vecs[i] = vec_cache[sha]
    batch["image_vec"] = vecs.to(device)
    batch["targets"] = case_targets(cases, device=device)
    return batch


@torch.no_grad()
def evaluate_mm(model, tokenizer, cases, media, vec_cache, *, device, max_length, batch_decisions=16) -> dict:
    model.eval()
    vision_dim = model.tower.dim
    correct = n = 0
    nll_sum = 0.0
    for start in range(0, len(cases), batch_decisions):
        chunk = [c for c in cases[start : start + batch_decisions] if c.target_option]
        if not chunk:
            continue
        batch = encode_mm_batch(tokenizer, chunk, media, vec_cache,
                                max_length=max_length, device=device, vision_dim=vision_dim)
        logits = model(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
            batch["decision_index"], batch["option_slot"],
            batch["num_decisions"], batch["max_options"], batch["image_vec"],
        )
        targets = batch["targets"]
        correct += int((logits.argmax(-1) == targets).sum().item())
        nll_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        n += len(chunk)
    model.train()
    return {"n": n, "accuracy": correct / n if n else None, "mean_nll": nll_sum / n if n else None}


def train_multimodal(
    *,
    model: MultimodalOptionScorer,
    tokenizer,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    media: dict[str, dict],
    vec_cache: dict[str, torch.Tensor],
    output_dir: str | Path,
    epochs: int = 3,
    microbatch: int = 4,
    grad_accumulation: int = 8,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    max_length: int = 256,
    seed: int = 42,
    amp: bool = True,
    eval_case_types: tuple[str, ...] = ("vision_needed", "text_suffices", "blurry", "no_valid_option"),
    resume_from: str | Path | None = None,
    case_types: dict[str, str] | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vision_dim = model.tower.dim
    steps_per_epoch = math.ceil(len(train_cases) / (microbatch * grad_accumulation))
    total_steps = max(steps_per_epoch * epochs, 1)
    optimizer = torch.optim.AdamW(model.parameter_groups(encoder_lr, head_lr), weight_decay=weight_decay)
    warmup = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - progress) + 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best = -1.0
    history: list[dict] = []
    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        random.setstate(payload["rng_python"])
        torch.set_rng_state(payload["rng_torch"])
        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        start_epoch = payload["epoch"]
        global_step = payload["global_step"]
        best = payload.get("metrics", {}).get("best_dev_accuracy", -1.0)
        history = payload.get("metrics", {}).get("history", [])
        print(f"[mm-resume] epoch={start_epoch} step={global_step}")

    rng = random.Random(seed ^ 0xBEEF)
    model = model.to(device)
    model.train()

    def save_checkpoint(path: Path, epoch: int) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "global_step": global_step,
                "rng_python": random.getstate(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "metrics": {"best_dev_accuracy": best, "history": history},
            },
            tmp,
        )
        tmp.replace(path)

    types = case_types or {}
    dev_typed = [
        c for c in dev_cases
        if c.target_option and types.get(c.id, "") in eval_case_types
    ]
    for epoch in range(start_epoch, epochs):
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for mb_idx, start in enumerate(range(0, len(order), microbatch)):
            chunk = [train_cases[i] for i in order[start : start + microbatch]]
            chunk = [c for c in chunk if c.target_option]  # ambiguous rows excluded from CE
            if not chunk:
                continue
            batch = encode_mm_batch(tokenizer, chunk, media, vec_cache,
                                    max_length=max_length, device=device, vision_dim=vision_dim)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(
                    batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                    batch["decision_index"], batch["option_slot"],
                    batch["num_decisions"], batch["max_options"], batch["image_vec"],
                )
                loss = F.cross_entropy(logits, batch["targets"]) / grad_accumulation
            scaler.scale(loss).backward()
            if (mb_idx + 1) % grad_accumulation == 0 or start + microbatch >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    print(
                        f"[mm] epoch={epoch} step={global_step} loss={float(loss.item())*grad_accumulation:.4f} "
                        f"vram={torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0:.2f}GB",
                        flush=True,
                    )
        dev_metrics = evaluate_mm(model, tokenizer, dev_typed, media, vec_cache,
                                 device=device, max_length=max_length)
        dev_metrics["epoch_time_s"] = time.time() - t0
        history.append({"epoch": epoch, "step": global_step, "dev": dev_metrics})
        print(f"[mm epoch {epoch}] dev={dev_metrics}", flush=True)
        if (dev_metrics.get("accuracy") or 0.0) >= best:
            best = dev_metrics.get("accuracy") or 0.0
            from safetensors.torch import save_file

            save_file(dict(model.state_dict()), str(out / "model_best.safetensors"), metadata={"format": "pt"})
        save_checkpoint(out / "checkpoint_latest.pt", epoch + 1)

    report = {"epochs": epochs, "final_step": global_step, "best_dev_accuracy": best, "history": history}
    (out / "run_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
