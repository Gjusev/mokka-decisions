"""Supervised trainer for the OptionScorer decision head.

Single implementation used both locally (CPU smoke tests) and on Kaggle: the
notebook/script installs this package and calls :func:`run_training` with a
config dict — there is no second, divergent training code path.

Resume contract: a checkpoint contains weights, optimizer, scheduler, scaler,
epoch, global_step, sampler progress, RNG states (CPU/CUDA) and the hashes of
config and data manifest — everything needed to continue a segment (state_dict
of weights alone is explicitly not enough; see docs/evaluation-protocol.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .contracts import DecisionCase
from .model import OptionScorer, encode_cases, case_targets


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CaseDataset(Dataset):
    def __init__(self, cases: Sequence[DecisionCase]):
        self.cases = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> DecisionCase:
        return self.cases[idx]


def collate_chunk(cases: Sequence[DecisionCase], tokenizer, max_length: int, device):
    batch = encode_cases(tokenizer, cases, max_length=max_length, device=device)
    batch["targets"] = case_targets(cases, device=device)
    return batch


@dataclass
class TrainConfig:
    encoder_name: str = "jhu-clsp/mmBERT-base"
    max_length: int = 256
    microbatch: int = 4  # decisions per step (each expands to ~K sequences)
    grad_accumulation: int = 8  # effective batch = microbatch * accumulation
    epochs: int = 3
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    freeze_encoder: bool = False
    amp: bool = True
    eval_every_steps: int = 0  # 0 = only at epoch end
    save_every_minutes: float = 30.0
    log_every_steps: int = 20
    output_dir: str = "runs/train"
    max_options_per_step: int = 12  # skip train cases with more options (none by construction)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


def config_hash(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@torch.no_grad()
def evaluate_model(
    model: OptionScorer,
    tokenizer,
    cases: Sequence[DecisionCase],
    *,
    device: str,
    max_length: int,
    batch_decisions: int = 16,
) -> dict:
    """Forced accuracy + mean NLL on a dev/eval set (no policy, no threshold)."""
    model.eval()
    correct = 0
    nll_sum = 0.0
    n = 0
    for start in range(0, len(cases), batch_decisions):
        chunk = cases[start : start + batch_decisions]
        batch = encode_cases(tokenizer, chunk, max_length=max_length, device=device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            decision_index=batch["decision_index"],
            num_decisions=batch["num_decisions"],
            max_options=batch["max_options"],
            option_slot=batch["option_slot"],
        )
        targets = case_targets(chunk, device=device)
        preds = logits.argmax(dim=-1)
        correct += int((preds == targets).sum().item())
        nll = F.cross_entropy(logits, targets, reduction="sum")
        nll_sum += float(nll.item())
        n += len(chunk)
    model.train()
    return {"n": n, "accuracy": correct / n if n else None, "mean_nll": nll_sum / n if n else None}


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_full_checkpoint(
    path: Path,
    *,
    model: OptionScorer,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    sampler_seed_state: int,
    metrics: dict,
    cfg: TrainConfig,
    data_manifest_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "sampler_seed_state": sampler_seed_state,
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metrics": metrics,
        "config": cfg.to_dict(),
        "config_hash": config_hash(cfg.to_dict()),
        "data_manifest_hash": data_manifest_hash,
        "saved_at": time.time(),
    }
    _atomic_torch_save(payload, path)


def run_training(
    cfg: TrainConfig,
    *,
    train_cases: Sequence[DecisionCase],
    dev_cases: Sequence[DecisionCase],
    tokenizer,
    resume_from: str | Path | None = None,
    data_manifest_hash: str = "",
    progress_cb=None,
) -> dict:
    """Train and return a report; checkpoints land in cfg.output_dir."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    model = OptionScorer(cfg.encoder_name, freeze_encoder=cfg.freeze_encoder).to(device)

    steps_per_epoch = math.ceil(len(train_cases) / (cfg.microbatch * cfg.grad_accumulation))
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(cfg.encoder_lr, cfg.head_lr), weight_decay=cfg.weight_decay
    )
    warmup = max(int(total_steps * cfg.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - progress) + 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    history: list[dict] = []

    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        random.setstate(payload["rng_python"])
        torch.set_rng_state(payload["rng_torch"])
        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        start_epoch = payload["epoch"]
        global_step = payload["global_step"]
        best_metric = payload.get("metrics", {}).get("best_dev_accuracy", -1.0)
        history = payload.get("metrics", {}).get("history", [])
        _ = payload.get("data_manifest_hash")
        print(f"[resume] epoch={start_epoch} global_step={global_step}")

    log_path = out_dir / "train_log.jsonl"
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    ckpt_best = out_dir / "checkpoint_best.pt"

    rng = random.Random(cfg.seed ^ 0xC0FFEE)  # shuffling rng, checkpointed via seed+epoch
    last_save = time.time()
    stop = False

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        microbatches = [
            order[i : i + cfg.microbatch]
            for i in range(0, len(order), cfg.microbatch)
        ]
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        t_epoch = time.time()

        for mb_idx, indices in enumerate(microbatches):
            cases = [train_cases[i] for i in indices]
            if any(len(c.options) > cfg.max_options_per_step for c in cases):
                continue
            batch = collate_chunk(cases, tokenizer, cfg.max_length, device)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    decision_index=batch["decision_index"],
                    num_decisions=batch["num_decisions"],
                    max_options=batch["max_options"],
                    option_slot=batch["option_slot"],
                )
                loss = F.cross_entropy(logits, batch["targets"]) / cfg.grad_accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * cfg.grad_accumulation

            if (mb_idx + 1) % cfg.grad_accumulation == 0 or mb_idx + 1 == len(microbatches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    entry = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": running_loss / (cfg.log_every_steps * cfg.grad_accumulation),
                        "lr": scheduler.get_last_lr()[0],
                        "vram_gb": (
                            torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available()
                            else None
                        ),
                    }
                    running_loss = 0.0
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry) + "\n")
                    if progress_cb:
                        progress_cb(entry)

                if cfg.save_every_minutes and (time.time() - last_save) / 60 >= cfg.save_every_minutes:
                    save_full_checkpoint(
                        ckpt_latest,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        sampler_seed_state=rng.randrange(2**31),
                        metrics={"best_dev_accuracy": best_metric, "history": history},
                        cfg=cfg,
                        data_manifest_hash=data_manifest_hash,
                    )
                    last_save = time.time()

                if cfg.eval_every_steps and global_step % cfg.eval_every_steps == 0:
                    dev_metrics = evaluate_model(
                        model, tokenizer, dev_cases, device=device, max_length=cfg.max_length
                    )
                    history.append({"epoch": epoch, "step": global_step, "dev": dev_metrics})

        dev_metrics = evaluate_model(
            model, tokenizer, dev_cases, device=device, max_length=cfg.max_length
        )
        dev_metrics["epoch_time_s"] = time.time() - t_epoch
        history.append({"epoch": epoch, "step": global_step, "dev": dev_metrics})
        print(f"[epoch {epoch}] dev={dev_metrics}")

        dev_acc = dev_metrics.get("accuracy") or 0.0
        if dev_acc >= best_metric:
            best_metric = dev_acc
            from .model import save_checkpoint

            save_checkpoint(
                model,
                ckpt_best.with_suffix(".safetensors"),
                extra={"epoch": epoch, "global_step": global_step, "dev": dev_metrics,
                       "encoder": cfg.encoder_name, "config_hash": config_hash(cfg.to_dict())},
            )
        save_full_checkpoint(
            ckpt_latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            sampler_seed_state=rng.randrange(2**31),
            metrics={"best_dev_accuracy": best_metric, "history": history},
            cfg=cfg,
            data_manifest_hash=data_manifest_hash,
        )
        last_save = time.time()

    report = {
        "config": cfg.to_dict(),
        "config_hash": config_hash(cfg.to_dict()),
        "data_manifest_hash": data_manifest_hash,
        "final_global_step": global_step,
        "epochs_completed": cfg.epochs,
        "best_dev_accuracy": best_metric,
        "history": history,
        "device": device,
        "stop": stop,
    }
    (out_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report
