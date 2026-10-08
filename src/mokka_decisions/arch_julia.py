"""Faithful Julia-1 architecture port for the equal-size Small control.

Ported from Julia-1's published inference code (Apache-2.0;
`runs/julia-research/upstream-source/julia/{model,data,inference}.py`,
checkpoint julia_config.json: head_layers=2, n_act=2, dropout=0.1).

Faithful: encoder via AutoModel (mmBERT); 2 extra pre-norm TransformerEncoder
layers over the full sequence (src_key_padding_mask); type_emb(3) added to
every position; scorer LayerNorm->Linear->GELU->Linear on the PRE-EXISTING
MASK-marker positions; serialization "{type} question: ..." with one MASK
marker per option (48-token option contract, head_length 256).

Declared divergences (quality arms only):
- act_head (RL auxiliary) and the per-qtype temperature buffer are omitted —
  neither participates in argmax quality.
- Training uses our trainer recipe (AMP on encoder); upstream training is
  private and NOT reproduced (their model card says so).

Architecture vs supervision stay separate variables: this class trains under
either the artificial ordinal targets or the dataset-native distributions.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import DecisionCase

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class JuliaArchScorer(nn.Module):
    """JuliaDecisionModel minus RL auxiliaries (see module docstring)."""

    def __init__(self, encoder_name: str, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name, attn_implementation="sdpa")
        width = self.encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(
            width, max(1, width // 64), 4 * width, dropout,
            batch_first=True, norm_first=True,
        )
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, width)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())
        positions = marker_pos[:, :, None].expand(-1, -1, hidden.shape[-1])
        markers = hidden.gather(1, positions)
        scores = self.scorer(markers).squeeze(-1).float()
        return scores.masked_fill(~marker_mask, -1e4)

    def parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        groups = []
        if enc:
            groups.append({"params": enc, "lr": encoder_lr})
        if rest:
            groups.append({"params": rest, "lr": head_lr})
        return groups


def julia_sequence(tokenizer, state: str, question: str, options: list[str], qtype: str,
                   max_length: int = 1024, head_length: int = 256) -> dict:
    """Port of julia/data.py sequence() (non-strict training path)."""
    clean = lambda text: text.replace(tokenizer.mask_token, " ")
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    head = encode(f"{qtype} question: {clean(question)}")
    option_ids = [encode(" " + clean(x)) for x in options]
    options = [[tokenizer.mask_token_id] + x[:48] for x in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [x[:per_option] for x in options]
        budget = head_length - sum(map(len, options))
    ids = [tokenizer.cls_token_id] + head[:max(8, budget)] + [tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    state_ids = encode(clean(state))
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("Question/options exceed sequence budget; shorten descriptions")
    return dict(ids=ids + state_ids[:room] + [tokenizer.sep_token_id],
                markers=markers, qtype=QTYPES[qtype])


def qtype_of(case: DecisionCase) -> str:
    """Typed question type stored by prepare_typed_train/prepare_typed_probs rows."""
    return getattr(case, "question_type", None) or "choice"


def collate_julia(tokenizer, cases: Sequence[DecisionCase], qtypes: dict[str, str],
                  max_length: int, head_length: int, device):
    encoded, targets = [], []
    for case in cases:
        qt = qtypes.get(case.id, "choice")
        item = julia_sequence(tokenizer, case.state, case.question,
                              [o.description for o in case.options], qt, max_length, head_length)
        encoded.append(item)
        targets.append(case.option_ids.index(case.target_option) if case.target_option else -100)
    length = min(max_length, ((max(len(x["ids"]) for x in encoded) + 7) // 8 * 8))
    count = max(len(x["markers"]) for x in encoded)
    pad = tokenizer.pad_token_id or 0
    ids = torch.full((len(cases), length), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((len(cases), count), dtype=torch.long)
    mask = torch.zeros_like(positions, dtype=torch.bool)
    for i, item in enumerate(encoded):
        n, k = len(item["ids"]), len(item["markers"])
        ids[i, :n] = torch.tensor(item["ids"])
        attention[i, :n] = 1
        positions[i, :k] = torch.tensor(item["markers"])
        mask[i, :k] = True
    return {
        "input_ids": ids.to(device), "attention_mask": attention.to(device),
        "marker_pos": positions.to(device), "marker_mask": mask.to(device),
        "qtype": torch.tensor([x["qtype"] for x in encoded], device=device),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
    }


def row_loss(logits: torch.Tensor, targets: torch.Tensor, cases, soft_targets: dict | None,
             ordinal_case_ids: set[str] | None, tau: float = 1.0) -> torch.Tensor:
    """Soft-CE identical in spirit to train.run_training's branch (fp32 math)."""
    from .train import ordinal_soft_target

    logp = F.log_softmax(logits.float(), dim=-1)
    losses = []
    for r, case in enumerate(cases):
        k = len(case.options)
        gold = int(targets[r].item())
        vec = None
        if soft_targets is not None and case.id in soft_targets:
            vec = soft_targets[case.id]
        elif ordinal_case_ids and case.id in ordinal_case_ids and gold >= 0:
            vec = ordinal_soft_target(k, gold, tau)
        if vec is not None:
            soft = torch.as_tensor(vec, device=logits.device, dtype=logp.dtype)
            losses.append(-(soft * logp[r, :k]).sum())
        else:
            losses.append(-logp[r, gold])
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_julia_arch(model, tokenizer, cases, qtypes, *, device,
                        max_length=1024, head_length=256, batch_decisions=16) -> dict:
    model.eval()
    correct = n = nll_sum = 0
    for s in range(0, len(cases), batch_decisions):
        chunk = cases[s : s + batch_decisions]
        batch = collate_julia(tokenizer, chunk, qtypes, max_length, head_length, device)
        logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}).float()
        valid = batch["targets"] != -100
        preds = logits.argmax(-1)
        correct += int(((preds == batch["targets"]) & valid).sum().item())
        nll_sum += float(F.cross_entropy(logits, batch["targets"], reduction="sum", ignore_index=-100).item())
        n += int(valid.sum().item())
    model.train()
    return {"n": n, "accuracy": correct / n if n else None, "mean_nll": nll_sum / n if n else None}


def train_julia_arch(model, tokenizer, train_cases, dev_cases, qtypes, *, output_dir,
                     epochs=3, microbatch=4, grad_accumulation=8, encoder_lr=2e-5, head_lr=1e-4,
                     weight_decay=0.01, warmup_ratio=0.05, max_grad_norm=1.0, seed=42,
                     device="cuda", soft_targets=None, ordinal_case_ids=None,
                     max_length=1024, head_length=256) -> dict:
    """Same recipe/checkpoint pattern as the A0/A1 trainers (comparable runs)."""
    import json
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    model.train()
    usable = [c for c in train_cases if c.target_option]
    steps_per_epoch = math.ceil(len(usable) / (microbatch * grad_accumulation))
    total_steps = max(steps_per_epoch * epochs, 1)
    optimizer = torch.optim.AdamW(model.parameter_groups(encoder_lr, head_lr), weight_decay=weight_decay)
    warmup = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.05, 0.95 * (1 - progress) + 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    rng = random.Random(seed ^ 0xA17A)
    best = -1.0
    history = []
    global_step = 0
    for epoch in range(epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for mb_idx, start in enumerate(range(0, len(order), microbatch)):
            chunk = [usable[i] for i in order[start : start + microbatch]]
            batch = collate_julia(tokenizer, chunk, qtypes, max_length, head_length, device)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
                loss = row_loss(logits, batch["targets"], chunk, soft_targets, ordinal_case_ids) / grad_accumulation
            scaler.scale(loss).backward()
            if (mb_idx + 1) % grad_accumulation == 0 or start + microbatch >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]], max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 25 == 0:
                    print(f"[julia-arch] epoch={epoch} step={global_step} loss={float(loss.item())*grad_accumulation:.4f}", flush=True)
        dev_metrics = evaluate_julia_arch(model, tokenizer, dev_cases, qtypes, device=device,
                                          max_length=max_length, head_length=head_length)
        history.append({"epoch": epoch, "step": global_step, "dev": dev_metrics})
        print(f"[julia-arch epoch {epoch}] dev={dev_metrics}", flush=True)
        if (dev_metrics.get("accuracy") or 0.0) >= best:
            best = dev_metrics.get("accuracy") or 0.0
            from safetensors.torch import save_file

            save_file(dict(model.state_dict()), str(out / "checkpoint_best.safetensors"),
                      metadata={"format": "pt"})
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epochs, "global_step": global_step,
                "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
                "metrics": {"best_dev_accuracy": best, "history": history}},
               out / "checkpoint_latest.pt")
    report = {"epochs": epochs, "final_step": global_step, "best_dev_accuracy": best, "history": history}
    (out / "run_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
