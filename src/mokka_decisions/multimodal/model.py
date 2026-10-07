"""Multimodal OptionScorer: text encoder + frozen SigLIP 2 vision tower.

Design (per the pilot brief):

- the image is encoded **once per decision** with a frozen SigLIP 2 vision
  tower (google/siglip2-base-patch16-224, ~86M) — never per option;
- each option is scored from the mmBERT CLS of (state, question+option)
  **fused** with the projected global image vector:
  ``score = head([text_cls ; dropout(act(proj(img_vec)))])``
  (global vector first; patches/regions only if this loses information);
- absent image -> a learned null embedding (the fusion must handle "no
  screenshot" as a first-class input, not a crash);
- same masked softmax over valid options as the textual scorer.

What is inherited: mmBERT encoder (fine-tuned by this project already) and
the SigLIP 2 tower (frozen, Google). What is ours: projection, fusion, head,
training, calibration, policy.
"""

from __future__ import annotations

import torch
from torch import nn

from ..model import DEFAULT_ENCODER

SIGLIP2_MODEL = "google/siglip2-base-patch16-224"


class ImageTower(nn.Module):
    """Frozen SigLIP 2 vision tower returning one pooled vector per image."""

    def __init__(self, model_name: str = SIGLIP2_MODEL):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name).vision_model
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.dim = self.backbone.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, 224, 224) normalised for SigLIP 2 -> (B, dim)."""
        out = self.backbone(pixel_values=pixel_values)
        return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]


class MultimodalOptionScorer(nn.Module):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        *,
        vision_dim: int | None = None,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.tower = ImageTower()
        if vision_dim is None:
            vision_dim = self.tower.dim
        # trainable bridge: image space -> decision space (+ learned null image)
        self.img_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.Tanh(), nn.Dropout(dropout)
        )
        self.null_image = nn.Parameter(torch.zeros(vision_dim))
        with torch.no_grad():
            self.null_image.normal_(0, 0.02)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids, attention_mask, token_type_ids,
        decision_index, option_slot,
        num_decisions: int, max_options: int,
        image_vec: torch.Tensor | None,   # (num_decisions, vision_dim) or None
    ) -> torch.Tensor:
        """image_vec None or all-missing rows use the learned null embedding."""
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        if image_vec is None:
            g = self.img_proj(self.null_image.unsqueeze(0)).expand(num_decisions, -1)
        else:
            proj = self.img_proj(image_vec)                       # (D, hidden)
            g = proj.new_empty(num_decisions, proj.shape[1])
            missing = ~torch.isfinite(image_vec).all(dim=1)
            if missing.any():
                null = self.img_proj(self.null_image.unsqueeze(0)).expand(int(missing.sum()), -1)
                g[missing] = null
            g[~missing] = proj[~missing]
        # expand image context to every option sequence
        g_per_seq = g[decision_index]                             # (S, hidden)
        features = torch.cat([pooled, g_per_seq], dim=-1)
        scores = self.head(features).squeeze(-1)
        logits = torch.full(
            (num_decisions, max_options),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        logits[decision_index, option_slot] = scores
        return logits

    def parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        groups = []
        if enc:
            groups.append({"params": enc, "lr": encoder_lr})
        if rest:
            groups.append({"params": rest, "lr": head_lr})
        return groups
