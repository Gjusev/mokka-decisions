"""Decision contract: the one schema every backend, file and CLI agrees on.

stdlib-only on purpose: the contract must load without torch/sklearn installed.

Example JSON (one training/inference case)::

    {
      "id": "bank-tr-00001-k6",
      "group_id": "bank-tr-00001",
      "source": "banking77",
      "language": "en",
      "domain": "banking",
      "state": "I was charged twice for the same order.",
      "question": "Which banking support intent does this customer request express?",
      "options": [
        {"id": "card_payment_wrong_exchange_rate", "description": "..."},
        {"id": "none", "description": "None of the listed options applies to this request."}
      ],
      "target_kind": "single",
      "target_option": "card_payment_wrong_exchange_rate",
      "split": "train"
    }

Inference answers distinguish the *candidate* (argmax) from the accepted
*choice* (null when the policy abstains).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping, Sequence

# Splits recognised by the pipeline. ``test`` is opened once, at the end.
SPLITS = ("train", "dev", "cal_temperature", "cal_policy", "test")

# Special option id meaning "no listed option applies". Always present in the
# option catalog; whether it wins is an explicit decision, distinct from
# abstaining for low confidence.
NONE_ID = "none"

NONE_DESCRIPTION = "None of the listed options applies to this request."

VALID_TARGET_KINDS = ("single", "none", "ambiguous")


class ContractError(ValueError):
    """Raised when a case or response violates the decision contract."""


def _require_str(value: Any, name: str, *, max_len: int = 200_000) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ContractError(f"{name} must not be blank")
    if len(value) > max_len:
        raise ContractError(f"{name} exceeds {max_len} characters ({len(value)})")
    return value


@dataclass(frozen=True)
class Option:
    id: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "description": self.description}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Option":
        return cls(
            id=_require_str(data.get("id"), "option.id", max_len=200),
            description=_require_str(data.get("description"), "option.description", max_len=2_000),
        )


@dataclass(frozen=True)
class DecisionCase:
    """One decision problem (training example or inference request)."""

    id: str
    group_id: str
    source: str
    language: str
    domain: str
    state: str
    question: str
    options: tuple[Option, ...]
    target_kind: str | None = None
    target_option: str | None = None
    label_origin: str = "public-dataset"
    split: str | None = None

    def __post_init__(self) -> None:
        _require_str(self.id, "id", max_len=300)
        _require_str(self.group_id, "group_id", max_len=300)
        _require_str(self.source, "source", max_len=100)
        _require_str(self.language, "language", max_len=8)
        _require_str(self.domain, "domain", max_len=100)
        _require_str(self.state, "state")
        _require_str(self.question, "question", max_len=2_000)
        if not 2 <= len(self.options) <= 128:
            raise ContractError(f"{self.id}: need 2..128 options, got {len(self.options)}")
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ContractError(f"{self.id}: duplicate option ids {sorted(ids)}")
        if self.target_kind is not None:
            if self.target_kind not in VALID_TARGET_KINDS:
                raise ContractError(f"{self.id}: bad target_kind {self.target_kind!r}")
            if self.target_kind != "ambiguous":
                if self.target_option not in ids:
                    raise ContractError(
                        f"{self.id}: target {self.target_option!r} not among option ids"
                    )
        if self.split is not None and self.split not in SPLITS:
            raise ContractError(f"{self.id}: unknown split {self.split!r}")

    @property
    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["options"] = [o.to_dict() for o in self.options]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionCase":
        return cls(
            id=data["id"],
            group_id=data["group_id"],
            source=data.get("source", "unknown"),
            language=data.get("language", "en"),
            domain=data.get("domain", "general"),
            state=data["state"],
            question=data["question"],
            options=tuple(Option.from_dict(o) for o in data["options"]),
            target_kind=data.get("target_kind"),
            target_option=data.get("target_option"),
            label_origin=data.get("label_origin", "public-dataset"),
            split=data.get("split"),
        )

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class DecisionResponse:
    """Inference answer: candidate vs accepted choice, never conflated."""

    candidate: str
    probabilities: dict[str, float]
    choice: str | None
    confidence: float
    abstain: bool
    reason_code: str
    model_revision: str = ""
    calibration_revision: str = ""

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ContractError("probabilities must not be empty")
        for oid, p in self.probabilities.items():
            if not math.isfinite(p) or p < 0.0:
                raise ContractError(f"probability for {oid!r} must be finite and >= 0, got {p}")
        total = math.fsum(self.probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ContractError(f"probabilities sum to {total!r}, expected 1.0 within 1e-6")
        if self.candidate not in self.probabilities:
            raise ContractError(f"candidate {self.candidate!r} missing from probabilities")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ContractError(f"confidence {self.confidence!r} outside [0, 1]")
        if self.abstain and self.choice is not None:
            raise ContractError("an abstaining response must have choice=None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# reason codes produced by the policy; anything else is a backend error
REASON_CODES = (
    "accepted",
    "below_validated_threshold",
    "invalid_input",
    "unsupported_length",
    "language_not_validated",
    "oos_detected",
)


def validate_request_payload(payload: Mapping[str, Any]) -> DecisionCase:
    """Build an inference :class:`DecisionCase` from a raw request payload."""
    state = _require_str(payload.get("state"), "state")
    question = payload.get("question") or "Which option applies to this request?"
    options = [Option.from_dict(o) for o in payload.get("options", [])]
    if not options:
        raise ContractError("options must not be empty")
    case = DecisionCase(
        id=str(payload.get("id") or "request"),
        group_id=str(payload.get("group_id") or payload.get("id") or "request"),
        source=str(payload.get("source") or "api"),
        language=str(payload.get("language") or "en"),
        domain=str(payload.get("domain") or "general"),
        state=state,
        question=_require_str(question, "question", max_len=2_000),
        options=tuple(options),
    )
    if len(state) > 20_000:
        raise ContractError("state exceeds supported length (20,000 characters)")
    return case


def load_jsonl(path: str | Iterable[str]) -> list[dict[str, Any]]:
    """Load JSONL from a path or an iterable of already-read lines."""
    if isinstance(path, str):
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in (raw.strip() for raw in fh) if ln]
    else:
        lines = [ln for ln in (raw.strip() for raw in path) if ln]
    return [json.loads(ln) for ln in lines]


def dump_jsonl(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
