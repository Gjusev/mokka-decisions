"""Laya multilingual as a zero-shot comparator on the same decision contract.

One choice question per case; criteria = our option descriptions (the same text
our model sees — same information, same language, only the model differs).
Laya returns per-option probabilities plus its calibrated answer_confidence;
we use the raw probabilities and calibrate the threshold ourselves on the same
cal_policy split as everyone else.
"""

from __future__ import annotations

from ..contracts import DecisionCase
from .base import Backend


class LayaBackend(Backend):
    name = "laya_multilingual_zero_shot"

    def __init__(self, *, revision: str | None = None):
        from laya import Router  # lazy

        kwargs = {"default": "multilingual"}
        if revision:
            kwargs["revision"] = revision
        self.router = Router(**kwargs)
        self.model_revision = f"laya-multilingual@{revision or 'default'}"
        self._qid = "decision"

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        questions = {
            self._qid: {
                "type": "choice",
                "instructions": f"{case.question} Answer for `request`.",
                "criteria": {o.id: o.description for o in case.options},
            }
        }
        payload = self.router.predict({"request": case.state}, questions)
        answer = payload["answers"][self._qid]
        probs = dict(answer.get("probabilities") or {})
        if not probs:  # fall back to a point mass on the choice
            return {answer["choice"]: 1.0}
        # keep only present options; renormalised by the base class
        return {oid: p for oid, p in probs.items() if oid in case.option_ids}
