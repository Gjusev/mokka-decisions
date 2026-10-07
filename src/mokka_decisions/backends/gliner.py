"""GLiNER2.5 multilingual as a zero-shot comparator via its classify API.

Score semantics (verified against the GLiNER2 README, fastino-ai/GLiNER2):
``classify_text`` with ``multi_label=True`` and ``cls_threshold=0.0`` returns
per-label *independent* confidences, not an exclusive distribution. We keep
those raw scores in ``raw_scores`` and use a defined, order-preserving
transformation for the decision: divide each raw score by the sum over the
present options (argmax unchanged, since the divisor is common). The
transformation is documented here and evaluated; we never present the raw
sigmoids themselves as calibrated decision probabilities.

Labels passed to GLiNER are the option *descriptions* (the semantic content);
``none`` uses its standard description like every other backend.
"""

from __future__ import annotations

from ..contracts import DecisionCase
from .base import Backend

GLINER_MODEL = "fastino/gliner2.5-multi-v1"


class GlinerBackend(Backend):
    name = "gliner25_multilingual_zero_shot"

    def __init__(self, model_name: str = GLINER_MODEL, revision: str | None = None):
        from gliner2 import AutoExtractor  # pip install "gliner2[local]"

        name = model_name if revision is None else f"{model_name}@{revision}"
        self.model = AutoExtractor.from_pretrained(name)
        self.model_revision = f"gliner2.5@{revision or 'default'}"
        self._qid = "decision"

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        schema = {
            self._qid: {
                "labels": [o.description for o in case.options],
                "multi_label": True,
                "cls_threshold": 0.0,
            }
        }
        result = self.model.classify_text(
            case.state, schema, include_confidence=True
        )
        payload = result[self._qid]
        # multi_label + confidence -> [{'label': desc, 'confidence': score}, ...]
        if isinstance(payload, dict):
            payload = [payload]
        desc_to_id = {o.description: o.id for o in case.options}
        scores: dict[str, float] = {}
        for item in payload:
            oid = desc_to_id.get(item.get("label"))
            if oid is not None:
                scores[oid] = float(item.get("confidence", 0.0))
        missing = [o.id for o in case.options if o.id not in scores]
        for oid in missing:  # labels below the floor still appear with 0.0
            scores[oid] = 0.0
        return scores
