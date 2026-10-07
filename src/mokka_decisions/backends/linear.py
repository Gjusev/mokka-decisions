"""TF-IDF + logistic regression: the always-present fixed-taxonomy baseline.

A single classifier over the union of training labels across sources; at
decision time it keeps only the probabilities of the options present (the
standard restriction trick). The shared ``none`` option has no training
signal, so this baseline can never choose it — a structural limitation we
report, not hide.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..contracts import NONE_ID, DecisionCase
from .base import Backend


class LinearBackend(Backend):
    name = "tfidf_lr"

    def __init__(self, train_rows: Sequence[dict], max_features: int = 50_000, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        texts = [r["text"] for r in train_rows]
        if not texts:
            raise ValueError("LinearBackend needs training rows with labels")
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000, C=4.0, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, [r["label"] for r in train_rows])
        self.classes_ = list(self.pipeline.named_steps["lr"].classes_)
        # cross-language training pool for the multilingual variant
        counts = Counter(r["language"] for r in train_rows)
        self.model_revision = f"tfidf-lr-{'+'.join(f'{k}{v}' for k, v in sorted(counts.items()))}"

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        import numpy as np

        probs = self.pipeline.predict_proba([case.state])[0]
        by_label = {label: float(p) for label, p in zip(self.classes_, probs)}
        present = {o.id: by_label.get(o.id, 0.0) for o in case.options if o.id != NONE_ID}
        return present  # none omitted: no signal; normalise_probabilities fills the rest


class LinearRejectBackend(LinearBackend):
    """TF-IDF + LR *with* a trainable reject: `none` becomes a real class.

    The closed-world LinearBackend scores 0 % on gold=none sets because we
    excluded `none` from its label space — a modelling choice, not an inherent
    limit of linear classifiers. This variant trains `none` on the OOS training
    utterances (CLINC's own out-of-scope annotations, train split only), so it
    can reject. Both variants are reported separately.
    """

    name = "tfidf_lr_none"

    def __init__(self, train_rows: Sequence[dict], **kwargs):
        from ..contracts import NONE_ID as _NONE

        augmented = []
        for row in train_rows:
            label = row["label"]
            new = dict(row)
            new["label"] = _NONE if str(label).strip().lower() in {"oos", "out_of_scope"} else label
            augmented.append(new)
        if not any(r["label"] == _NONE for r in augmented):
            raise ValueError("LinearRejectBackend needs OOS training rows (clinc oos)")
        super().__init__(augmented, **kwargs)

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        probs = self.pipeline.predict_proba([case.state])[0]
        by_label = {label: float(p) for label, p in zip(self.classes_, probs)}
        return {o.id: by_label.get(o.id, 0.0) for o in case.options}
