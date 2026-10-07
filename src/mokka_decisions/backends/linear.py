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
