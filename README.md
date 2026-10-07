# mokka-decisions

A small open **decision model**: given a state, a question and a list of
described options, it returns calibrated probabilities, a choice and — when
confidence is insufficient — an explicit abstention.

```
state:    "Me cobraron dos veces el mismo pedido."
question: "Which banking support intent does this customer request express?"
options:  [ transaction_charged_twice — "The same transaction was charged twice"
            top_up_reverted — "A top-up was reversed or reverted"
            ... 4 more ... ]
→  probabilities over the options (temperature-calibrated)
   choice (or null + reason_code when the policy abstains)
```

It is **not** a chat model and does not generate text for the decision: one
encoder pass per option, a shared scorer head, softmax over the valid options.

## What is ours, what is inherited

| Component | Origin |
|---|---|
| encoder (mmBERT-base, 307M) | pretrained by JHU CLSP (MIT); **fine-tuned by this project** |
| option-scoring head, loss, masking | **this project** |
| decision data contract, candidate-set construction, splits | **this project** |
| temperature calibration + abstention policy | **this project** |
| training/eval harness, Kaggle recipe | **this project** |
| baselines (TF-IDF+LR, Laya, GLiNER2.5) | third-party models, our adapters |

See `docs/model-card.md` for the full statement, metrics and limits.

## Install

```bash
pip install -e .[data,train,baselines]
```

## Build the dataset (deterministic)

```bash
python scripts/prepare_data.py --config configs/pilot.yaml --out data/processed/v1
```

Downloads BANKING77, MASSIVE (en/de/es) and CLINC150/OOS from their official
sources, splits by semantic group before any expansion, and generates
candidate-set decision instances. One command reproduces the same instances
and hashes.

## Train (Kaggle GPU, free tier)

```bash
python scripts/prepare_kaggle.py --bundle kaggle-bundle     # wheel + configs + data
python -m kaggle datasets version -p kaggle-bundle --dir-mode zip -m "v2"
python -m kaggle kernels push -p kaggle                     # starts the job
```

The kernel installs the pinned wheel and runs the same trainer used locally
(`kaggle/run.py`, stages: smoke / train / resume_test / eval / baselines).
Checkpoints are resumable: weights + optimizer + scheduler + scaler + RNG.

## Decide

```bash
echo '{"state": "I was charged twice", "options": [{"id": "a", "description": "..."}, {"id": "b", "description": "..."}]}' \
  | python -m mokka_decisions.cli decide --model-dir release/mokka-decisions-v0.1.0
```

## Evaluation

Protocol: `docs/evaluation-protocol.md` (frozen before the test pass).
Metrics: forced accuracy, macro-F1, NLL, Brier, ECE, selective risk/coverage,
OOS false acceptance, option-permutation flip rate, latency; bootstrap CIs by
semantic group. Per-language reporting — a weak language cannot hide in the
average.

## Documentation

- `docs/dataset-card.md` — provenance, construction, leakage guards
- `docs/evaluation-protocol.md` — metrics, comparators, success criteria
- `docs/model-card.md` — what was trained, measured results, limitations
- `docs/runbook-kaggle.md` — the free-tier GPU operating recipe

## License

Apache-2.0 for code. Datasets and third-party models keep their own licenses
(`THIRD_PARTY_NOTICES.md`).
