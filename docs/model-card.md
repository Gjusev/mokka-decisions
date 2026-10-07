# Model card — mokka-decisions OptionScorer v0.1

Status: **draft** — every metric cell below is filled only from produced
artifacts (`eval/rows/*.jsonl`, `eval_results.json`); nothing is projected.

## Model

- **Architecture**: option scorer over a multilingual encoder. Each option is
  encoded as `(state, question + option description)` with **shared** encoder
  and head weights; a scalar score per option; softmax restricted to the valid
  options of the decision (padding masked). No per-option parameters, no
  label-id embeddings — the model scores option sets it never saw in training.
- **Encoder (inherited)**: `jhu-clsp/mmBERT-base` (307M params), pretrained by
  JHU CLSP. **Fine-tuned by this project** jointly with the new head.
- **Trained by this project**: the scoring head, the training objective and
  recipe, the candidate-set data construction, temperature calibration,
  abstention policy.
- **Not claimed**: pretraining from scratch; general decision ability beyond
  the evaluated scenarios; calibrated urgency/scores without labels.

## Training data

BANKING77 (EN) + MASSIVE 1.0 (EN/DE/ES) + CLINC150/OOS (EN), converted to
candidate-set decisions with a shared `none` option; official test splits
untouched. Full provenance and construction: `docs/dataset-card.md`,
`manifests/sources.json` (licenses per source).

Recipe: see `configs/train.yaml` and the run report (`runs/**/run_report.json`)
for the exact hyperparameters, steps and data hash of the shipped checkpoint.

## Intended use

Choosing among 2–12 described options for a request (EN/DE/ES evaluated),
returning calibrated probabilities and abstaining when confidence is below a
validated threshold. Useful for triage/routing with a human fallback.

## Metrics

(filled after the frozen test pass; see `docs/evaluation-protocol.md`)

| Set | Forced acc | Macro-F1 | NLL | ECE | Coverage@risk≤5% | OOS false acc. | Flip rate |
|---|---|---|---|---|---|---|---|
| banking77 candidates (test) | — | — | — | — | — | — | — |
| banking77 full 77-way (test) | — | — | — | — | — | — | — |
| massive candidates en/de/es (test) | — | — | — | — | — | — | — |
| clinc candidates (test) | — | — | — | — | — | — | — |
| clinc OOS (test) | — | — | — | — | — | — | — |
| cross-domain probe | — | — | — | — | — | — | — |

Latency/memory: measured on the stated hardware in the final report.

## Calibration & abstention

Temperature fitted on `cal_temperature` only; threshold on `cal_policy` only
(target selective risk 5 %). Choosing `none` is a decision; abstaining returns
`choice=null` with a reason code.

## Known limitations

- Option descriptions are English for all languages (schema-described-once
  scenario); cross-lingual transfer is measured, not assumed.
- CLINC option texts are humanized intent names, not authored glosses.
- Trained taxonomies dominate: the model has not been evaluated on
  domains beyond banking/assistant intents.
- TF-IDF+LR is a strong comparator when labels exist; this model's edge must
  come from `none`/OOS handling, cross-lingual transfer and unseen options —
  measured, not asserted.

## Verdict

`pending` — `qualified_for_declared_scope` or `experimental` after the final
test pass, per the protocol's success criteria.
