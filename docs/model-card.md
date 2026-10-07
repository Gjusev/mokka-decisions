# Model card — mokka-decisions OptionScorer v0.1.0

Release: `release/mokka-decisions-v0.1.0` (self-contained; offline-verified in a
fresh process with an empty HF cache and `HF_HUB_OFFLINE=1`).

## Model

- **Architecture**: option scorer over a multilingual encoder. Every option of
  a decision is encoded as the pair `(state, question + " Option: " +
  description)` with **shared** encoder and head weights; a scalar score per
  option; softmax restricted to the valid options. No per-option parameters,
  no label-id embeddings — the model scores option sets it never saw.
- **Encoder (inherited)**: `jhu-clsp/mmBERT-base` (307M params, ModernBERT),
  pretrained by JHU CLSP; **fine-tuned by this project** jointly with the new
  head (1,575 steps, 3 epochs, best epoch by dev).
- **Trained by this project**: scoring head, objective, candidate-set data
  construction, temperature calibration (T = 2.349 on 4,088 held-out cases),
  abstention policy (threshold 0.300 on 4,280 held-out cases).
- **Not claimed**: pretraining from scratch; general decision ability beyond
  the evaluated scenarios; typed decisions (see benchmark note below).

## Training data

BANKING77 (EN) + MASSIVE 1.0 (EN/DE/ES) + CLINC150/OOS (EN) as candidate-set
decisions with a shared `none` option; official test pools untouched; splits
by semantic group before expansion (leakage guards active; 7 exact train/test
duplicates in BANKING77 excluded from the train side). Full provenance:
`docs/dataset-card.md`, manifest hashes in the release.

Recipe: `configs/train.yaml` (16,784 instances, eff. batch 32, LR 2e-5/1e-4,
fp16, seed 42); run report and per-epoch history in `runs/**/run_report.json`.

## Measured results (official test, full budgets, metrics v2)

Candidate sets contain the gold option **by construction** (selection, not
recall). Fixed threshold policy fitted on calibration data only.

| Scenario | n | Forced acc | Macro-F1 | NLL | Brier(multiclass) | ECE | Cov@thr | Risk@thr |
|---|---|---|---|---|---|---|---|---|
| banking candidates (K=6) | 1,000 | **0.986** | 0.985 | 0.049 | 0.023 | 0.010 | 1.000 | 0.014 |
| massive candidates (en/de/es) | 1,500 | **0.969** | 0.937 | 0.095 | 0.049 | 0.011 | 0.999 | 0.031 |
| clinc candidates (K=6) | 800 | **0.978** | 0.975 | 0.067 | 0.034 | 0.005 | 1.000 | 0.023 |
| banking full 77-way | 1,000 | 0.812 | 0.808 | 0.633 | 0.271 | 0.028 | 0.972 | 0.174 |
| clinc OOS (gold none) | 400 | 0.933 | — | 0.252 | 0.116 | 0.036 | 1.000 | 0.068 |
| cross-domain probe (de/es) | 400 | 0.785 | — | 0.861 | 0.389 | 0.150 | 0.968 | 0.215 |

Per language (massive candidates): de 0.967 / en 0.968 / es 0.971 — no weak
language hidden in the average. Option-order permutation: **0 flips / 500**.
Latency (T4, batch=1, K=6): p50 22–26 ms; 78-option decisions: p50 318 ms.

Comparators on identical IDs/options: TF-IDF+LR (closed-world) banking
candidates 0.984 / full-77 0.872; it cannot choose `none` by construction
(`tfidf_lr_none` variant: see evaluation protocol). Laya zero-shot: banking
candidates 0.725, clinc OOS 0.985, ECE up to 0.51. GLiNER2.5: pending final
run (isolated env). Rows and per-backend policies: `runs/final-report/`.

## Typed decisions (Julia-1 benchmark) — honest labelling

- Julia-1 reproduced locally with its official script (isolated venv, CPU):
  **0.7255** (1,451/2,000) vs published 0.7315.
- Our v0.1 zero-shot under the frozen adapter protocol: see
  `docs/typed-decisions-protocol.md` and `runs/typed-decisions/` — a
  **different adaptation budget** (never trained on this format), reported as
  `zero_shot_domain_shift`, never as a like-for-like defeat or victory.
  Adapted arms (equal data) are the next experiment.

## Calibration & abstention

Temperature on `cal_temperature` only; threshold on `cal_policy` only. The
5 % selective-risk target is met on calibration data (2.37 %) and holds on
candidate-set test scenarios (1.4–3.1 %); it does **not** hold on the
78-option scenario (17.4 %) or the cross-domain probe (21.5 %) — those are
outside the declared scope. Choosing `none` is a decision; abstaining returns
`choice=null` + reason code.

## Known limitations

- Option descriptions are English across languages (schema-described-once
  scenario); cross-lingual transfer measured, not assumed.
- CLINC option texts are humanized intent names, not authored glosses.
- Cross-domain OOS false acceptance 20.8 % (6.5 % of probe labels audited as
  ambiguous by exact-name collision).
- 77-way full-taxonomy decisions degrade vs fixed classifiers trained on the
  same labels (0.812 vs 0.872).
- Not evaluated on boolean/ordinal typed-decision formats in this release.

## Verdict

**`qualified_for_declared_scope`** — declared scope: candidate-set decisions
(2–12 described options, gold present) on banking and assistant intent
domains, EN/DE/ES, with calibrated probabilities and permutation-invariant
scoring; `none`-selection works on in-distribution OOS (93.3 %) but
out-of-domain abstention is **not** within the validated risk budget
(`experimental` for xprobe/77-way scenarios).

## License

Apache-2.0 (code and trained weights of this release). Datasets and the
inherited encoder keep their own licenses (`THIRD_PARTY_NOTICES.md`).
