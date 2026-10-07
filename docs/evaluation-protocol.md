# Evaluation protocol

Version: 1.0 (frozen before the final test pass). Defines splits, metrics,
comparator classes and selection criteria. The results table starts empty and
every cell references a produced artifact; anything not yet run stays `not_run`.

## 1. Data and splits

Sources (official splits preserved; see `docs/dataset-card.md` and
`manifests/sources.json` for licenses and revisions):

| Source | Language | Official test kept as | Purpose |
|---|---|---|---|
| BANKING77 | en | final test (3,080) | primary in-domain taxonomy (77 intents) |
| MASSIVE en-US/de-DE/es-ES | en/de/es | final test per locale | multilingual intents (60 intents) |
| CLINC150+OOS | en | final test | unseen intents + out-of-scope |

Split assignment is a pure function of `group_id` (sha256 → uniform), applied
**before** any candidate-set expansion, question augmentation or translation
use. MASSIVE translations share one `group_id` across locales, so a translated
utterance can never cross splits. Ratios over the non-test pool:
train 80 %, dev 8 %, cal_temperature 6 %, cal_policy 6 %.

Guarantees asserted at build time (`python scripts/prepare_data.py`):
group–split disjunction, no identical text across train/test or dev/test,
gold option always present among options, official test pools untouched.

## 2. Decision instances

Three scenario classes — reported separately, never averaged together:

1. **Candidate-set decisions** (primary): K=6 options (gold + sampled
   distractors + `none`), sampled deterministically per case. Training varies
   K∈[4,8] with `none` present in ~50 % of training instances (always when
   the gold is `none`). This is the scenario the model is designed for.
2. **Fixed-taxonomy decisions**: all 77 banking intents + `none` (BANKING77
   official test). Comparison with fixed-classification baselines.
3. **Generalisation probes**:
   - cross-domain OOS: MASSIVE de/es test utterances scored against CLINC
     options + `none` (gold `none`) — descriptions known, utterance domain never
     seen with this schema;
   - CLINC OOS test (gold `none` with CLINC distractors);
   - option-order permutation: same case, permuted options; a *flip* is a
     changed argmax.

## 3. Metrics (fixed definitions)

- **forced accuracy** — argmax vs gold, abstention ignored;
- **macro-F1** — over option ids present in the eval set (forced);
- **NLL, Brier** — of the gold option's calibrated probability, finite rows only;
- **ECE** — 15 equal-width bins over max-probability;
- **selective risk / coverage** — with the frozen threshold: risk = error rate
  among accepted decisions, coverage = accepted fraction; choosing `none` is an
  accepted decision (correct iff gold is `none`);
- **OOS false acceptance** — on gold=`none` sets: accepted non-`none` decisions / OOS cases;
- **flip rate** — argmax changes under option permutation;
- **bootstrap 95 % CI** — forced accuracy, resampling whole `group_id`s
  (1,000 resamples, seed 42);
- **latency p50/p95** — per case, batch=1, wall clock inside the backend
  (tokenisation + scoring + policy inputs); throughput measured separately at
  batch sizes 1/8/32; CUDA-synchronised where applicable; hardware stated.

Model selection: best dev (candidate-set, fixed K=6) accuracy; ties broken by
mean NLL. Temperature fitted on `cal_temperature` only. Threshold picked on
`cal_policy` only (smallest threshold meeting target selective risk 5 %,
grid 0.30–0.995 step 0.005). If the target is unreachable, the safest
threshold ships with `met_target=false` and the release is downgraded.

## 4. Comparator classes

| Backend | Class | Notes |
|---|---|---|
| TF-IDF + logistic regression | supervised, fixed taxonomy | restricted to present options; cannot choose `none` (structural, reported) |
| **mokka OptionScorer (ours)** | supervised, option-set conditioned | mmBERT encoder (pretrained, third-party) + our trained head, jointly fine-tuned |
| Laya multilingual (zero-shot) | option-set conditioned | same option texts as ours; laya probabilities, our threshold calibration |
| GLiNER2.5 multi (zero-shot) | option-set conditioned | independent per-label confidences; defined order-preserving renormalisation (raw scores kept) |
| laya-triage fine-tuned (banking77) | supervised, fixed taxonomy, other project | only in the banking fixed-taxonomy scenario |
| Jeff, Clef-Flash | not_run unless access exists | 0.8 B / 9 B; require resources not yet available under the 0 € budget |

All backends run through the same runner: one row per case with probabilities,
candidate, latency and explicit error. Failed cases stay in the denominator.

## 5. Success criteria (declared before the test pass)

- accepted-decision error ≤ 5 % with coverage ≥ 60 % in candidate-set scenarios
  (banking + massive en/de/es), reported per language — a failed language
  cannot hide in the average;
- OOS false acceptance ≤ 5 % on both OOS sets;
- permutation flip rate ≤ 2 %;
- practical improvement over the best small comparator: +10 pp coverage at the
  same selective risk, or materially lower latency at equal quality;
- otherwise the release is marked `experimental` with the shortfall stated.

## 6. Results

Pilot pass (pilot-budget data, pilot model, pilot eval budgets — the test sets
were opened by this run; the full-model pass below repeats them and both are
reported). Artifacts: `runs/local-baselines/` (tfidf), `runs/kaggle-eval/`
(mokka pilot).

| Set | Backend | Forced acc | Macro-F1 | NLL | ECE | Coverage | Risk | OOS false acc. | Status |
|---|---|---|---|---|---|---|---|---|---|
| banking77_test_candidates (n=600) | tfidf_lr | 0.987 | 0.986 | — | — | — | — | n/a | run |
| banking77_test_candidates (n=600) | mokka pilot | 0.970 | 0.969 | 0.086 | 0.011 | 1.000 | 0.030 | n/a | run |
| banking77_test_full (n=600) | tfidf_lr | 0.865 | 0.865 | — | — | — | — | n/a | run |
| banking77_test_full (n=600) | mokka pilot | 0.705 | 0.692 | 0.988 | 0.035 | 0.965 | 0.278 | n/a | run |
| massive_test_candidates (n=900) | tfidf_lr | 0.960 | 0.940 | — | — | — | — | n/a | run |
| massive_test_candidates (n=900) | mokka pilot | 0.954 | 0.924 | 0.113 | 0.015 | 1.000 | 0.046 | n/a | run |
| clinc_test_candidates (n=500) | tfidf_lr | 0.988 | 0.988 | — | — | — | — | n/a | run |
| clinc_test_candidates (n=500) | mokka pilot | 0.958 | 0.954 | 0.138 | 0.019 | 1.000 | 0.042 | n/a | run |
| clinc_oos_test (n=300) | tfidf_lr | 0.000 | 0.000 | — | — | — | — | 1.000 (always picks an option) | run |
| clinc_oos_test (n=300) | mokka pilot | 0.937 | — | 0.206 | 0.042 | 1.000 | 0.063 | 0.063 | run |
| xprobe_massive_vs_clinc (n=300) | tfidf_lr | 0.000 | 0.000 | — | — | — | — | 1.000 | run |
| xprobe_massive_vs_clinc (n=300) | mokka pilot | 0.797 | — | 0.844 | 0.141 | 0.997 | 0.204 | 0.203 | run |
| permutation probe (n=500) | mokka pilot | 0 flips | | | | | | | run |

Calibration (pilot): temperature 1.763 (cal_temperature n=4,088), threshold
0.300 on cal_policy (n=4,280) → risk 3.86 % at 99.86 % coverage (target met).
Pilot temperature was fitted on pilot-budget calibration splits.

Laya / GLiNER2.5 rows: pending (kernel `baselines` running).
Full-model pass (train.yaml budgets): pending.

Reading so far, honestly stated:

- candidate-set decisions: our model is 1–3 pp behind TF-IDF+LR with labels —
  expected for a scorer vs a fixed classifier on its home turf;
- the model's differentiator is real and measured: OOS `none`-selection 93.7 %
  where TF-IDF structurally scores 0, and perfect option-order invariance
  (0/500 flips);
- the 77-way fixed-taxonomy scenario is the model's weakest (70.5 % vs 86.5 %):
  training with K=4–8 candidate sets transfers imperfectly to 78-option
  decisions — an honest limitation, improvable with full-taxonomy training
  instances;
- cross-domain OOS (xprobe) false acceptance 20.3 % is above the 5 % target —
  the pilot model under-abstains on out-of-domain text in de/es.

Verdict: pending full-model pass + comparators.
