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

| Set | Backend | Forced acc | Macro-F1 | NLL | ECE | Coverage | Risk | Status |
|---|---|---|---|---|---|---|---|---|
| banking77_test_candidates | — | — | — | — | — | — | — | not_run |
| banking77_test_full | — | — | — | — | — | — | — | not_run |
| massive_test_candidates (en) | — | — | — | — | — | — | — | not_run |
| massive_test_candidates (de) | — | — | — | — | — | — | — | not_run |
| massive_test_candidates (es) | — | — | — | — | — | — | — | not_run |
| clinc_test_candidates | — | — | — | — | — | — | — | not_run |
| clinc_oos_test | — | — | — | — | — | — | — | not_run |
| xprobe_massive_vs_clinc | — | — | — | — | — | — | — | not_run |
| permutation probe | — | flips: — | | | | | | not_run |

Verdict: pending. Artifacts: `eval/rows/*.jsonl`, `eval_results.json`.
