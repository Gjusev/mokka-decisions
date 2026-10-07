# Dataset card — mokka-decisions v1

Built by `scripts/prepare_data.py`; deterministic from the pinned sources and
config. This card documents provenance, licensing and construction; see
`manifests/sources.json` and `manifests/splits.json` in the build output for
machine-readable versions.

## Sources

| Source | Version/URL | License | Use |
|---|---|---|---|
| BANKING77 | PolyAI-LDN/task-specific-datasets `master` CSVs (same URLs the official HF script downloads) | CC BY 4.0 | primary EN taxonomy (77 intents), official test kept |
| MASSIVE 1.0 | `amazon-massive-dataset-1.0.tar.gz` (Amazon S3, official release) | CC BY 4.0 | EN/DE/ES intents (60 intents), aligned ids across locales |
| CLINC150+OOS | HF `clinc/clinc_oos` parquet conversion, `plus` config | CC BY 3.0 | unseen intents + out-of-scope requests |

Sources keep their own licenses and attributions independently of this
repository's code license. Downloaded files are cached in `data/raw/`
(gitignored); only manifests, statistics and hashes are committed.

## Construction

1. Each source is normalised to one row per utterance with its **official**
   split preserved. Official test pools become our final test, untouched.
2. Split assignment for non-test rows is a pure function of `group_id`
   (sha256 → uniform; train 80 / dev 8 / cal_temperature 6 / cal_policy 6).
   MASSIVE rows across locales share a `group_id`, so translations of one
   utterance always share a split.
3. Only after splitting are decision instances expanded: K candidate options
   (gold + seeded distractors + optional `none`), option order permuted with a
   seeded RNG. All randomness derives from (group_id, split, salt) — the build
   is reproducible bit-for-bit.
4. Training instances: K∈[4,8], `none` present in ~50 % (always for OOS gold).
   Eval instances: fixed K=6 (+ full-taxonomy variants and cross-domain probes,
   see the evaluation protocol).

## Option descriptions

Authored for this project (English, one line), keyed by the canonical label
spellings of each dataset:

- BANKING77: 77 authored glosses (`taxonomies.BANKING77_DESCRIPTIONS`). The
  two canonical quirks (`Refund_not_showing_up`, `reverted_card_payment?`) are
  preserved exactly as the dataset spells them.
- MASSIVE: 60 authored glosses (`taxonomies.MASSIVE_DESCRIPTIONS`).
- CLINC150: humanized intent names (`transfer_money` → `Transfer money`),
  not authored glosses — CLINC serves as the unseen-domain/OOS probe, not a
  headline domain. This is a documented quality difference in option text.
- `none`: one shared description ("None of the listed options applies to this
  request.").

Descriptions are English for all languages: the deployed scenario is
"schema described once, requests in any language". Cross-lingual transfer is
measured, not assumed.

## Label origins

All labels are the public datasets' gold labels (`label_origin:
public-dataset`). The OOS rows are CLINC's own out-of-scope annotations.
Nothing is model-generated; no synthetic paraphrases or translations were
added in v1.

## Leakage guards (asserted at build time)

- group–split disjunction across all sources;
- no identical normalised text across train/test or dev/test;
- gold always present among the expanded option set;
- per-locale MASSIVE splits consistent through the shared group id.

## Statistics

See `manifests/splits.json` in the build output (counts per source/split/
language are printed and stored there on every build).
