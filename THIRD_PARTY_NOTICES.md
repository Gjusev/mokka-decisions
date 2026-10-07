# Third-party notices

This project builds on open models and datasets from third parties. Their
licenses and attributions apply to their artifacts independently of this
repository's code license.

## Models

| Artifact | License | Use here |
|---|---|---|
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (JHU CLSP) | MIT (per model card) | pretrained encoder; fine-tuned by this project with a new decision head |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | Apache-2.0 | zero-shot comparator (not modified) |
| [gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) | Apache-2.0 | zero-shot comparator via [GLiNER2](https://github.com/fastino-ai/GLiNER2) (not modified) |

The trained weights produced by this project fine-tune mmBERT-base; the model
card states exactly what was inherited and what was trained.

## Datasets

| Dataset | License | Use |
|---|---|---|
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI) | CC BY 4.0 | banking intent taxonomy (EN) |
| [MASSIVE 1.0](https://github.com/alexa/massive) (Amazon) | CC BY 4.0 | multilingual intents (EN/DE/ES) |
| [CLINC150/OOS](https://huggingface.co/datasets/clinc/clinc_oos) | CC BY 3.0 | unseen intents + out-of-scope |

## Software

Python packages used at build/train/eval time (pinned versions in the run
logs): torch, transformers, datasets, safetensors, scikit-learn, pyarrow,
pyyaml, pytest, gliner2, laya, kaggle CLI. The `laya-triage` sibling project
informed the Kaggle operating recipe (its own repository was not modified).
