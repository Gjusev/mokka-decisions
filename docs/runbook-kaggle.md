# Runbook — Kaggle jobs

Operational recipe for the free-tier GPU jobs. Commands run from the repo root
with the project venv; authentication comes from the environment
(`KAGGLE_API_TOKEN`), never from files in the repo.

## 1. Build the bundle (wheel + configs + prepared data)

```powershell
.\.venv\Scripts\python.exe -m pip install --quiet build
.\.venv\Scripts\python.exe -m build --wheel --outdir kaggle-bundle\dist
.\.venv\Scripts\python.exe scripts\prepare_kaggle.py --config configs\train.yaml --bundle kaggle-bundle
```

`scripts/prepare_kaggle.py` assembles the private dataset folder:

```
kaggle-bundle/
  BUNDLE_MANIFEST.json     sha256 of every file + config + data hash
  dist/mokka_decisions-*.whl
  configs/{train,pilot,evaluation,release}.yaml
  data/processed/v1/**     instances, eval sets, manifests
```

## 2. Create/version the bundle dataset (private)

First time: put a `dataset-metadata.json` inside the folder (see script), then

```powershell
.\.venv\Scripts\python.exe -m kaggle datasets create -p kaggle-bundle --dir-mode zip
```

Later versions:

```powershell
.\.venv\Scripts\python.exe -m kaggle datasets version -p kaggle-bundle --dir-mode zip -m "data vN"
```

## 3. Push the kernel (this starts the job — never use push to poll)

```powershell
.\.venv\Scripts\python.exe -m kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
```

`kaggle/kernel-metadata.json` pins `code_file: run.py`, private, GPU, internet,
T4. The stage is selected by editing the `--stage` default in run.py (or the
pushed file) — stages: `smoke`, `train`, `resume_test`, `eval`, `baselines`.

## 4. Poll status (moderate intervals), then collect

```powershell
.\.venv\Scripts\python.exe -m kaggle kernels status gjusev/mokka-decisions-pilot
.\.venv\Scripts\python.exe -m kaggle kernels output gjusev/mokka-decisions-pilot -p runs/kaggle-pilot
```

Verify before trusting: `run_status.json` present and the expected stage name,
`final_global_step` > 0 for training, checkpoint file exists, metrics files
non-empty. A successful push is *not* a completed run.

## 5. Transporting checkpoints between jobs

Preferred mechanism (per operational notes): private versioned datasets, not
`kernel_sources`.

```powershell
# after collecting runs/kaggle-pilot/train/checkpoint_latest.pt
.\.venv\Scripts\python.exe scripts\prepare_kaggle.py --checkpoint runs\kaggle-pilot\train\checkpoint_latest.pt --bundle kaggle-checkpoint
.\.venv\Scripts\python.exe -m kaggle datasets version -p kaggle-checkpoint --dir-mode zip -m "checkpoint step N"
```

Attach `gjusev/mokka-decisions-checkpoint` in `dataset_sources` of the next
job; `run.py --stage train` auto-discovers and resumes from it.

## 6. Quota hygiene

- Check remaining GPU quota before each push (`kaggle quota` or the MCP
  `get_accelerator_quota` tool); T4x2 counts double speed.
- Segments target < 8–9 h; the trainer checkpoints every 30 min
  (`save_every_minutes`).
- If the job dies: collect output, upload `checkpoint_latest.pt` as a new
  dataset version, push again — `--stage train` resumes.
